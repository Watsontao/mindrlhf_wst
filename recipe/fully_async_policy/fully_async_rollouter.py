# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
import time
import json
import re
from pprint import pformat
from collections import defaultdict
from typing import Any, Optional, List, Dict, Union

import numpy as np
import ray
import torch
from ray import ObjectRef

from verl import DataProto
from recipe.fully_async_policy.detach_utils import (
    RolloutSample,
    ValidateMetrics,
    prepare_single_generation_data,
)
from recipe.fully_async_policy.message_queue import MessageQueueClient
from recipe.fully_async_policy.ray_trainer import FullyAsyncRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.profiler import marked_timer
from verl.utils.tracking import ValidationGenerationsLogger
import copy

class MockWorkerGroup:
    """Mock WorkerGroup to wrap sliced server handles."""
    def __init__(self, world_size, workers):
        self.world_size = world_size
        self.workers = workers


class ManagerRoutingView:
    """
    轻量级路由视图：包装一组专属的 agent_loop_workers，
    提供与 FullyAsyncAgentLoopManager 兼容的 generate_single_sample_async 接口。
    每组 workers 只绑定特定的 server_handles（即特定的 GPU replicas），
    从而实现 Heavy/Fast 的物理隔离。
    """

    def __init__(self, underlying_manager, workers, pool_name: str):
        self.underlying_manager = underlying_manager
        self.agent_loop_workers = workers
        self.pool_name = pool_name
        self._worker_index = 0

    async def generate_single_sample_async(self, sample, partial_output_list=None):
        """选择一个 worker（round-robin），发送生成请求"""
        worker = self._select_best_worker()
        output_future = worker.generate_sequences_no_post.remote(sample, partial_output_list)
        return await asyncio.wrap_future(output_future.future())

    def _select_best_worker(self):
        """Round-robin 选择 worker"""
        worker = self.agent_loop_workers[self._worker_index]
        self._worker_index = (self._worker_index + 1) % len(self.agent_loop_workers)
        return worker

    # 以下方法委托给底层 Manager
    async def cancel(self):
        worker_cancel_tasks = [w.cancel_agent_loops.remote() for w in self.agent_loop_workers]
        await asyncio.gather(*worker_cancel_tasks)

    async def resume(self):
        worker_resume_tasks = [w.resume_agent_loops.remote() for w in self.agent_loop_workers]
        await asyncio.gather(*worker_resume_tasks)

    async def wake_up(self):
        await self.underlying_manager.wake_up()

    async def sleep(self):
        await self.underlying_manager.sleep()

    async def clear_kv_cache(self):
        await self.underlying_manager.clear_kv_cache()

@ray.remote(num_cpus=10, max_concurrency=100)
class FullyAsyncRollouter(FullyAsyncRayPPOTrainer):
    """
    Asynchronous sample generator, responsible for continuously generating training samples
    and putting them into MessageQueue
    Based on the mature implementation improvements of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        device_name=None,
    ):
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        self.val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine
        assert self.config.data.train_batch_size == 0, "train_batch_size must be zero"
        assert self.config.data.gen_batch_size == 1, "gen_batch_size must be one"
        assert self.config.async_training.staleness_threshold >= 0, "staleness_threshold must larger than 0"
        assert self.config.async_training.trigger_parameter_sync_step >= 1, (
            "trigger_parameter_sync_step must larger than 1"
        )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = False
        self.kl_ctrl_in_reward = False
        self.use_critic = False
        self.use_reference_policy = False
        self.use_rm = False

        print("[FullyAsyncRollouter] Creating datasets...")
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        self._validate_config()
        print(f"[FullyAsyncRollouter] Rollouter _create_dataloader...\n{train_dataset}\n{val_dataset}")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # ==================== fully async config ====================

        self.total_rollout_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.rollout.total_rollout_steps is not None:
            self.total_rollout_steps = min(self.config.rollout.total_rollout_steps, self.total_rollout_steps)
        print(f"[FullyAsyncRollouter] Total rollout steps: {self.total_rollout_steps}")
        self.total_train_steps = None

        # Rollouter parameter configuration
        self.message_queue_client = None

        # Worker groups: rollout_wg is same to actor_rollout_wg
        self.rollout_wg = None
        self.actor_rollout_wg = None
        self.async_rollout_manager = None

        # Config
        self.staleness_threshold: float = config.async_training.get("staleness_threshold", 1)
        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        self.max_required_samples = None
        self.max_concurrent_samples = None
        # queue size
        self.max_queue_size = None

        # Statistics
        self.current_param_version = 0
        self.total_generated_samples = 0
        self.staleness_samples = 0
        self.dropped_stale_samples = 0
        self.processed_sample_count = 0
        # we start from step 1
        self.global_steps = 1
        self.idle_start_time = None
        self.version_start_time = None

        # Concurrency control
        # Modified by self.pause() or self._should_pause_generation()
        self.paused = False
        self.running = True
        self.monitor_loop_trigger = True

        # Add dataloader lock
        self.dataloader_lock = asyncio.Lock()

        # Statistics for long-tail analysis
        self.total_long_probes = 0
        self.total_long_matches_sum = 0.0 # Sum of ratios (count_long_rest / (n-1))
        self.long_tail_threshold = 512 # Define long-tail as > 512 tokens

        # P-DSR State
        self.sample_buffer = {}
        
        # Batch aggregator: collects probe results, sorts by length, dispatches
        self.batch_aggregator = {}  # batch_id -> list of (sample_id, probe_ret, probe_len)
        self.fake_batch_size = 64  # How many probes to aggregate before flushing

        # Initialize async queues
        # Initialize async queues
        self.probe_queue = asyncio.Queue(maxsize=0)
        self.fast_re_queue = asyncio.Queue(maxsize=0)
        self.heavy_re_queue = asyncio.Queue(maxsize=0)
        self.active_tasks = set()
        self.cancel_queue = asyncio.Queue()

    def _init_async_objects(self):
        # Initialize asyncio synchronization primitives.
        # We let asyncio.Condition create the Lock internally to ensure they share the same Event Loop.
        # This avoids 'ValueError: loop argument must agree with lock' which can occur in Ray environments
        # where the lock's captured loop (get_running_loop) differs from Condition's default loop check.
        # Explicitly passing the loop is deprecated/removed in Python 3.10+, so this reverse-initialization
        # is the most robust workaround.
        self.condition = asyncio.Condition()
        self.lock = self.condition._lock

    async def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        async with self.lock:
            self.message_queue_client = message_queue_client

    async def set_max_required_samples(self):
        async with self.lock:
            self.max_required_samples = int(
                self.required_samples
                * (self.staleness_threshold + 1)
                * self.config.async_training.trigger_parameter_sync_step
            )
            self.total_train_steps = int(
                self.total_rollout_steps
                / (self.required_samples * self.config.async_training.trigger_parameter_sync_step)
            )

            # Cap max concurrent based on total handles across both managers
            total_handles = len(self.rollout_wg.workers)
            self.max_concurrent_samples = total_handles * 128
            self.max_concurrent_samples = min(self.max_concurrent_samples, self.max_required_samples)
            self.max_queue_size = self.max_required_samples

            print(
                f"[FullyAsyncRollouter] required_samples : {self.required_samples} "
                f"max_required_samples: {self.max_required_samples} "
                f"max_queue_size: {self.max_queue_size} "
                f"total_train_steps: {self.total_train_steps} "
                f"total_rollout_steps: {self.total_rollout_steps} "
                f"max_concurrent_samples: {self.max_concurrent_samples} "
            )

    def get_rollout_wg(self):
        """Get rollout worker group"""
        return self.rollout_wg

    def get_max_queue_size(self):
        return self.max_queue_size

    def get_total_train_steps(self):
        return self.total_train_steps

    async def update_param_version(self, version: int, validate: bool = False, global_steps: int = 0):
        """Update current parameter version.

        NOTE: This method must return quickly so that resume() (which depends on
        this method's ObjectRef) can fire promptly. Validation is spawned as a
        background asyncio task to avoid blocking the return.
        """
        async with self.lock:
            old_version = self.current_param_version
            self.current_param_version = version
            # every time param change, reset staleness_samples
            self.staleness_samples = (
                len(self.active_tasks) + self.cancel_queue.qsize() + await self.message_queue_client.get_queue_size()
            )
            timing_raw = {}
            idle_ratio = None
            if self.idle_start_time is not None and self.version_start_time is not None:
                rollout_active_time = self.idle_start_time - self.version_start_time
                rollout_version_time = time.time() - self.version_start_time
                idle_ratio = 1 - rollout_active_time / rollout_version_time
                timing_raw["rollouter/active_time"] = rollout_active_time
                timing_raw["rollouter/version_time"] = rollout_version_time
                timing_raw["rollouter/idle_ratio"] = idle_ratio
                self.idle_start_time = None
            print(
                f"[FullyAsyncRollouter][Public][update_param_version] "
                f"Parameter version updated from {old_version} to {version} "
                f",reset staleness_samples to: {self.staleness_samples}"
                f",idle_ratio: {idle_ratio}"
            )
            self.version_start_time = time.time()

        # Validation runs OUTSIDE the lock and as a background task,
        # so it does NOT block this method from returning.
        # This is critical: resume.remote() depends on this method's ObjectRef,
        # if we block here, resume never fires → deadlock.
        need_validate = (
            self.val_reward_fn is not None
            and self.config.rollout.test_freq > 0
            and version % self.config.rollout.test_freq == 0
            and version > 0
        ) or (validate and self.val_reward_fn is not None)

        if need_validate:
            asyncio.create_task(self._run_validation_background(timing_raw, global_steps, version))
        else:
            data = ValidateMetrics(
                timing_raw=timing_raw, metrics=None, global_steps=global_steps, param_version=version
            )
            await self.message_queue_client.put_validate(ray.cloudpickle.dumps(data))

    async def _run_validation_background(self, timing_raw: dict, global_steps: int, version: int):
        """Run validation in background so it doesn't block resume()."""
        try:
            print(f"[FullyAsyncRollouter][Validate] Starting validation for version {version}...")
            loop = asyncio.get_event_loop()
            val_metrics = await loop.run_in_executor(None, self._validate)
            print(f"[FullyAsyncRollouter][Validate] Validation done for version {version}")
            with marked_timer("rollouter/validate_time", timing_raw, color="green"):
                pass  # timing already captured by run_in_executor wall time
            data = ValidateMetrics(
                timing_raw=timing_raw, metrics=val_metrics, global_steps=global_steps, param_version=version
            )
            await self.message_queue_client.put_validate(ray.cloudpickle.dumps(data))
        except Exception as e:
            import traceback
            print(f"[FullyAsyncRollouter][Validate] Error during validation: {e}")
            print(traceback.format_exc())
            # Still send metrics even on error so trainer doesn't hang
            data = ValidateMetrics(
                timing_raw=timing_raw, metrics=None, global_steps=global_steps, param_version=version
            )
            await self.message_queue_client.put_validate(ray.cloudpickle.dumps(data))

    async def save_checkpoint(self, local_global_step_folder: str):
        # WARNING!: Due to the asynchronous nature, there are some in-flight samples
        # (pending/cancel/result queue and message queue).
        # Therefore, directly saving the state of the dataloader will result in losing these
        # samples when resuming training.
        # TODO: Implement dataloader recovery without losing in-flight samples.
        from verl.utils.fs import local_mkdir_safe

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        async with self.dataloader_lock:
            dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        print(f"[FullyAsyncRollouter] Saved dataloader checkpoint to {dataloader_local_path}")

    def load_checkpoint(self):
        """Load checkpoint including dataloader state based on resume mode"""

        if self.config.trainer.resume_mode == "disable":
            print("[FullyAsyncRollouter] Resume mode is disabled, starting from scratch")
            return 0

        # Determine checkpoint folder path
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("[FullyAsyncRollouter] Load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)

            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        # Find and validate global_step_folder based on resume mode
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("[FullyAsyncRollouter] Training from scratch (no checkpoint found)")
                return 0
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), (
                "[FullyAsyncRollouter] resume_from_path must be str type"
            )
            assert "global_step_" in self.config.trainer.resume_from_path, (
                "[FullyAsyncRollouter] resume_from_path must specify the global_steps"
            )
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
        else:
            raise ValueError(f"[FullyAsyncRollouter] Unknown resume_mode: {self.config.trainer.resume_mode}")

        print(f"[FullyAsyncRollouter] Loading checkpoint from: {global_step_folder}")

        # Extract and set global step
        trainer_global_steps = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = (
            trainer_global_steps * self.required_samples * self.config.async_training.trigger_parameter_sync_step + 1
        )
        print(f"[FullyAsyncRollouter] Setting global_steps to {self.global_steps}")

        # Load dataloader state
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
            print(f"[FullyAsyncRollouter] Loaded dataloader state from {dataloader_local_path}")
        else:
            print(
                f"[FullyAsyncRollouter] Warning: No dataloader state found at {dataloader_local_path}, "
                f"will start from scratch"
            )

    def _validate_config(self):
        # Validate asynchronous training configuration
        if not hasattr(self.config, "async_training"):
            raise ValueError("[FullyAsyncRollouter] Missing async_training configuration")
        assert self.config.actor_rollout_ref.rollout.calculate_log_probs, "must rollout calculate log_probs"

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_async_objects()
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        await self._init_async_rollout_manager()

    def _create_actor_rollout_classes(self):
        # only create rollout
        for role in [Role.Rollout]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self):
        self.rollout_wg = self.all_wg[str(Role.Rollout)]
        self.rollout_wg.init_model()
        self.actor_rollout_wg = self.rollout_wg

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.rollout.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    async def _init_async_rollout_manager(self):
        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        from recipe.fully_async_policy.agent_loop import FullyAsyncAgentLoopManager, FullyAsyncAgentLoopWorker

        self.async_rollout_mode = True
        
        # ====================================================================
        # Replica-Level Routing: Physical Isolation of Heavy/Fast Worker Pools
        # ====================================================================
        # 1. 创建一个 Manager（包含所有 replica/vLLM server）
        # 2. 按 replica 维度拆分 server_handles 为 Heavy 和 Fast 两组
        # 3. 为每组创建专属的 agent_loop_workers（只能访问该组的 server）
        # 4. 用 ManagerRoutingView 包装，提供与 Manager 相同的接口
        # ====================================================================
        
        manager = await FullyAsyncAgentLoopManager.create(
            config=self.config,
            worker_group=self.rollout_wg,
        )
        
        # 保存原始 Manager 用于 wake_up/sleep 等全局操作
        self.async_rollout_manager = manager
        
        num_replicas = len(manager.server_handles)
        print(f"[FullyAsyncRollouter] Manager 创建完成，共 {num_replicas} 个 replica")
        
        if num_replicas >= 2:
            # 拆分 replica：1/4 给 Heavy（至少 1 个），其余给 Fast
            num_heavy_replicas = max(1, num_replicas // 4)
            heavy_handles = manager.server_handles[:num_heavy_replicas]
            fast_handles = manager.server_handles[num_heavy_replicas:]
            
            print(f"[FullyAsyncRollouter] 双池模式: "
                  f"Heavy={num_heavy_replicas} replicas, Fast={num_replicas - num_heavy_replicas} replicas")
            
            # 为每个池创建专属的 agent_loop_workers
            num_workers_per_pool = max(1, self.config.actor_rollout_ref.rollout.agent.num_workers // 2)
            
            heavy_workers = self._create_pool_workers(
                FullyAsyncAgentLoopWorker, heavy_handles, num_workers_per_pool, "heavy"
            )
            fast_workers = self._create_pool_workers(
                FullyAsyncAgentLoopWorker, fast_handles, num_workers_per_pool, "fast"
            )
            
            self.heavy_manager = ManagerRoutingView(manager, heavy_workers, "Heavy")
            self.fast_manager = ManagerRoutingView(manager, fast_workers, "Fast")
        else:
            # 只有 1 个 replica，无法拆分，共享同一个 Manager
            print(f"[FullyAsyncRollouter] 单池模式（仅 {num_replicas} 个 replica，需要 ≥2 才能拆分）")
            self.heavy_manager = manager
            self.fast_manager = manager

    def _create_pool_workers(self, worker_class, server_handles, num_workers, pool_name):
        """为指定池创建专属的 agent_loop_workers，只绑定该池的 server_handles"""
        import ray
        workers = []
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            node_id = node_ids[i % len(node_ids)]
            workers.append(
                worker_class.options(
                    name=f"agent_loop_worker_{pool_name}_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, server_handles, self.async_rollout_manager.reward_router_address)
            )
        print(f"[FullyAsyncRollouter] {pool_name} 池: 创建了 {num_workers} 个 agent_loop_workers，"
              f"绑定 {len(server_handles)} 个 server_handles")
        return workers

    # Add samples to the pending_queue
    async def _feed_samples(self):
        continuous_iterator = self._create_continuous_iterator()
        
        sample_global_idx = 0

        for epoch, batch_dict in continuous_iterator:
            # P-DSR: Generate PROBE first (N=1)
            full_batch = prepare_single_generation_data(batch_dict, self.config, override_n=1)

            sample_id = f"sample_{epoch}_{self.global_steps}"
            sample_global_idx += 1
            
            # Cache the raw batch_dict for generating the rest (N-1) later
            self.sample_buffer[sample_id] = batch_dict
            
            # Assign to a fake batch for aggregation
            fake_batch_id = f"batch_{self.global_steps // self.fake_batch_size}"

            rollout_sample = RolloutSample(
                full_batch=full_batch,
                agent_loop_output_list=[None] * 1, # Probe only needs 1 slot
                sample_id=sample_id,
                epoch=epoch,
                param_version=0,
                param_version_start=[],
                param_version_end=[],
                processing_times=[],
                tool_calls=[],
                rollout_status={'is_probe': True, 'priority': 'FAST', 'fake_batch_id': fake_batch_id},
            )

            await self.probe_queue.put(rollout_sample)

            # Check if have reached the last step
            if self.global_steps >= self.total_rollout_steps:
                print(
                    f"[FullyAsyncRollouter][Feed] "
                    f"Maximum count has been reached, stop adding new samples"
                    f"{self.global_steps} >= {self.total_rollout_steps}"
                )
                break

            self.global_steps += 1

        # End signal
        await self.probe_queue.put("DONE")
        print(f"[FullyAsyncRollouter][Feed] Sample addition is complete, {self.global_steps} samples have been added")

    async def _fast_processor_worker(self):
        """
        Streaming worker coroutine for FAST pool, fetching from probe_queue and fast_re_queue
        """
        while True:
            try:
                # 1. 确定下一个任务的来源
                simple_from_cancel_queue = False
                from_re_queue = False
                rollout_sample = None

                # 【核心修改 1】：赋予存量任务“特权豁免”。
                # 无论是否 Pause，只要 cancel_queue 或 fast_re_queue 里有任务，无条件优先执行！
                if not self.cancel_queue.empty():
                    rollout_sample = self.cancel_queue.get_nowait()
                    simple_from_cancel_queue = True
                    if not isinstance(rollout_sample.agent_loop_output_list, list):
                        rollout_sample.agent_loop_output_list = [None] * len(rollout_sample.full_batch)
                elif not self.fast_re_queue.empty():
                    rollout_sample = self.fast_re_queue.get_nowait()
                    from_re_queue = True
                
                # 2. 如果没有存量任务，再去管探针（Probe）和系统状态
                if rollout_sample is None:
                    is_paused = self.paused or await self._should_pause_generation()

                    if is_paused:
                        # Flush aggregator → 把 probe 结果转化为 REST 任务放入 re_queue
                        for b_id in list(self.batch_aggregator.keys()):
                            if self.batch_aggregator.get(b_id):
                                print(f"[FullyAsyncRollouter][FastProcessor] Flushing aggregator batch {b_id}")
                                await self._flush_aggregator_batch(b_id, force=True)

                        # Flush 后 re_queue 可能有新任务，回到循环顶部让步骤 1 优先消费
                        if not self.fast_re_queue.empty() or not self.cancel_queue.empty():
                            continue

                        # re_queue 空了，等 active_tasks 逐步完成（不设 self.paused，不影响 heavy worker）
                        while self.active_tasks:
                            async with self.lock:
                                tasks_to_wait = list(self.active_tasks)
                            if not tasks_to_wait:
                                break
                            done_tasks, _ = await asyncio.wait(
                                tasks_to_wait, return_when=asyncio.FIRST_COMPLETED
                            )
                            async with self.lock:
                                for t in done_tasks:
                                    self.active_tasks.discard(t)
                            for t in done_tasks:
                                await t
                            # task 完成可能产出新的 re_queue 任务，优先去处理
                            if not self.fast_re_queue.empty() or not self.cancel_queue.empty():
                                break

                        # 再检查一次 re_queue
                        if not self.fast_re_queue.empty() or not self.cancel_queue.empty():
                            continue

                        # 全部排空，短暂 sleep 等待 resume 或新任务
                        async with self.lock:
                            self.idle_start_time = time.time()
                            try:
                                await asyncio.wait_for(self.condition.wait(), timeout=0.5)
                            except asyncio.TimeoutError:
                                pass
                        continue
                    
                    # 没拉闸，放心从 probe_queue 拿新探针
                    rollout_sample = await self.probe_queue.get()
                    if rollout_sample != "DONE":
                        self.staleness_samples += 1

                # 3. 处理终止信号
                if rollout_sample == "DONE":
                    print("[FullyAsyncRollouter][FastProcessor] Received termination signal...")
                    while True:
                        async with self.lock:
                            tasks_to_wait = list(self.active_tasks)
                        if not tasks_to_wait:
                            break
                        done_tasks, _ = await asyncio.wait(
                            tasks_to_wait, return_when=asyncio.FIRST_COMPLETED
                        )
                        async with self.lock:
                            for t in done_tasks:
                                self.active_tasks.discard(t)
                        for t in done_tasks:
                            await t
                    break

                # 4. 并发控制（正常等满载的并发跑完）
                while True:
                    async with self.lock:
                        if len(self.active_tasks) < self.max_concurrent_samples:
                            break
                        tasks_to_wait = list(self.active_tasks)
                    if not tasks_to_wait:
                        break
                    done_tasks, _ = await asyncio.wait(
                        tasks_to_wait, return_when=asyncio.FIRST_COMPLETED
                    )
                    async with self.lock:
                        for t in done_tasks:
                            self.active_tasks.discard(t)
                    for t in done_tasks:
                        await t

                # 5. 【核心修改 3】：剥离死锁元凶的 is_probe 和 while True 阻塞网络，直接派发任务！
                async with self.lock:
                    task = asyncio.create_task(
                        self._process_single_sample_streaming(rollout_sample, self.fast_manager, 'FAST'),
                        name=f"fast_{rollout_sample.sample_id}",
                    )
                    self.active_tasks.add(task)

                # 6. 清理队列标志
                if simple_from_cancel_queue:
                    self.cancel_queue.task_done()
                elif from_re_queue:
                    self.fast_re_queue.task_done()
                else:
                    self.probe_queue.task_done()

            except Exception as e:
                import traceback
                print(f"[FullyAsyncRollouter][FastProcessor] crashed: {e}")
                print(traceback.format_exc())
                await asyncio.sleep(1)
                
        # Fast processor 彻底处理完所有的 probe，发送信号让 Heavy 也收工
        await self.heavy_re_queue.put("DONE")


    async def _heavy_processor_worker(self):
        """
        Streaming worker coroutine for HEAVY pool, fetching from heavy_queue and re_queue
        """
        while True:
            try:
                # 1. 确定下一个任务的来源
                simple_from_cancel_queue = False
                from_re_queue = False
                rollout_sample = None

                # 【核心修改 1】：优先消化 cancel 和 heavy_re_queue 的存量任务，无视 Pause！
                if not self.cancel_queue.empty():
                    rollout_sample = self.cancel_queue.get_nowait()
                    simple_from_cancel_queue = True
                    if not isinstance(rollout_sample.agent_loop_output_list, list):
                        rollout_sample.agent_loop_output_list = [None] * len(rollout_sample.full_batch)
                elif not self.heavy_re_queue.empty():
                    rollout_sample = self.heavy_re_queue.get_nowait()
                    from_re_queue = True
                
                # 2. 如果暂时没任务，处理静默等待逻辑
                if rollout_sample is None:
                    is_paused = self.paused or await self._should_pause_generation()

                    if is_paused:
                        # Flush aggregator
                        for b_id in list(self.batch_aggregator.keys()):
                            if self.batch_aggregator.get(b_id):
                                print(f"[FullyAsyncRollouter][HeavyProcessor] Flushing aggregator batch {b_id}")
                                await self._flush_aggregator_batch(b_id, force=True)

                        # Flush 后 re_queue 可能有新任务，回到循环顶部让步骤 1 优先消费
                        if not self.heavy_re_queue.empty() or not self.cancel_queue.empty():
                            continue

                        # re_queue 空了，等 active_tasks 逐步完成
                        while self.active_tasks:
                            async with self.lock:
                                tasks_to_wait = list(self.active_tasks)
                            if not tasks_to_wait:
                                break
                            done_tasks, _ = await asyncio.wait(
                                tasks_to_wait, return_when=asyncio.FIRST_COMPLETED
                            )
                            async with self.lock:
                                for t in done_tasks:
                                    self.active_tasks.discard(t)
                            for t in done_tasks:
                                await t
                            # task 完成可能产出新的 re_queue 任务，优先去处理
                            if not self.heavy_re_queue.empty() or not self.cancel_queue.empty():
                                break

                        # 再检查一次 re_queue
                        if not self.heavy_re_queue.empty() or not self.cancel_queue.empty():
                            continue

                    # should_pause 但所有存量任务已排空，或者不在 pause 但也没任务
                    # 短暂 sleep 等待新任务或 resume
                    async with self.lock:
                        self.idle_start_time = time.time()
                        try:
                            await asyncio.wait_for(self.condition.wait(), timeout=0.5)
                        except asyncio.TimeoutError:
                            pass
                    continue


                # 3. 处理终止信号
                if rollout_sample == "DONE":
                    print("[FullyAsyncRollouter][HeavyProcessor] Received termination signal...")
                    while True:
                        async with self.lock:
                            tasks_to_wait = list(self.active_tasks)
                        if not tasks_to_wait:
                            break
                        done_tasks, _ = await asyncio.wait(
                            tasks_to_wait, return_when=asyncio.FIRST_COMPLETED
                        )
                        async with self.lock:
                            for t in done_tasks:
                                self.active_tasks.discard(t)
                        for t in done_tasks:
                            await t
                    break

                # 4. 并发控制
                while True:
                    async with self.lock:
                        if len(self.active_tasks) < self.max_concurrent_samples:
                            break
                        tasks_to_wait = list(self.active_tasks)
                    if not tasks_to_wait:
                        break
                    done_tasks, _ = await asyncio.wait(
                        tasks_to_wait, return_when=asyncio.FIRST_COMPLETED
                    )
                    async with self.lock:
                        for t in done_tasks:
                            self.active_tasks.discard(t)
                    for t in done_tasks:
                        await t

                # 5. 【核心修改 3】：拿到了存量任务，直接派发！不再受任何全局 Pause 的牵制。
                async with self.lock:
                    task = asyncio.create_task(
                        self._process_single_sample_streaming(rollout_sample, self.heavy_manager, 'HEAVY'),
                        name=f"heavy_{rollout_sample.sample_id}",
                    )
                    self.active_tasks.add(task)

                # 6. 清理队列标志
                if simple_from_cancel_queue:
                    self.cancel_queue.task_done()
                elif from_re_queue:
                    self.heavy_re_queue.task_done()

            except Exception as e:
                import traceback
                print(f"[FullyAsyncRollouter][HeavyProcessor] crashed: {e}")
                print(traceback.format_exc())
                await asyncio.sleep(1)

    async def _flush_aggregator_batch(self, batch_id: str, force=False):
        """Sort probes by length, split into FAST/HEAVY, dispatch REST (N-1) samples."""
        if batch_id not in self.batch_aggregator:
            return
        entries = self.batch_aggregator[batch_id]
        if not entries and not force:
            return

        # Only flush when we have enough entries or when forced
        if not force and len(entries) < self.fake_batch_size:
            return

        # Pop the entries out
        entries = self.batch_aggregator.pop(batch_id, [])
        if not entries:
            return

        # Sort by probe_len descending
        entries.sort(key=lambda x: x[2], reverse=True)

        # Top 20% are HEAVY, rest are FAST
        n_heavy = max(1, len(entries) // 5)
        heavy_entries = entries[:n_heavy]
        fast_entries = entries[n_heavy:]

        print(f"[P-DSR BatchFlush] batch_id={batch_id} | total={len(entries)} | heavy={len(heavy_entries)} | fast={len(fast_entries)}")

        for sample_id, probe_ret, probe_len in heavy_entries:
            await self._dispatch_rest(sample_id, probe_ret, 'HEAVY', cap=None)

        for sample_id, probe_ret, probe_len in fast_entries:
            # FAST cap = median of FAST group * 1.5
            if fast_entries:
                median_len = fast_entries[len(fast_entries) // 2][2]
                cap = int(median_len * 1.5)
            else:
                cap = 2048
            await self._dispatch_rest(sample_id, probe_ret, 'FAST', cap=cap)

    async def _dispatch_rest(self, sample_id: str, probe_ret, priority: str, cap=None):
        """Dispatch REST (N-1) samples for a single sample after probe completes."""
        if sample_id not in self.sample_buffer:
            print(f"[P-DSR] ❌ Error: Buffer missing for sample {sample_id}, skipping dispatch.")
            return

        buf_data = self.sample_buffer[sample_id]
        input_batch = buf_data['input'] if isinstance(buf_data, dict) and 'input' in buf_data else buf_data

        rest_n = self.config.actor_rollout_ref.rollout.n - 1
        if rest_n <= 0:
            print(f"[P-DSR] ℹ️ N=1 mode, no REST dispatch needed for {sample_id}.")
            return

        full_batch_rest = prepare_single_generation_data(input_batch, self.config, override_n=rest_n)
        if cap:
            if not getattr(full_batch_rest, 'meta_info', None):
                full_batch_rest.meta_info = {}
            full_batch_rest.meta_info['max_tokens'] = cap
        else:
            if getattr(full_batch_rest, 'meta_info', None) and 'max_tokens' in full_batch_rest.meta_info:
                del full_batch_rest.meta_info['max_tokens']

        rest_sample = RolloutSample(
            full_batch=full_batch_rest,
            agent_loop_output_list=[None] * rest_n,
            sample_id=sample_id,
            epoch=0,
            param_version=self.current_param_version,
            param_version_start=[],
            param_version_end=[],
            processing_times=[],
            tool_calls=[],
            rollout_status={'priority': priority, 'is_rest': True},
        )
        if priority == 'HEAVY':
            await self.heavy_re_queue.put(rest_sample)
        else:
            await self.fast_re_queue.put(rest_sample)
        async with self.lock:
            self.condition.notify_all()

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample, manager, priority: str):
        """Process a single sample streamingly"""
        print(f"[P-DSR Debug] Start processing {rollout_sample.sample_id}")
        
        # Fix: If retrying from cancel_queue, agent_loop_output_list might be a DataProto. Reset it.
        if not isinstance(rollout_sample.agent_loop_output_list, list):
            rollout_sample.agent_loop_output_list = [None] * len(rollout_sample.full_batch)

        # Calling asynchronous generation methods
        rollout_sample.full_batch.non_tensor_batch["param_version"] = [self.current_param_version] * len(
            rollout_sample.full_batch
        )
        
        # --- P-DSR: Inject Priority into Meta Info ---
        priority = rollout_sample.rollout_status.get('priority', 'HEAVY')
        rollout_sample.full_batch.meta_info['priority'] = priority


        _gen_start = time.time()
        ret, is_cancel = await manager.generate_single_sample_async(
            rollout_sample.full_batch, rollout_sample.agent_loop_output_list
        )
        _gen_time = time.time() - _gen_start
        print(f"[P-DSR DEBUG EXTREME] sample_id={rollout_sample.sample_id}, priority={priority}, is_cancel={is_cancel}, ret={ret}")
        
        
        # --- P-DSR: Circuit Breaker & Retry (熔断重试) ---
        if priority == 'FAST' and is_cancel:
            # 此时 ret 实质上是 list[AgentLoopOutput], 代表被 cap 截断的残差生成
            print(f"[P-DSR DIAG] 🚨 样本 {rollout_sample.sample_id} 触发熔断 Repack！ -> 正在带缓存重回 Heavy 队列...")
            
            # 1. 保存已生成的 AgentLoopOutput 状态
            rollout_sample.agent_loop_output_list = ret
            
            # 2. 升级优先级
            rollout_sample.rollout_status['priority'] = 'HEAVY'
            
            # 3. 清除动态 cap (max_tokens)，否则 HEAVY 任务还会被截断！
            if 'max_tokens' in rollout_sample.full_batch.meta_info:
                del rollout_sample.full_batch.meta_info['max_tokens']
            
            # 4. 重新入队 (使用 heavy_re_queue)
            await self.heavy_re_queue.put(rollout_sample)
            
            # 5. 唤醒 Worker
            async with self.lock:
                self.condition.notify_all()
                print("Worker唤醒完毕!")
                
            return 
        elif is_cancel:
            # For HEAVY, or other unhandled cancellations
            rollout_sample.agent_loop_output_list = ret
            await self.cancel_queue.put(rollout_sample)
            async with self.lock:
                self.condition.notify_all()
            return
        # ------------------------------------------------
        
        # --- P-DSR Logic: Reassembly (Rollouter-Side) ---
        
        # 1. Handle Probe Completion — Batch Aggregator Routing
        is_probe = rollout_sample.rollout_status.get('is_probe', False)
        if is_probe:
            # Get probe generation length
            if 'response_mask' in ret.batch:
                response_mask = ret.batch['response_mask']
                if hasattr(response_mask, 'cpu'): response_mask = response_mask.cpu()
                probe_len = response_mask.sum(dim=-1).item()
            else:
                probe_len = 0

            # Cache probe result in sample_buffer for later reassembly
            sid = rollout_sample.sample_id
            if sid in self.sample_buffer:
                stored_data = self.sample_buffer[sid]
                if isinstance(stored_data, dict) and 'input' not in stored_data:
                    self.sample_buffer[sid] = {'input': stored_data, 'probe_ret': ret}
                else:
                    self.sample_buffer[sid]['probe_ret'] = ret

            # --- Batch Aggregator: Collect probe results ---
            fake_batch_id = rollout_sample.rollout_status.get('fake_batch_id', 'default')
            if fake_batch_id not in self.batch_aggregator:
                self.batch_aggregator[fake_batch_id] = []
            self.batch_aggregator[fake_batch_id].append((sid, ret, probe_len))
            
            print(f"[P-DSR Probe] {sid} | probe_len={probe_len} | batch={fake_batch_id} ({len(self.batch_aggregator[fake_batch_id])}/{self.fake_batch_size})")
            
            # Flush if batch is full
            if len(self.batch_aggregator[fake_batch_id]) >= self.fake_batch_size:
                await self._flush_aggregator_batch(fake_batch_id)
            return

        # 2. Handle Rest Completion (Reassembly)
        is_rest = rollout_sample.rollout_status.get('is_rest', False)
        if is_rest:
            if rollout_sample.sample_id in self.sample_buffer:
                stored_data = self.sample_buffer.pop(rollout_sample.sample_id)
                probe_ret = stored_data['probe_ret']
                
                # Concatenate Probe (1) + Rest (N-1) -> Total (N)
                # DataProto.concat expects a list
                merged_batch = DataProto.concat([probe_ret, ret])
                
                # Update the sample with the merged batch
                rollout_sample.full_batch = merged_batch
                # Update status
                rollout_sample.rollout_status['is_merged'] = True
            else:
                print(f"[P-DSR] ❌ Error: Buffer missing for Rest sample {rollout_sample.sample_id}")
                return
        else:
            # P-DSR: Normal single-pass generation (bypassed Probe/Rest)
            rollout_sample.full_batch = ret

        # ------------------------------------------------------------------
        # [日志统计] 抽离出的独立方法
        # ------------------------------------------------------------------
        await self._log_rollout_metrics(rollout_sample, priority)

        rollout_sample.full_batch.non_tensor_batch["uid"] = np.array(
            [f"uid_{rollout_sample.sample_id}"] * len(rollout_sample.full_batch), dtype=object
        )
        rollout_sample.param_version = self.current_param_version
        rollout_sample.rollout_status = await self.get_statistics()
        rollout_sample.agent_loop_output_list = []

        success = await self.message_queue_client.put_sample(
            sample=ray.cloudpickle.dumps(rollout_sample),
            param_version=rollout_sample.param_version,
        )
        if success:
            self.total_generated_samples += 1
        else:
            self.dropped_stale_samples += 1

        self.processed_sample_count += 1

    async def _log_rollout_metrics(self, rollout_sample, priority):
        """Helper method to log length statistics and decoded responses."""
        try:
            # Use the merged full_batch to get all 8 responses
            if 'response_mask' in rollout_sample.full_batch.batch:
                response_mask = rollout_sample.full_batch.batch['response_mask']
                if hasattr(response_mask, 'cpu'):
                    response_mask = response_mask.cpu()
                lengths = response_mask.sum(dim=-1).tolist()
                
                # 1. Processing times (Note: full_batch might not have processing_times merged perfectly, 
                # but we try to get what we can or pad 0s)
                # For simplicity, we just log 0s if missing, as merging times is complex
                rounded_times = [0.0] * len(lengths)

                # 2. Calculate Conditional Long-tail Probability
                if len(lengths) > 1:
                    is_first_long = lengths[0] > self.long_tail_threshold
                    if is_first_long:
                        self.total_long_probes += 1
                        rest_lengths = lengths[1:]
                        long_rest_count = sum(1 for l in rest_lengths if l > self.long_tail_threshold)
                        match_ratio = long_rest_count / len(rest_lengths)
                        self.total_long_matches_sum += match_ratio
                        
                        avg_prob = (self.total_long_matches_sum / self.total_long_probes) * 100
                        print(f"[P-DSR Analysis] Sample: {rollout_sample.sample_id} | Probe is LONG. "
                              f"Conditional Prob (N-1 also long): {avg_prob:.2f}% (Total Long Probes: {self.total_long_probes})")

                # 3. Decode Text
                try:
                    responses_tensor = rollout_sample.full_batch.batch['responses']
                    if hasattr(responses_tensor, 'cpu'): responses_tensor = responses_tensor.cpu()
                    decoded_texts = self.tokenizer.batch_decode(responses_tensor, skip_special_tokens=True)
                except:
                    decoded_texts = ["<Decode Error>"] * len(lengths)

                # 4. Dynamic Filename
                import re
                import os
                output_dir = os.path.join(os.path.expanduser("~"), "work", "verl_070_fully_async", "Rollout_response", "fully_async")
                os.makedirs(output_dir, exist_ok=True)
                model_path = self.config.actor_rollout_ref.model.path
                model_match = re.search(r'(\d+(?:\.\d+)?b)', model_path.lower())
                model_suffix = model_match.group(1) if model_match else "model"
                filename = os.path.join(output_dir, f"rollout_length_stats_async_{model_suffix}.jsonl")
                filename_text = os.path.join(output_dir, f"rollout_responses_async_{model_suffix}.jsonl")

                log_entry = {
                    "timestamp": time.time(),
                    "param_version": self.current_param_version,
                    "sample_id": rollout_sample.sample_id,
                    "n_responses": len(lengths),
                    "lengths": lengths,
                    "times": rounded_times,
                    "priority": priority
                }
                
                with open(filename, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    
                log_entry_text = {
                    "sample_id": rollout_sample.sample_id,
                    "responses": decoded_texts
                }
                with open(filename_text, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry_text, ensure_ascii=False) + "\n")
                    
        except Exception as e:
            print(f"[FullyAsyncRollouter] Failed to log stats: {e}")

    async def _streaming_generation_main(self):
        """The main entry method for stream processing"""

        if getattr(self, "heavy_manager", None) is None:
            await self._init_async_rollout_manager()

        # Start the streaming loop
        print(f"[FullyAsyncRollouter] Start streaming mode, maximum concurrent samples: {self.max_concurrent_samples}")

        # Start sample feed coroutine, streaming process coroutines
        self.feed_task = asyncio.create_task(self._feed_samples())
        self.fast_processor_task = asyncio.create_task(self._fast_processor_worker())
        self.heavy_processor_task = asyncio.create_task(self._heavy_processor_worker())

        try:
            # Wait for sample feed to complete
            done, pending = await asyncio.wait(
                [self.feed_task, self.fast_processor_task, self.heavy_processor_task], return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                if task.exception():
                    import traceback
                    print(f"[FullyAsyncRollouter] Streaming process exception: {repr(task.exception())}")
                    print(traceback.format_exc())
                    raise task.exception()

            if self.feed_task not in done:
                raise RuntimeError("Processor task exited prematurely")

            print("[FullyAsyncRollouter] Sample feed completed")

            # Wait for streaming to complete
            await asyncio.gather(self.fast_processor_task, self.heavy_processor_task)
            print("[FullyAsyncRollouter] Streaming process completed")

        except Exception as e:
            print(f"[FullyAsyncRollouter] Critical error: {e}")

        finally:
            if hasattr(self, "fast_processor_task") and self.fast_processor_task:
                self.fast_processor_task.cancel()
            if hasattr(self, "heavy_processor_task") and self.heavy_processor_task:
                self.heavy_processor_task.cancel()

            await asyncio.gather(self.fast_processor_task, self.heavy_processor_task, return_exceptions=True)

        # Send a finish signal
        await self.message_queue_client.put_sample(
            sample=None,
            param_version=self.current_param_version,
        )

        async with self.lock:
            self.running = False

    async def fit(self):
        """
        Start the async rollouter - entry point that sets up and runs async tasks
        Main async fit method that coordinates all coroutines
        """

        print("[FullyAsyncRollouter] Starting FullyAsyncRollouter...")

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")

        # Set the running status flag
        async with self.lock:
            self.paused = False
            self.running = True

        # Create the main asynchronous task
        generation_task = asyncio.create_task(self._streaming_generation_main())
        monitor_task = asyncio.create_task(self._async_monitor_loop())

        try:
            # Run build and monitoring tasks concurrently
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)
        except Exception as e:
            print(f"[FullyAsyncRollouter] Asynchronous task execution error: {e}")
        finally:
            if not generation_task.done():
                generation_task.cancel()
            if not monitor_task.done():
                monitor_task.cancel()

            # Wait for the task to complete
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)

        print("[FullyAsyncRollouter] Rollouter fit completed")

    async def _async_monitor_loop(self):
        """
        Async coroutine for monitoring:
        Function 1: Log information output
        Function 2: Trigger rollout recovery
        """
        last_stats_time = time.time()
        stats_interval = 60.0
        check_interval = 10.0

        while True:
            async with self.lock:
                if not self.running:
                    break
            await asyncio.sleep(check_interval)
            # Print statistics periodically
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                stats = await self.get_statistics()
                print(f"[FullyAsyncRollouter][MonitorLoop][Statistics] {pformat(stats)}")
                last_stats_time = current_time

            # Trigger rollout recovery
            if self.monitor_loop_trigger:
                if not await self._should_pause_generation():
                    async with self.lock:
                        self.paused = False
                        self.condition.notify_all()

    async def _should_pause_generation(self) -> bool:
        """Determine whether the build should be paused"""
        queue_stats = self.message_queue_client.get_statistics_sync()
        queue_size = queue_stats["queue_size"]

        if queue_size >= self.max_queue_size:
            if not self.paused:
                print(
                    f"[FullyAsyncRollouter][ShouldPause]  "
                    f"due to full queue: size={queue_size}, max={self.max_queue_size}"
                )
            return True

        if self.staleness_samples >= self.max_required_samples:
            if not self.paused:
                print(
                    "[FullyAsyncRollouter][ShouldPause] "
                    f"due to "
                    f"staleness_samples {self.staleness_samples} >= max_required_samples {self.max_required_samples} "
                )
            return True

        return False

    async def pause(self):
        """pause rollout"""
        print("[FullyAsyncRollouter][Public][Pause]")

        # Step 1: 设置 paused 标志，通知 workers 开始排空
        async with self.lock:
            self.paused = True
            # Cancel all rollout tasks (partial rollout specific)
            if self.config.async_training.partial_rollout:
                await asyncio.gather(self.heavy_manager.cancel(), self.fast_manager.cancel(), return_exceptions=True)

        # Step 2: 在锁外等待 workers 自行排空 active_tasks
        # Workers 检测到 paused=True 后会自己等 active_tasks 完成并清空，
        # 这里做防御性等待，正常情况下 active_tasks 已被 workers 清空，几乎瞬间完成。
        for _ in range(600):  # 最多等 300s
            async with self.lock:
                remaining = len(self.active_tasks)
            if remaining == 0:
                break
            await asyncio.sleep(0.5)
        else:
            # 超时兜底：强制等待剩余 tasks
            async with self.lock:
                tasks_snapshot = list(self.active_tasks)
            if tasks_snapshot:
                print(f"[FullyAsyncRollouter][Pause] Timeout waiting for workers, force gathering {len(tasks_snapshot)} tasks")
                await asyncio.gather(*tasks_snapshot, return_exceptions=True)
                async with self.lock:
                    self.active_tasks.clear()

        # Step 3: 清理 aggregator 和 kv_cache（此时所有 tasks 已完成）
        async with self.lock:
            self.batch_aggregator.clear()
            self.active_tasks.clear()
        await asyncio.gather(self.heavy_manager.clear_kv_cache(), self.fast_manager.clear_kv_cache(), return_exceptions=True)
        async with self.lock:
            self.monitor_loop_trigger = False
        print("[FullyAsyncRollouter][Public][Pause] Done")

    async def resume(self, dependency_ref: ObjectRef = None):
        if dependency_ref is not None:
            ray.get(dependency_ref)
        print("[FullyAsyncRollouter][Public][Resume]")
        async with self.lock:
            if self.config.async_training.partial_rollout:
                await asyncio.gather(self.heavy_manager.resume(), self.fast_manager.resume(), return_exceptions=True)
            self.paused = False
            self.monitor_loop_trigger = True
            self.condition.notify_all()
        # 延迟再 notify 一次，确保 workers 不会错过
        await asyncio.sleep(0.1)
        async with self.lock:
            self.condition.notify_all()

    async def get_statistics(self) -> dict:
        queue_stats = self.message_queue_client.get_statistics_sync()

        stats = {
            # monitor stats
            "monitor/active_tasks_size": len(self.active_tasks),
            "monitor/queue/probe_queue_size": self.probe_queue.qsize(),
            "monitor/queue/fast_re_queue_size": self.fast_re_queue.qsize(),
            "monitor/queue/heavy_re_queue_size": self.heavy_re_queue.qsize(),
            "monitor/queue/cancel_queue_size": self.cancel_queue.qsize(),
            "monitor/queue/mq_queue_size": queue_stats["queue_size"],
            "monitor/aggregator_state": str({k: len(v) for k, v in self.batch_aggregator.items()}),
            # counting stats
            "count/current_param_version": self.current_param_version,
            "count/total_generated_samples": self.total_generated_samples,
            "count/staleness_samples": self.staleness_samples,
            "count/dropped_stale_samples": self.dropped_stale_samples,
            # static stats
            "static/max_required_samples": self.max_required_samples,
            "static/required_samples": self.required_samples,
            "static/staleness_threshold": self.staleness_threshold,
            "static/max_queue_size": self.max_queue_size,
            "static/max_concurrent_samples": self.max_concurrent_samples,
            "static/fake_batch_size": self.fake_batch_size,
        }

        return stats

    # --- Re-add missing methods from 0.7.0 for completeness ---
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[torch.utils.data.Sampler]):
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
            collate_fn = default_collate_fn
        from torchdata.stateful_dataloader import StatefulDataLoader
        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset, batch_size=self.config.data.get("gen_batch_size", 1), num_workers=self.config.data.get("dataloader_num_workers", 0), drop_last=True, collate_fn=collate_fn, sampler=train_sampler)
        self.val_dataloader = StatefulDataLoader(dataset=self.val_dataset, batch_size=self.config.data.get("val_batch_size", 1), num_workers=self.config.data.get("dataloader_num_workers", 0), shuffle=self.config.data.get("validation_shuffle", True), drop_last=False, collate_fn=collate_fn)
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        self.total_training_steps = total_training_steps
        from omegaconf import open_dict
        try:
            with open_dict(self.config):
                if "actor_rollout_ref" in self.config and "actor" in self.config.actor_rollout_ref and "optim" in self.config.actor_rollout_ref.actor:
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if "critic" in self.config and "optim" in self.config.critic:
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e: pass

