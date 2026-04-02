"""
Dynamic Segment Rollout Trainer: Adaptive Over-Provision scheduling.

Replaces the Budget-Constrained Dual-Channel design with a more generalizable approach:

Key design changes vs. original:
  1. over_provision_ratio replaces cont_ratio + total_token_capacity:
       Each pipeline step dispatches N_dispatch = target_completions * over_provision_ratio
       requests total (buffer continuations first, then new samples).
       Once target_completions finish, proceed (APRIL-style).
       No manual budget split needed — hardware adapts automatically.

  2. target_finish_rate replaces seg1_min_len:
       seg1_len is adjusted each step via EMA so that the observed seg1 finish rate
       tracks a user-specified target (default 0.6). No dataset-specific tuning needed.

  3. No pending_batch_info:
       Each step always injects new samples up to N_dispatch - N_cont.
       No throttling / deferral needed since budget is controlled by N_dispatch.

  4. GRESO pre-filtering (借鉴 arxiv:2506.02177):
       Before injection, prompts with zero-variance reward history (all-correct or
       all-wrong in recent K rollouts) are skipped — they provide no gradient signal.
       History is updated after each training step.

  5. SortedRL length-aware scheduling (借鉴 arxiv:2603.23414):
       When assembling training batch, complete GRPO groups are sorted by average
       response length (shortest first). Shorter groups train first, longer groups
       remain in pool for next step. Reduces gradient variance and forms natural
       easy-to-hard curriculum.

Parameters:
    target_completions      : target number of completed samples per training step
                              Default: n * min_train_groups
    over_provision_ratio    : total dispatched = target_completions * this ratio
                              Default: 2.0 (same as APRIL)
    target_finish_rate      : desired fraction of seg1 samples that finish in seg1
                              Default: 0.6  (hardware/dataset agnostic)
    seg1_len_init           : initial seg1_len before EMA adapts it
                              Default: max_response_length // 4
    ema_alpha               : EMA smoothing factor for seg1_len adaptation
                              Default: 0.3
    l_min                   : hard minimum for any segment length
                              Default: 64
    l_max                   : hard maximum = max_response_length
    min_train_groups        : minimum complete GRPO groups to trigger training
                              Default: 8
    greso_window            : number of recent rollouts to track per prompt for GRESO
                              Default: 5
    greso_min_history       : minimum history length before filtering activates
                              Default: 3
    sorted_rl               : whether to sort groups by length before training
                              Default: True
"""

import time
import uuid
from collections import defaultdict, deque
from pprint import pprint

import numpy as np
import ray
import torch
from tensordict import TensorDict
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_timing_metrics, compute_throughout_metrics
from verl.trainer.ppo.reward import compute_reward
from verl.utils.debug import marked_timer

from recipe.segment_rollout.segment_trainer import SegmentRolloutTrainer


class DynamicSegmentRolloutTrainer(SegmentRolloutTrainer):
    """Adaptive Over-Provision Dynamic Segment Rollout trainer.

    Each pipeline step:
      1. Dispatch N_dispatch = target_completions * over_provision_ratio requests
         (buffer continuations first, new samples fill the rest)
      2. cont_len = l_max (always give continuations full budget — let them EOS naturally)
      3. seg1_len adapts via EMA to hit target_finish_rate
      4. Once target_completions samples finish → assemble training batch
    """

    # ───────────────────────── config ─────────────────────────

    def _get_dynamic_segment_config(self):
        """Read dynamic segment rollout config with defaults."""
        seg_cfg = self.config.get("dynamic_segment_rollout", {})
        max_response_length = self.config.data.max_response_length
        n = self.config.actor_rollout_ref.rollout.n

        l_min = seg_cfg.get("l_min", 64)
        l_max = seg_cfg.get("l_max", max_response_length)
        min_train_groups = seg_cfg.get("min_train_groups", 8)

        # Over-provision parameters (replaces cont_ratio + total_token_capacity)
        target_completions = seg_cfg.get("target_completions", n * min_train_groups)
        over_provision_ratio = seg_cfg.get("over_provision_ratio", 2.0)

        # Adaptive seg1_len parameters (replaces seg1_min_len)
        target_finish_rate = seg_cfg.get("target_finish_rate", 0.6)
        seg1_len_init = seg_cfg.get("seg1_len_init", max_response_length // 4)
        ema_alpha = seg_cfg.get("ema_alpha", 0.3)

        # GRESO pre-filtering parameters
        greso_window = seg_cfg.get("greso_window", 5)
        greso_min_history = seg_cfg.get("greso_min_history", 3)

        # SortedRL length-aware scheduling
        sorted_rl = seg_cfg.get("sorted_rl", True)

        return {
            "l_min": l_min,
            "l_max": l_max,
            "min_train_groups": min_train_groups,
            "max_response_length": max_response_length,
            "target_completions": target_completions,
            "over_provision_ratio": over_provision_ratio,
            "target_finish_rate": target_finish_rate,
            "seg1_len_init": seg1_len_init,
            "ema_alpha": ema_alpha,
            "greso_window": greso_window,
            "greso_min_history": greso_min_history,
            "sorted_rl": sorted_rl,
        }

    # ───────────────────────── adaptive seg1_len ─────────────────────────

    def _update_seg1_len(self, current_seg1_len, actual_finish_rate,
                          target_finish_rate, ema_alpha, l_min, l_max):
        """Adjust seg1_len via EMA so actual finish rate tracks target_finish_rate.

        If too many samples finish in seg1 (finish_rate > target + tolerance):
            → seg1 is too long → shorten it to push more samples into unfinished_pool
        If too few samples finish in seg1 (finish_rate < target - tolerance):
            → seg1 is too short → lengthen it

        Returns updated seg1_len (int).
        """
        tolerance = 0.05
        if actual_finish_rate > target_finish_rate + tolerance:
            # Too many finishing → seg1 too long → shrink
            adjustment = 1.0 - ema_alpha * (actual_finish_rate - target_finish_rate)
        elif actual_finish_rate < target_finish_rate - tolerance:
            # Too few finishing → seg1 too short → grow
            adjustment = 1.0 + ema_alpha * (target_finish_rate - actual_finish_rate)
        else:
            adjustment = 1.0

        new_len = int(current_seg1_len * adjustment)
        return max(l_min, min(l_max, new_len))

    # ───────────────────────── GRESO 预过滤 ─────────────────────────

    def _greso_is_zero_variance(self, prompt_key, reward_history, greso_min_history):
        """判断一个 prompt 是否属于零方差样本（全对或全错），应被跳过。

        Args:
            prompt_key: prompt 的唯一标识（tuple(raw_prompt_ids)，跨步稳定）
            reward_history: dict[prompt_key -> deque[float]]
            greso_min_history: 历史积累不足时不过滤

        Returns:
            True 表示应跳过（零方差），False 表示保留。
        """
        history = reward_history.get(prompt_key)
        if history is None or len(history) < greso_min_history:
            return False  # 历史不足，不过滤，保守策略
        recent = list(history)
        all_correct = all(r >= 1.0 for r in recent)
        all_wrong = all(r <= 0.0 for r in recent)
        return all_correct or all_wrong

    def _greso_update_history_by_keys(self, trained_batch, train_prompt_keys,
                                       reward_history, greso_window, n):
        """训练完成后，用 prompt_key（raw_prompt_ids 的 tuple）更新 GRESO 历史记录。

        Args:
            trained_batch: 训练后的 DataProto，包含 token_level_scores
            train_prompt_keys: list[tuple(int)]，每个 group 对应的 prompt token ids
            reward_history: 要更新的历史字典 {prompt_key -> deque[float]}
            greso_window: 每个 prompt 保留的历史长度
            n: 每个 prompt 的 rollout 数量
        """
        scores = trained_batch.batch.get("token_level_scores", None)
        if scores is None:
            return
        if hasattr(scores, "cpu"):
            scores = scores.cpu()
        # 每个样本的 reward = token_level_scores 各 token 的最大值（outcome reward）
        sample_rewards = scores.max(dim=-1).values.tolist()

        for gi, prompt_key in enumerate(train_prompt_keys):
            group_rewards = sample_rewards[gi * n: (gi + 1) * n]
            avg_reward = sum(group_rewards) / len(group_rewards) if group_rewards else 0.0
            if prompt_key not in reward_history:
                reward_history[prompt_key] = deque(maxlen=greso_window)
            reward_history[prompt_key].append(avg_reward)

    def _greso_update_history(self, trained_batch, complete_puids, experience_pool_snapshot,
                               reward_history, greso_window, n):
        """训练完成后，把本批次每个 prompt 的平均 reward 写入历史记录。

        Args:
            trained_batch: 训练后的 DataProto，包含 token_level_scores
            complete_puids: 本次训练的 prompt_uid 列表
            experience_pool_snapshot: 训练前的 experience_pool 快照（已删除）
            reward_history: 要更新的历史字典
            greso_window: 每个 prompt 保留的历史长度
            n: 每个 prompt 的 rollout 数量
        """
        scores = trained_batch.batch.get("token_level_scores", None)
        if scores is None:
            return
        if hasattr(scores, "cpu"):
            scores = scores.cpu()
        # 每个样本的 reward = token_level_scores 中非零部分的最大值（对应 outcome reward）
        sample_rewards = scores.max(dim=-1).values.tolist()

        for gi, puid in enumerate(complete_puids):
            group_rewards = sample_rewards[gi * n: (gi + 1) * n]
            avg_reward = sum(group_rewards) / len(group_rewards) if group_rewards else 0.0
            # prompt_key: 取 puid 的 batch_id 部分（去掉 sample index）
            prompt_key = puid
            if prompt_key not in reward_history:
                reward_history[prompt_key] = deque(maxlen=greso_window)
            reward_history[prompt_key].append(avg_reward)

    # ───────────────────────── SortedRL 长度排序 ─────────────────────────

    def _sorted_rl_select_groups(self, complete_puids, experience_pool, min_train_groups, n):
        """按 group 平均响应长度从短到长排序，取前 min_train_groups 个训练。

        短样本优先训练，长样本留在 pool 等下一步，形成自然易到难课程。

        Returns:
            train_puids: 本步实际训练的 puid 列表
        """
        def group_avg_len(puid):
            group = experience_pool[puid]
            return sum(len(s["accumulated_ids"]) for s in group) / max(len(group), 1)

        sorted_puids = sorted(complete_puids, key=group_avg_len)
        train_puids = sorted_puids[:min_train_groups]
        return train_puids

    # ───────────────────────── continuation override ─────────────────────────

    def _process_continuation_results_dynamic(self, cont_futures, seg_len, max_response_length,
                                               unfinished_pool, experience_pool):
        """Process continuation results WITHOUT fixed num_segments limit.

        Samples finish only on EOS or max_response_length.

        Returns (newly_finished_count, eos_count, maxlen_count,
                 cont_new_token_lengths, cont_accumulated_lengths).
        """
        newly_finished = []
        new_eos = 0
        new_max = 0
        cont_new_token_lengths = []
        cont_accumulated_lengths = []

        for uid, future in cont_futures.items():
            output = ray.get(future)
            new_tokens = list(output.token_ids)
            state = unfinished_pool[uid]
            state["accumulated_ids"].extend(new_tokens)
            state["seg_count"] += 1
            cont_new_token_lengths.append(len(new_tokens))
            cont_accumulated_lengths.append(len(state["accumulated_ids"]))

            hit_eos = len(new_tokens) < seg_len
            hit_max = len(state["accumulated_ids"]) >= max_response_length

            if hit_eos or hit_max:
                state["accumulated_ids"] = state["accumulated_ids"][:max_response_length]
                state["finished"] = True
                experience_pool.setdefault(state["prompt_uid"], []).append(state)
                newly_finished.append(uid)
                if hit_eos:
                    new_eos += 1
                else:
                    new_max += 1

        for uid in newly_finished:
            del unfinished_pool[uid]

        return len(newly_finished), new_eos, new_max, cont_new_token_lengths, cont_accumulated_lengths

    # ───────────────────────── main loop ─────────────────────────

    def fit(self):
        """Adaptive Over-Provision pipeline training loop."""
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        dcfg = self._get_dynamic_segment_config()
        l_min = dcfg["l_min"]
        l_max = dcfg["l_max"]
        max_response_length = dcfg["max_response_length"]
        min_train_groups = dcfg["min_train_groups"]
        target_completions = dcfg["target_completions"]
        over_provision_ratio = dcfg["over_provision_ratio"]
        target_finish_rate = dcfg["target_finish_rate"]
        ema_alpha = dcfg["ema_alpha"]
        n = self.config.actor_rollout_ref.rollout.n
        greso_window = dcfg["greso_window"]
        greso_min_history = dcfg["greso_min_history"]
        sorted_rl = dcfg["sorted_rl"]

        # N_dispatch: total requests per pipeline step
        n_dispatch = int(target_completions * over_provision_ratio)
        # cont_len: always give continuations full l_max so they finish naturally
        cont_len = l_max
        # seg1_len: starts at seg1_len_init, adapts via EMA
        seg1_len = dcfg["seg1_len_init"]

        print(f"[动态分段] 配置: "
              f"target_completions={target_completions}, over_provision_ratio={over_provision_ratio}, "
              f"n_dispatch={n_dispatch}, cont_len={cont_len}, seg1_len_init={seg1_len}, "
              f"target_finish_rate={target_finish_rate}, ema_alpha={ema_alpha}, "
              f"l_min={l_min}, l_max={l_max}, n={n}, min_train_groups={min_train_groups}, "
              f"greso_window={greso_window}, greso_min_history={greso_min_history}, sorted_rl={sorted_rl}")

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(
            total=self.total_training_steps, initial=self.global_steps,
            desc="Dynamic Segment Pipeline",
        )
        self.global_steps += 1
        self.max_steps_duration = 0

        prompt_length = self.config.data.max_prompt_length
        pad_token_id = self.tokenizer.pad_token_id or 0

        # ── Cross-step pipeline state ──
        unfinished_pool = {}
        experience_pool = {}
        batch_counter = 0
        gen_time_accum = 0.0
        # GRESO: 每个 prompt 的历史 reward 记录，key=tuple(raw_prompt_ids)
        reward_history = {}

        dataloader_iter = iter(self.train_dataloader)
        current_epoch = 0
        pipeline_step = 0

        while self.global_steps <= self.total_training_steps:
            pipeline_step += 1

            # ═══════════════════════════════════════════════════════
            # Phase 1: Determine how many new samples to inject
            #   n_cont = min(buffer size, n_dispatch)  -- continuations first
            #   n_inject = n_dispatch - n_cont          -- new samples fill the rest
            # ═══════════════════════════════════════════════════════
            n_cont = min(len(unfinished_pool), n_dispatch)
            n_inject_needed = n_dispatch - n_cont

            # Round n_inject_needed to multiple of num_workers
            num_workers = len(self.async_rollout_manager.agent_loop_workers)
            n_inject_needed = (n_inject_needed // num_workers) * num_workers
            n_inject_needed = max(0, n_inject_needed)

            # Fetch enough new batches to fill n_inject_needed
            new_batch_infos = []
            n_collected = 0
            while n_collected < n_inject_needed:
                try:
                    batch_dict = next(dataloader_iter)
                    info = self._prepare_new_batch(batch_dict, batch_counter, n)
                    batch_counter += 1
                    new_batch_infos.append(info)
                    n_collected += info["total_samples"]
                except StopIteration:
                    current_epoch += 1
                    if current_epoch < self.config.trainer.total_epochs:
                        dataloader_iter = iter(self.train_dataloader)
                    else:
                        break

            # Merge collected batch infos into one
            if new_batch_infos:
                if len(new_batch_infos) == 1:
                    new_batch_info = new_batch_infos[0]
                else:
                    new_batch_info = {
                        "gen_batch": DataProto.concat([b["gen_batch"] for b in new_batch_infos]),
                        "batch_repeated": DataProto.concat([b["batch_repeated"] for b in new_batch_infos]),
                        "total_samples": sum(b["total_samples"] for b in new_batch_infos),
                        "prompt_uids": sum([b["prompt_uids"] for b in new_batch_infos], []),
                    }
                # Trim to exactly n_inject_needed
                n_inject = min(new_batch_info["total_samples"], n_inject_needed)
                if n_inject < new_batch_info["total_samples"]:
                    new_batch_info = {
                        "gen_batch": new_batch_info["gen_batch"][:n_inject],
                        "batch_repeated": new_batch_info["batch_repeated"][:n_inject],
                        "total_samples": n_inject,
                        "prompt_uids": new_batch_info["prompt_uids"][:n_inject],
                    }
            else:
                new_batch_info = None
                n_inject = 0

            has_new = n_inject > 0
            has_cont = n_cont > 0

            if not has_new and not has_cont:
                complete_puids = [p for p, r in experience_pool.items() if len(r) >= n]
                if not complete_puids:
                    print("[动态分段] 无更多数据，结束训练")
                    break

            print(f"[动态分段] 流水线步 {pipeline_step} | "
                  f"续生={n_cont} cont_len={cont_len} | "
                  f"新注入={n_inject} seg1_len={seg1_len} | "
                  f"n_dispatch={n_dispatch}")

            # ═══════════════════════════════════════════════════════
            # Phase 2: Generation
            # ═══════════════════════════════════════════════════════
            if has_new or has_cont:
                gen_start = time.time()
                self.async_rollout_manager.wake_up()

                # ── 2a: New prompts → AgentLoop workers ──
                seg1_futures = None
                if has_new:
                    gen_batch = new_batch_info["gen_batch"]
                    gen_batch.meta_info["max_tokens"] = seg1_len
                    workers = self.async_rollout_manager.agent_loop_workers
                    chunks = gen_batch.chunk(len(workers))
                    seg1_futures = [
                        worker.generate_sequences.remote(chunk)
                        for worker, chunk in zip(workers, chunks)
                    ]

                # ── 2b: Continuations → direct vLLM server calls ──
                cont_futures = {}
                if has_cont:
                    sampling_params = self._get_sampling_params_for_continuation(cont_len)
                    server_handles = self.async_rollout_manager.server_handles
                    num_servers = len(server_handles)
                    cont_uids = list(unfinished_pool.keys())[:n_cont]
                    for idx, uid in enumerate(cont_uids):
                        state = unfinished_pool[uid]
                        continuation_ids = state["raw_prompt_ids"] + state["accumulated_ids"]
                        server = server_handles[idx % num_servers]
                        cont_futures[uid] = server.generate.remote(
                            prompt_ids=continuation_ids,
                            sampling_params=sampling_params,
                            request_id=str(uuid.uuid4()),
                        )

                # ── Collect seg1 results ──
                seg1_finished = 0
                seg1_total = 0
                seg1_lengths = []
                if seg1_futures:
                    seg1_outputs = ray.get(seg1_futures)
                    seg1_output = DataProto.concat(seg1_outputs)
                    seg1_total = n_inject
                    seg1_finished, seg1_lengths = self._process_seg1_results(
                        seg1_output, new_batch_info, seg1_len, n,
                        unfinished_pool, experience_pool,
                    )
                    print(f"[动态分段] 流水线步 {pipeline_step} | Seg1: "
                          f"{seg1_finished}/{seg1_total} 已完成 "
                          f"({seg1_finished * 100 // max(seg1_total, 1)}%)")

                # ── Collect continuation results ──
                cont_total = 0
                cont_finished = 0
                cont_new_token_lengths = []
                cont_accumulated_lengths = []
                if cont_futures:
                    cont_total = len(cont_futures)
                    cont_finished, eos_count, max_count, cont_new_token_lengths, cont_accumulated_lengths = \
                        self._process_continuation_results_dynamic(
                            cont_futures, cont_len, max_response_length,
                            unfinished_pool, experience_pool,
                        )
                    print(f"[动态分段] 流水线步 {pipeline_step} | 续生结果: "
                          f"{cont_finished}/{cont_total} 已完成 "
                          f"(eos={eos_count}, maxlen={max_count}), "
                          f"仍在unfinished_pool中: {len(unfinished_pool)}")

                self.async_rollout_manager.sleep()
                gen_elapsed = time.time() - gen_start
                gen_time_accum += gen_elapsed

                # ── Adaptive seg1_len update (EMA) ──
                if seg1_total > 0:
                    actual_finish_rate = seg1_finished / seg1_total
                    seg1_len = self._update_seg1_len(
                        seg1_len, actual_finish_rate,
                        target_finish_rate, ema_alpha, l_min, l_max,
                    )

                # ── Per-pipeline-step metrics ──
                step_metrics = {
                    "pipe/step": pipeline_step,
                    "pipe/n_cont": n_cont,
                    "pipe/n_inject": n_inject,
                    "pipe/n_dispatch": n_dispatch,
                    "pipe/seg1_len": seg1_len,
                    "pipe/cont_len": cont_len,
                    "pipe/gen_time": gen_elapsed,
                    "pipe/seg1_total": seg1_total,
                    "pipe/seg1_finished": seg1_finished,
                    "pipe/seg1_finish_rate": seg1_finished / seg1_total if seg1_total > 0 else 0,
                    "pipe/seg1_target_finish_rate": target_finish_rate,
                    "pipe/cont_total": cont_total,
                    "pipe/cont_finished": cont_finished,
                    "pipe/cont_finish_rate": cont_finished / cont_total if cont_total > 0 else 0,
                    "pipe/unfinished_pool_size": len(unfinished_pool),
                }
                step_metrics.update(self._length_distribution_metrics(seg1_lengths, "pipe_seg1_len"))
                step_metrics.update(self._length_distribution_metrics(cont_new_token_lengths, "pipe_cont_new_len"))
                step_metrics.update(self._length_distribution_metrics(cont_accumulated_lengths, "pipe_cont_accum_len"))
                unfinished_accum = [len(s["accumulated_ids"]) for s in unfinished_pool.values()]
                step_metrics.update(self._length_distribution_metrics(unfinished_accum, "pipe_unfinished_accum"))
                all_new_tokens = seg1_lengths + cont_new_token_lengths
                total_new_tokens = sum(all_new_tokens)
                step_metrics["pipe/tokens_generated"] = total_new_tokens
                step_metrics["pipe/tokens_per_sec"] = total_new_tokens / gen_elapsed if gen_elapsed > 0 else 0
                logger.log(data=step_metrics, step=pipeline_step)

            # ═══════════════════════════════════════════════════════
            # Phase 3: Train if enough complete GRPO groups
            # ═══════════════════════════════════════════════════════
            complete_puids = [p for p, r in experience_pool.items() if len(r) >= n]

            # ── GRESO 预过滤：跳过零方差 prompt 组 ──
            if greso_window > 0 and complete_puids:
                greso_kept = []
                greso_skipped = 0
                for puid in complete_puids:
                    group = experience_pool[puid]
                    prompt_key = tuple(group[0]["raw_prompt_ids"])
                    if self._greso_is_zero_variance(prompt_key, reward_history, greso_min_history):
                        greso_skipped += 1
                    else:
                        greso_kept.append(puid)
                if greso_skipped > 0:
                    print(f"[动态分段] GRESO: 跳过 {greso_skipped} 个零方差prompt组，"
                          f"保留 {len(greso_kept)} 个")
                complete_puids = greso_kept

            # ── SortedRL：按平均响应长度从短到长排序，取前 min_train_groups 个 ──
            if sorted_rl and len(complete_puids) > min_train_groups:
                complete_puids = self._sorted_rl_select_groups(
                    complete_puids, experience_pool, min_train_groups, n
                )
                print(f"[动态分段] SortedRL: 选取最短 {len(complete_puids)} 个group训练")

            should_train = (
                len(complete_puids) >= min_train_groups
                or (len(unfinished_pool) == 0 and len(complete_puids) > 0)
            )

            if should_train:
                metrics = {}
                timing_raw = {}
                train_start = time.time()
                is_last_step = self.global_steps >= self.total_training_steps
                num_train_groups = len(complete_puids)
                num_train_samples = num_train_groups * n

                print(f"[动态分段] 训练步 {self.global_steps}: "
                      f"{num_train_groups} 个group ({num_train_samples} 个样本)")

                with marked_timer("train", timing_raw, color="green"):
                    train_batch = self._assemble_training_batch_from_pool(
                        complete_puids, experience_pool, n,
                        prompt_length, max_response_length, pad_token_id,
                    )
                    self._log_rollout_responses(train_batch, folder_suffix="dynamic_segment")

                    # 训练前收集 prompt_keys，用于 GRESO 历史更新（pool 清空后无法获取）
                    train_prompt_keys = []
                    for puid in complete_puids:
                        group = experience_pool[puid]
                        train_prompt_keys.append(tuple(group[0]["raw_prompt_ids"]))

                    for puid in complete_puids:
                        del experience_pool[puid]

                    trained_batch = self._do_train_step(train_batch, metrics, timing_raw)

                # ── GRESO 历史更新：记录本批次每个 prompt 的平均 reward ──
                if greso_window > 0:
                    self._greso_update_history_by_keys(
                        trained_batch, train_prompt_keys, reward_history, greso_window, n
                    )

                # ── Metrics ──
                metrics.update(compute_data_metrics(batch=trained_batch, use_critic=self.use_critic))

                train_duration = time.time() - train_start
                timing_raw["gen"] = gen_time_accum
                timing_raw["step"] = gen_time_accum + train_duration
                self.max_steps_duration = max(self.max_steps_duration, timing_raw["step"])
                gen_time_accum = 0.0

                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_timing_metrics(batch=trained_batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(
                    batch=trained_batch, timing_raw=timing_raw, n_gpus=n_gpus,
                ))

                timing_str = " | ".join(f"{k}={v:.1f}s" for k, v in timing_raw.items())
                pool_remaining = sum(len(r) for r in experience_pool.values())
                print(f"[动态分段] === 训练步 {self.global_steps} "
                      f"(流水线步={pipeline_step}) === "
                      f"groups={num_train_groups} | "
                      f"unfinished={len(unfinished_pool)} | "
                      f"pool剩余={pool_remaining} | "
                      f"{timing_str}")

                # Train batch length distribution
                train_response_mask = trained_batch.batch.get("response_mask", None)
                if train_response_mask is not None:
                    if hasattr(train_response_mask, 'cpu'):
                        train_response_mask = train_response_mask.cpu()
                    train_lengths = train_response_mask.sum(dim=-1).tolist()
                    metrics.update(self._length_distribution_metrics(train_lengths, "train_resp_len"))
                    actual_segs = [max(1, (l + l_min - 1) // l_min) for l in train_lengths]
                    metrics["train_resp_len/avg_segments_used"] = sum(actual_segs) / len(actual_segs)
                    metrics["train_resp_len/pct_1seg"] = sum(1 for l in train_lengths if l <= l_min) / len(train_lengths)
                    metrics["train_resp_len/pct_sprint"] = sum(1 for l in train_lengths if l > l_max * 0.8) / len(train_lengths)
                    grpo_length_cvs = []
                    for gi in range(0, len(train_lengths), n):
                        group = train_lengths[gi:gi + n]
                        if len(group) >= 2:
                            g_mean = sum(group) / len(group)
                            if g_mean > 0:
                                g_std = (sum((x - g_mean) ** 2 for x in group) / len(group)) ** 0.5
                                grpo_length_cvs.append(g_std / g_mean)
                    if grpo_length_cvs:
                        metrics["train_resp_len/grpo_cv_mean"] = sum(grpo_length_cvs) / len(grpo_length_cvs)
                        metrics["train_resp_len/grpo_cv_max"] = max(grpo_length_cvs)

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": current_epoch,
                    "dynamic_segment/pipeline_step": pipeline_step,
                    "dynamic_segment/train_groups": num_train_groups,
                    "dynamic_segment/train_samples": num_train_samples,
                    "dynamic_segment/unfinished_pool_size": len(unfinished_pool),
                    "dynamic_segment/experience_pool_samples": pool_remaining,
                    "dynamic_segment/n_dispatch": n_dispatch,
                    "dynamic_segment/over_provision_ratio": over_provision_ratio,
                    "dynamic_segment/seg1_len_adaptive": seg1_len,
                    "dynamic_segment/target_finish_rate": target_finish_rate,
                    "dynamic_segment/l_min": l_min,
                    "dynamic_segment/l_max": l_max,
                    "dynamic_segment/greso_tracked_prompts": len(reward_history),
                })
                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)

                # ── Validation ──
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)
                    logger.log(data=val_metrics, step=self.global_steps)

                # ── Checkpoint ──
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                self.global_steps += 1
                if self.global_steps > self.total_training_steps:
                    break
            else:
                pool_total = sum(len(r) for r in experience_pool.values())
                print(f"[动态分段] 步 {pipeline_step}: 仅生成 | "
                      f"unfinished={len(unfinished_pool)} | "
                      f"pool样本={pool_total} | "
                      f"完整groups={len(complete_puids)} (需要 {min_train_groups})")

        progress_bar.close()
        print("[动态分段] 训练完成。")
