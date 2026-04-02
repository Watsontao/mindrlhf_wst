"""
Segment Rollout Trainer: UloRL cross-step pipeline implementation.

Based on the UloRL paper (arxiv 2507.19766), this trainer implements a cross-step
pipeline where generation is split into fixed-length segments. Instead of generating
all responses for a batch completely before training, each pipeline step:
  1. Injects new prompts (seg1 via AgentLoop) + continues unfinished (direct vLLM)
  2. Finished samples go to the experience pool
  3. Training fires when enough complete GRPO groups accumulate

Key difference from single-batch segment rollout:
  - New prompts flow in continuously while old unfinished samples continue
  - Complete groups train immediately, not waiting for the entire batch to finish
  - New prompt generation and continuation happen concurrently in vLLM's continuous batching
"""

import time
import uuid
from collections import defaultdict
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


class SegmentRolloutTrainer(RayPPOTrainer):
    """UloRL cross-step pipeline trainer with segment-based rollout."""

    # ───────────────────────── config ─────────────────────────

    def _get_segment_config(self):
        """Read segment rollout config with defaults."""
        seg_cfg = self.config.get("segment_rollout", {})
        num_segments = seg_cfg.get("num_segments", 4)
        max_response_length = self.config.data.max_response_length
        seg_len = max_response_length // num_segments
        min_train_groups = seg_cfg.get("min_train_groups", 8)
        return {
            "num_segments": num_segments,
            "seg_len": seg_len,
            "max_response_length": max_response_length,
            "min_train_groups": min_train_groups,
        }

    def _get_sampling_params_for_continuation(self, seg_len):
        """Build sampling_params dict for direct vLLM server calls."""
        config = self.config.actor_rollout_ref.rollout
        return {
            "max_tokens": seg_len,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "repetition_penalty": 1.0,
        }

    # ───────────────────────── training ─────────────────────────

    def _do_train_step(self, batch, metrics, timing_raw):
        """Run full training pipeline: reward → log_prob → ref → advantage → update.

        Returns the batch after training (contains token_level_scores, advantages, etc.).
        """
        with marked_timer("reward", timing_raw, color="yellow"):
            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                batch, reward_fn=self.reward_fn, return_dict=False
            )

        with marked_timer("old_log_prob", timing_raw, color="blue"):
            old_log_prob, _ = self._compute_old_log_prob(batch)
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            with marked_timer("ref_log_prob", timing_raw, color="olive"):
                ref_log_prob = self._compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                values = self._compute_values(batch)
                batch = batch.union(values)

        with marked_timer("adv", timing_raw, color="brown"):
            batch.batch["token_level_scores"] = reward_tensor
            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
            if self.config.algorithm.use_kl_in_reward:
                from verl.trainer.ppo.ray_trainer import apply_kl_penalty
                batch, kl_metrics = apply_kl_penalty(
                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                )
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                config=self.config.algorithm,
            )

        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                critic_output = self._update_critic(batch)

        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor", timing_raw, color="red"):
                actor_output = self._update_actor(batch)

        return batch

    # ───────────────────────── pipeline helpers ─────────────────────────

    def _prepare_new_batch(self, batch_dict, batch_id, n):
        """Prepare a new dataloader batch for pipeline injection.

        Returns dict with gen_batch (for AgentLoop workers) and metadata for tracking.
        """
        batch = DataProto.from_single_dict(batch_dict)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
        )

        gen_batch = self._get_gen_batch(batch)
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch_repeated = gen_batch.repeat(repeat_times=n, interleave=True)
        batch_repeated = batch.repeat(repeat_times=n, interleave=True)

        total_samples = len(batch.batch) * n
        num_prompts = len(batch.batch)

        # Unique prompt_uids for GRPO grouping
        prompt_uids = []
        for p_idx in range(num_prompts):
            puid = f"b{batch_id}_p{p_idx}"
            for _ in range(n):
                prompt_uids.append(puid)

        return {
            "gen_batch": gen_batch_repeated,
            "batch_repeated": batch_repeated,
            "total_samples": total_samples,
            "prompt_uids": prompt_uids,
        }

    def _process_seg1_results(self, seg1_output, batch_info, seg_len, n,
                               unfinished_pool, experience_pool):
        """Process segment 1 output: extract tokens, route to pools.

        Returns number of finished samples.
        """
        total_samples = batch_info["total_samples"]
        batch_repeated = batch_info["batch_repeated"]
        prompt_uids = batch_info["prompt_uids"]

        prompt_length = seg1_output.batch["prompts"].shape[1]
        seg1_finished = 0

        for i in range(total_samples):
            # Extract unpadded prompt ids (left-padded → strip leading pads)
            prompts_i = seg1_output.batch["prompts"][i]
            attn_prompt = seg1_output.batch["attention_mask"][i][:prompt_length]
            raw_prompt_ids = prompts_i[attn_prompt.bool()].tolist()

            # Extract actual response tokens
            attn_resp = seg1_output.batch["attention_mask"][i][prompt_length:]
            actual_len = int(attn_resp.sum().item())
            accumulated_ids = seg1_output.batch["responses"][i][:actual_len].tolist()

            finished = actual_len < seg_len
            if finished:
                seg1_finished += 1

            uid = str(uuid.uuid4())
            state = {
                "uid": uid,
                "prompt_uid": prompt_uids[i],
                "group_idx": i % n,
                "raw_prompt_ids": raw_prompt_ids,
                "accumulated_ids": accumulated_ids,
                "non_tensor_data": {
                    k: batch_repeated.non_tensor_batch[k][i]
                    for k in batch_repeated.non_tensor_batch
                },
                "seg_count": 1,
                "finished": finished,
            }

            if finished:
                experience_pool.setdefault(prompt_uids[i], []).append(state)
            else:
                unfinished_pool[uid] = state

        return seg1_finished

    def _process_continuation_results(self, cont_futures, seg_len, max_response_length,
                                       num_segments, unfinished_pool, experience_pool):
        """Process continuation results: update tokens, move finished to experience pool.

        Returns (newly_finished_count, eos_count, maxlen_count).
        """
        newly_finished = []
        new_eos = 0
        new_max = 0

        for uid, future in cont_futures.items():
            output = ray.get(future)
            new_tokens = list(output.token_ids)
            state = unfinished_pool[uid]
            state["accumulated_ids"].extend(new_tokens)
            state["seg_count"] += 1

            hit_eos = len(new_tokens) < seg_len
            hit_max = len(state["accumulated_ids"]) >= max_response_length
            hit_seg_limit = state["seg_count"] >= num_segments

            if hit_eos or hit_max or hit_seg_limit:
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

        return len(newly_finished), new_eos, new_max

    def _assemble_training_batch_from_pool(self, complete_puids, experience_pool, n,
                                            prompt_length, response_length, pad_token_id):
        """Build training DataProto from complete GRPO groups in experience pool.

        Samples within each group are sorted by group_idx (contiguous for GRPO advantage).
        """
        all_prompts = []
        all_responses = []
        all_attention_mask = []
        all_response_mask = []
        all_input_ids = []
        all_position_ids = []
        all_non_tensor = defaultdict(list)

        for puid in complete_puids:
            group = experience_pool[puid]
            group.sort(key=lambda s: s["group_idx"])

            for state in group:
                p_ids = state["raw_prompt_ids"]
                r_ids = state["accumulated_ids"][:response_length]
                actual_resp_len = len(r_ids)

                # Left-pad prompt
                pad_len = prompt_length - len(p_ids)
                if pad_len > 0:
                    padded_prompt = [pad_token_id] * pad_len + p_ids
                else:
                    padded_prompt = p_ids[-prompt_length:]
                    pad_len = 0

                # Right-pad response
                padded_response = r_ids + [pad_token_id] * (response_length - actual_resp_len)

                prompt_t = torch.tensor(padded_prompt, dtype=torch.long)
                response_t = torch.tensor(padded_response, dtype=torch.long)
                input_ids_t = torch.cat([prompt_t, response_t])

                attn_prompt = torch.zeros(prompt_length, dtype=torch.long)
                attn_prompt[pad_len:] = 1
                attn_resp = torch.zeros(response_length, dtype=torch.long)
                attn_resp[:actual_resp_len] = 1
                attention_mask_t = torch.cat([attn_prompt, attn_resp])

                response_mask_t = torch.zeros(response_length, dtype=torch.long)
                response_mask_t[:actual_resp_len] = 1

                position_ids_t = torch.zeros(prompt_length + response_length, dtype=torch.long)
                valid_count = attention_mask_t.sum().item()
                position_ids_t[attention_mask_t.bool()] = torch.arange(valid_count)

                all_prompts.append(prompt_t)
                all_responses.append(response_t)
                all_input_ids.append(input_ids_t)
                all_attention_mask.append(attention_mask_t)
                all_response_mask.append(response_mask_t)
                all_position_ids.append(position_ids_t)

                for k, v in state["non_tensor_data"].items():
                    all_non_tensor[k].append(v)

        batch_td = TensorDict(
            {
                "prompts": torch.stack(all_prompts),
                "responses": torch.stack(all_responses),
                "input_ids": torch.stack(all_input_ids),
                "attention_mask": torch.stack(all_attention_mask),
                "response_mask": torch.stack(all_response_mask),
                "position_ids": torch.stack(all_position_ids),
            },
            batch_size=len(all_prompts),
        )

        train_batch = DataProto(batch=batch_td)
        for k, v in all_non_tensor.items():
            train_batch.non_tensor_batch[k] = np.array(v, dtype=object)
        train_batch.meta_info["global_token_num"] = torch.sum(
            train_batch.batch["attention_mask"], dim=-1
        ).tolist()

        return train_batch

    # ───────────────────────── main loop ─────────────────────────

    def fit(self):
        """UloRL cross-step pipeline training loop.

        Each pipeline step:
          1. Inject new prompts → seg1 via AgentLoop workers
          2. Continue unfinished from previous steps → direct vLLM server calls
          3. Both happen concurrently in one wake_up/sleep cycle (vLLM continuous batching)
          4. Complete GRPO groups accumulate in experience pool
          5. Train when enough complete groups are ready
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        seg_cfg = self._get_segment_config()
        num_segments = seg_cfg["num_segments"]
        seg_len = seg_cfg["seg_len"]
        max_response_length = seg_cfg["max_response_length"]
        min_train_groups = seg_cfg["min_train_groups"]
        n = self.config.actor_rollout_ref.rollout.n

        print(f"[SegmentPipeline] config: num_segments={num_segments}, seg_len={seg_len}, "
              f"max_response_length={max_response_length}, n={n}, "
              f"min_train_groups={min_train_groups}")

        self.global_steps = 0
        self._load_checkpoint()

        # Validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(
            total=self.total_training_steps, initial=self.global_steps,
            desc="Segment Pipeline Training",
        )
        self.global_steps += 1
        self.max_steps_duration = 0

        prompt_length = self.config.data.max_prompt_length
        pad_token_id = self.tokenizer.pad_token_id or 0

        # ── Cross-step pipeline state ──
        unfinished_pool = {}     # uid → sample state dict
        experience_pool = {}     # prompt_uid → [sample state dict, ...]
        batch_counter = 0
        gen_time_accum = 0.0     # generation time accumulated since last training

        dataloader_iter = iter(self.train_dataloader)
        current_epoch = 0
        pipeline_step = 0

        while self.global_steps <= self.total_training_steps:
            pipeline_step += 1

            # ═══════════════════════════════════════════════════════
            # Phase 1: Fetch new batch from dataloader
            # ═══════════════════════════════════════════════════════
            new_batch_info = None
            try:
                batch_dict = next(dataloader_iter)
                new_batch_info = self._prepare_new_batch(batch_dict, batch_counter, n)
                batch_counter += 1
            except StopIteration:
                current_epoch += 1
                if current_epoch < self.config.trainer.total_epochs:
                    dataloader_iter = iter(self.train_dataloader)
                    try:
                        batch_dict = next(dataloader_iter)
                        new_batch_info = self._prepare_new_batch(batch_dict, batch_counter, n)
                        batch_counter += 1
                    except StopIteration:
                        pass  # truly no more data

            has_new = new_batch_info is not None
            has_cont = len(unfinished_pool) > 0

            # Nothing to generate — check if we can still train on remaining pool
            if not has_new and not has_cont:
                complete_puids = [p for p, r in experience_pool.items() if len(r) >= n]
                if not complete_puids:
                    print("[SegmentPipeline] No data remaining, ending training")
                    break
                # Fall through to training phase below

            # ═══════════════════════════════════════════════════════
            # Phase 2: Generation (single wake_up/sleep cycle)
            #   - seg1 for new prompts via AgentLoop workers
            #   - continuation for unfinished via direct vLLM server calls
            #   - both dispatched concurrently to vLLM's continuous batching
            # ═══════════════════════════════════════════════════════
            if has_new or has_cont:
                gen_start = time.time()
                self.async_rollout_manager.wake_up()

                # ── 2a: New prompts → AgentLoop workers ──
                seg1_futures = None
                if has_new:
                    gen_batch = new_batch_info["gen_batch"]
                    gen_batch.meta_info["max_tokens"] = seg_len
                    workers = self.async_rollout_manager.agent_loop_workers
                    chunks = gen_batch.chunk(len(workers))
                    seg1_futures = [
                        worker.generate_sequences.remote(chunk)
                        for worker, chunk in zip(workers, chunks)
                    ]

                # ── 2b: Continuations → direct vLLM server calls ──
                cont_futures = {}
                if has_cont:
                    sampling_params = self._get_sampling_params_for_continuation(seg_len)
                    server_handles = self.async_rollout_manager.server_handles
                    num_servers = len(server_handles)
                    for idx, (uid, state) in enumerate(unfinished_pool.items()):
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
                if seg1_futures:
                    seg1_outputs = ray.get(seg1_futures)
                    seg1_output = DataProto.concat(seg1_outputs)
                    seg1_total = new_batch_info["total_samples"]
                    seg1_finished = self._process_seg1_results(
                        seg1_output, new_batch_info, seg_len, n,
                        unfinished_pool, experience_pool,
                    )
                    print(f"[SegmentPipeline] Step {pipeline_step} | Seg1: "
                          f"{seg1_finished}/{seg1_total} finished "
                          f"({seg1_finished * 100 // seg1_total}%)")

                # ── Collect continuation results ──
                cont_total = 0
                cont_finished = 0
                if cont_futures:
                    cont_total = len(cont_futures)
                    cont_finished, eos_count, max_count = self._process_continuation_results(
                        cont_futures, seg_len, max_response_length, num_segments,
                        unfinished_pool, experience_pool,
                    )
                    print(f"[SegmentPipeline] Step {pipeline_step} | Continuation: "
                          f"{cont_finished}/{cont_total} finished "
                          f"(eos={eos_count}, maxlen={max_count}), "
                          f"{len(unfinished_pool)} still unfinished")

                self.async_rollout_manager.sleep()
                gen_time_accum += time.time() - gen_start

            # ═══════════════════════════════════════════════════════
            # Phase 3: Train if enough complete GRPO groups
            # ═══════════════════════════════════════════════════════
            complete_puids = [p for p, r in experience_pool.items() if len(r) >= n]

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

                print(f"[SegmentPipeline] Training step {self.global_steps}: "
                      f"{num_train_groups} groups ({num_train_samples} samples)")

                with marked_timer("train", timing_raw, color="green"):
                    train_batch = self._assemble_training_batch_from_pool(
                        complete_puids, experience_pool, n,
                        prompt_length, max_response_length, pad_token_id,
                    )
                    # Remove trained groups from pool BEFORE training
                    # (so pool reflects post-train state in logs)
                    for puid in complete_puids:
                        del experience_pool[puid]

                    trained_batch = self._do_train_step(train_batch, metrics, timing_raw)

                # ── Metrics ──
                metrics.update(compute_data_metrics(batch=trained_batch, use_critic=self.use_critic))

                train_duration = time.time() - train_start
                timing_raw["gen"] = gen_time_accum
                timing_raw["step"] = gen_time_accum + train_duration
                self.max_steps_duration = max(self.max_steps_duration, timing_raw["step"])
                gen_time_accum = 0.0  # reset after training

                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_timing_metrics(batch=trained_batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(
                    batch=trained_batch, timing_raw=timing_raw, n_gpus=n_gpus,
                ))

                # Step summary
                timing_str = " | ".join(f"{k}={v:.1f}s" for k, v in timing_raw.items())
                pool_remaining = sum(len(r) for r in experience_pool.values())
                print(f"[SegmentPipeline] === Train Step {self.global_steps} "
                      f"(pipeline={pipeline_step}) === "
                      f"groups={num_train_groups} | "
                      f"unfinished={len(unfinished_pool)} | "
                      f"pool_remaining={pool_remaining} | "
                      f"{timing_str}")

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": current_epoch,
                    "segment_rollout/pipeline_step": pipeline_step,
                    "segment_rollout/train_groups": num_train_groups,
                    "segment_rollout/train_samples": num_train_samples,
                    "segment_rollout/unfinished_pool_size": len(unfinished_pool),
                    "segment_rollout/experience_pool_samples": pool_remaining,
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
                # No training this pipeline step — just log progress
                pool_total = sum(len(r) for r in experience_pool.values())
                print(f"[SegmentPipeline] Step {pipeline_step}: gen only | "
                      f"unfinished={len(unfinished_pool)} | "
                      f"pool_samples={pool_total} | "
                      f"complete_groups={len(complete_puids)} (need {min_train_groups})")

        progress_bar.close()
        print("[SegmentPipeline] Training complete.")
