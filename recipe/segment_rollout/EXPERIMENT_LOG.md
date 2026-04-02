# Segment Rollout 实验记录

基于 UloRL 论文 (arxiv 2507.19766) 的 cross-step pipeline 实现，在 verl v0.7.0 同步 GRPO 训练框架上。

## 实验环境

- 模型: Qwen2.5-0.5B-Instruct
- 硬件: 2x NPU (Ascend)
- 数据集: MATH (train.parquet / test.parquet)
- 训练配置:
  - train_batch_size=128, max_prompt_length=2048, max_response_length=3072
  - rollout n=8 (每个 prompt 生成 8 个回复, GRPO)
  - lr=5e-7, ppo_mini_batch_size=64, ppo_micro_batch_size_per_gpu=2
  - use_kl_loss=True, kl_loss_coef=0.001, entropy_coeff=0.001
  - tensor_model_parallel_size=2, gpu_memory_utilization=0.7
  - total_epochs=1 (58 training steps), test_freq=5

---

## 实验 1: Baseline (同步 GRPO)

- 日志文件: `sync_training_log_20260322_214746.txt`
- 运行命令: 标准 verl GRPO 训练 (无 segment rollout)
- 每个 step 固定处理 128 groups (1024 samples)

### 时间指标 (稳定后 step 2-58 平均)

| 指标 | 数值 |
|------|------|
| gen (生成) | ~338s |
| old_log_prob | ~73s |
| ref_log_prob | ~66s |
| update_actor | ~213s |
| train (总训练) | ~353s |
| **step (总)** | **~691s** |

### 精度指标 (val acc on MATH)

| Step | Accuracy |
|:----:|:--------:|
| 0 (初始) | 0.3244 |
| 5 | 0.3238 |
| 10 | 0.3172 |
| 15 | 0.3236 |
| 20 | 0.3214 |
| 25 | 0.3258 |
| 30 | 0.3292 |
| 35 | 0.3284 |
| 50 | 0.3356 |
| 55 | 0.3288 |
| **58 (最终)** | **0.3396** |

### 总训练时间

约 58 × 691s = ~40078s ≈ **11.1 小时** (不含 validation 时间)

---

## 实验 2: Segment Rollout (min_train_groups=8)

- 日志文件: `segment_train_log_20260323_173342.txt`
- 运行命令: `python3 -m recipe.segment_rollout.main_segment` (segment_rollout.num_segments=4, segment_rollout.min_train_groups=8)
- Segment 配置: num_segments=4, seg_len=768

### 流程说明

UloRL cross-step pipeline:
1. 每个 pipeline step 注入 128 个新 prompt (seg1, 最多生成 768 token)
2. 同时对上一轮未完成的样本进行 continuation (续写 768 token)
3. seg1 和 continuation 在同一个 vLLM wake_up/sleep 周期内并发执行
4. 完成的样本进入 experience_pool, 当某个 prompt 的 8 个回复全部完成时形成 complete group
5. complete groups >= min_train_groups 时触发训练

### 关键观察

- seg1 完成率: ~77% (约 790/1024 个样本在第一段内完成)
- continuation 完成率: ~93% (上一轮未完成的样本, 大部分在一轮 continuation 后完成)
- 每个 pipeline step 都触发训练 (因为 min_train_groups=8 很低)
- 训练 groups 范围: 68-152, 平均 ~128

### 时间指标 (step 3-58 稳定后平均)

| 指标 | 数值 | vs Baseline |
|------|------|-------------|
| gen (生成) | ~133s | **-61%** |
| old_log_prob | ~72s | 基本持平 |
| ref_log_prob | ~65s | 基本持平 |
| update_actor | ~210s | 基本持平 |
| train (总训练) | ~348s | 基本持平 |
| **step (总)** | **~482s** | **-30%** |

### 精度指标 (val acc on MATH)

| Step | Accuracy |
|:----:|:--------:|
| 0 (初始) | 0.3192 |
| 5 | 0.3218 |
| 10 | 0.3232 |
| 15 | 0.3262 |
| 20 | 0.3274 |
| 25 | 0.3332 |
| 30 | 0.3294 |
| 35 | 0.3330 |
| 40 | 0.3288 |
| 45 | 0.3322 |
| 50 | 0.3340 |
| 55 | 0.3306 |
| **58 (最终)** | **0.3296** |

### 每步训练 Groups 数

| Train Step | Groups | Train Step | Groups | Train Step | Groups | Train Step | Groups |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **68** | 16 | 124 | 31 | 113 | 46 | 137 |
| 2 | **97** | 17 | 133 | 32 | 130 | 47 | 128 |
| 3 | 124 | 18 | 120 | 33 | 143 | 48 | 132 |
| 4 | 128 | 19 | 123 | 34 | 109 | 49 | 118 |
| 5 | 126 | 20 | 130 | 35 | 131 | 50 | 124 |
| 6 | 134 | 21 | 129 | 36 | 136 | 51 | 152 |
| 7 | 131 | 22 | 131 | 37 | 122 | 52 | 121 |
| 8 | 124 | 23 | 140 | 38 | 129 | 53 | 126 |
| 9 | 119 | 24 | 134 | 39 | 137 | 54 | 125 |
| 10 | 128 | 25 | 102 | 40 | 130 | 55 | 136 |
| 11 | 129 | 26 | 136 | 41 | 126 | 56 | 120 |
| 12 | 138 | 27 | 141 | 42 | 125 | 57 | 130 |
| 13 | 124 | 28 | 120 | 43 | 116 | 58 | 130 |
| 14 | 143 | 29 | 127 | 44 | 132 | | |
| 15 | 121 | 30 | 139 | 45 | 122 | | |

- 前两步偏少 (68, 97): pipeline 刚启动, pool 未积累, Step 1 只有 seg1 没有 continuation
- Step 3 起稳定在 109-152, **平均约 128**, 与 baseline 的固定 128 基本一致
- min_train_groups=8 实际上并未导致训练 batch 过小 (除启动阶段), 因为每个 pipeline step 自然产出 ~128 个 complete groups

### 加速比与精度对比

| | Baseline | Segment (min=8) | 差异 |
|---|:---:|:---:|:---:|
| 平均 step 时间 | 691s | 482s | **-30% (1.43x)** |
| 预估总训练时间 | ~11.1h | ~7.8h | **-3.3h** |
| 最终精度 | 0.3396 | 0.3296 | **-1.0%** |

### 分析

- 生成时间大幅降低: 338s → 133s, 因为 seg1 只生成 768 token (而非 3072), 不用等长尾
- 训练时间基本不变: train pipeline (old_log_prob + ref + update_actor) 不受 segment rollout 影响
- 精度略有下降 (-1.0%): 可能因为 step 1 只有 68 groups (544 samples) 训练, 且训练 batch size 波动大 (68-152)

---

## 实验 3: Segment Rollout (min_train_groups=128)

- 日志文件: `segment_train_log_20260324_151332.txt` (**未跑完, 只到 step 10**)
- 运行命令:
```bash
python3 -m recipe.segment_rollout.main_segment \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/ma-user/work/data/math/train.parquet \
    data.val_files=/home/ma-user/work/data/math/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/home/ma-user/work/models/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='segment_rollout_grpo_math' \
    trainer.experiment_name='qwen2_0.5b_segment_rollout_update_log_martrics_min_train_groups_128' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.device=npu \
    +segment_rollout.num_segments=4 \
    +segment_rollout.min_train_groups=128
```

### 关键观察: 大小步交替问题

min_train_groups=128 导致了明显的"大小步交替"现象:

- 每个 pipeline step 大约产出 100-130 个 complete groups
- 有时刚好超过 128 → 立即训练 (小 step, ~130 groups)
- 有时差一点不够 128 → 需要再跑一个 pipeline step 积累 → 攒到 ~250 groups (大 step)

| Train Step | Pipeline Step | Groups | Train(s) | Gen(s) | Step(s) | 类型 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2 | 164 | 441 | 248 | 689 | 中 |
| 2 | 4 | **257** | **689** | 275 | **964** | 大 |
| 3 | 5 | 139 | 371 | 134 | 505 | 小 |
| 4 | 7 | **249** | **665** | 263 | **928** | 大 |
| 5 | 8 | 135 | 361 | 131 | 493 | 小 |
| 6 | 10 | **237** | **641** | 271 | **912** | 大 |
| 7 | 11 | 139 | 379 | 138 | 517 | 小 |
| 8 | 12 | 133 | 363 | 134 | 496 | 小 |
| 9 | 13 | 129 | 351 | 134 | 485 | 小 |
| 10 | 15 | **251** | **687** | 263 | **950** | 大 |

### 问题分析

- 大 step (~250 groups): 训练 ~680s, 总 ~940s → **比 baseline 还慢!**
- 小 step (~130 groups): 训练 ~360s, 总 ~500s → 与 min_train_groups=8 相当
- **平均 step 时间: 694s ≈ baseline 的 691s, 加速效果被完全抵消**

### 根本原因

seg1 完成率 ~77%, 每个 pipeline step 大约只能产出 128×77%≈98 个 complete groups (加上 continuation 完成的约 120-130 个)。这个数字刚好在 128 附近波动:
- 够了 → 训练 ~130 groups (正常)
- 差一点不够 → 多等一轮 → 积累到 ~250 groups → 训练量翻倍

---

## 总结

| 实验 | min_train_groups | 平均 step 时间 | 最终精度 | 加速比 |
|------|:---:|:---:|:---:|:---:|
| Baseline | - | 691s | 0.3396 | 1.00x |
| Segment (min=8) | 8 | 482s | 0.3296 | **1.43x** |
| Segment (min=128) | 128 | 694s | 未完成 | ~1.00x |

### 结论

1. **Segment rollout 方案有效**, gen 时间从 338s 降至 133s (-61%)
2. **效果取决于 min_train_groups 参数**: 设得太高 (128) 导致大小步交替, 抵消加速效果
3. 精度略有下降 (-1.0%), 可能与训练 batch size 波动有关
4. 同步框架下的理论加速上限: 130(gen) + 350(train) = 480s → **1.44x** (受限于 gen/train 串行)

### 待办

- [ ] 尝试 min_train_groups=64 或 96, 平衡触发频率和训练 batch size 稳定性
- [ ] 动态 continuation: seg_len 改为动态 (remaining = max_response_length - 已生成长度), 减少 continuation 轮次
- [ ] 验证精度差距是否可通过稳定训练 batch size 缩小
- [ ] 评估 POIS 和 DMMPTs 的效果
- [ ] 评估 P-DSR 探路者分流思路在同步框架下的可行性
