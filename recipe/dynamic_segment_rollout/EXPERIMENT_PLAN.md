# Dynamic Segment Rollout — 实验方案

## 方法概述

**Budget-Aware Dynamic Segment Rollout**：在 cross-step pipeline 训练中，用 Token 预算动态计算每步的段长，替代固定段长。

**核心公式**：
```
L_next = Clamp(Total_Token_Capacity / N_active, L_min, L_max)
```

- **清洗阶段**：N_active 大 → L 小（如 64）→ 快速筛除简单样本
- **冲刺阶段**：N_active 小 → L 大（如 3072）→ 全速跑完长尾

---

## 一、动机实验

> **目的**：证明响应长度分布有显著不均匀性，固定段长存在浪费，动态方法有理论必要性。

### 1.1 响应长度分布分析

- 跑一次标准 GRPO 训练（`ray_trainer.py`），收集 `Rollout_response/sync/` 下的长度数据
- 画出长度分布直方图
- 计算关键统计量：CV（变异系数）、p10/p50/p90 分位数、max/median 比值

**预期结论**：CV > 0.5，p90/p10 > 5，说明响应长度分布高度不均匀。

### 1.2 固定段长的效率浪费分析

- 用 `segment_trainer.py` 跑一次训练，读取 tensorboard 数据
- 重点观察指标：
  - `train_resp_len/pct_1seg`：一段就完成的样本比例（越高说明浪费越严重）
  - `train_resp_len/pct_max_seg`：跑满所有段的样本比例
  - `pipe/seg1_finish_rate`：第一段就结束的比率

**预期结论**：存在大量样本一段内即完成（短样本），同时存在少量样本跑满所有段（长尾），两者混在同一个固定段长下效率低下。

---

## 二、主对比实验

> **目的**：证明 Dynamic Segment 在达到相同准确率时，时钟时间/吞吐量更优。

### 实验设置

| 对照组 | Trainer | 配置 |
|--------|---------|------|
| Baseline | `ray_trainer.py` | 标准 GRPO，无流水线 |
| Fixed Segment | `segment_trainer.py` | `num_segments=4`，固定 `seg_len=768` |
| **Dynamic Segment** | `dynamic_segment_trainer.py` | `l_min=64, l_max=3072, capacity=393216` |

**统一控制变量**：
- 模型：Qwen2.5-0.5B-Instruct
- 数据：MATH Level 1–3 过滤子集
- `train_batch_size=128`，`n=8`，`max_response_length=3072`
- 总训练 steps 相同

### 需要记录的指标

#### 第一类：训练效果（回答"有没有更好"）

| Tensorboard 指标 | 记录粒度 | 说明 |
|-----------------|---------|------|
| `critic/score/mean` | train step | 平均奖励，反映模型学到了多少 |
| `actor/pg_loss` | train step | policy gradient loss |
| `actor/kl` | train step | KL 散度，训练稳定性 |

#### 第二类：吞吐效率（回答"有没有更快"，核心 claim）

| Tensorboard 指标 | 记录粒度 | 说明 |
|-----------------|---------|------|
| `pipe/tokens_per_sec` | pipeline step | 每步实际生成速度 |
| `pipe/gen_time` | pipeline step | 每步生成耗时 |
| `pipe/budget_utilization` | pipeline step | 实际生成 tokens / capacity，越接近 1 越好 |
| `dynamic_segment/pipeline_step` | train step | 每个 train step 消耗的 pipeline step 数，越小越高效 |

> Fixed Segment 对照：手动计算 `实际生成tokens / (n_active × fixed_seg_len)` 作为利用率参考

#### 第三类：动态段长行为（回答"为什么更好"）

| Tensorboard 指标 | 记录粒度 | 说明 |
|-----------------|---------|------|
| `pipe/dynamic_seg_len` | pipeline step | 段长随时间变化，验证 cleaning→sprint 相变 |
| `pipe/seg1_finish_rate` | pipeline step | 第一段完成率，cleaning 阶段应显著高于 Fixed |
| `train_resp_len/cv` | train step | 训练批次响应长度变异系数 |
| `train_resp_len/grpo_cv_mean` | train step | GRPO 组内长度方差，越小越好 |
| `train_resp_len/pct_1seg` | train step | 一段内完成的样本比例 |

#### 最简判断（只看 5 个）

```
1. critic/score/mean              ← 效果有没有差
2. pipe/tokens_per_sec            ← 速度有没有提升
3. pipe/dynamic_seg_len           ← 动态行为是否按预期变化
4. dynamic_segment/pipeline_step  ← 收敛效率
5. train_resp_len/cv              ← 批次质量
```

前两个决定论文能不能发，第三个决定故事能不能讲。

---

## 三、消融实验

> **目的**：验证各设计选择的有效性，确定最优超参。

### 3.1 `total_token_capacity` 取值

| 组别 | capacity | 物理含义 |
|------|----------|---------|
| 0.5x | 196608 | 激进清洗，段长更小 |
| **1x（默认）** | **393216** | `128 × 3072` |
| 2x | 786432 | 温和过渡，段长更大 |

### 3.2 `l_min` 的影响

取值：32 / **64** / 128 / 256

- 过小：清洗阶段生成极短，调度开销比例上升
- 过大：简单样本无法在第一段内完成，失去清洗效果

### 3.3 `l_max` 的影响

取值：`max_response_length` 的 50% / 75% / **100%**

- 小于 max_response_length：长尾样本仍需多轮
- 等于 max_response_length：冲刺阶段一次跑完

### 3.4 是否移除 `num_segments` 限制

| 组别 | 结束条件 |
|------|---------|
| 有限制（Fixed Segment） | EOS \| maxlen \| seg_count >= num_segments |
| **无限制（Dynamic Segment）** | EOS \| maxlen |

验证去掉硬性段数限制是否提升长样本的完成质量。

### 3.5 `min_train_groups` 的影响

取值：8 / 32 / **64** / 128

- 过小：训练批次小，梯度噪声大
- 过大：需要积累更多样本，pipeline 延迟增加

---

## 四、分析实验

> **目的**：解释为什么有效，提供可视化证据。

### 4.1 动态段长变化曲线

- Tensorboard 指标：`pipe/dynamic_seg_len`
- 观察训练过程中段长从 `l_min` 向 `l_max` 过渡的过程
- **预期图形**：早期（N_active 大）段长在 l_min 附近震荡；随着简单样本被清洗，段长逐渐升高趋近 l_max

### 4.2 Budget 利用率分析

- Tensorboard 指标：`pipe/budget_utilization`（实际生成 tokens / total_token_capacity）
- 对比 Fixed Segment 的实际 token 利用率
- **预期结论**：Dynamic Segment 的利用率更均匀，Fixed Segment 在早期有大量 padding 浪费

### 4.3 GRPO 组内长度方差对比

- Tensorboard 指标：`train_resp_len/grpo_cv_mean`、`train_resp_len/grpo_cv_max`
- 对比 Fixed vs Dynamic 的组内长度 CV
- **预期结论**：Dynamic Segment 减少组内长度不均衡（同一 prompt 的 N 个响应长度更集中）

### 4.4 清洗效率分析

- Tensorboard 指标：`pipe/seg1_finish_rate`（第一段就完成的比例）
- 对比 Fixed vs Dynamic 在清洗阶段的快速淘汰率
- **预期结论**：Dynamic Segment 清洗阶段的 finish_rate 更高（因为 l_min 更小，简单样本更容易在第一段完成）

---

## 五、扩展性实验（加分项）

### 5.1 不同模型规模

| 模型 | 参数量 |
|------|--------|
| Qwen2.5-0.5B-Instruct | 0.5B |
| Qwen2.5-1.5B-Instruct | 1.5B |
| Qwen2.5-7B-Instruct | 7B |

验证动态段长收益是否随模型规模变化（越大模型响应越长，理论上收益越大）。

### 5.2 不同数据集

| 数据集 | 特点 | 预期 |
|--------|------|------|
| GSM8K | 短响应（200–600 token），分布窄 | Dynamic 收益小 |
| MATH L1–3 | 中长响应（500–2000 token），分布宽 | Dynamic 收益大 |

**预期结论**：响应长度分布越宽（CV 越高），Dynamic Segment 相对 Fixed 的收益越大。这可以作为论文的一个 general claim。

### 5.3 不同 `n`（GRPO 组大小）

取值：n = 4 / **8** / 16

验证 n 越大时，Dynamic Segment 是否能更好地利用组内完成时序的差异。

---

## 六、优先级与实验顺序

```
阶段一（必须）：
  1. 跑 segment_trainer，收集长度分布数据（验证动机）
  2. 三组主对比实验（Baseline / Fixed / Dynamic）

阶段二（强烈建议）：
  3. 消融：total_token_capacity × l_min（3.1 + 3.2）
  4. 分析：段长变化曲线 + budget 利用率（4.1 + 4.2）

阶段三（有时间做）：
  5. 消融：l_max + num_segments 限制（3.3 + 3.4）
  6. 分析：GRPO 组内方差（4.3 + 4.4）
  7. 扩展：不同数据集（5.2）
  8. 扩展：不同模型规模（5.1）
```

---

## 七、核心 Claim（论文贡献点）

1. **效率 Claim**：在达到相同准确率的前提下，Dynamic Segment 比 Fixed Segment 时钟时间更短
2. **适用性 Claim**：响应长度分布越不均匀（CV 越高），收益越大；可以用 CV 作为是否使用 Dynamic Segment 的判断依据
3. **分析 Claim**：两阶段（cleaning + sprint）的语义可以通过 `pipe/dynamic_seg_len` 曲线直接可视化验证
