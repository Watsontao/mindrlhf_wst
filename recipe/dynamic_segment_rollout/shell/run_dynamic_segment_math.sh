#!/usr/bin/env bash
set -x

export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ATTENTION_BACKEND=XFORMERS

# 1. Project and experiment naming
PROJECT_NAME="dynamic_segment_rollout_grpo"
EXPERIMENT_NAME="8npu_7b_math_gsm8k_demo02"

# 2. Log directory
LOG_DIR="/home/ma-user/work/verl/mylog/${PROJECT_NAME}/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/dynamic_segment_train_log_$(date +%Y%m%d_%H%M%S).txt"
echo "Logs will be saved to: ${LOG_FILE}"



gsm8k_train_path=/home/ma-user/work/data/gsm8k/train.parquet
gsm8k_test_path=/home/ma-user/work/data/gsm8k/test.parquet
math_train_path=/home/ma-user/work/data/math/train.parquet
math_test_path=/home/ma-user/work/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"


# Dynamic segment config (Adaptive Over-Provision):
#   over_provision_ratio = 2.0  (dispatch 2x target_completions requests each step, APRIL-style)
#   target_completions = 512    (n * min_train_groups = 8 * 64)
#   target_finish_rate = 0.6    (EMA adapts seg1_len so ~60% of seg1 samples finish naturally)
#   seg1_len_init = 768         (initial seg1_len before EMA adapts it)
#   ema_alpha = 0.3             (EMA smoothing factor)
#   l_min = 64, l_max = 3072
python3 -m recipe.dynamic_segment_rollout.main_dynamic_segment \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/home/ma-user/work/models/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 \
    trainer.device=npu \
    +dynamic_segment_rollout.l_min=64 \
    +dynamic_segment_rollout.l_max=3072 \
    +dynamic_segment_rollout.over_provision_ratio=2.0 \
    +dynamic_segment_rollout.target_completions=512 \
    +dynamic_segment_rollout.target_finish_rate=0.6 \
    +dynamic_segment_rollout.seg1_len_init=768 \
    +dynamic_segment_rollout.ema_alpha=0.3 \
    +dynamic_segment_rollout.min_train_groups=128 \
    +dynamic_segment_rollout.greso_window=3 \
    +dynamic_segment_rollout.greso_min_history=2 \
    +dynamic_segment_rollout.sorted_rl=True \
    $@ 2>&1 | tee ${LOG_FILE}
