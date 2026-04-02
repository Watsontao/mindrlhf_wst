#!/usr/bin/env bash
set -x

export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ATTENTION_BACKEND=XFORMERS

LOG_FILE="/home/ma-user/work/verl//segment_train_log_$(date +%Y%m%d_%H%M%S).txt"
echo "Logs will be saved to: ${LOG_FILE}"

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
    +segment_rollout.min_train_groups=128 \
    $@ 2>&1 | tee ${LOG_FILE}
