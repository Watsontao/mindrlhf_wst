set -x

export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ATTENTION_BACKEND=XFORMERS

# 1. 提取公共变量，方便统一管理
PROJECT_NAME="qwen2-5-0-5b-baseline"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# 将基础名字加上时间戳，作为实验名和日志名
EXPERIMENT_NAME="2npu-baseline01_${TIMESTAMP}"

# 2. 定义日志外层文件夹 (名字和 project_name 保持一致)
LOG_DIR="/home/ma-user/work/verl/mylog/${PROJECT_NAME}"

# 确保这个日志文件夹存在，如果不存在则自动创建
mkdir -p "${LOG_DIR}"

# 3. 定义日志文件名 (和 experiment_name 保持完全一致)
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.txt"
echo "Logs will be saved to: ${LOG_FILE}"

# 使用 2>&1 | tee 将标准输出和错误输出同时写入文件和屏幕
python3 -m verl.trainer.main_ppo \
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
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.device=npu $@ 2>&1 | tee "${LOG_FILE}"