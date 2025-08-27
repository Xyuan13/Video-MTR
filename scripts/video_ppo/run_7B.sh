# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0

export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NCCL_SOCKET_IFNAME=lo
QWEN_VL_7B_MAX_MODEL_LEN=8192

# ADD IMAGE_FACTOR as enviroment variables
export VIDEOENV_IMAGE_FACTOR=14


MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"


if [ "$VIDEOENV_IMAGE_FACTOR" == "14" ]; then
    MAX_PROMPT_LENGTH=4096
    MAX_TRAJECTORY_LENGTH=5120
else
    MAX_PROMPT_LENGTH=6400
    MAX_TRAJECTORY_LENGTH=7168
fi


MAX_RESPONSE_LENGTH=512
GPU_NUM=8 
TRAIN_BATCH_SIZE=$((2 * GPU_NUM)) 
MAX_MODEL_LEN=$QWEN_VL_7B_MAX_MODEL_LEN


MAX_NUM_BATCHED_TOKENS=$((QWEN_VL_7B_MAX_MODEL_LEN * GPU_NUM )) 

MAX_TURNS=3
WINDOW_SIZE=2 
EXPERIMENT_BASE_NAME="video_ppo_7B_0826" 

export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "MAX_TRAJECTORY_LENGTH: $MAX_TRAJECTORY_LENGTH"
echo "EXPERIMENT_NAME: $EXPERIMENT_BASE_NAME"

# Print video_data_config_path yaml file content
echo "Reading env_config from: $SCRIPT_DIR/env_config.yaml"
VIDEO_DATA_CONFIG_PATH=$(grep -v "^[[:space:]]*#" "$SCRIPT_DIR/env_config.yaml" | grep "video_data_config_path:" | head -1 | sed 's/.*video_data_config_path: *"\(.*\)".*/\1/')
# Expand $SCRIPT_DIR in the config path
VIDEO_DATA_CONFIG_PATH=$(eval echo "$VIDEO_DATA_CONFIG_PATH")
echo "=================================================="
echo "video_data_config_path: $VIDEO_DATA_CONFIG_PATH"
echo "=================================================="
cat "$VIDEO_DATA_CONFIG_PATH"
echo "=================================================="

# create dataset
python -m vagen.env.create_dataset \
    --yaml_path "$SCRIPT_DIR/env_config.yaml" \
    --train_path "data/parquet_files/train.parquet" \
    --test_path "data/parquet_files/test.parquet" \
    --force_gen

# # train
python3 -m vagen.trainer.main_ppo \
    algorithm.adv_estimator=bi_level_gae \
    algorithm.high_level_gamma=1 \
    data.train_files=data/parquet_files/train.parquet \
    data.val_files=data/parquet_files/test.parquet \
    data.val_batch_size=$GPU_NUM \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.max_trajectory_length=$MAX_TRAJECTORY_LENGTH \
    data.image_key=images \
    data.truncation=left \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.limit_mm_per_prompt=32 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.temperature=0.7 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$MODEL_PATH \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_mini_batch_size=8 \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='video-mtr_demo' \
    trainer.experiment_name="$EXPERIMENT_BASE_NAME" \
    trainer.n_gpus_per_node=$GPU_NUM \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_training_steps=500 \
    rollout_manager.max_turns=$MAX_TURNS \
    rollout_manager.window_size=$WINDOW_SIZE \
    rollout_manager.use_multi_turn_reward=True \
    rollout_manager.use_loss_mask=True \
    rollout_manager.use_gae_mask=True \
    trainer.val_before_train=True \
    trainer.val_generations_to_log_to_wandb=4 \
    rollout_manager.n_trajectory=2 \
    2>&1 | tee train.log
