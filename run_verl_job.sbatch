#!/bin/bash
#SBATCH --partition=mcml-dgx-a100-40x8,mcml-hgx-a100-80x4,mcml-hgx-a100-80x4-mig,mcml-hgx-h100-94x4
#SBATCH --cpus-per-task=8
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --time=1:35:10
#SBATCH --job-name=verl_init_training



LOG_DIR="logs/$(date +%Y-%m-%d)"
mkdir -p "$LOG_DIR"


DATE=$(date +%H%M%S) # Using H-M-S for the filename timestamp
JOB_ID="${SLURM_JOB_ID}"
LOG_OUT="${LOG_DIR}/verl_${DATE}_${JOB_ID}.out"
LOG_ERR="${LOG_DIR}/verl_${DATE}_${JOB_ID}.err"
SYSINFO_LOG="${LOG_DIR}/sysinfo_${DATE}_${JOB_ID}.log"


exec > "$LOG_OUT" 2> "$LOG_ERR"

set -x


source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl


{
    echo "========== SYSTEM INFO =========="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "User: $(whoami)"
    echo "SLURM Node: ${SLURMD_NODENAME}"
    echo "SLURM Job ID: ${SLURM_JOB_ID}"
    echo "SLURM Job Node List: ${SLURM_JOB_NODELIST}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-'(not set)'}"
    echo ""

    echo "----- GPU Info -----"
    nvidia-smi
    echo ""
    echo "Driver Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    echo "GPU Name(s): $(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)"
    echo "CUDA Architecture(s): $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | uniq)"
    echo "nvcc Version (from env): $(nvcc --version | grep release)"
    echo ""

    echo "----- CPU Info -----"
    lscpu
    echo ""
    echo "CPU Cores: $(lscpu | grep '^CPU(s):' | awk '{print $2}')"
    echo "Model Name: $(lscpu | grep 'Model name' | awk -F ':' '{print $2}' | xargs)"
    echo "Threads per core: $(lscpu | grep 'Thread(s) per core' | awk -F ':' '{print $2}' | xargs)"
    echo "Architecture: $(lscpu | grep 'Architecture' | awk -F ':' '{print $2}' | xargs)"
    echo "Total RAM: $(free -h | grep Mem | awk '{print $2}')"
    echo ""

    echo "----- Python Environment -----"
    which python
    python --version
    echo "Installed packages:"
    pip list
} > "${SYSINFO_LOG}"



cd ~/verl

export CUDA_LAUNCH_BLOCKING=1
export NCCL_P2P_DISABLE=1

ENGINE=${1:-vllm}

HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.val_files=test_docvideo_large_dict_prompt_allsame_nframes.parquet \
    data.train_files=train_docvideo_large_dict_prompt_allsame_nframes.parquet \
    data.train_batch_size=4 \
    data.max_prompt_length=8000 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_geo3k' \
    trainer.experiment_name='qwen2_5_vl_3b_function_rm' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 "$@" 


# +actor_rollout_ref.model.dtype=half \
# +actor_rollout_ref.model.torch_dtype=float16   
# +actor_rollout_ref.rollout.engine_kwargs.vllm.dtype=half    
#     actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
#     ray_init.num_cpus=8 \

