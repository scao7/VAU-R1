export WANDB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
export PYTHONPATH=".:$PYTHONPATH"
OUTDIR=./checkpoints/$WANDB_NAME

# export DEBUG_MODE="true"
export LOG_PATH="./logs/${WANDB_NAME}.log"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1
# export OMP_NUM_THREADS=1
# export DISABLE_ADDMM_CUDA_LT=1
# export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1


# IMPORTANT: skip JIT/CUDA builds for DS & Torch extensions
# export DS_SKIP_CUDA_BUILD=1
# export DS_BUILD_OPS=0
# export TORCH_CUDA_ARCH_LIST=8.6 
# # NCCL safety for single-node, multi-GPU
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=0
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

OUTDIR=./checkpoints/$WANDB_NAME

torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=10668 \
  src/open_r1/grpo_qa.py \
  --deepspeed /home/scao/myproject/VAU-R1/scripts/training/zero3_offload.json \
  --output_dir "$OUTDIR" \
  --model_name_or_path "Qwen/Qwen2-0.5B-Instruct" \
  --train_data_path /home/scao/myproject/VAU-R1/annotations/train.csv \
  --eval_data_path /home/scao/myproject/VAU-R1/annotations/val.csv \
  --train_video_folder /home/scao/myproject/VAU-R1/organized_data \
  --eval_video_folder /home/scao/myproject/VAU-R1/organized_data \
  --dataset_name All \
  --max_prompt_length 1024 \
  --max_completion_length 1024 \
  --num_generations 4 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --logging_steps 1 \
  --bf16 True\
  --data_seed 42 \
  --gradient_checkpointing \
  --attn_implementation flash_attention_2 \
  --num_train_epochs 1 \
  --run_name "$WANDB_NAME" \
  --report_to tensorboard \
  --save_steps 100 \
  --save_total_limit 3 \
  --save_only_model
