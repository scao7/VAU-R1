
# export WANDB_PROJECT=Video-GRPO
export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
# export NCCL_SOCKET_IFNAME=bond0
# # export NCCL_DEBUG="INFO"
# export NCCL_IB_HCA=mlx5_0

export WANDB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")

export PYTHONPATH=".:$PYTHONPATH"
OUTDIR=./checkpoints/$WANDB_NAME

export DEBUG_MODE="true"
export LOG_PATH="./logs/${WANDB_NAME}.log"
export TOKENIZERS_PARALLELISM=false


# srun accelerate launch --config_file=/mnt/petrelfs/yanziang/videoo1/TimeZero/configs/zero3.yaml 
torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12951 \
    src/sft/sft_cls.py \
    --deepspeed training_scripts/zero2_offload.json \
    --model_name_or_path ../huggingface/Qwen2-VL-2B-Instruct \
    --train_data_path path_to_train_csv \
    --eval_data_path path_to_val_csv \
    --train_video_folder /home/zhuliyun/dataset/msad/MSAD_train \
    --eval_video_folder /home/zhuliyun/dataset/msad/MSAD_train \
    --dataset_name all \
    --learning_rate 2.0e-5 \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing false\
    --bf16 \
    --torch_dtype bfloat16 \
    --logging_steps 5 \
    --eval_strategy no \
    --report_to tensorboard \
    --output_dir $OUTDIR \
    --save_steps 200 \
    --save_only_model true\
    --use_peft true \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_r 8 

