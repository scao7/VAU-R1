export WANDB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
export PYTHONPATH=".:$PYTHONPATH"
OUTDIR=./checkpoints/$WANDB_NAME

export DEBUG_MODE="true"
export LOG_PATH="./logs/${WANDB_NAME}.log"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
export TORCH_CUDA_ARCH_LIST="86"  
OUTDIR=./checkpoints/$WANDB_NAME


    
    python src/open_r1/grpo_qa.py \
    --deepspeed scripts/training/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --train_data_path /data/Shengting/AFTER_GRADUATE/VAU-R1/annotations/ucf_subset_train.csv \
    --eval_data_path /data/Shengting/AFTER_GRADUATE/VAU-R1/annotations/ucf_subset_val.csv \
    --train_video_folder /data/Shengting/AFTER_GRADUATE/VAU-R1-old/video_data/train \
    --eval_video_folder /data/Shengting/AFTER_GRADUATE/VAU-R1-old/video_data/val \
    --dataset_name All \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --num_generations 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation sdpa \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --report_to tensorboard \
    --save_steps 100 \
    --save_total_limit 3 \


