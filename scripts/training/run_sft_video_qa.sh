# export WANDB_PROJECT=Video-GRPO
export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1

export WANDB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")

export PYTHONPATH=".:$PYTHONPATH"
OUTDIR=./checkpoints/$WANDB_NAME

export DEBUG_MODE="true"
export LOG_PATH="./logs/${WANDB_NAME}.log"
export TOKENIZERS_PARALLELISM=false

torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12951 \
    src/sft/sft_qa.py \
    --deepspeed training_scripts/zero3_offload.json \
    --model_name_or_path ./path/to/model/ \
    --train_data_path /path/to/train.csv \
    --eval_data_path /path/to/val.csv \
    --train_video_folder /path/to/train/videos \
    --eval_video_folder /path/to/val/videos \
    --dataset_name all \
    --learning_rate 2e-5 \
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
    --save_only_model true

