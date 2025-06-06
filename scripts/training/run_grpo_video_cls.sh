
export WANDB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
export PYTHONPATH=".:$PYTHONPATH"
OUTDIR=./checkpoints/$WANDB_NAME

export DEBUG_MODE="true"
export LOG_PATH="./logs/${WANDB_NAME}.log"
export TOKENIZERS_PARALLELISM=false


torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=10668 \
    src/open_r1/grpo_cls.py \
    --deepspeed training_scripts/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path ./path/to/model/ \
    --train_data_path /path/to/train.csv \
    --eval_data_path /path/to/val.csv \
    --train_video_folder /path/to/train/videos \
    --eval_video_folder /path/to/val/videos \
    --dataset_name all \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --report_to tensorboard \
    --save_steps 100 \
    --learning_rate 2e-6 \
    --save_total_limit 3 \
    --save_only_model true