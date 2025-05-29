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
    --deepspeed training_scripts/zero2_offload.json \
    --model_name_or_path /root/autodl-tmp/huggingface/Qwen2.5-VL-3B-Instruct \
    --train_data_path /root/autodl-tmp/gt_csv/merged-train-final-2.csv \
    --eval_data_path /root/autodl-tmp/gt_csv/merged-val-new-final.csv \
    --train_video_folder /home/zhuliyun/dataset/msad/MSAD_train \
    --eval_video_folder /home/zhuliyun/dataset/msad/MSAD_train \
    --dataset_name temporal_bench \
    --learning_rate 2e-5 \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --bf16 \
    --torch_dtype bfloat16 \
    --logging_steps 5 \
    --eval_strategy no \
    --report_to tensorboard \
    --save_total_limit 3 \
    --output_dir $OUTDIR \
    --save_steps 100 \
    --save_only_model true 

