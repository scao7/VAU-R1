#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
# dataset_path="path/to/your/dataset"
# model_path="path/to/your/model"
# test_gt_path="path/to/your/test_gt.csv"

dataset_path="/root/autodl-tmp/dataset/msad" 
model_path="/root/autodl-tmp/huggingface/Qwen2.5-VL-3B-Instruct"
test_gt_path="/root/autodl-tmp/VAU-R1/annotations/test_msad.csv"
api_key="your_api_key"

# init output path
model_tag=$(echo "$model_path" | cut -d'/' -f2- | tr '/' '_')
output_csv="./results/reasoning/${model_tag}/output_$(basename "$dataset_path").csv"

# Run inference script
python src/evaluation/inference_reasoning_qwen.py \
    --dataset "msad" \
    --video_folder "$dataset_path" \
    --model_path "$model_path" \
    --test_gt_path "$test_gt_path" \
    --output_csv "$output_csv" 


# Evaluate the results
# python src/evaluation/evaluate_reasoning_deepseek.py \
#     --test_gt_path "$test_gt_path" \
#     --pred_path "$output_csv" \
#     --api_key "$api_key" 