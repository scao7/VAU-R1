#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
dataset_path="/root/autodl-tmp/dataset/msad" 
model_path="/root/autodl-tmp/huggingface/Qwen2.5-VL-3B-Instruct"
test_gt_path="/root/autodl-tmp/VAU-R1/annotations/test_msad.csv"
use_think=false  # set to false to disable --think

# init output path
model_tag=$(echo "$model_path" | cut -d'/' -f2- | tr '/' '_')
think_tag=$( [ "$use_think" = true ] && echo "w_think" || echo "no_think" )
output_csv="./results/cls/${model_tag}/output_$(basename "$dataset_path")_${think_tag}.csv"

# Run inference script
python src/evaluation/inference_cls_qwen.py \
    --dataset "msad" \
    --video_folder "$dataset_path" \
    --model_path "$model_path" \
    --test_gt_path "$test_gt_path" \
    --output_csv "$output_csv" \
    $( [ "$use_think" = true ] && echo "--think" )

# Evaluate the results
python src/evaluation/evaluation_cls.py \
    --test_gt_path "$test_gt_path" \
    --pred_path "$output_csv"
