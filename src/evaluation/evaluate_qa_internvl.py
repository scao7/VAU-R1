import csv
import json
import numpy as np
from tools.generate_csv import  get_index
import os
from transformers import AutoModel, AutoTokenizer
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
from PIL import Image
import re
import argparse

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def load_data(csv_file):
    csv_content = []
    with open(csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            csv_content.append(row)
    return csv_content

def generate_prompt(data):
    return f"""
            You are a helpful AI assistant performing video anomaly detection reasoning. 
            Below is a video description and a reasoning process. Based on this, answer the following multiple-choice question correctly.

            ### Question:
            {data["Question"]}

            ### Options:
            "A. "{data["Option 1"]}
            "B. "{data["Option 2"]}
            "C. "{data["Option 3"]}
            "D. "{data["Option 4"]}

            When the user asks a question, you should first carefully reason through the problem internally, and then present the final option.
            The expected format is: <think> your detailed reasoning process here </think><answer> Please only output one option letter! (A, B, C, or D) </answer>. 
            """

def generate_prompt_w_o_think(data):
    return f"""
            You are a helpful AI assistant performing video anomaly detection reasoning. 
            Below is a video description and a reasoning process. Based on this, answer the following multiple-choice question correctly.

            ### Question:
            {data["Question"]}

            ### Options:
            "A. "{data["Option 1"]}
            "B. "{data["Option 2"]}
            "C. "{data["Option 3"]}
            "D. "{data["Option 4"]}

            Please only output one option letter (A, B, C, or D). Please do not output other words.
            """


def get_model_answer(model, tokenizer, prompt, data, dataset_name, generate_think):
    if generate_think:
        prompt = generate_prompt(data)
    else:
        prompt = generate_prompt_w_o_think(data)
    if dataset_name == "ucf":
        video_folder_path = "../dataset/ucf_crime/all_videos_2/"
    if dataset_name == "msad":
        video_folder_path = "../dataset/msad/MSAD_test/"
    if dataset_name == "ecva":
        video_folder_path = "../dataset/ecva/"
    video_path = video_folder_path + data["Video Name"]
    if dataset_name == "ecva":
        video_path = video_folder_path + data["Video Name"].split(".")[0] + ".mp4"
    pixel_values, num_patches_list = load_video(video_path, num_segments=16, max_num=1)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    question = video_prefix + prompt
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    response, history = model.chat(
        tokenizer, pixel_values, question, generation_config,
        num_patches_list=num_patches_list, history=None, return_history=True
    )
    print(f'User: {question}\nAssistant: {response}')
    return response


def clean_model_answer(model_answer):
    match = re.search(r"<answer>\s*([A-D])\s*</answer>", model_answer, re.IGNORECASE | re.DOTALL)
    if not match:
        # fallback: try to match a lone letter
        match = re.search(r"\b([A-D])\b", model_answer)
    return match.group(1).upper() if match else ""

def evaluate_model(model, tokenizer, dataset, dataset_name, generate_think):
    correct_count = 0
    total = len(dataset)

    for data in dataset:
        prompt = generate_prompt(data)
        model_answer = get_model_answer(model, tokenizer, prompt, data, dataset_name, generate_think)
        model_answer = clean_model_answer(model_answer)
        correct_option = data["Correct Option"]

        print(f"Question: {data['Question']}")
        print(f"Model Answer: {model_answer}, Correct Answer: {correct_option}")

        if model_answer == correct_option:
            correct_count += 1

        current_accuracy = (correct_count / (dataset.index(data) + 1)) * 100
        print(f"Current Accuracy: {current_accuracy:.2f}%\n")

    final_accuracy = (correct_count / total) * 100
    return final_accuracy, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset name: ucf/msad')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to dataset GT CSV file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to pre-trained model')
    parser.add_argument('--summary_txt_path', type=str, required=True, help='Path to the summary txt')
    parser.add_argument('--gpu', type=str, default="0", help='GPU device ID (default: 0)')
    parser.add_argument('--w_o_think', action='store_false', help='Disable think mode (default: True)')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # torch.cuda.set_device(0)

    # Load model
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    dataset = load_data(args.csv_file)
    summary_txt_path = args.summary_txt_path
    generate_think = args.w_o_think
    accuracy, total = evaluate_model(model, tokenizer, dataset, args.dataset_name, generate_think=args.w_o_think)

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"CSV File: {args.csv_file}\n")
        f.write(f"Total Questions: {total}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")

    print(f"Summary written to: {summary_txt_path}")


if __name__ == "__main__":
    main()