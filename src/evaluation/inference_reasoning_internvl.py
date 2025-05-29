import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import re
import json
import argparse
import csv

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

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    # the starting point/ ending point
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -1000000000, 100000000000   # if bound is None, let the entire video as input
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

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

class inference_beam_search():
    def __init__(self, model, tokenizer, generation_config, video_path):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.video_path = video_path
        
    def get_video_segments(self, num_of_seg=3):
        # n is the Number of segments to divide the video into. 
        vr = VideoReader(self.video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1  # num of total frame
        fps = float(vr.get_avg_fps())  
        duration = max_frame / fps  # calculate the duration of video(s)
        segment_length = duration / num_of_seg  
        bounds = [(i * segment_length, (i + 1) * segment_length) for i in range(num_of_seg)]
        print(duration, bounds)
        return bounds

    def get_video_description(self, num_of_seg=3):
        bounds = self.get_video_segments(num_of_seg)

        responses = []
        description_prompt = 'You are an advanced anomaly detector assigned to analyze a video. Please identify any unusual or suspicious events that occur in the scene. Do not focus on any text or watermarks that might appear. Provide a description in three sentences.'
        scoring_prompt = "You are an intelligent anomaly detector, and your task is to identify abnormal events in videos. Here are the descriptions of different video clips. Please rate the anomaly level of each description on a scale from 0 to 1:\n\n"
        
        # devide the video into multiple clips
        for bound in bounds:
            pixel_values, num_patches_list = load_video(self.video_path, bound=bound, input_size=448, max_num=1, num_segments=8)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])  
            question = video_prefix + description_prompt
            response = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config, history=None, return_history=False)
            print(f'User: {question}\nAssistant: {response}')
            responses.append(response)

        # output the anomaly scores for each video clip
        for i, response in enumerate(responses):
            scoring_prompt += f"Clip Description {i+1}: {response}\n"

        scoring_prompt += (
                            f"\nThere are exactly {num_of_seg} clips. For each clip, provide an anomaly score between 0 and 1, "
                            f"where higher values indicate a greater degree of anomaly. "
                            f"Return the scores as a JSON object, strictly formatted as: "
                            f'{{"anomaly_scores": [score_1, score_2, ..., score_{num_of_seg}]}}. '
                            f"The list must contain exactly {num_of_seg} float values."
                        )
        scoring_response = self.model.chat(self.tokenizer, None, scoring_prompt, self.generation_config, history=None, return_history=False)
        print(f'User: {scoring_prompt} \nAssistant: {scoring_response}')

        match = re.search(r"```json\n(.*?)\n```", scoring_response, re.DOTALL)
        if match:
            json_str = match.group(1)
            anomaly_scores = json.loads(json_str)["anomaly_scores"] # get the anomaly score list 
            print(anomaly_scores)
        else:
            anomaly_scores = [0.0] * num_of_seg

        max_anomaly_index = anomaly_scores.index(max(anomaly_scores))
        anomaly_response = responses[max_anomaly_index]
        anomaly_bound = bounds[max_anomaly_index]
        print(f"The response with the highest score is the clip {max_anomaly_index + 1}. The description for this video clip is {anomaly_response}")
        return anomaly_response, anomaly_scores, anomaly_bound

    def get_reasoning_process(self, best_of_n=3, num_of_seg=3):
        anomaly_response, anomaly_scores, anomaly_bound = self.get_video_description(num_of_seg)
        best_response = None
        best_score = float('-inf')
        responses = []
        ano_pixel_values, num_patches_list = load_video(self.video_path, bound=anomaly_bound, input_size=448, max_num=1, num_segments=16)
        ano_pixel_values = ano_pixel_values.to(torch.bfloat16).cuda()

        for _ in range(best_of_n):
            reasoning_prompt = f"""
                You are given a description of a potential anomaly segment from a video:

                {anomaly_response}

                Please do the following in your response:
                1. **Description**: Provide a concise summary of what is happening in the video segment. 
                2. **Analysis**: Explain in detail whether (and how) any anomaly occurs, and the reasoning behind it.
                Finally, output your response **exclusively** as a valid JSON object with two fields: "Description" and "Analysis".

                Example of the JSON format:
                {{
                "Description": "...",
                "Analysis": "..."
                }}

                Important: Output **only** the JSON (no extra text).
                """

            # refine the reasoning process
            response = self.model.chat(self.tokenizer, ano_pixel_values, reasoning_prompt, self.generation_config, history=None, return_history=False)
            print(f'User: {reasoning_prompt} \nAssistant: {response}')
            responses.append(response)
            # Assuming the response contains a score, you can parse it and compare
        
        description, analysis = self.evaluate_response(responses)  # Implement this method to evaluate the response

        return description, analysis
    
    def evaluate_response(self, responses):
        # evaluate the quality of the reasoning process
        # evaluation_prompt = f"Please evaluate the reasoning process of the assistant. Please output the best reasoning response in all candidates. {responses}"
        
        evaluation_prompt = f"""You have multiple candidate reasoning responses. Read them carefully, then select the best one.
        Finally, output your chosen response **exclusively** as a valid JSON object with two fields: "Description" and "Analysis".

        Example of the JSON format:
        {{
        "Description": "...",
        "Analysis": "..."
        }}

        Important:
        - Output **only** the JSON (no extra text).
        Here are the candidate responses: {responses}
        """
        
        evaluation_response = self.model.chat(self.tokenizer, None, evaluation_prompt, self.generation_config, history=None, return_history=False)
        print(f'User: {evaluation_prompt} \n\n Assistant: {evaluation_response}')
        # match = re.search(r"```json\n(.*?)\n```", evaluation_response, re.DOTALL)
        # if match:
        #     json_str = match.group(1)
        # else:
        #     print("No valid JSON found in the response!")
        match = re.search(r"\{.*?\}", evaluation_response, re.DOTALL)
        if match:
            json_str = match.group(0)
            # data = json.loads(json_str)
            # print("Parsed JSON:", data)
        else:
            print("No valid JSON found in the response!")
            return None, None
        try:
            result = json.loads(json_str)
            description = result["Description"]
            analysis = result["Analysis"]
            print("Description:", description)
            print("Analysis:", analysis)
        except json.JSONDecodeError:
            print("Failed to parse JSON output!")
            description, analysis = None, None
        return description, analysis

    def get_output_w_o_cot(self, num_of_seg=1):
        bounds = self.get_video_segments(num_of_seg)

        responses = []
        description_prompt = 'You are an advanced anomaly detector assigned to analyze a video. Please identify any unusual or suspicious events that occur in the scene. Do not focus on any text or watermarks that might appear. Provide a description in five sentences.'
        analysis_prompt = 'Explain in detail whether (and how) any anomaly occurs, and the reasoning behind it.'
        # devide the video into multiple clips
        for bound in bounds:
            pixel_values, num_patches_list = load_video(self.video_path, bound=bound, input_size=448, max_num=1, num_segments=16)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])  
            question = video_prefix + description_prompt
            des_response = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config, history=None, return_history=False)
            print(f'User: {question}\nAssistant: {des_response}')
        
            question = video_prefix + analysis_prompt
            ana_response = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config, history=None, return_history=False)
            print(f'User: {question}\nAssistant: {ana_response}')
        return des_response, ana_response 


def main(args):
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    videos_list = []
    existing_videos = set()

    if os.path.exists(args.output_csv):
        with open(args.output_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_videos.add(row["Video Name"])

    with open(args.test_list_txt, 'r', encoding='utf-8') as f:
        for line in f:
            videos_list.append(line.strip())

    write_header = not os.path.exists(args.output_csv)
    with open(args.output_csv, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Video Name", "Description", "Analysis"])

        for video in videos_list:
            video_path = os.path.join(args.video_folder, video)
            video_name = video.split('/')[-1].replace('.mp4', '')

            if video_name in existing_videos:
                print(f"Skipping already processed video: {video_name}")
                continue

            print(f"Processing: {video_path}")
            inference_result = inference_beam_search(model, tokenizer, generation_config, video_path)

            if args.w_o_cot and hasattr(inference_result, 'get_output_w_o_cot'):
                description, analysis = inference_result.get_output_w_o_cot(num_of_seg=1)
            
            elif hasattr(inference_result, 'get_reasoning_process'):
                description, analysis = inference_result.get_reasoning_process(best_of_n=3, num_of_seg=4)
                
            else:
                print(f"Error: inference_beam_search did not return the expected object for {video}")
                description, analysis = None, None

            writer.writerow([str(video_name), str(description), str(analysis)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run anomaly detection on UCF-Crime videos")
    parser.add_argument('--gpu', type=str, default="0", help="Specify which GPU(s) to use")
    parser.add_argument('--model_path', type=str, help="Path to the pretrained model", default="/home/zhuliyun/huggingface/Internvl2_5-8B-MPO-SFT04052-lora-merge")
    parser.add_argument('--video_folder', type=str, help="Path to folder containing video files", default="/home/zhuliyun/dataset/msad/MSAD_test/")
    parser.add_argument('--test_list_txt', type=str, help="Path to Anomaly_Test.txt file", default="/home/zhuliyun/VAD-CoT/json/msad_test.txt")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to output CSV file")
    parser.add_argument('--w_o_cot', action='store_true', help="Use get_output_w_o_cot() instead of reasoning")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)


