import os 
import csv
import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm
import re
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class QwenVL:
    def __init__(self, model_path = None, max_new_tokens = 1024, min_pixels = 256*28*28, max_pixels = 768*28*28):
        if "Qwen2.5" in model_path:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        elif "Qwen2-VL" in model_path:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )  
        self.model.config.use_cache=True

        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
        self.gen_config = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True
        }
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
    
    def parse_input(self, query=None, imgs=None, vid=None):
        if imgs is None and vid is None:
            messages = [{"role": "user", "content": query}]
            return messages
        content = []
        if imgs is not None:
            if isinstance(imgs, str):
                imgs = [imgs]
            
            for img in imgs:
                content.append({"type": "image", "image": img})
        if vid is not None:
            content.append({"type": "video", "video": vid, "max_pixels": self.max_pixels, "fps":1.0})

        content.append({"type": "text", "text": query})
        messages = [{"role": "user", "content": content}]
        return messages

    def chat(self, query = None, imgs = None, vid = None, history = None):
        if history is None:
            history = []
            
        user_query = self.parse_input(query, imgs, vid)
        history.extend(user_query)

        text = self.processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True, add_vision_id=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(history, return_video_kwargs=True)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")
        print("input_ids:", inputs.input_ids.shape)
        generated_ids = self.model.generate(**inputs, **self.gen_config)
        print("output_ids:", generated_ids.shape)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        history.append({"role": "assistant", "content": response})

        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        return response, history


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


def inference(chat_model, vid_path, data, think):
    history = None

    if think:
        prompt = generate_prompt(data)
    else:
        prompt = generate_prompt_w_o_think(data)
    
    response, history = chat_model.chat(query=prompt, vid=vid_path, history=history)
    print("Response:", response)
    return response

_STANDALONE_OPT_RE = re.compile(r'(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])', re.I)

def clean_model_answer(text: str) -> str:
    m = re.search(r'<answer>\s*([A-D])\s*</answer>', text, re.I)
    if m:
        return m.group(1).upper()

    m = re.search(r'<\s*([A-D])\s*/?\s*>', text, re.I)
    if m:
        return m.group(1).upper()

    parts = re.split(r'</think>', text, flags=re.I)
    tail = parts[-1] if len(parts) > 1 else ''
    opts_in_tail = _STANDALONE_OPT_RE.findall(tail)
    if opts_in_tail:
        return opts_in_tail[-1].upper()        

    opts_all = _STANDALONE_OPT_RE.findall(text)
    if opts_all:
        return opts_all[-1].upper()

    return ""



def main(args):
    model = QwenVL(model_path=args.model_path)
    
    # read the video list
    df = pd.read_csv(args.test_gt_path)
    videos_list = df["Video Name"].tolist()

    existing_videos = set()
    if os.path.exists(args.output_csv):
        with open(args.output_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_videos.add(row["Video Name"])

    for video in tqdm(videos_list):
        video_path = os.path.join(args.video_folder, video)
        if video in existing_videos:
            print(f"Skipping already processed video: {video}")
            continue

        data = df[df["Video Name"] == video].iloc[0].to_dict()
        prompt = generate_prompt(data)
        response = None
        try:
            response = inference(model, video_path, data, think=args.think)
        except Exception as e:
            print(f"Error processing video {video}: {e}")
            traceback.print_exc()
            continue

        response = clean_model_answer(response)
        correct_option = data["Correct Option"]

        # process each video
        with open(args.output_csv, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if os.stat(args.output_csv).st_size == 0:
                writer.writerow(["Video Name", "QA answer"])
            writer.writerow([str(video), str(response)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["msad", "ucf", "ecva"])
    parser.add_argument('--model_path', type=str, required=True, help='Path to pre-trained model')
    parser.add_argument("--video_folder", type=str, required=True, help="Path to the video folder.")
    parser.add_argument('--test_gt_path', type=str, required=True, help='Path to dataset GT CSV file')
    parser.add_argument("--output_csv", type=str, help="Path to the output CSV file.")
    parser.add_argument("--think", action="store_true", help="Whether to use the think process.") 
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = f"./results/qa/{'_'.join(args.model_path.split('/')[1:])}/output_{args.dataset}_gd_{'w_think' if args.think else 'no_think'}.csv"
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    main(args)