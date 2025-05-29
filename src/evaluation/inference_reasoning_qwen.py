import os 
import csv
import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm
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
        self.model.config.use_cache = True

        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
        self.gen_config = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
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
        generated_ids = self.model.generate(**inputs, **self.gen_config)
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


def inference(chat_model, vid_path):
    history = None
    description_prompt = 'You are an advanced anomaly detector assigned to analyze a video. Please identify any unusual or suspicious events that occur in the scene. Do not focus on any text or watermarks that might appear. Provide a description in five sentences.'
    analysis_prompt = 'Explain in detail whether (and how) any anomaly occurs in the video, and the reasoning behind it.'
    des_response, history = chat_model.chat(query=description_prompt, vid=vid_path, history=history)
    ana_response, history = chat_model.chat(query=analysis_prompt, vid=vid_path, history=history)
    return des_response, ana_response

def main(args):
    chat_model = QwenVL(model_path=args.model_path)

    videos_list = []
    existing_videos = set()

    if os.path.exists(args.output_csv):
        with open(args.output_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_videos.add(row["Video Name"])

    # read the video list
    df = pd.read_csv(args.test_gt_path)
    videos_list = df["Video Name"].tolist()

    for video in tqdm(videos_list):
        video_path = os.path.join(args.video_folder, video)
        if video in existing_videos:
            print(f"Skipping already processed video: {video}")
            continue

        description, analysis = None, None
        try:
            description, analysis = inference(chat_model, video_path)
        except Exception as e:
            print(f"Error processing {video}: {e}")

        # save the response to the output CSV file
        with open(args.output_csv, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if os.stat(args.output_csv).st_size == 0:
                writer.writerow(["Video Name", "Description", "Analysis"])
            
            writer.writerow([str(video), str(description), str(analysis)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video inference with QwenVL.")
    parser.add_argument("--dataset", type=str, choices=["msad", "ucf", "ecva"], required=True)
    parser.add_argument("--video_folder", type=str, required=True, help="Path to the folder containing videos.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the QwenVL model.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file.")
    parser.add_argument("--test_gt_path", type=str, required=True, help="Path to txt file with video list.")
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = f"./results/reasoning/{'_'.join(args.model_path.split('/')[1:])}/output_{args.dataset}.csv"
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    main(args)
