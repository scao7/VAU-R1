import os 
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
import csv
from tqdm import tqdm
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"


THINK_FIXED_QUESTION = "Is there any anomaly in the video? If yes, please answer 'Anomaly' and present the precise anomaly time period. If no, please answer 'Normal' and output time period as <glue>[-1.0, -1.0]</glue>."


THINK_QUESTION_TEMPLATE = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from :[OPTION].

Output your thought process within the <think> </think> tags with detailed analysis.

Then, provide your answer within the <answer> </answer> tags, output the corresponding option. At the same time, in the <glue> </glue> tags, present the precise time period in seconds of the video clips on which you base your answer to this question in the format of [s1, e1]. 

For example: <think> your detailed reasoning process here </think><answer> your final answer here </answer><glue>[5.2, 10.4]</glue>.
"""


NO_THINK_FIXED_QUESTION = "Is there any anomaly in the video? If yes, output time period as <glue>[start_time, end_time]</glue>, replace start_time and end_time with the actual time period in seconds. If no anomaly, output <glue>[-1.0, -1.0]</glue>."


NO_THINK_QUESTION_TEMPLATE = """Answer the question: "[QUESTION]" according to the content of the video.

Provide your answer within the <glue> </glue> tags, present the precise time period in seconds of the video clips on which you base your answer to this question in the format of [s1, e1]. 

For example: <glue>[5.2, 10.4]</glue>.
"""


class QwenVL:
    def __init__(
        self, model_path,
        max_new_tokens = 1024,
        max_pixels = 768 * 28 * 28,
        min_pixels = 256 * 28 * 28,
    ):
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
            "use_cache": True
        }
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        

    def _wrap_video(self, path: str):
        return {
            "type": "video",
            "video": path,
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_pixels,
        }


    def parse_input(self, query=None, imgs=None, vid=None):
        # pure-text question
        if imgs is None and vid is None:
            return [{"role": "user", "content": query}]

        # multimodal question
        content = []

        if imgs is not None:
            if isinstance(imgs, str):
                imgs = [imgs]
            for img in imgs:
                content.append({"type": "image", "image": img})

        if vid is not None:
            content.append(self._wrap_video(vid))

        content.append({"type": "text", "text": query})
        return [{"role": "user", "content": content}]


    @torch.inference_mode()
    def chat(self, query=None, imgs=None, vid=None, history=None):
        if history is None:
            history = [{"role": "system", "content": "You are a helpful assistant."}]

        history.extend(self.parse_input(query, imgs, vid))

        prompt_text = self.processor.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [history], return_video_kwargs=True
        )
        fps_inputs = video_kwargs.get("fps")

        proc_kwargs = dict(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if fps_inputs:
            proc_kwargs["fps"] = fps_inputs

        inputs = self.processor(**proc_kwargs).to("cuda")
        # print("input_ids:", inputs.input_ids.shape)

        outputs = self.model.generate(**inputs, **self.gen_config)
        # print("output_ids:", outputs.shape)
        trim = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        response = self.processor.batch_decode(
            trim, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        history.append({"role": "assistant", "content": response})

        # tidy-up
        del inputs, outputs, trim
        torch.cuda.empty_cache()
        return response, history


def inference(chat_model, vid_path, think):
    history = None

    if think:
        temporal_gd_prompt = THINK_QUESTION_TEMPLATE.replace("[QUESTION]", THINK_FIXED_QUESTION).replace("[OPTION]", "[Anomaly, Normal]")
    else:
        temporal_gd_prompt = NO_THINK_QUESTION_TEMPLATE.replace("[QUESTION]", NO_THINK_FIXED_QUESTION)

    response, history = chat_model.chat(query=temporal_gd_prompt, vid=vid_path, history=history)
    return response


def main(args):
    chat_model = QwenVL(model_path=args.model_path)

    videos_list = []
    existing_videos = set()

    # skip the videos that are already in the csv file
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

        response = None
        try:
            response = inference(chat_model, video_path, args.think)
        except Exception as e:
            print(f"Error processing video {video}: {e}")
            continue

        # save the response to the output CSV file
        with open(args.output_csv, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if os.stat(args.output_csv).st_size == 0:
                writer.writerow(["Video Name", "Temporal Grounding"])

            writer.writerow([str(video), str(response)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video inference with Qwen2VL.")
    parser.add_argument("--dataset", type=str, choices=["msad", "ucf", "ecva"], required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--video_folder", type=str, required=True, help="Path to the video folder.")
    parser.add_argument("--test_gt_path", type=str, required=True, help="Path to the test ground truth CSV file.")
    parser.add_argument("--output_csv", type=str, help="Path to the output CSV file.")
    parser.add_argument("--think", action="store_true", help="Whether to use the think process.")
    args = parser.parse_args()
    
    if args.output_csv is None:
        args.output_csv = f"./results/tag/{'_'.join(args.model_path.split('/')[1:])}/output_{args.dataset}_{'w_think' if args.think else 'no_think'}.csv"
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    main(args)