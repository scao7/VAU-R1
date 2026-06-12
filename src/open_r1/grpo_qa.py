# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import Qwen2VLForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from src.open_r1.trainer import Qwen2VLGRPOTrainer_Video_QA as Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from src.open_r1.my_qwen_utils import process_vision_info
from tqdm import tqdm
import torch
import json
import random
import ast
import csv


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'iou', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "answer"],
        metadata={"help": "List of reward functions. Possible values: 'iou', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

    train_data_path: str = field(
        default="/share/wy/Video/Charades/charades_annotation/train.json",
        metadata={"help": "Path to the training data JSON file."},
    )
    eval_data_path: str = field(
        default="/share/wy/Video/Charades/charades_annotation/val.json",
        metadata={"help": "Path to the evaluation data JSON file."},
    )

    train_video_folder: str = field(
        default="/share/wy/Video/Charades/Charades_v1",  # Replace with your actual video folder path
        metadata={"help": "Path to the folder containing video files."},
    )
    eval_video_folder: str = field(
        default="/home/zhuliyun/datasets/msad",
        metadata={"help": "Path to the folder containing evaluation video files."},
    )


def is_valid_two_d_list_format(s):
    pattern = r'^\[(\(\d+(\.\d+)?,\s*\d+(\.\d+)?\)(,\s*\(\d+(\.\d+)?,\s*\d+(\.\d+)?\))*(,)?|)\]$'
    if not re.match(pattern, s):
        return False
    try:
        lst = ast.literal_eval(s)
        if not isinstance(lst, list):
            return False
        for item in lst:
            if not isinstance(item, tuple):
                return False
            if len(item) != 2:
                return False
            for num in item:
                if not isinstance(num, (int, float)):
                    return False
            if item[0] > item[1]:
                return False
        return True
    except:
        return False
        

def answer_reward(completions, solution, **kwargs):
    """Reward function that checks multiple-choice correctness via <answer>...</answer>."""

    def extract_characters_regex(s):
        s = s.strip()
        answer_prefixes = [
            "The best answer is",
            "The correct answer is",
            "The answer is",
            "The answer",
            "The best option is",
            "The correct option is",
            "Best answer:" "Best option:",
        ]
        for answer_prefix in answer_prefixes:
            s = s.replace(answer_prefix, "")

        if len(s.split()) > 10 and not re.search("[ABCDEFG]", s):
            return ""

        matches = re.search(r"[ABCDEFG]", s)
        if matches is None:
            return ""
        return matches[0]
    
    rewards = []
    for content, sol in zip(completions, solution):
        reward = 0.0
        pattern_answer = r'<answer>(.*?)</answer>'
        match_answer = re.search(pattern_answer, content, re.DOTALL)
        if match_answer:
            answer = match_answer.group(1)
            if extract_characters_regex(answer) == extract_characters_regex(sol['answer']):
                reward = 1.0
        rewards.append(reward)
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has <think> and <answer> correctly."""
    pattern = re.compile(r'<think>.*?</think>\s*<answer>.*?</answer>', re.DOTALL)
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]

    reward_list = []
    for i, match in enumerate(matches):
        r = 1.0 if match else 0.0
        reward_list.append(r)
    return reward_list


reward_funcs_registry = {
    "answer": answer_reward,
    "format": format_reward,
}


def load_csv_dataset(train_data_path, eval_data_path, train_video_folder, eval_video_folder):
    def create_dataset_from_csv(file_path, split_name):
        if split_name == "train":
            video_folder = train_video_folder
        elif split_name == "eval":
            video_folder = eval_video_folder
        else:
            raise ValueError(f"Unknown split name: {split_name}")

        examples = []
        with open(file_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                options = [
                    "A. " + row['Option 1'],
                    "B. " + row['Option 2'],
                    "C. " + row['Option 3'],
                    "D. " + row['Option 4'],
                ]
                
                msad_video_folder = "/home/scao/myproject/VAU-R1/organized_data/msad"
                ucf_video_folder = "/home/scao/myproject/VAU-R1/organized_data/ucf"
                ecva_video_folder = "/home/scao/myproject/VAU-R1/organized_data/ecva"

                original_name = row['Video Name']

                if 'msad' in original_name.lower():
                    video_folder = msad_video_folder
                elif 'ucf' in original_name.lower():
                    video_folder = ucf_video_folder
                elif 'ecva' in original_name.lower():
                    video_folder = ecva_video_folder
                else:
                    raise ValueError(f"Unknown dataset prefix in video name: {original_name}")

                for prefix in ['msad_', 'ucf_', 'ecva_']:
                    if original_name.lower().startswith(prefix):
                        original_name = original_name[len(prefix):]

                video_path = os.path.join(video_folder, original_name)
                example = {
                    "problem": {
                        "question": row['Question'],
                        "options": options
                    },
                    "solution": {
                        "answer": row['Correct Option'],
                    },
                    "video_path": video_path,
                }
                examples.append(example)

        random.shuffle(examples)
        print(len(examples))
        print(examples[:1])

        dataset = Dataset.from_list(examples)
        # 关键：强制以 Python 对象返回，避免 pyarrow 自动包装导致的 batch 组装问题
        dataset = dataset.with_format("python")
        dataset.client = None

        def __getitem__(self, idx):
            if isinstance(idx, list):
                single_items = [self.__getitem__(i) for i in idx]
                keys = single_items[0].keys()
                batched = {k: [it[k] for it in single_items] for k in keys}
                return batched

            retry = 0
            MAX_RETRY = 20
            n = len(examples)

            while True:
                ex = examples[idx]  # 用闭包中的 examples，避免 self 递归
                out = {k: v for k, v in ex.items()}
                try:
                    msgs = [{
                        "role": "user",
                        "content": [{
                            "type": "video",
                            "video": ex["video_path"],     # 传入字符串路径，不加 [0]
                            "total_pixels": 3584 * 28 * 28,
                            "min_pixels": 16 * 28 * 28,
                        }],
                    }]
                    print("ex[video_path]: ", ex["video_path"])
                    _, video_inputs, video_kwargs = process_vision_info(
                        [msgs], return_video_kwargs=True, client=self.client
                    )
                    # 返回扁平对象，避免多套一层 list
                    out["video_inputs"] = video_inputs
                    out["video_kwargs"] = video_kwargs
                    return out
                except Exception as e:
                    print(
                        f"Warning: Error loading video from {ex['video_path']}, skipping. Error: {e}"
                    )
                    retry += 1
                    if retry > MAX_RETRY:
                        raise RuntimeError(f"Tried {MAX_RETRY} times but still failed at idx={idx}.")
                    idx = (idx + 1) % n

        dataset.__getitem__ = __getitem__.__get__(dataset, Dataset)
        return dataset

    train_dataset = create_dataset_from_csv(train_data_path, "train")
    eval_dataset = create_dataset_from_csv(eval_data_path, "eval")
    return DatasetDict({"train": train_dataset, "eval": eval_dataset})


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    dataset = load_csv_dataset(
        script_args.train_data_path,
        script_args.eval_data_path,
        script_args.train_video_folder,
        script_args.eval_video_folder
    )

    if not training_args.use_vllm:
        trainer_cls = Qwen2VLGRPOTrainer
    else:
        raise NotImplementedError
    
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)