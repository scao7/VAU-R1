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

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""

import logging
import os
import sys

import datasets
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
import trl
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from tqdm import tqdm
import json
import random
import csv

from src.open_r1.my_qwen_utils import process_vision_info
logger = logging.getLogger(__name__)

@dataclass
class SFTScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'iou', 'format'.
    """

    train_data_path: str = field(
        default="./Charades/charades_annotation/train.json",
        metadata={"help": "Path to the training data JSON file."},
    )
    eval_data_path: str = field(
        default="./Charades/charades_annotation/val.json",
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
    preprocessed_data_path: Optional[str] = field( # Add preprocessed_data_path argument
        default="",
        metadata={"help": "Path to the preprocessed dataset directory. If provided, load preprocessed data instead of raw videos."},
    )

@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})


def load_csv_dataset(train_data_path, eval_data_path, train_video_folder, eval_video_folder):#, preprocessed_data_path=None): # Modified to accept preprocessed_data_path
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
                # print(row)
                msad_video_folder = "/root/autodl-tmp/dataset/msad"
                ucf_video_folder = "/root/autodl-tmp/dataset/ucf-crime/all_videos"
                ecva_video_folder = "/root/autodl-tmp/dataset/ecva"
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

                if not original_name.endswith('.mp4'):
                    original_name += '.mp4'

                video_path = os.path.join(video_folder, original_name)
                example = {
                    "solution": {
                        "answer": row['Description'],
                    },
                    "video_path": video_path,  
                }

                examples.append(example)

        random.shuffle(examples)
        print(len(examples))
        print(examples[:1])
        dataset = Dataset.from_list(examples)


        dataset.client = None
        def __getitem__(self, idx): # Define getitem within the scope where dataset is available
            retry_count = 0
            example = dataset[idx]
            data_to_return = {k: v for k, v in example.items()} # Create a copy to avoid modifying original dataset

            try:
                messages = [{"role": "user", "content": [{"type": "video", "video": example["video_path"][0], "total_pixels": 3584 * 28 * 28, "min_pixels": 16 * 28 * 28,},]}]
                image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True, client=self.client)
                fps_inputs = video_kwargs['fps']
                # # data_to_return["image_inputs"] = [torch.load(os.path.join(example["video_path"][0], "image_inputs.pt"))]
                data_to_return["video_inputs"] = [video_inputs]
                # with open(os.path.join(example["video_path"][0], "video_kwargs.json"), 'r') as f:
                data_to_return["video_kwargs"] = [video_kwargs]
            except Exception as e:
                print(f"Warning: Error loading preprocessed data from {example['video_path'][0]}, falling back to video_path. Error: {e}")
                retry_count += 1
                MAX_RETRY = 20
                if retry_count > MAX_RETRY:
                    raise RuntimeError(f"Tried {MAX_RETRY} times but still failed.")
                print(idx)
                idx = idx + 1
                return self.__getitem__(idx)

            return data_to_return

        dataset.__getitem__ = __getitem__.__get__(dataset, Dataset) # Bind getitem to the dataset

        return dataset

    train_dataset = create_dataset_from_csv(train_data_path, "train")
    eval_dataset = create_dataset_from_csv(eval_data_path, "eval")
    return DatasetDict({"train": train_dataset, "eval": eval_dataset})

processor = None


FIXED_QUESTION = "You are an advanced anomaly detector assigned to analyze a video. Please identify any unusual or suspicious events that occur in the scene. Do not focus on any text or watermarks that might appear. Please provide a description for the video."

def convert_example(example):
    """
    correct example into "messages" 
    eg:
    {
      "system": "You are a helpful assistant.",
      "conversations": [
          {"from": "user", "value": "How many objects are included in this image?",
           "image_path": "/path/to/image.png"},
          {"from": "assistant", "value": "<think>\nI can see 10 objects\n</think>\n<answer>\n10\n</answer>"}
      ]
    }
    """
    messages = []
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": FIXED_QUESTION},
            {"type": "video", 
            "video": example["video_path"], 
            "total_pixels": 3584 * 28 * 28, 
            "min_pixels": 16 * 28 * 28,
            },
            ]
    })

    answer_text = f'{example["solution"]["answer"]}'
    messages.append({
        "role": "assistant",
        "content": answer_text,
    })

    example["messages"] = messages
    # import pdb; pdb.set_trace()
    return example


def collate_fn(examples):
    texts = [
        processor.apply_chat_template( convert_example(example)["messages"], tokenize=False, add_generation_prompt=True )
        for example in examples
    ]

    video_inputs = [x["video_inputs"] for x in examples]
    fps_inputs = [x["video_kwargs"]["fps"] for x in examples]

    video_inputs = video_inputs[0]
    fps_inputs = fps_inputs[0]

    image_inputs = None
    batch = processor(
        text=[texts[0]],
        images=image_inputs, 
        videos=[video_inputs[0]], 
        fps=[fps_inputs[0]], 
        return_tensors="pt",
        padding=True,
    )
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    video_token_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
    labels[labels == video_token_id] = -100
    batch["labels"] = labels

    return batch


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################

    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # import pdb; pdb.set_trace()
    dataset = load_csv_dataset(
        script_args.train_data_path,
        script_args.eval_data_path,
        script_args.train_video_folder,
        script_args.eval_video_folder # Pass preprocessed_data_path
    )

    ################
    # Load tokenizer
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        )
        logger.info("Using AutoTokenizer for text-only model.")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    ###################
    # Model init kwargs
    ###################
    from peft import LoraConfig, get_peft_model

    # lora_config = LoraConfig(
    #     task_type="CAUSAL_LM",
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #     inference_mode=False,
    #     r=16,
    #     lora_alpha=16,
    #     lora_dropout=0.05,
    #     bias="none",
    # )

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_sliding_window=True,
    )
    # training_args.model_init_kwargs = model_kwargs
    from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, 
        # torch_dtype=torch.bfloat16,
        **model_kwargs
    )
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     model_args.model_name_or_path, 
    #     # torch_dtype=torch.bfloat16,
    #     **model_kwargs
    # )
    # model = get_peft_model(model, lora_config)  
    # model.print_trainable_parameters()

    ############################
    # Initialize the SFT Trainer
    ############################
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if training_args.eval_strategy != "no" else None,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    # import pdb; pdb.set_trace()
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["VAU-R1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    #############
    # push to hub
    #############

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)
        processor.push_to_hub(training_args.hub_model_id)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)