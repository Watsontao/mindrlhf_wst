# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Preprocess DAPO-Math-17k for single-turn GRPO training.

The dataset is already in verl-compatible format, this script:
  1. Appends the instruction suffix to each prompt
  2. Normalizes reward_model.style to "rule" for verl compatibility
  3. Saves train.parquet
  4. Uses MATH-lighteval test set as val (reuse existing val data, or pass --val_files)

Usage:
    # Download from HuggingFace:
    python dapo_math_17k.py --local_save_dir ~/data/dapo_math

    # From local cache:
    python dapo_math_17k.py --local_dataset_path /path/to/dapo_math_17k --local_save_dir ~/data/dapo_math

    # Also save a val split (random 1000 samples from train):
    python dapo_math_17k.py --local_save_dir ~/data/dapo_math --val_size 1000
"""

import argparse
import json
import os
import random

import datasets

from verl.utils.hdfs_io import copy, makedirs


INSTRUCTION_SUFFIX = "Let's think step by step and output the final answer within \\boxed{}."
DATA_SOURCE = "BytedTsinghua-SIA/DAPO-Math-17k"


def process_fn(example, idx):
    """Normalize one DAPO-Math example to verl format."""
    # prompt is already a list of {role, content} dicts
    prompt = example["prompt"]
    # Append instruction suffix to the last user turn
    for turn in reversed(prompt):
        if turn["role"] == "user":
            if INSTRUCTION_SUFFIX not in turn["content"]:
                turn["content"] = turn["content"].strip() + "\n" + INSTRUCTION_SUFFIX
            break

    reward_model = dict(example["reward_model"])
    # verl rule-based reward expects style="rule"
    reward_model["style"] = "rule"

    extra_info = dict(example.get("extra_info") or {})
    extra_info["index"] = idx
    extra_info["split"] = "train"

    return {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        "ability": example.get("ability", "math"),
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess DAPO-Math-17k for single-turn GRPO")
    parser.add_argument("--local_dataset_path", default=None,
                        help="Local path to raw dataset (skip HuggingFace download)")
    parser.add_argument("--local_save_dir", default="~/data/dapo_math",
                        help="Output directory for processed parquet files")
    parser.add_argument("--hdfs_dir", default=None,
                        help="Optional HDFS path to copy processed files")
    parser.add_argument("--val_size", type=int, default=1000,
                        help="Split this many samples from train as val set. "
                             "Default 1000. Set 0 to skip (use external val like math_hard/test.parquet)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    # ── Load ──
    print(f"Loading {DATA_SOURCE} ...")
    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path, "default")
    else:
        dataset = datasets.load_dataset(DATA_SOURCE, "default")

    train_dataset = dataset["train"]
    print(f"  Raw train size: {len(train_dataset):,}")

    # ── Process ──
    train_dataset = train_dataset.map(process_fn, with_indices=True, num_proc=4,
                                      desc="Processing")

    # Keep only needed columns
    keep_cols = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    train_dataset = train_dataset.select_columns(keep_cols)

    # ── Optional val split ──
    if args.val_size > 0:
        random.seed(args.seed)
        n = len(train_dataset)
        val_indices = sorted(random.sample(range(n), min(args.val_size, n)))
        train_indices = [i for i in range(n) if i not in set(val_indices)]
        val_dataset   = train_dataset.select(val_indices)
        train_dataset = train_dataset.select(train_indices)
        val_path = os.path.join(local_save_dir, "val.parquet")
        val_dataset.to_parquet(val_path)
        print(f"  Val   split: {len(val_dataset):,}  -> {val_path}")

    # ── Save ──
    train_path = os.path.join(local_save_dir, "train.parquet")
    train_dataset.to_parquet(train_path)
    print(f"  Train split: {len(train_dataset):,}  -> {train_path}")

    # Save one example for inspection
    example_path = os.path.join(local_save_dir, "train_example.json")
    with open(example_path, "w", encoding="utf-8") as f:
        json.dump(train_dataset[0], f, indent=2, ensure_ascii=False)
    print(f"  Example     : {example_path}")

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        print(f"  Copied to HDFS: {args.hdfs_dir}")

    print("Done.")
