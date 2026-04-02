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
Preprocess MATH-lighteval, filtering to hard levels only (default: Level 4 + Level 5).

Compared to math_dataset.py, adds:
  --min_level  : minimum difficulty level to keep (1-5, default 4)
  --keep_test_all : if set, val set keeps ALL levels (for fair comparison with full MATH baseline)

Usage:
    # From local cache (recommended on server):
    python math_hard.py         --local_dataset_path C:/Users/2025/data/math         --local_save_dir C:/Users/2025/data/math_hard         --min_level 4

    # Download from HuggingFace:
    python math_hard.py --local_save_dir /home/ma-user/work/data/math_hard --min_level 4
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default="C:/Users/2025/data/math",
                        help="Local path to raw dataset (skips HuggingFace download)")
    parser.add_argument("--local_save_dir", default="C:/Users/2025/data/math_hard",
                        help="Output directory for processed parquet files")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--min_level", type=int, default=4, choices=[1, 2, 3, 4, 5],
                        help="Minimum difficulty level to keep in train set (1-5). Default: 4")
    parser.add_argument("--keep_test_all", action="store_true",
                        help="If set, val set keeps all levels (for fair comparison with full MATH baseline). "
                             "Default: val also filtered to min_level.")
    args = parser.parse_args()

    data_source = "DigitalLearningGmbH/MATH-lighteval"
    keep_levels = [f"Level {i}" for i in range(args.min_level, 6)]
    print(f"Loading {data_source} ...")
    print(f"Keeping levels: {keep_levels}")

    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset  = dataset["test"]

    # ── Level distribution before filtering ──
    from collections import Counter
    # Support both raw HF format (has "level" column) and
    # already-processed format (level inside extra_info)
    def get_level(example):
        if "level" in example and example["level"]:
            return example["level"]
        ei = example.get("extra_info") or {}
        return ei.get("level", "Unknown") if isinstance(ei, dict) else "Unknown"

    train_levels = Counter(get_level(ex) for ex in train_dataset)
    test_levels  = Counter(get_level(ex) for ex in test_dataset)
    print("\nTrain level distribution (before filter):")
    for lv in sorted(train_levels): print(f"  {lv}: {train_levels[lv]}")
    print("Test level distribution (before filter):")
    for lv in sorted(test_levels):  print(f"  {lv}: {test_levels[lv]}")

    # ── Filter ──
    train_dataset = train_dataset.filter(lambda x: get_level(x) in keep_levels)
    if not args.keep_test_all:
        test_dataset = test_dataset.filter(lambda x: get_level(x) in keep_levels)

    print(f"\nAfter filter → train: {len(train_dataset)}, test: {len(test_dataset)}")

    # ── Detect format: raw HF (has "problem"/"solution") or already processed ──
    sample = train_dataset[0]
    is_raw = "problem" in sample

    if is_raw:
        # Raw HuggingFace format — need full preprocessing
        instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

        def make_map_fn(split):
            def process_fn(example, idx):
                question = example.pop("problem")
                question = question + " " + instruction_following
                answer   = example.pop("solution")
                solution = extract_solution(answer)
                return {
                    "data_source": data_source,
                    "prompt": [{"role": "user", "content": question}],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": solution},
                    "extra_info": {
                        "split": split,
                        "index": idx,
                        "level": example.get("level", ""),
                        "type":  example.get("type", ""),
                    },
                }
            return process_fn

        train_dataset = train_dataset.map(make_map_fn("train"), with_indices=True)
        test_dataset  = test_dataset.map(make_map_fn("test"),  with_indices=True)
    else:
        # Already-processed format (prompt/reward_model already exist) — just pass through
        print("Detected already-processed format, skipping preprocessing.")

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir,  "test.parquet"))

    with open(os.path.join(local_save_dir, "train_example.json"), "w") as f:
        json.dump(train_dataset[0], f, indent=2, ensure_ascii=False)
    with open(os.path.join(local_save_dir, "test_example.json"), "w") as f:
        json.dump(test_dataset[0], f, indent=2, ensure_ascii=False)

    # ── Summary ──
    print(f"\nSaved to: {local_save_dir}")
    print(f"  train.parquet : {len(train_dataset)} samples")
    print(f"  test.parquet  : {len(test_dataset)} samples")

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        print(f"  Copied to HDFS: {args.hdfs_dir}")
