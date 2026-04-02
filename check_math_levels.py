"""
查看 MATH 数据集的 level / type 分布。
支持本地 parquet 文件或 HuggingFace 直接加载。

用法:
    # 从本地 parquet（服务器上）
    python check_math_levels.py --parquet /home/ma-user/work/data/math/train.parquet

    # 同时看 train + test
    python check_math_levels.py \
        --parquet /home/ma-user/work/data/math/train.parquet \
                  /home/ma-user/work/data/math/test.parquet

    # 从 HuggingFace 加载原始数据集
    python check_math_levels.py --hf

    # 本地 HuggingFace 缓存
    python check_math_levels.py --hf --local_path /home/ma-user/work/data/math_raw
"""

import argparse
import os
from collections import Counter

BAR_WIDTH = 40


def bar(count, total):
    filled = int(round(count / total * BAR_WIDTH)) if total > 0 else 0
    return "#" * filled + "-" * (BAR_WIDTH - filled)


def print_distribution(title, counter, total):
    print(f"\n  {title}  (total={total:,})")
    print("  " + "-" * 65)
    for key in sorted(counter.keys()):
        cnt = counter[key]
        pct = cnt / total * 100
        print(f"  {str(key):>20} : {cnt:>5}  ({pct:5.1f}%)  {bar(cnt, total)}")


def analyze_records(records, split_name):
    total = len(records)
    print(f"\n{'='*70}")
    print(f"  Split: {split_name}  |  Total: {total:,}")
    print(f"{'='*70}")

    level_counter = Counter()
    type_counter  = Counter()
    level_type    = {}   # level -> Counter of types

    for rec in records:
        # Support both raw HF format and processed parquet format
        level = rec.get("level") or rec.get("extra_info", {}).get("level") or "Unknown"
        typ   = rec.get("type")  or rec.get("extra_info", {}).get("type")  or "Unknown"
        level_counter[level] += 1
        type_counter[typ]    += 1
        level_type.setdefault(level, Counter())[typ] += 1

    print_distribution("Level 分布", level_counter, total)
    print_distribution("Type 分布",  type_counter,  total)

    # Level x Type cross table
    all_levels = sorted(level_counter.keys())
    all_types  = sorted(type_counter.keys())
    print(f"\n  Level x Type 交叉表")
    print("  " + "-" * 80)
    header = f"  {'':>12}" + "".join(f"{t[:8]:>10}" for t in all_types)
    print(header)
    for lv in all_levels:
        row = f"  {lv:>12}"
        for t in all_types:
            cnt = level_type.get(lv, Counter()).get(t, 0)
            row += f"{cnt:>10}"
        print(row)

    # Hard subset summary
    hard_levels = ["Level 4", "Level 5"]
    hard_count  = sum(level_counter.get(lv, 0) for lv in hard_levels)
    print(f"\n  Level 4+5 子集: {hard_count:,} / {total:,}  ({hard_count/total*100:.1f}%)")


def load_from_parquet(path):
    import pandas as pd
    df = pd.read_parquet(path)
    return df.to_dict("records")


def load_from_parquet_pyarrow(path):
    """Fallback: read only needed columns via pyarrow."""
    import pyarrow.parquet as pq
    # Try reading level/type from extra_info or direct columns
    f = pq.ParquetFile(path)
    schema_names = f.schema_arrow.names
    records = []
    # Read in batches
    for batch in f.iter_batches(batch_size=1000):
        d = batch.to_pydict()
        n = len(next(iter(d.values())))
        for i in range(n):
            rec = {k: d[k][i] for k in d}
            records.append(rec)
    return records


def load_from_hf(local_path=None):
    import datasets
    data_source = "DigitalLearningGmbH/MATH-lighteval"
    if local_path:
        ds = datasets.load_dataset(local_path)
    else:
        print(f"从 HuggingFace 加载 {data_source} ...")
        ds = datasets.load_dataset(data_source)
    result = {}
    for split in ds:
        result[split] = [dict(row) for row in ds[split]]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", nargs="+", default=None,
                        help="一个或多个 parquet 文件路径")
    parser.add_argument("--hf", action="store_true",
                        help="从 HuggingFace 加载原始数据集")
    parser.add_argument("--local_path", default=None,
                        help="本地 HuggingFace 缓存路径（配合 --hf 使用）")
    args = parser.parse_args()

    if args.hf or (not args.parquet):
        splits = load_from_hf(args.local_path)
        for split_name, records in splits.items():
            analyze_records(records, split_name)
    else:
        for path in args.parquet:
            path = os.path.expanduser(path)
            split_name = os.path.basename(path).replace(".parquet", "")
            print(f"\n加载 {path} ...")
            try:
                records = load_from_parquet(path)
            except Exception as e:
                print(f"  pandas 读取失败 ({e})，尝试 pyarrow ...")
                try:
                    records = load_from_parquet_pyarrow(path)
                except Exception as e2:
                    print(f"  pyarrow 也失败: {e2}")
                    continue
            analyze_records(records, split_name)
