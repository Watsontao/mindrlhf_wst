"""
Rollout Response Analyzer
Usage:
    python analyze_rollout.py <folder>
    python analyze_rollout.py Rollout_response/dynamic_segment_rollout_grpo/8npu_7b_math/dynamic_segment

The folder should contain:
    rollout_length_stats_*.jsonl   -- per-group length records
    rollout_responses_*.jsonl      -- actual response texts
"""

import argparse
import json
import os
import statistics
import sys
from glob import glob

# Windows 终端强制 UTF-8 输出
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# --------------------------- I/O helpers ----------------------------

def find_jsonl(folder, keyword):
    matches = glob(os.path.join(folder, f"*{keyword}*.jsonl"))
    if not matches:
        return None
    return sorted(matches)[-1]


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# --------------------------- stats helpers --------------------------

def percentile(sorted_lst, p):
    if not sorted_lst:
        return 0
    idx = min(int(len(sorted_lst) * p / 100), len(sorted_lst) - 1)
    return sorted_lst[idx]


def describe(lengths):
    if not lengths:
        return {}
    s = sorted(lengths)
    n = len(s)
    mean = sum(s) / n
    med  = s[n // 2]
    std  = statistics.stdev(s) if n > 1 else 0
    cv   = std / mean if mean > 0 else 0
    return {
        "n": n, "mean": mean, "median": med, "std": std, "cv": cv,
        "min": s[0], "max": s[-1],
        "p10": percentile(s, 10), "p25": percentile(s, 25),
        "p75": percentile(s, 75), "p90": percentile(s, 90),
        "p95": percentile(s, 95), "p99": percentile(s, 99),
    }


def bar(value, total, width=40):
    filled = int(round(value / total * width)) if total > 0 else 0
    return "#" * filled + "-" * (width - filled)


# --------------------------- main analysis --------------------------

def analyze(folder):
    folder = os.path.expanduser(folder)
    if not os.path.isdir(folder):
        print(f"[错误] 文件夹不存在: {folder}")
        sys.exit(1)

    stats_path = find_jsonl(folder, "length_stats")
    resp_path  = find_jsonl(folder, "responses")

    if not stats_path:
        print(f"[错误] 未找到 length_stats jsonl 文件: {folder}")
        sys.exit(1)

    print("=" * 70)
    print(f"  Rollout 分析报告: {folder}")
    print("=" * 70)

    # -- 1. 加载长度统计 --
    stats_records = load_jsonl(stats_path)
    all_lengths = []
    per_step = {}
    for rec in stats_records:
        gs = int(rec["global_steps"])
        ls = rec["lengths"]
        all_lengths.extend(ls)
        per_step.setdefault(gs, []).extend(ls)

    n_steps = len(per_step)
    n_total = len(all_lengths)
    sl = sorted(all_lengths)

    print(f"\n  记录条数: {len(stats_records):,}  |  响应总数: {n_total:,}  |  训练步数: {n_steps}")

    # -- 2. 总体分布 --
    d = describe(sl)
    print(f"\n{'-'*70}")
    print("  总体分布统计")
    print(f"{'-'*70}")
    print(f"  均值={d['mean']:.1f}  中位数={d['median']}  标准差={d['std']:.1f}  变异系数CV={d['cv']:.3f}")
    print(f"  min={d['min']}  p10={d['p10']}  p25={d['p25']}  p50={d['median']}  "
          f"p75={d['p75']}  p90={d['p90']}  p95={d['p95']}  p99={d['p99']}  max={d['max']}")

    # -- 3. 直方图 --
    actual_max = d["max"]
    max_len = actual_max
    for candidate in [512, 1024, 2048, 3072, 4096, 8192]:
        if actual_max <= candidate * 1.02:
            max_len = candidate
            break

    buckets = []
    step = max_len // 8
    lo = 0
    while lo < max_len:
        hi = lo + step
        buckets.append((lo, hi))
        lo = hi
    buckets.append((max_len, 999999))

    print(f"\n{'-'*70}")
    print(f"  长度分布直方图  (推断 l_max={max_len})")
    print(f"{'-'*70}")
    for lo, hi in buckets:
        cnt = sum(1 for l in sl if lo <= l < hi)
        pct = cnt / n_total * 100
        label = f"{lo}-{hi}" if hi < 999999 else f">={lo}"
        print(f"  {label:>12} : {cnt:>6}  ({pct:5.1f}%)  {bar(pct, 100, 36)}")

    # -- 4. 截断率 --
    trunc = sum(1 for l in sl if l >= max_len)
    near  = sum(1 for l in sl if l >= max_len * 0.95)
    print(f"\n{'-'*70}")
    print("  截断分析")
    print(f"{'-'*70}")
    print(f"  达到 l_max ({max_len})           : {trunc:>6}  ({trunc/n_total*100:.2f}%)")
    print(f"  接近 l_max (>={max_len*0.95:.0f})   : {near:>6}  ({near/n_total*100:.2f}%)")
    trunc_rate = trunc / n_total
    if trunc_rate < 0.01:
        print("  [!] 截断率极低: 绝大多数样本一段就完成，分段机制几乎未被使用")
    elif trunc_rate < 0.05:
        print("  [~] 截断率较低: 少量样本需要多段，分段机制参与度有限")
    elif trunc_rate < 0.20:
        print("  [+] 截断率适中: 分段机制有效参与，训练数据有意义")
    else:
        print("  [*] 截断率较高: 大量样本需要多段续写，分段机制是关键")

    # -- 5. 长尾判定 --
    skew_ratio     = (d["mean"] - d["median"]) / d["median"] if d["median"] > 0 else 0
    p99_over_med   = d["p99"] / d["median"] if d["median"] > 0 else 0
    reasons = []
    if skew_ratio > 0.3:
        reasons.append(f"均值/中位数偏差 {skew_ratio*100:.1f}% (>30%) -> 右偏分布")
    if d["cv"] > 0.6:
        reasons.append(f"变异系数 CV={d['cv']:.3f} > 0.6 -> 方差过大")
    if p99_over_med > 4.0:
        reasons.append(f"p99/中位数 = {p99_over_med:.1f}x > 4x -> 存在重尾")
    if trunc_rate > 0.05:
        reasons.append(f"截断率 {trunc_rate*100:.1f}% > 5% -> 大量样本被截断")

    score = len(reasons)
    if score == 0:
        verdict = "无长尾  [OK]   分布紧凑均衡"
    elif score == 1:
        verdict = "轻微    [WARN] 存在轻微偏斜，暂不影响训练"
    elif score == 2:
        verdict = "中等    [WARN] 明显长尾，需关注"
    else:
        verdict = "严重    [BAD]  强长尾，可能导致训练不稳定"

    print(f"\n{'-'*70}")
    print("  长尾问题判定")
    print(f"{'-'*70}")
    print(f"  严重程度: {verdict}")
    for r in reasons:
        print(f"    - {r}")
    if not reasons:
        print("    - 未检测到长尾指标")

    # -- 6. 分布形态诊断 --
    short_pct = sum(1 for l in sl if l < max_len * 0.25) / n_total * 100
    long_pct  = sum(1 for l in sl if l > max_len * 0.75) / n_total * 100

    print(f"\n{'-'*70}")
    print("  分布形态诊断")
    print(f"{'-'*70}")
    print(f"  短响应 (<{max_len*0.25:.0f}, 不足 l_max 25%): {short_pct:.1f}%")
    print(f"  长响应 (>{max_len*0.75:.0f}, 超过 l_max 75%): {long_pct:.1f}%")
    print(f"  中位数 / l_max: {d['median']/max_len*100:.1f}%")

    if d["median"] / max_len < 0.25:
        print("  [!] 左侧堆积: 大多数响应很短，数据对该模型可能过于简单")
        print("      建议: 换用更难的数据集，或换用更小的模型")
    elif d["median"] / max_len > 0.70:
        print("  [!] 右侧堆积: 响应普遍接近 l_max，建议增大 max_response_length")
    elif long_pct > 15:
        print("  [~] 双峰迹象: 短响应和长响应共存")
        print("      符合预期: 简单样本快速完成，难样本需要多段续写")
    else:
        print("  [+] 分布均衡: 长度分布合理，适合分段 Rollout")

    # -- 7. 逐步趋势 --
    print(f"\n{'-'*70}")
    print("  逐训练步响应长度趋势 (均值)")
    print(f"{'-'*70}")
    step_means = []
    for gs in sorted(per_step.keys()):
        ls = per_step[gs]
        m = sum(ls) / len(ls)
        step_means.append((gs, m, len(ls)))

    for i, (gs, m, cnt) in enumerate(step_means):
        suffix = f"  step{gs:>3}: {m:>6.1f} (n={cnt})"
        if (i + 1) % 5 == 0:
            print(suffix)
        else:
            print(suffix, end="")
    print()

    third = max(1, len(step_means) // 3)
    early = [m for _, m, _ in step_means[:third]]
    late  = [m for _, m, _ in step_means[-third:]]
    early_avg = sum(early) / len(early)
    late_avg  = sum(late)  / len(late)
    trend_pct = (late_avg - early_avg) / early_avg * 100
    print(f"\n  前 {third} 步均值: {early_avg:.1f}")
    print(f"  后 {third} 步均值: {late_avg:.1f}")
    if trend_pct > 10:
        print(f"  [+] 响应长度增长 (+{trend_pct:.1f}%): 模型逐步使用更长的思考链")
    elif trend_pct < -10:
        print(f"  [~] 响应长度缩短 ({trend_pct:.1f}%): 模型趋向更简洁的回答")
    else:
        print(f"  [=] 响应长度稳定 ({trend_pct:+.1f}%): 无明显变化趋势")

    # -- 8. 文本内容分析 --
    if resp_path:
        print(f"\n{'-'*70}")
        print("  响应文本内容分析")
        print(f"{'-'*70}")
        resp_records = load_jsonl(resp_path)
        all_texts = []
        for rec in resp_records:
            raw = rec.get("responses", [])
            if isinstance(raw, list):
                all_texts.extend(raw)
            elif isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        all_texts.extend(parsed)
                except Exception:
                    all_texts.append(raw)

        if all_texts:
            text_lens  = [len(t) for t in all_texts]
            token_lens = [len(t.split()) for t in all_texts]
            print(f"  响应文本总数   : {len(all_texts):,}")
            print(f"  字符长度       : 均值={sum(text_lens)/len(text_lens):.0f}  "
                  f"中位数={sorted(text_lens)[len(text_lens)//2]}  最大={max(text_lens)}")
            print(f"  单词数         : 均值={sum(token_lens)/len(token_lens):.0f}  "
                  f"中位数={sorted(token_lens)[len(token_lens)//2]}  最大={max(token_lens)}")

            boxed    = sum(1 for t in all_texts if "\\boxed{" in t or "boxed" in t.lower())
            truncated = sum(1 for t in all_texts
                            if not any(t.rstrip().endswith(e) for e in [".", "}", "]", "\\]", ")"]))
            print(f"  含 \\boxed{{}}    : {boxed} ({boxed/len(all_texts)*100:.1f}%)")
            print(f"  疑似截断 (无结尾标点): {truncated} ({truncated/len(all_texts)*100:.1f}%)")

    # -- 汇总 --
    print(f"\n{'='*70}")
    print("  汇总")
    print(f"{'='*70}")
    print(f"  响应总数         : {n_total:,}")
    print(f"  长度 中位数/均值  : {d['median']} / {d['mean']:.1f}")
    print(f"  l_max 利用率     : {d['median']/max_len*100:.1f}% (中位数)  {d['mean']/max_len*100:.1f}% (均值)")
    print(f"  截断率           : {trunc_rate*100:.2f}%")
    print(f"  长尾严重程度      : {verdict.split()[0]}")
    shape = "左侧堆积" if d["median"] / max_len < 0.25 else ("右侧堆积" if d["median"] / max_len > 0.70 else "均衡")
    trend = "增长" if trend_pct > 10 else ("缩短" if trend_pct < -10 else "稳定")
    print(f"  分布形态         : {shape}")
    print(f"  长度趋势         : {trend}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析 Rollout 响应长度分布")
    parser.add_argument("folder", help="包含 rollout jsonl 文件的文件夹路径")
    args = parser.parse_args()
    analyze(args.folder)
