#!/usr/bin/env python
"""比较两个 sweep log：计算每个 prompt_length 点的中位数延迟及比值。

用法: python sweep_ratio.py sweep1.log sweep2.log
解析 long_latency_sweep.sh 的输出：
  - "===== prompt_length = N =====" 标记一个点
  - "Time: X.XXs" 为该点的一次延迟测量
"""
import argparse
import re
import statistics
from collections import OrderedDict

LEN_RE = re.compile(r"prompt_length\s*=\s*(\d+)")
TIME_RE = re.compile(r"Time:\s*([\d.]+)\s*s")


def parse_log(path):
    """返回 OrderedDict[length] -> list[float]"""
    points = OrderedDict()
    cur = None
    with open(path) as f:
        for line in f:
            m = LEN_RE.search(line)
            if m:
                cur = int(m.group(1))
                points.setdefault(cur, [])
                continue
            m = TIME_RE.search(line)
            if m and cur is not None:
                points[cur].append(float(m.group(1)))
    return points


def main():
    ap = argparse.ArgumentParser(description="比较两个 sweep log 每个点的中位数延迟比值")
    ap.add_argument("log1")
    ap.add_argument("log2")
    args = ap.parse_args()

    p1 = parse_log(args.log1)
    p2 = parse_log(args.log2)

    lengths = sorted(set(p1) | set(p2))

    print(f"{'length':>8} | {'med1(s)':>9} {'n1':>3} | {'med2(s)':>9} {'n2':>3} | {'ratio(2/1)':>10}")
    print("-" * 56)
    for L in lengths:
        v1 = p1.get(L, [])
        v2 = p2.get(L, [])
        m1 = statistics.median(v1) if v1 else None
        m2 = statistics.median(v2) if v2 else None
        s1 = f"{m1:9.3f}" if m1 is not None else f"{'-':>9}"
        s2 = f"{m2:9.3f}" if m2 is not None else f"{'-':>9}"
        if m1 is not None and m2 is not None and m2 != 0:
            ratio = f"{m2 / m1:10.3f}"
        else:
            ratio = f"{'-':>10}"
        print(f"{L:>8} | {s1} {len(v1):>3} | {s2} {len(v2):>3} | {ratio}")


if __name__ == "__main__":
    main()
