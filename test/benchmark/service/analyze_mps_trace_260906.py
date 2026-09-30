#!/usr/bin/env python3
"""Analyze worker CUDA-forward traces from the 260906 MPS probe.

The report intentionally separates model-forward execution from HTTP queueing:
CUDA-event ``model_forward_ms`` is used for slowdown, while merged worker
intervals are used to verify whether TP2/TP4 were actually busy.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from pathlib import Path


FLAG_RE = re.compile(r"pd_fake_decode=(True|False)")


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    x = (len(values) - 1) * p
    lo, hi = int(x), min(len(values) - 1, int(x) + 1)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def load_rows(trace_dir: Path):
    raw = []
    for filename in glob.glob(str(trace_dir / "*.jsonl")):
        with open(filename) as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "forward_end_wall" in row:
                    raw.append(row)

    # Every TP rank writes the same batch.  Rank replicas differ by a few ms,
    # so group by port/shape and a 30-ms start-time bucket, retaining the
    # fastest rank as a conservative representative of that batch.
    raw.sort(key=lambda r: (r["worker_port"], r["forward_start_wall"]))
    rows = []
    for row in raw:
        match = None
        for index in range(max(0, len(rows) - 32), len(rows)):
            other = rows[index]
            if (
                other["worker_port"] == row["worker_port"]
                and other["batch_tokens"] == row["batch_tokens"]
                and other["batch_size"] == row["batch_size"]
                and tuple(other["request_input_lens"])
                == tuple(row["request_input_lens"])
                and abs(other["forward_start_wall"] - row["forward_start_wall"]) < 0.03
            ):
                match = index
                break
        if match is None:
            rows.append(row)
        elif row["model_forward_ms"] < rows[match]["model_forward_ms"]:
            rows[match] = row
    return rows


def validate_decode_mode(log_paths, purpose):
    """Validate the decode mode, with a narrowly scoped MPS exception."""
    paths = []
    for item in log_paths:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.log")))
        else:
            paths.append(path)
    values = []
    for path in paths:
        if not path.exists():
            raise ValueError(f"server log does not exist: {path}")
        values.extend(FLAG_RE.findall(path.read_text(errors="replace")))
    if not values:
        raise ValueError("no pd_fake_decode flag found in server logs; refusing an unverified trace")
    if "True" in values and purpose != "mps-slowdown":
        raise ValueError(
            "pd_fake_decode=True is only allowed for the explicit mps-slowdown purpose"
        )
    return {
        "pd_fake_decode_values": sorted(set(values)),
        "server_logs": [str(p) for p in paths],
        "purpose": purpose,
        "fake_decode_exception": purpose == "mps-slowdown" and "True" in values,
    }


def merge(intervals, max_gap=0.0):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + max_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def intersect_intervals(left, right):
    """Return the interval list for the intersection of two merged lists."""
    output = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            output.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return merge(output)


def overlap(left, right):
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def clipped_intervals(rows, start, end):
    intervals = []
    for row in rows:
        left = max(float(row["forward_start_wall"]), start)
        right = min(float(row["forward_end_wall"]), end)
        if right > left:
            intervals.append((left, right))
    return merge(intervals)


def select_saturation_segment(
    rows_by_port, start, end, mode, target_len, selection_gap, min_segment_seconds
):
    """Find the longest initial saturated event cluster without crediting gaps."""
    candidates_by_port = {
        port: merge(
            [
                (
                    max(float(row["forward_start_wall"]), start),
                    min(float(row["forward_end_wall"]), end),
                )
                for row in rows
                if row["forward_end_wall"] > start and row["forward_start_wall"] < end
            ],
            max_gap=selection_gap,
        )
        for port, rows in rows_by_port.items()
    }
    if mode == "overlap":
        candidates = intersect_intervals(
            intersect_intervals(candidates_by_port[8000], candidates_by_port[8001]),
            candidates_by_port[8002],
        )
    elif int(target_len) <= 4000:
        candidates = intersect_intervals(candidates_by_port[8000], candidates_by_port[8001])
    else:
        candidates = candidates_by_port[8002]
    candidates = [item for item in candidates if item[1] - item[0] >= min_segment_seconds]
    return max(candidates, key=lambda item: item[1] - item[0], default=None)


def shape_summary(rows):
    groups = {}
    for row in rows:
        key = (
            int(row["batch_tokens"]),
            int(row["batch_size"]),
            tuple(int(x) for x in row.get("request_input_lens", [])),
        )
        groups.setdefault(key, []).append(row)
    output = []
    for (batch_tokens, batch_size, request_lens), group in sorted(groups.items()):
        values = [float(row["model_forward_ms"]) for row in group]
        output.append(
            {
                "batch_tokens": batch_tokens,
                "batch_size": batch_size,
                "request_input_lens": list(request_lens),
                "count": len(group),
                "model_forward_p50_ms": percentile(values, 0.50),
                "model_forward_p90_ms": percentile(values, 0.90),
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=Path("_/mps_real_260906/highqps_trace"))
    parser.add_argument("--runs", type=Path, default=Path("_/mps_real_260906/highqps_runs.jsonl"))
    parser.add_argument(
        "--server-log",
        action="append",
        required=True,
        help="server log or directory; fake Decode is accepted only for --purpose mps-slowdown",
    )
    parser.add_argument(
        "--purpose",
        choices=("serving", "mps-slowdown"),
        default="serving",
        help="serving rejects pd_fake_decode; mps-slowdown permits it only for worker-side MPS analysis",
    )
    parser.add_argument("--min-busy", type=float, default=0.90)
    parser.add_argument("--min-overlap", type=float, default=0.90)
    parser.add_argument(
        "--segment-gap-ms",
        type=float,
        default=100.0,
        help="maximum event gap used only to locate a saturation segment; gaps are not credited as busy time",
    )
    parser.add_argument(
        "--min-segment-seconds",
        type=float,
        default=10.0,
        help="minimum saturation-segment duration required for a stable slowdown sample",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.min_busy <= 1 or not 0 < args.min_overlap <= 1:
        parser.error("--min-busy and --min-overlap must be in (0, 1]")
    if args.segment_gap_ms < 0 or args.min_segment_seconds <= 0:
        parser.error("--segment-gap-ms must be non-negative and --min-segment-seconds must be positive")
    decode_validation = validate_decode_mode(args.server_log, args.purpose)
    rows = load_rows(args.trace_dir)
    runs = [json.loads(line) for line in args.runs.read_text().splitlines() if line.strip()]
    output = []
    for run in runs:
        start, end = run["start"], run["end"]
        selected = [
            r
            for r in rows
            if r["forward_end_wall"] > start and r["forward_start_wall"] < end
        ]
        tp2_by_port = {
            port: [r for r in selected if r["worker_port"] == port]
            for port in (8000, 8001)
        }
        tp2 = [r for values in tp2_by_port.values() for r in values]
        tp4 = [r for r in selected if r["worker_port"] == 8002]
        tp2_by_port_busy = {
            port: clipped_intervals(values, start, end)
            for port, values in tp2_by_port.items()
        }
        tp2_busy = clipped_intervals(tp2, start, end)
        tp4_busy = clipped_intervals(tp4, start, end)
        duration = end - start
        tp2_port_fractions = {
            str(port): sum(b - a for a, b in intervals) / duration
            for port, intervals in tp2_by_port_busy.items()
        }
        tp2_fraction = min(tp2_port_fractions.values(), default=0.0)
        tp2_union_fraction = sum(b - a for a, b in tp2_busy) / duration
        tp4_fraction = sum(b - a for a, b in tp4_busy) / duration
        overlap_fraction = overlap(tp2_busy, tp4_busy) / duration
        # Compute the strict three-way intersection directly.  The previous
        # aggregate TP2 union was insufficient to prove both TP2 workers were busy.
        all_workers_intervals = []
        for left, right in tp2_by_port_busy[8000]:
            for left2, right2 in tp2_by_port_busy[8001]:
                for left3, right3 in tp4_busy:
                    begin = max(left, left2, left3)
                    finish = min(right, right2, right3)
                    if finish > begin:
                        all_workers_intervals.append((begin, finish))
        all_workers_fraction = sum(
            b - a for a, b in merge(all_workers_intervals)
        ) / duration
        if run.get("mode") == "overlap":
            required_fractions = list(tp2_port_fractions.values()) + [tp4_fraction]
        elif int(run.get("target_len", 0)) <= 4000:
            required_fractions = list(tp2_port_fractions.values())
        else:
            required_fractions = [tp4_fraction]
        full_load = all(value >= args.min_busy for value in required_fractions)
        if run.get("mode") == "overlap":
            full_load = full_load and all_workers_fraction >= args.min_overlap
        rows_by_port = {8000: tp2_by_port[8000], 8001: tp2_by_port[8001], 8002: tp4}
        segment = select_saturation_segment(
            rows_by_port,
            start,
            end,
            run.get("mode"),
            run.get("target_len", 0),
            args.segment_gap_ms / 1000.0,
            args.min_segment_seconds,
        )
        segment_result = None
        segment_rows_by_port = rows_by_port
        segment_full_load = False
        if segment is not None:
            segment_start, segment_end = segment
            segment_duration = segment_end - segment_start
            segment_busy = {
                port: clipped_intervals(values, segment_start, segment_end)
                for port, values in rows_by_port.items()
            }
            segment_fractions = {
                str(port): sum(b - a for a, b in intervals) / segment_duration
                for port, intervals in segment_busy.items()
            }
            segment_all_workers = intersect_intervals(
                intersect_intervals(segment_busy[8000], segment_busy[8001]),
                segment_busy[8002],
            )
            segment_overlap_fraction = sum(
                b - a for a, b in segment_all_workers
            ) / segment_duration
            if run.get("mode") == "overlap":
                segment_required = list(segment_fractions.values())
            elif int(run.get("target_len", 0)) <= 4000:
                segment_required = [segment_fractions["8000"], segment_fractions["8001"]]
            else:
                segment_required = [segment_fractions["8002"]]
            segment_full_load = all(value >= args.min_busy for value in segment_required)
            if run.get("mode") == "overlap":
                segment_full_load = segment_full_load and segment_overlap_fraction >= args.min_overlap
            segment_rows_by_port = {
                port: [
                    row
                    for row in values
                    if row["forward_end_wall"] > segment_start
                    and row["forward_start_wall"] < segment_end
                ]
                for port, values in rows_by_port.items()
            }
            segment_result = {
                "start": segment_start,
                "end": segment_end,
                "duration_seconds": segment_duration,
                "busy_fraction_by_port": segment_fractions,
                "all_workers_overlap_fraction": segment_overlap_fraction,
                "full_load_eligible": segment_full_load,
                "trace_batches": sum(len(values) for values in segment_rows_by_port.values()),
            }
        result = {
            "mode": run["mode"],
            "target_len": run["target_len"],
            "background_len": run.get("background_len"),
            "trace_batches": len(selected),
            "tp2_busy_fraction": tp2_fraction,
            "tp2_union_busy_fraction": tp2_union_fraction,
            "tp2_port_busy_fraction": tp2_port_fractions,
            "tp4_busy_fraction": tp4_fraction,
            "both_busy_fraction": overlap_fraction,
            "all_workers_overlap_fraction": all_workers_fraction,
            "min_busy_fraction": args.min_busy,
            "min_overlap_fraction": args.min_overlap,
            "full_window_full_load_eligible": full_load,
            "full_load_eligible": segment_full_load if segment is not None else full_load,
            "selected_saturation_segment": segment_result,
            "batch_shapes": {},
        }
        segment_tp2 = [
            row for port in (8000, 8001) for row in segment_rows_by_port[port]
        ]
        segment_tp4 = segment_rows_by_port[8002]
        shape_rows = {"tp2": segment_tp2, "tp4": segment_tp4}
        for label, subset in (("tp2", tp2), ("tp4", tp4)):
            values = [r["model_forward_ms"] for r in subset]
            result[f"{label}_forward_p50_ms"] = percentile(values, 0.50)
            result[f"{label}_forward_p90_ms"] = percentile(values, 0.90)
            result[f"{label}_batch_tokens_p50"] = percentile([r["batch_tokens"] for r in subset], 0.50)
            result["batch_shapes"][label] = (
                shape_summary(shape_rows[label]) if segment_full_load else []
            )
        output.append(result)
    payload = {
        "trace_rows_after_rank_dedup": len(rows),
        "decode_validation": decode_validation,
        "selection_policy": {
            "min_busy_fraction": args.min_busy,
            "min_overlap_fraction_for_overlap_runs": args.min_overlap,
            "segment_gap_ms_for_candidate_selection": args.segment_gap_ms,
            "min_segment_seconds": args.min_segment_seconds,
            "overlap_requires_each_tp2_port_and_tp4": True,
            "solo_requires_only_the_routed_worker_group": True,
            "only_full_load_runs_contribute_batch_shapes": True,
        },
        "runs": output,
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
