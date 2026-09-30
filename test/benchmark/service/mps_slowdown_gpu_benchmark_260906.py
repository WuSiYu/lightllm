#!/usr/bin/env python3
"""Measure CUDA wall-clock slowdown from two clients sharing one GPU with MPS.

This is deliberately independent of the FlexTP simulator.  Each worker runs
two bf16-style GEMMs over ``[input_tokens, width]`` activations, which gives a
repeatable prefill-shaped workload without loading the 70B model.  For every
input length the script measures a single worker first, then two synchronized
workers under a private CUDA MPS pipe.  The reported slowdown is the ratio of
the per-worker median elapsed time in those two conditions.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time


def _positive_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be comma-separated positive integers")
    return values


def _worker(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        print("CUDA is unavailable in worker", file=sys.stderr)
        return 2
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    length = int(args.length)
    width = int(args.width)
    # Two projections approximate the dominant prefill GEMM path.  Keep the
    # tensors resident so timings capture kernels rather than allocation.
    x = torch.randn((length, width), device=device, dtype=torch.float16)
    weight = torch.randn((width, width), device=device, dtype=torch.float16)
    proj = torch.randn((width, width), device=device, dtype=torch.float16)
    for _ in range(args.warmup):
        y = torch.mm(x, weight)
        y = torch.mm(y, proj)
        y.relu_()
    torch.cuda.synchronize(device)

    ready = Path(args.ready)
    go = Path(args.go)
    ready.touch()
    deadline = time.monotonic() + args.start_timeout
    while not go.exists():
        if time.monotonic() >= deadline:
            print("worker start barrier timed out", file=sys.stderr)
            return 3
        time.sleep(0.002)

    samples = []
    for _ in range(args.repeats):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(args.iterations):
            y = torch.mm(x, weight)
            y = torch.mm(y, proj)
            y.relu_()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) / args.iterations)
    Path(args.result).write_text(json.dumps({"samples_s": samples}) + "\n")
    return 0


def _start_workers(
    *,
    length: int,
    args: argparse.Namespace,
    root: Path,
    count: int,
    mps_env: dict[str, str] | None,
) -> list[float]:
    group = root / f"length-{length}-n{count}-{time.time_ns()}"
    group.mkdir(parents=True)
    processes = []
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "OMP_NUM_THREADS": "1"})
    if mps_env:
        env.update(mps_env)
    for index in range(count):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--length",
            str(length),
            "--width",
            str(args.width),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--iterations",
            str(args.iterations),
            "--start-timeout",
            str(args.start_timeout),
            "--ready",
            str(group / f"ready-{index}"),
            "--go",
            str(group / "go"),
            "--result",
            str(group / f"result-{index}.json"),
        ]
        processes.append(subprocess.Popen(command, env=env))

    try:
        deadline = time.monotonic() + args.start_timeout
        ready_paths = [group / f"ready-{index}" for index in range(count)]
        while not all(path.exists() for path in ready_paths):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"worker readiness timed out: length={length} count={count}")
            time.sleep(0.01)
        (group / "go").touch()
        for process in processes:
            process.wait(timeout=args.worker_timeout)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError(
                f"worker failed: length={length} count={count} codes={[p.returncode for p in processes]}"
            )
        samples = []
        for index in range(count):
            payload = json.loads((group / f"result-{index}.json").read_text())
            samples.append(float(statistics.median(payload["samples_s"])))
        return samples
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


def _start_mps(pipe: Path) -> None:
    pipe.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    subprocess.run(
        ["nvidia-cuda-mps-control"],
        input="quit\n",
        text=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    subprocess.run(
        ["nvidia-cuda-mps-control", "-d"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    deadline = time.monotonic() + 10
    while not (pipe / "control").exists():
        if time.monotonic() >= deadline:
            raise RuntimeError(f"MPS control socket did not appear: {pipe}")
        time.sleep(0.1)


def _stop_mps(pipe: Path) -> None:
    if not pipe.exists():
        return
    env = os.environ.copy()
    env["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe)
    subprocess.run(
        ["nvidia-cuda-mps-control"],
        input="quit\n",
        text=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _write_svg(rows: list[dict], output: Path) -> None:
    lengths = sorted({int(row["input_tokens"]) for row in rows})
    width, height = max(360, 220 * len(lengths)), 330
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="10" y="20" font-size="15">Measured MPS slowdown (GPU wall-clock)</text>',
    ]
    for panel, length in enumerate(lengths):
        x0, y0, pw, ph = panel * 220 + 30, 40, 175, 230
        values = [float(row["slowdown"]) for row in rows if int(row["input_tokens"]) == length]
        lo, hi = min(0.0, min(values)), max(1.0, max(values))
        span = max(hi - lo, 1e-9)
        parts.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#999"/>')
        parts.append(f'<text x="{x0 + 4}" y="{y0 + 15}" font-size="11">input={length}</text>')
        points = []
        for row in sorted((r for r in rows if int(r["input_tokens"]) == length), key=lambda r: float(r["concurrent_workers"])):
            factor = float(row["concurrent_workers"])
            value = float(row["slowdown"])
            x = x0 + 10 + (factor - 1.0) / max(max(float(r["concurrent_workers"]) for r in rows) - 1.0, 1e-9) * (pw - 20)
            y = y0 + ph - 10 - (value - lo) / span * (ph - 35)
            points.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#2563eb" stroke-width="2"/>')
        parts.append(f'<text x="{x0 + 4}" y="{y0 + ph + 15}" font-size="9">workers 1..{max(float(r["concurrent_workers"]) for r in rows):g}</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts) + "\n")


def _write_summary(rows: list[dict], output: Path) -> None:
    width = int(rows[0]["width"]) if rows else 0
    lines = [
        "# GPU MPS slowdown (260906)",
        "",
        f"Workload: two resident fp16 GEMMs over `[input_tokens, {width}]`; GPU wall-clock, not simulator output.",
        "Slowdown = median per-worker elapsed time with two MPS clients divided by the single-worker median.",
        "",
        "| input tokens | concurrent MPS workers | single worker (s) | two-worker median (s) | slowdown |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['input_tokens'])} | {row['concurrent_workers']:g} | "
            f"{row['single_worker_s']:.6f} | {row['two_worker_s']:.6f} | {row['slowdown']:.3f} |"
        )
    output.write_text("\n".join(lines) + "\n")


def _measure(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; cannot measure GPU MPS slowdown")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise SystemExit(f"invalid GPU index {args.gpu}; count={torch.cuda.device_count()}")
    torch.cuda.set_device(args.gpu)
    if not args.width % 8 == 0:
        raise SystemExit("width must be divisible by 8")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="mps-slowdown-", dir="/tmp") as temp:
        root = Path(temp)
        pipe = root / "mps-pipe"
        try:
            for length in args.lengths:
                single = _start_workers(length=length, args=args, root=root, count=1, mps_env=None)[0]
                _start_mps(pipe)
                try:
                    overlap = _start_workers(
                        length=length,
                        args=args,
                        root=root,
                        count=2,
                        mps_env={"CUDA_MPS_PIPE_DIRECTORY": str(pipe)},
                    )
                finally:
                    _stop_mps(pipe)
                two_worker = statistics.median(overlap)
                rows.append(
                    {
                        "input_tokens": int(length),
                        "width": int(args.width),
                        "gpu": int(args.gpu),
                        "concurrent_workers": 2,
                        "single_worker_s": single,
                        "two_worker_s": two_worker,
                        "slowdown": two_worker / single,
                        "single_samples_s": [single],
                        "two_worker_samples_s": overlap,
                    }
                )
                print(
                    f"input={length} single={single:.6f}s overlap={two_worker:.6f}s "
                    f"slowdown={two_worker / single:.3f}",
                    flush=True,
                )
        finally:
            _stop_mps(pipe)

    csv_path = args.output_dir / "mps_slowdown_gpu.csv"
    fields = ["input_tokens", "width", "gpu", "concurrent_workers", "single_worker_s", "two_worker_s", "slowdown"]
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    (args.output_dir / "mps_slowdown_gpu.json").write_text(json.dumps(rows, indent=2) + "\n")
    driver = "unknown"
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", str(args.gpu)],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    metadata = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": args.gpu,
        "device_name": torch.cuda.get_device_name(args.gpu),
        "driver_version": driver,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "width": args.width,
        "lengths": args.lengths,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "iterations": args.iterations,
        "concurrent_workers": 2,
        "definition": "median(two-worker MPS wall-clock) / median(single-worker wall-clock)",
    }
    (args.output_dir / "mps_slowdown_gpu_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _write_svg(rows, args.output_dir / "mps_slowdown_gpu.svg")
    _write_summary(rows, args.output_dir / "summary.md")
    print(f"MPS_GPU_SLOWDOWN_OK rows={len(rows)} csv={csv_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--lengths", type=_positive_int_list, default=[256, 1024, 4096, 8192, 16384])
    parser.add_argument("--width", type=int, default=8192)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--start-timeout", type=float, default=60.0)
    parser.add_argument("--worker-timeout", type=float, default=300.0)
    parser.add_argument("--ready", default="")
    parser.add_argument("--go", default="")
    parser.add_argument("--result", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("_/flex_tp_paper_analysis/260906-mps-slowdown-gpu"))
    args = parser.parse_args()
    if args.worker:
        if not args.ready or not args.go or not args.result:
            parser.error("worker requires --ready, --go and --result")
        if args.length <= 0 or args.width <= 0 or args.warmup < 0 or args.repeats <= 0 or args.iterations <= 0:
            parser.error("worker dimensions and counts are invalid")
        return _worker(args)
    if args.width <= 0 or args.warmup < 0 or args.repeats <= 0 or args.iterations <= 0:
        parser.error("dimensions and counts are invalid")
    if not math.isfinite(args.start_timeout) or args.start_timeout <= 0:
        parser.error("start-timeout must be positive")
    return _measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
