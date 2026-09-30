#!/usr/bin/env python3
"""Real LightLLM request saturation probe for worker-side MPS slowdown.

The worker instrumentation (LIGHTLLM_MPS_TRACE_DIR) is the primary metric;
HTTP timings are retained only as an auxiliary observable.
"""
import argparse
import asyncio
import json
import os
import statistics
import time
import uuid
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer


LENGTHS = [32, 64, 128, 256, 1024, 2048, 4096, 8192, 16384]


def make_prompt(tokenizer, target: int, nonce: str) -> tuple[str, int]:
    # Keep a recognizable nonce while converging to the requested tokenizer
    # length.  The server trace records the actual input length used.
    # Repeat the neutral token only; embedding the nonce once avoids turning a
    # 16K-token request into a >200K-token prompt.
    text = f"mps_probe_{nonce} " + ("token " * max(1, target + 8))
    for _ in range(6):
        ids = tokenizer(text, add_special_tokens=True).input_ids
        if len(ids) > target + 2:
            ids = ids[: target]
            text = tokenizer.decode(ids, skip_special_tokens=True)
        elif len(ids) < target:
            text += " token" * (target - len(ids) + 2)
        else:
            break
    return text, len(tokenizer(text, add_special_tokens=True).input_ids)


async def one_request(session, url, prompt, target, stream, out, timeout_s):
        start = time.perf_counter()
        first = None
        tokens = 0
        error = None
        payload = {"inputs": prompt, "parameters": {"do_sample": False, "ignore_eos": True, "max_new_tokens": 1}}
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_s, connect=timeout_s)
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                async for raw, _ in resp.content.iter_chunks():
                    if first is None and raw:
                        first = time.perf_counter()
                    tokens += raw.count(b'"text"')
        except Exception as exc:  # retain failures without stopping saturation
            error = repr(exc)
        end = time.perf_counter()
        out.append({"target_len": target, "stream": stream, "start": start, "first": first,
                    "end": end, "ttft_s": None if first is None else first - start,
                    "latency_s": end - start, "tokens_seen": tokens, "error": error})


async def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    lengths = [int(x) for x in args.lengths.split(",")]
    if args.mode == "overlap" and args.background_len is None:
        raise ValueError("--background-len is required in overlap mode")
    url = args.url.rstrip("/") + "/generate_stream"
    records = []
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        counters = {"target": 0, "background": 0}
        streams = ["target"] + (["background"] if args.mode == "overlap" else [])
        stream_len = {"target": args.target_len,
                      "background": int(args.background_len) if args.background_len is not None else None}
        async def producer(stream):
            rate = args.rate_target if stream == "target" else args.rate_background
            deadline = time.perf_counter() + args.duration
            interval = 1.0 / rate
            next_send = time.perf_counter()
            while next_send < deadline:
                delay = next_send - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                nonce = f"{args.label}_{stream}_{counters[stream]}_{uuid.uuid4().hex[:6]}"
                prompt, _ = make_prompt(tokenizer, stream_len[stream], nonce)
                tasks.append(asyncio.create_task(one_request(session, url, prompt, stream_len[stream], stream, records, args.request_timeout)))
                counters[stream] += 1
                next_send += interval
        await asyncio.gather(*(producer(s) for s in streams))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {"mode": args.mode, "target_len": args.target_len, "background_len": args.background_len,
               "rate_target": args.rate_target, "rate_background": args.rate_background,
               "duration_s": args.duration, "max_inflight": None,
               "sent": counters, "records": records}
    out_path.write_text(json.dumps(summary, indent=2))
    good = [r for r in records if not r["error"]]
    if good:
        p50 = statistics.median(r["latency_s"] for r in good)
        print(json.dumps({"output": str(out_path), "sent": counters, "completed": len(good), "latency_p50_s": p50}, indent=2))
    else:
        print(json.dumps({"output": str(out_path), "sent": counters, "completed": 0}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:60011")
    p.add_argument("--model-dir", default="/mtc/wusiyu/models/Llama-3.3-70B-Instruct")
    p.add_argument("--mode", choices=["solo", "overlap"], required=True)
    p.add_argument("--target-len", type=int, required=True)
    p.add_argument("--background-len", type=int)
    p.add_argument("--lengths", default=','.join(map(str, LENGTHS)))
    p.add_argument("--rate", type=float, default=None, help="legacy alias for both streams")
    p.add_argument("--rate-target", type=float, default=2000.0)
    p.add_argument("--rate-background", type=float, default=2000.0)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--max-inflight", type=int, default=None, help="deprecated; ignored")
    p.add_argument("--request-timeout", type=float, default=180.0)
    p.add_argument("--label", default="mps-real")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.rate is not None:
        args.rate_target = args.rate_background = args.rate
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
