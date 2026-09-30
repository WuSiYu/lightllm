#!/usr/bin/env python3
"""Run a warmed, live fake-Decode comparison against a shared Prefill pool.

The worker pool is expected to be running already. This program owns each
temporary PD master, waits for all Prefill workers to register, performs a
mixed short/long warmup before every measured scenario, and replays the same
length trace for every policy.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import aiohttp
from transformers import AutoTokenizer

from flex_tp_step_sim import (
    WorkloadRequest,
    _repo_root,
    generate_servegen_mm_image,
    generate_synthetic_5pct,
)


POLICY_TO_SELECTOR = {
    # The fixed policies use the V6 lifecycle with a homogeneous worker pool;
    # topology (and therefore TP size) is supplied by the worker profile.
    "fixed_tp2": "flex_tp_v6",
    "fixed_tp4": "flex_tp_v6",
    "v3": "flex_tp_v3",
    "v4": "flex_tp_v4",
    "v5": "flex_tp_v5",
    "v6": "flex_tp_v6",
    "v7": "flex_tp_v7",
    "v8": "flex_tp_v8",
    "v9": "flex_tp_v9",
    "v10": "flex_tp_v10",
    "v11": "flex_tp_v11",
    "v12": "flex_tp_v12",
    "v13": "flex_tp_v13",
    "naive": "flex_tp_naive",
    "naive_switch": "flex_tp_naive_switch",
}

# V7 led simulated mean token goodput; V6 led SLO, so both run first.
DEFAULT_POLICY_ORDER = ("fixed_tp2", "fixed_tp4", "naive", "naive_switch", "v6", "v12", "v13")
SHORT_WARMUP_LENGTHS = (128, 512, 1024, 2048)
# Include a value above the V13 default threshold so its dedicated TP4
# fallback is exercised during every live policy warmup.
LONG_WARMUP_LENGTHS = (4096, 6000, 16000)


@dataclass(frozen=True)
class TraceRequest:
    request_id: int
    arrival_s: float
    input_tokens: int
    is_long: bool


@dataclass
class RequestResult:
    request_id: int
    input_tokens: int
    is_long: bool
    success: bool
    status: int
    ttft_s: Optional[float]
    e2e_s: float
    send_lag_s: float
    prompt_tokens: Optional[int]
    finished: bool
    response_bytes: int
    error: Optional[str]


class ManagedMaster:
    def __init__(self, args: argparse.Namespace, policy: str, log_path: Path) -> None:
        self.args = args
        self.policy = policy
        self.log_path = log_path
        self.process: Optional[subprocess.Popen] = None
        self.log_file = None

    def start(self) -> None:
        if _port_is_open(self.args.host, self.args.port):
            raise RuntimeError(f"refusing to start master: {self.args.host}:{self.args.port} is already open")

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("w")
        command = [
            sys.executable,
            "-u",
            "-m",
            "lightllm.server.api_server",
            "--model_dir",
            str(self.args.model_dir),
            "--max_req_total_len",
            str(self.args.max_req_total_len),
            "--run_mode",
            "pd_master",
            "--select_p_d_node_strategy",
            POLICY_TO_SELECTOR[self.policy],
            "--flex_tp_threshold",
            str(self.args.long_threshold),
            "--flex_tp_long_threshold",
            str(self.args.long_threshold),
            "--flex_tp_slo_ttft",
            str(self.args.slo_ttft),
            "--flex_tp_mps_slowdown",
            str(self.args.mps_slowdown),
            "--flex_tp_bundle_window_ms",
            "20",
            "--flex_tp_bundle_token_cap",
            "8192",
            "--flex_tp_bundle_token_trigger",
            "4096",
            "--flex_tp_max_inflight",
            "64",
            "--flex_tp_instance_token_credit",
            "16384",
            "--flex_tp_prediction_margin",
            "0.08",
            "--flex_tp_v10_short_weight",
            str(self.args.v10_short_weight),
            "--flex_tp_v10_long_weight",
            str(self.args.v10_long_weight),
            "--flex_tp_v10_slack_weight",
            str(self.args.v10_slack_weight),
            "--flex_tp_v10_overlap_slack_ratio",
            str(self.args.v10_overlap_slack_ratio),
            "--flex_tp_v11_short_weight",
            str(self.args.v11_short_weight),
            "--flex_tp_v11_long_weight",
            str(self.args.v11_long_weight),
            "--flex_tp_v11_slack_weight",
            str(self.args.v11_slack_weight),
            "--flex_tp_v11_overlap_slack_ratio",
            str(self.args.v11_overlap_slack_ratio),
            "--flex_tp_v11_routing_cost_weight",
            str(self.args.v11_routing_cost_weight),
            "--flex_tp_v12_routing_cost_weight",
            str(self.args.v12_routing_cost_weight),
            "--flex_tp_v12_tp4_service_ratio_limit",
            str(self.args.v12_tp4_service_ratio_limit),
            "--flex_tp_v13_latency_scale",
            str(self.args.v13_latency_scale),
            "--flex_tp_v13_long_threshold",
            str(self.args.v13_long_threshold),
            "--flex_tp_v13_routing_cost_weight",
            str(self.args.v13_routing_cost_weight),
            "--flex_tp_v13_tp4_service_ratio_limit",
            str(self.args.v13_tp4_service_ratio_limit),
            "--flex_tp_v13_tp4_pressure_threshold",
            str(self.args.v13_tp4_pressure_threshold),
            "--pd_fake_decode",
            "--pd_fake_decode_kv_transfer_fixed_ms",
            str(self.args.fake_kv_fixed_ms),
            "--pd_fake_decode_kv_transfer_us_per_token",
            str(self.args.fake_kv_us_per_token),
            "--host",
            self.args.host,
            "--port",
            str(self.args.port),
        ]
        env = os.environ.copy()
        for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
            env.pop(name, None)
        self.process = subprocess.Popen(
            command,
            cwd=_repo_root(),
            env=env,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def wait_for_workers(self) -> None:
        assert self.process is not None
        deadline = time.monotonic() + self.args.master_ready_timeout
        last_registration_count = 0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"master {self.policy} exited with {self.process.returncode}:\n{_tail(self.log_path)}"
                )
            if self.log_path.exists():
                text = self.log_path.read_text(errors="replace")
                last_registration_count = text.count('"GET /pd_register 1.1" 101')
                if "server start up ok" in text and last_registration_count >= self.args.expected_prefill_workers:
                    print(
                        f"[master:{self.policy}] ready with {last_registration_count} Prefill registrations",
                        flush=True,
                    )
                    return
            time.sleep(1)
        raise TimeoutError(
            f"master {self.policy} did not receive {self.args.expected_prefill_workers} workers; "
            f"observed {last_registration_count}:\n{_tail(self.log_path)}"
        )

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        if self.log_file is not None:
            self.log_file.close()
        self.process = None
        deadline = time.monotonic() + 20
        while _port_is_open(self.args.host, self.args.port) and time.monotonic() < deadline:
            time.sleep(0.25)
        if _port_is_open(self.args.host, self.args.port):
            raise RuntimeError(f"master port {self.args.host}:{self.args.port} remained open after stop")

    def __enter__(self) -> "ManagedMaster":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _default_host() -> str:
    """Use the host address when resolvable, with a local-test fallback."""
    try:
        return socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        return "127.0.0.1"


def _tail(path: Path, lines: int = 50) -> str:
    if not path.exists():
        return "<log not created>"
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _normalize_trace(
    requests: Iterable[WorkloadRequest], max_input_tokens: int, long_threshold: int
) -> List[TraceRequest]:
    source = list(requests)
    if not source:
        raise ValueError("workload generator returned no requests")
    first_arrival = min(item.arrival_s for item in source)
    trace: List[TraceRequest] = []
    long_span = max(1, max_input_tokens - long_threshold)
    for item in source:
        input_tokens = max(4, int(item.input_tokens))
        if input_tokens > max_input_tokens:
            if input_tokens >= long_threshold:
                input_tokens = long_threshold + 1 + (
                    (input_tokens - long_threshold - 1) % long_span
                )
            else:
                input_tokens = max_input_tokens
        trace.append(
            TraceRequest(
                request_id=len(trace) + 1,
                arrival_s=max(0.0, float(item.arrival_s) - first_arrival),
                input_tokens=input_tokens,
                is_long=input_tokens >= long_threshold,
            )
        )
    return trace


def _policy_long_threshold(args: argparse.Namespace, policy: str) -> int:
    return args.v13_long_threshold if policy == "v13" else args.long_threshold


def _make_scenarios(args: argparse.Namespace) -> Dict[str, List[TraceRequest]]:
    scenarios: Dict[str, List[TraceRequest]] = {}
    if args.scenarios in ("all", "servegen-mm-image"):
        servegen_rates = args.servegen_rates or (args.servegen_rate,)
        for rate in servegen_rates:
            scenarios[f"servegen-mm-image-r{rate:g}"] = _normalize_trace(
                generate_servegen_mm_image(
                    repo_root=_repo_root(),
                    request_rate=rate,
                    duration_s=args.servegen_duration,
                    seed=args.seed,
                ),
                args.max_input_tokens,
                args.long_threshold,
            )
    if args.scenarios in ("all", "synthetic-5pct"):
        synthetic_rates = args.synthetic_rates or (args.synthetic_rate,)
        for rate in synthetic_rates:
            scenarios[f"synthetic-5pct-r{rate:g}"] = _normalize_trace(
                generate_synthetic_5pct(
                    request_rate=rate,
                    num_prompts=args.synthetic_prompts,
                    seed=args.seed,
                ),
                args.max_input_tokens,
                args.long_threshold,
            )
    return scenarios


def _trace_fingerprint(trace: Sequence[TraceRequest]) -> str:
    payload = json.dumps(
        [(round(item.arrival_s, 6), item.input_tokens) for item in trace],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class ExactPromptFactory:
    """Build unique strings whose encoded length exactly matches the trace."""

    def __init__(self, model_dir: Path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        prefix = "0123456789abcdef"
        prefix_tokens = len(self.tokenizer.encode(prefix, add_special_tokens=True))
        extended_tokens = len(self.tokenizer.encode(prefix + " word", add_special_tokens=True))
        if extended_tokens != prefix_tokens + 1:
            raise RuntimeError("the model tokenizer does not encode leading-space ' word' as one token")

    def build(self, namespace: str, request_id: int, target_length: int) -> str:
        # The per-request prefix defeats cross-policy radix-cache reuse. Its
        # variable BPE length is subtracted before adding one-token fillers.
        prefix = hashlib.sha256(f"{namespace}:{request_id}".encode()).hexdigest()[:16]
        prefix_tokens = len(self.tokenizer.encode(prefix, add_special_tokens=True))
        filler_count = target_length - prefix_tokens
        if filler_count < 0:
            raise ValueError(
                f"target length {target_length} is shorter than unique prefix length {prefix_tokens}"
            )
        prompt = prefix + " word" * filler_count
        encoded_length = len(self.tokenizer.encode(prompt, add_special_tokens=True))
        if encoded_length != target_length:
            raise AssertionError(
                f"exact prompt construction failed: target={target_length}, encoded={encoded_length}"
            )
        return prompt


async def _send_one(
    session: aiohttp.ClientSession,
    url: str,
    item: TraceRequest,
    prompt: str,
    scheduled_at: float,
) -> RequestResult:
    now = time.perf_counter()
    if scheduled_at > now:
        await asyncio.sleep(scheduled_at - now)
    started = time.perf_counter()
    send_lag = max(0.0, started - scheduled_at)
    payload = {
        "inputs": prompt,
        "parameters": {"do_sample": False, "ignore_eos": True, "max_new_tokens": 1},
    }
    first_data_at: Optional[float] = None
    status = 0
    response = bytearray()
    error: Optional[str] = None
    finished = False
    prompt_tokens: Optional[int] = None
    try:
        async with session.post(url, json=payload) as http_response:
            status = http_response.status
            async for chunk in http_response.content.iter_any():
                if chunk and first_data_at is None:
                    first_data_at = time.perf_counter()
                response.extend(chunk)
            if status != 200:
                error = f"HTTP {status}: {bytes(response[:300]).decode(errors='replace')}"
            else:
                for line in bytes(response).decode(errors="replace").splitlines():
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[len("data:") :].strip())
                    finished = finished or bool(event.get("finished"))
                    token_info = event.get("token") or {}
                    value = token_info.get("prompt_tokens")
                    if value is not None:
                        prompt_tokens = int(value)
                if first_data_at is None:
                    error = "HTTP 200 response contained no SSE data"
                elif not finished:
                    error = "SSE stream did not contain a finished event"
    except Exception as exc:
        error = repr(exc)
    ended = time.perf_counter()
    success = error is None and status == 200 and first_data_at is not None and finished
    return RequestResult(
        request_id=item.request_id,
        input_tokens=item.input_tokens,
        is_long=item.is_long,
        success=success,
        status=status,
        ttft_s=(first_data_at - started) if first_data_at is not None else None,
        e2e_s=ended - started,
        send_lag_s=send_lag,
        prompt_tokens=prompt_tokens,
        finished=finished,
        response_bytes=len(response),
        error=error,
    )


async def _run_warmup(
    session: aiohttp.ClientSession,
    url: str,
    policy: str,
    scenario: str,
    args: argparse.Namespace,
    prompt_factory: ExactPromptFactory,
) -> Dict:
    all_results: List[RequestResult] = []
    started = time.perf_counter()
    route_threshold = _policy_long_threshold(args, policy)
    for round_index in range(args.warmup_rounds):
        short_requests: List[TraceRequest] = []
        for index in range(args.warmup_short_requests):
            length = min(
                route_threshold - 1,
                SHORT_WARMUP_LENGTHS[index % len(SHORT_WARMUP_LENGTHS)],
            )
            short_requests.append(TraceRequest(index + 1, 0.0, length, False))
        long_requests: List[TraceRequest] = []
        for index in range(args.warmup_long_requests):
            profiled_length = LONG_WARMUP_LENGTHS[index % len(LONG_WARMUP_LENGTHS)]
            length = min(
                args.max_input_tokens,
                max(route_threshold + 1, profiled_length),
            )
            long_requests.append(
                TraceRequest(args.warmup_short_requests + index + 1, 0.0, length, True)
            )
        namespace = f"warmup:{policy}:{scenario}:{round_index}:{time.time_ns()}"
        short_results = await _send_in_batches(
            session,
            url,
            short_requests,
            namespace + ":short",
            args.warmup_short_concurrency,
            prompt_factory,
        )
        short_failures = [result for result in short_results if not result.success]
        if short_failures:
            raise RuntimeError(
                f"short warmup failed: {[result.error for result in short_failures[:3]]}"
            )
        long_results = await _send_in_batches(
            session,
            url,
            long_requests,
            namespace + ":long",
            args.warmup_long_concurrency,
            prompt_factory,
        )
        round_results = short_results + long_results
        all_results.extend(round_results)
        failures = [result for result in round_results if not result.success]
        print(
            f"[warmup:{policy}:{scenario}] round={round_index + 1}/{args.warmup_rounds} "
            f"short={args.warmup_short_requests} long={args.warmup_long_requests} "
            f"concurrency={args.warmup_short_concurrency}/{args.warmup_long_concurrency} "
            f"ok={len(round_results) - len(failures)}/{len(round_results)}",
            flush=True,
        )
        if failures:
            raise RuntimeError(f"warmup failed: {[result.error for result in failures[:3]]}")
    ttfts = [result.ttft_s for result in all_results if result.ttft_s is not None]
    return {
        "rounds": args.warmup_rounds,
        "short_requests": args.warmup_rounds * args.warmup_short_requests,
        "long_requests": args.warmup_rounds * args.warmup_long_requests,
        "short_concurrency": args.warmup_short_concurrency,
        "long_concurrency": args.warmup_long_concurrency,
        "successful_requests": sum(result.success for result in all_results),
        "total_requests": len(all_results),
        "duration_s": time.perf_counter() - started,
        "ttft_p95_s": _percentile(ttfts, 95),
    }


async def _send_in_batches(
    session: aiohttp.ClientSession,
    url: str,
    requests: Sequence[TraceRequest],
    namespace: str,
    concurrency: int,
    prompt_factory: ExactPromptFactory,
) -> List[RequestResult]:
    results: List[RequestResult] = []
    prompts = [
        prompt_factory.build(namespace, item.request_id, item.input_tokens)
        for item in requests
    ]
    for offset in range(0, len(requests), concurrency):
        batch = requests[offset : offset + concurrency]
        prompt_batch = prompts[offset : offset + concurrency]
        scheduled_at = time.perf_counter()
        results.extend(
            await asyncio.gather(
                *(
                    _send_one(session, url, item, prompt, scheduled_at)
                    for item, prompt in zip(batch, prompt_batch)
                )
            )
        )
    return results


def _summarize(
    policy: str,
    scenario: str,
    trace: Sequence[TraceRequest],
    results: Sequence[RequestResult],
    elapsed_s: float,
    warmup: Dict,
    slo_ttft: float,
    long_threshold: int,
) -> Dict:
    successes = [result for result in results if result.success]
    ttfts = [result.ttft_s for result in successes if result.ttft_s is not None]
    e2es = [result.e2e_s for result in successes]
    transfer_tails = [
        result.e2e_s - result.ttft_s
        for result in successes
        if result.ttft_s is not None
    ]
    short_ttfts = [
        result.ttft_s
        for result in successes
        if result.input_tokens < long_threshold and result.ttft_s is not None
    ]
    long_ttfts = [
        result.ttft_s
        for result in successes
        if result.input_tokens >= long_threshold and result.ttft_s is not None
    ]
    observed_input_tokens = [
        result.prompt_tokens
        for result in successes
        if result.prompt_tokens is not None and result.prompt_tokens > 0
    ]
    input_tokens = sum(
        result.prompt_tokens
        if result.prompt_tokens is not None and result.prompt_tokens > 0
        else result.input_tokens
        for result in successes
    )
    route_class_matches = sum(
        result.prompt_tokens is not None
        and (result.prompt_tokens >= long_threshold)
        == (result.input_tokens >= long_threshold)
        for result in successes
    )
    return {
        "policy": policy,
        "selector": POLICY_TO_SELECTOR[policy],
        "scenario": scenario,
        "trace_fingerprint": _trace_fingerprint(trace),
        "offered_requests": len(trace),
        "successful_requests": len(successes),
        "failed_requests": len(trace) - len(successes),
        "completion_fraction": len(successes) / len(trace),
        "long_request_fraction": sum(item.input_tokens >= long_threshold for item in trace)
        / len(trace),
        "elapsed_s": elapsed_s,
        "request_throughput_rps": len(successes) / elapsed_s,
        "input_token_throughput_s": input_tokens / elapsed_s,
        "slo_ttft_s": slo_ttft,
        "offered_slo_attainment": sum(value <= slo_ttft for value in ttfts) / len(trace),
        "ttft_mean_s": statistics.fmean(ttfts) if ttfts else None,
        "ttft_p50_s": _percentile(ttfts, 50),
        "ttft_p95_s": _percentile(ttfts, 95),
        "ttft_p99_s": _percentile(ttfts, 99),
        "short_ttft_p95_s": _percentile(short_ttfts, 95),
        "long_ttft_p95_s": _percentile(long_ttfts, 95),
        "e2e_p95_s": _percentile(e2es, 95),
        "client_fake_transfer_tail_p50_s": _percentile(transfer_tails, 50),
        "client_fake_transfer_tail_p95_s": _percentile(transfer_tails, 95),
        "send_lag_p95_s": _percentile([result.send_lag_s for result in results], 95),
        "prompt_tokens_observed_fraction": (
            len(observed_input_tokens) / len(successes) if successes else 0.0
        ),
        "prompt_token_delta_mean": (
            statistics.fmean(
                result.prompt_tokens - result.input_tokens
                for result in successes
                if result.prompt_tokens is not None
            )
            if observed_input_tokens
            else None
        ),
        "route_class_match_fraction": route_class_matches / len(successes) if successes else 0.0,
        "warmup": warmup,
    }


async def _run_scenario(
    args: argparse.Namespace,
    policy: str,
    scenario: str,
    trace: Sequence[TraceRequest],
    prompt_factory: ExactPromptFactory,
) -> Tuple[Dict, List[RequestResult]]:
    timeout = aiohttp.ClientTimeout(total=args.request_timeout, connect=30)
    connector = aiohttp.TCPConnector(limit=0)
    url = f"http://{args.host}:{args.port}/generate_stream"
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as session:
        warmup = await _run_warmup(session, url, policy, scenario, args, prompt_factory)
        await asyncio.sleep(args.post_warmup_settle_s)
        namespace = f"measured:{policy}:{scenario}:{time.time_ns()}"
        prompts = [
            prompt_factory.build(namespace, item.request_id, item.input_tokens)
            for item in trace
        ]
        benchmark_started = time.perf_counter()
        tasks = [
            asyncio.create_task(
                _send_one(
                    session,
                    url,
                    item,
                    prompt,
                    benchmark_started + item.arrival_s,
                )
            )
            for item, prompt in zip(trace, prompts)
        ]
        results = await asyncio.gather(*tasks)
        elapsed_s = time.perf_counter() - benchmark_started
    summary = _summarize(
        policy,
        scenario,
        trace,
        results,
        elapsed_s,
        warmup,
        args.slo_ttft,
        _policy_long_threshold(args, policy),
    )
    p95_text = (
        f"{summary['ttft_p95_s']:.3f}s"
        if summary["ttft_p95_s"] is not None
        else "n/a"
    )
    print(
        f"[result:{policy}:{scenario}] ok={summary['successful_requests']}/{summary['offered_requests']} "
        f"tokens={summary['input_token_throughput_s']:.1f}/s "
        f"p95={p95_text} SLO={summary['offered_slo_attainment']:.4f}",
        flush=True,
    )
    return summary, results


def _parse_policies(value: str) -> Tuple[str, ...]:
    policies = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    unknown = [item for item in policies if item not in POLICY_TO_SELECTOR]
    if not policies or unknown or len(set(policies)) != len(policies):
        raise argparse.ArgumentTypeError(
            f"policies must be unique values from {','.join(POLICY_TO_SELECTOR)}; unknown={unknown}"
        )
    return policies


def _parse_rates(value: str) -> Tuple[float, ...]:
    rates = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not rates or any(not math.isfinite(item) or item <= 0 for item in rates):
        raise argparse.ArgumentTypeError("rates must be comma-separated positive finite values")
    if len(set(rates)) != len(rates):
        raise argparse.ArgumentTypeError("rates must not contain duplicates")
    return rates


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _at_least_two(value: str) -> int:
    result = int(value)
    if result < 2:
        raise argparse.ArgumentTypeError("mixed warmup requires at least two requests")
    return result


def _write_suite_results(
    args: argparse.Namespace,
    traces: Dict[str, List[TraceRequest]],
    policy_results: Dict[str, List[Dict]],
) -> None:
    for scenario in traces:
        fingerprints = {
            row["trace_fingerprint"]
            for rows in policy_results.values()
            for row in rows
            if row["scenario"] == scenario
        }
        if len(fingerprints) > 1:
            raise AssertionError(f"trace mismatch across policies for {scenario}: {fingerprints}")

    aggregates = []
    for policy, rows in policy_results.items():
        aggregates.append(
            {
                "policy": policy,
                "selector": POLICY_TO_SELECTOR[policy],
                "mean_input_token_throughput_s": statistics.fmean(
                    row["input_token_throughput_s"] for row in rows
                ),
                "mean_offered_slo_attainment": statistics.fmean(
                    row["offered_slo_attainment"] for row in rows
                ),
                "mean_ttft_p95_s": statistics.fmean(
                    row["ttft_p95_s"] for row in rows if row["ttft_p95_s"] is not None
                ),
                "all_requests_successful": all(row["failed_requests"] == 0 for row in rows),
                "all_warmups_successful": all(
                    row["warmup"]["successful_requests"] == row["warmup"]["total_requests"]
                    for row in rows
                ),
            }
        )
    ranking = sorted(
        aggregates,
        key=lambda row: (-row["mean_input_token_throughput_s"], row["mean_ttft_p95_s"]),
    )
    output = {
        "date_marker": "260905",
        "decode_mode": "production fake decode; one Prefill token plus modeled KV transfer",
        "policy_execution_order": list(args.policies),
        "best_first_basis": "policy order supplied by --policies",
        "warmup_invariant": {
            "before_every_policy_scenario": True,
            "rounds": args.warmup_rounds,
            "short_per_round": args.warmup_short_requests,
            "long_per_round": args.warmup_long_requests,
            "short_concurrency": args.warmup_short_concurrency,
            "long_concurrency": args.warmup_long_concurrency,
        },
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"model_dir", "output_dir", "policies"}
        }
        | {
            "model_dir": str(args.model_dir),
            "output_dir": str(args.output_dir),
            "policies": list(args.policies),
        },
        "results": policy_results,
        "aggregates": aggregates,
        "ranking_by_mean_input_token_throughput": ranking,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=_default_host())
    parser.add_argument("--port", type=int, default=16011)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/mtc/wusiyu/models/Llama-3.3-70B-Instruct"),
    )
    parser.add_argument("--policies", type=_parse_policies, default=DEFAULT_POLICY_ORDER)
    parser.add_argument(
        "--scenarios",
        choices=("all", "servegen-mm-image", "synthetic-5pct"),
        default="all",
    )
    parser.add_argument("--servegen-duration", type=_positive_int, default=20)
    parser.add_argument("--servegen-rate", type=float, default=8.0)
    parser.add_argument("--servegen-rates", type=_parse_rates, default=None)
    parser.add_argument("--synthetic-prompts", type=_positive_int, default=160)
    parser.add_argument("--synthetic-rate", type=float, default=8.0)
    parser.add_argument("--synthetic-rates", type=_parse_rates, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=_positive_int, default=16384)
    parser.add_argument("--max-req-total-len", type=_positive_int, default=65536)
    parser.add_argument("--long-threshold", type=_positive_int, default=4000)
    parser.add_argument("--slo-ttft", type=float, default=3.0)
    parser.add_argument("--mps-slowdown", type=float, default=2.0)
    parser.add_argument("--fake-kv-fixed-ms", type=float, default=20.0)
    parser.add_argument("--fake-kv-us-per-token", type=float, default=0.0)
    parser.add_argument("--v10-short-weight", type=float, default=1.0)
    parser.add_argument("--v10-long-weight", type=float, default=1.0)
    parser.add_argument("--v10-slack-weight", type=float, default=1.0)
    parser.add_argument("--v10-overlap-slack-ratio", type=float, default=0.10)
    parser.add_argument("--v11-short-weight", type=float, default=1.0)
    parser.add_argument("--v11-long-weight", type=float, default=1.0)
    parser.add_argument("--v11-slack-weight", type=float, default=1.0)
    parser.add_argument("--v11-overlap-slack-ratio", type=float, default=0.10)
    parser.add_argument("--v11-routing-cost-weight", type=float, default=0.50)
    parser.add_argument("--v12-routing-cost-weight", type=float, default=2.0)
    parser.add_argument("--v12-tp4-service-ratio-limit", type=float, default=0.60)
    parser.add_argument("--v13-latency-scale", type=float, default=1.0)
    parser.add_argument("--v13-long-threshold", type=int, default=12000)
    parser.add_argument("--v13-routing-cost-weight", type=float, default=2.0)
    parser.add_argument("--v13-tp4-service-ratio-limit", type=float, default=0.60)
    parser.add_argument("--v13-tp4-pressure-threshold", type=float, default=1.0)
    parser.add_argument("--expected-prefill-workers", type=_positive_int, default=3)
    parser.add_argument("--master-ready-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--warmup-rounds", type=_positive_int, default=2)
    parser.add_argument("--warmup-short-requests", type=_at_least_two, default=8)
    parser.add_argument("--warmup-long-requests", type=_at_least_two, default=4)
    parser.add_argument("--warmup-short-concurrency", type=_positive_int, default=4)
    parser.add_argument("--warmup-long-concurrency", type=_positive_int, default=1)
    parser.add_argument("--post-warmup-settle-s", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_/flex_tp_paper_analysis/260905-v12-v13-live-fake-decode"),
    )
    args = parser.parse_args()
    if args.max_input_tokens > args.max_req_total_len - 128:
        parser.error("--max-input-tokens must leave 128 tokens for the unique cache-bypass prefix")
    if args.long_threshold < 5 or args.max_input_tokens <= args.long_threshold:
        parser.error("--long-threshold must be >= 5 and less than --max-input-tokens")
    if "v13" in args.policies and args.max_input_tokens <= args.v13_long_threshold:
        parser.error("--max-input-tokens must exceed --v13-long-threshold when v13 is selected")
    if args.v13_long_threshold < 5:
        parser.error("--v13-long-threshold must be >= 5")
    if not math.isfinite(args.v13_latency_scale) or args.v13_latency_scale <= 0:
        parser.error("--v13-latency-scale must be positive and finite")
    if not math.isfinite(args.v13_routing_cost_weight) or args.v13_routing_cost_weight < 0:
        parser.error("--v13-routing-cost-weight must be finite and non-negative")
    if not math.isfinite(args.v13_tp4_service_ratio_limit) or not 0 <= args.v13_tp4_service_ratio_limit <= 1:
        parser.error("--v13-tp4-service-ratio-limit must be in [0, 1]")
    if not math.isfinite(args.v13_tp4_pressure_threshold) or not 0 <= args.v13_tp4_pressure_threshold <= 1:
        parser.error("--v13-tp4-pressure-threshold must be in [0, 1]")
    for name in ("servegen_rate", "synthetic_rate", "slo_ttft", "master_ready_timeout", "request_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("servegen_rates", "synthetic_rates"):
        values = getattr(args, name)
        if values is not None and any(not math.isfinite(value) or value <= 0 for value in values):
            parser.error(f"--{name.replace('_', '-')} must contain positive finite values")
    if args.post_warmup_settle_s < 0:
        parser.error("--post-warmup-settle-s must be non-negative")
    if not math.isfinite(args.post_warmup_settle_s):
        parser.error("--post-warmup-settle-s must be finite")
    if not math.isfinite(args.mps_slowdown) or args.mps_slowdown < 1:
        parser.error("--mps-slowdown must be finite and >= 1")
    if any(
        not math.isfinite(value) or value < 0
        for value in (
            args.v10_short_weight,
            args.v10_long_weight,
            args.v10_slack_weight,
            args.v10_overlap_slack_ratio,
            args.v11_short_weight,
            args.v11_long_weight,
            args.v11_slack_weight,
            args.v11_overlap_slack_ratio,
            args.v11_routing_cost_weight,
            args.v12_routing_cost_weight,
            args.v12_tp4_service_ratio_limit,
        )
    ):
        parser.error("V10 weights and overlap slack ratio must be finite and non-negative")
    if args.v12_tp4_service_ratio_limit > 1:
        parser.error("--v12-tp4-service-ratio-limit must be <= 1")
    if any(
        not math.isfinite(value) or value < 0
        for value in (args.fake_kv_fixed_ms, args.fake_kv_us_per_token)
    ):
        parser.error("fake KV transfer delays must be finite and non-negative")
    if args.warmup_short_concurrency > args.warmup_short_requests:
        parser.error("--warmup-short-concurrency cannot exceed --warmup-short-requests")
    if args.warmup_long_concurrency > args.warmup_long_requests:
        parser.error("--warmup-long-concurrency cannot exceed --warmup-long-requests")
    return args


def main() -> int:
    args = parse_args()
    scenario_traces = _make_scenarios(args)
    prompt_factory = ExactPromptFactory(args.model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy_results: Dict[str, List[Dict]] = {}
    print(f"policy order: {','.join(args.policies)}", flush=True)
    print(
        f"warmup invariant: before every scenario, {args.warmup_rounds} rounds x "
        f"({args.warmup_short_requests} short @ {args.warmup_short_concurrency} + "
        f"{args.warmup_long_requests} long @ {args.warmup_long_concurrency})",
        flush=True,
    )

    try:
        for policy in args.policies:
            log_path = args.output_dir / "server_logs" / f"master_{policy}.log"
            with ManagedMaster(args, policy, log_path) as master:
                master.wait_for_workers()
                rows: List[Dict] = []
                for scenario, trace in scenario_traces.items():
                    summary, request_results = asyncio.run(
                        _run_scenario(args, policy, scenario, trace, prompt_factory)
                    )
                    rows.append(summary)
                    detail_path = args.output_dir / policy / f"{scenario}.requests.json"
                    detail_path.parent.mkdir(parents=True, exist_ok=True)
                    detail_path.write_text(
                        json.dumps([asdict(result) for result in request_results], indent=2) + "\n"
                    )
                    if summary["failed_requests"]:
                        raise RuntimeError(
                            f"{policy}/{scenario} had {summary['failed_requests']} failed requests"
                        )
                policy_results[policy] = rows
                _write_suite_results(args, scenario_traces, policy_results)
    finally:
        _write_suite_results(args, scenario_traces, policy_results)

    print(f"LIVE_COMPARISON_OK {args.output_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
