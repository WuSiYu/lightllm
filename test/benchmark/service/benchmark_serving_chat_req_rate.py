# Adapted from benchmarks/benchmark_serving.py
# of the vllm-project/vllm GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 vLLM Team
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

import argparse
import asyncio
from collections import namedtuple
import dataclasses
import datetime
import json
import os
from queue import Empty, Queue
import random
import sys
import threading
import time
import uuid
import warnings
from typing import AsyncGenerator, Dict, List, Literal, NamedTuple, Tuple, Union

import aiohttp
import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

# from lightllm.server.tokenizer import _FAST_LLAMA_TOKENIZER

from fastchat.model.model_adapter import get_conversation_template
MODEL_TEMPLATE = 'llama-2'
SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
ADDR = ''
PORT = ''

def gen_prompt_from_conversation(conversation: List[Dict[str, str]]):
    conv = get_conversation_template(MODEL_TEMPLATE)
    for message in conversation:
        msg_role = message["role"]
        if msg_role == "system":
            conv.system_message = message["content"]
        elif msg_role == "user":
            conv.append_message(conv.roles[0], message["content"])
        elif msg_role == "assistant":
            conv.append_message(conv.roles[1], message["content"])
        else:
            raise ValueError(f"Unknown role: {msg_role}")

    # Add a blank message for the assistant.
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


_mid_start_times = []
_mid_end_times = []
_mid_end_rates = []
FAILED_REQUESTS = 0
MAX_REQ_TOTAL_TOKENS = 16000

# Predefined simple mixed uniform distributions for --dataset-type simple.X
# Each entry is a list of components: {weight, input_range (min,max), output_range (min,max)}
SIMPLE_DATASETS = {
    "1-0": [
        {"weight": 1.0, "input_range": (100, 1000),   "output_range": (50, 500)},
    ],
    "1-3": [
        {"weight": 0.97, "input_range": (100, 1000),   "output_range": (50, 500)},
        {"weight": 0.03, "input_range": (1000, 20000), "output_range": (50, 500)},
    ],
    "1-10": [
        {"weight": 0.90, "input_range": (100, 1000),   "output_range": (50, 500)},
        {"weight": 0.10, "input_range": (1000, 20000), "output_range": (50, 500)},
    ],
    "1-5": [
        {"weight": 0.95, "input_range": (100, 1000),   "output_range": (50, 500)},
        {"weight": 0.05, "input_range": (1000, 20000), "output_range": (50, 500)},
    ],
    "1-20": [
        {"weight": 0.80, "input_range": (100, 1000),   "output_range": (50, 500)},
        {"weight": 0.20, "input_range": (1000, 20000), "output_range": (50, 500)},
    ],
}

def get_tokenizer(
    tokenizer_name: str,
    tokenizer_mode: str = "auto",
    *args,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    kwargs['trust_remote_code'] = True
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError(
                "Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    if "llama" in tokenizer_name.lower() and kwargs.get("use_fast", True):
        pass
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, *args,
                                                  **kwargs)
    except TypeError as e:
        err_msg = (
            "Failed to load the tokenizer. If you are using a LLaMA-based "
            f"model, use _FAST_LLAMA_TOKENIZER instead of the original "
            "tokenizer.")
        raise RuntimeError(err_msg) from e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        pass
    return tokenizer

@dataclasses.dataclass
class Results():
    prompt: str
    prompt_len: int
    output: str
    output_len: int
    dataset_output_len: int
    latency: float
    token_latencys: List[float]
    chat_rounds: int


RESULTS: List[Results] = []

class Request(NamedTuple):
    prompts: List[Dict[str, str]]
    prompt_len: int
    dataset_output_len: int
    chat_rounds: int
    timestamp: float = -1.0


def _build_user_content_by_token_budget(input_tokens: int) -> str:
    user_tokens = max(4, input_tokens - 64)
    return ("token " * user_tokens).strip()


def sample_requests_from_servegen(args) -> List[Request]:
    if args.request_rate == float("inf") or args.request_rate <= 0:
        raise ValueError("For --dataset-type servegen, --request-rate must be a finite positive number.")

    try:
        from servegen import Category
        from servegen.clientpool import ClientPool
        from servegen.construct import generate_workload
        from servegen.utils import get_constant_rate_fn
    except ImportError:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        servegen_root = os.path.join(repo_root, "_", "ServeGen")
        if servegen_root not in sys.path:
            sys.path.insert(0, servegen_root)
        from servegen import Category
        from servegen.clientpool import ClientPool
        from servegen.construct import generate_workload
        from servegen.utils import get_constant_rate_fn

    duration = 300
    pool = ClientPool(Category.LANGUAGE, args.servegen_mode)
    rate_fn = get_constant_rate_fn(pool.span(0, duration), args.request_rate)
    sg_requests = generate_workload(pool, rate_fn, duration=duration, seed=args.seed)

    sampled_requests: List[Request] = []
    dropped_too_long = 0
    for req in sg_requests:
        input_tokens = max(4, int(req.data.get("input_tokens", 16)))
        output_tokens = max(4, int(req.data.get("output_tokens", 16)))
        if input_tokens + output_tokens > MAX_REQ_TOTAL_TOKENS:
            dropped_too_long += 1
            continue

        system_prompt = ""
        if args.bypass_cache:
            nonce = uuid.uuid4().hex
            system_prompt = f"{nonce} ( <-- cache_bypass, ignore it ) {system_prompt}"

        prompts = [
            dict(role="system", content=system_prompt),
            dict(role="user", content=_build_user_content_by_token_budget(input_tokens)),
        ]
        sampled_requests.append(
            Request(
                prompts=prompts,
                prompt_len=input_tokens,
                dataset_output_len=output_tokens,
                chat_rounds=1,
                timestamp=float(req.timestamp),
            )
        )

    print("done generating ServeGen workload")
    print(f"servegen duration: {duration}s")
    print(f"servegen generated requests: {len(sampled_requests)}")
    print(f"servegen dropped requests (> {MAX_REQ_TOTAL_TOKENS} total tokens): {dropped_too_long}")
    if sampled_requests:
        avg_in = sum(x.prompt_len for x in sampled_requests) / len(sampled_requests)
        avg_out = sum(x.dataset_output_len for x in sampled_requests) / len(sampled_requests)
        p_lens = [x.prompt_len for x in sampled_requests]
        o_lens = [x.dataset_output_len for x in sampled_requests]
        p50_in = np.percentile(p_lens, 50)
        p50_out = np.percentile(o_lens, 50)
        p90_in = np.percentile(p_lens, 90)
        p90_out = np.percentile(o_lens, 90)
        p95_in = np.percentile(p_lens, 95)
        p95_out = np.percentile(o_lens, 95)
        p99_in = np.percentile(p_lens, 99)
        p99_out = np.percentile(o_lens, 99)
        print(f"servegen avg / p50 / p90 / p95 / p99 input tokens: {avg_in:.2f} / {p50_in:.2f} / {p90_in:.2f} / {p95_in:.2f} / {p99_in:.2f}")
        print(f"servegen avg / p50 / p90 / p95 / p99 output tokens: {avg_out:.2f} / {p50_out:.2f} / {p90_out:.2f} / {p95_out:.2f} / {p99_out:.2f}")
    return sampled_requests

def sample_requests_simple(args) -> List[Request]:
    dataset_id = args.dataset_type.split(".", 1)[1]
    if dataset_id not in SIMPLE_DATASETS:
        raise ValueError(
            f"Unknown simple dataset '{args.dataset_type}'. "
            f"Available: {', '.join('simple.' + k for k in SIMPLE_DATASETS)}"
        )

    config = SIMPLE_DATASETS[dataset_id]
    weights = [c["weight"] for c in config]

    sampled_requests: List[Request] = []
    for _ in range(args.num_prompts):
        comp = random.choices(config, weights=weights, k=1)[0]
        input_len = random.randint(*comp["input_range"])
        output_len = random.randint(*comp["output_range"])

        system_prompt = ""
        if args.bypass_cache:
            nonce = uuid.uuid4().hex
            system_prompt = f"{nonce} ( <-- cache_bypass, ignore it )"

        prompts = [
            dict(role="system", content=system_prompt),
            dict(role="user", content=_build_user_content_by_token_budget(input_len)),
        ]
        sampled_requests.append(Request(
            prompts=prompts,
            prompt_len=input_len,
            dataset_output_len=output_len,
            chat_rounds=1,
        ))

    print(f"Generated {len(sampled_requests)} simple requests (dataset_type={args.dataset_type})")
    for idx, c in enumerate(config):
        print(f"  component {idx}: weight={c['weight']}, input=[{c['input_range'][0]},{c['input_range'][1]}], output=[{c['output_range'][0]},{c['output_range'][1]}]")
    p_lens = [r.prompt_len for r in sampled_requests]
    o_lens = [r.dataset_output_len for r in sampled_requests]
    print(f"  input  avg={np.mean(p_lens):.1f}, p50={np.percentile(p_lens,50):.1f}, p90={np.percentile(p_lens,90):.1f}, p99={np.percentile(p_lens,99):.1f}")
    print(f"  output avg={np.mean(o_lens):.1f}, p50={np.percentile(o_lens,50):.1f}, p90={np.percentile(o_lens,90):.1f}, p99={np.percentile(o_lens,99):.1f}")
    return sampled_requests


def sample_requests(
    dataset_path: str,
    num_requests: int,
    max_round: int,
    tokenizer: PreTrainedTokenizerBase,
    args
) -> List[Request]:   # (prompt, prompt_len, output_len, chat_rounds)
    force_long: bool = args.long
    force_long_1500: bool = args.long_1500
    force_long_out_3x: bool = args.long_out_3x

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Find the conversations >= 2 turns and starts from human
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2 and data["conversations"][0]["from"] == "human"
    ]

    # Random select turns.
    dataset_ = []
    for data in dataset:
        conversations = data["conversations"]
        rounds = len(conversations) // 2    # 1 round = 1 ask + 1 answer
        minimal_round = 1
        if force_long:
            minimal_round = rounds    # FIXME
        rounds_used = random.randint(minimal_round, min(max_round, rounds))
        prompt_list: List[Dict[str, str]] = []
        prompt_list.append(dict(
            role = 'system',
            content = SYSTEM_PROMPT,
        ))
        for i in range(rounds_used*2-1):
            prompt_list.append(dict(
                role = 'user' if conversations[i]['from'] == 'human' else 'assistant',
                content = conversations[i]['value'],
            ))
        if args.bypass_cache:
            nonce = uuid.uuid4().hex
            prompt_list[0]["content"] = f"{nonce} ( <-- cache_bypass, ignore it ) {prompt_list[0]['content']}"
        output = conversations[rounds_used*2-1]["value"]
        dataset_.append((prompt_list, output, rounds_used))
    dataset = dataset_

    print("done reading dataset")
    # Tokenize the prompts and completions.
    selnum = num_requests * 3
    if force_long: selnum *= 2
    dataset = random.sample(dataset, selnum)
    prompts = [prompt for prompt, _, _ in dataset]
    prompts_str = [gen_prompt_from_conversation(prompt) for prompt, _, _ in dataset]    # approximate, may not same with the real concat method on server side
    completions = [completion for _, completion, _ in dataset]
    chat_rounds = [chat_rounds for _, _, chat_rounds in dataset]

    prompt_token_ids = tokenizer(prompts_str).input_ids
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset: List[Request] = []
    for i in range(len(dataset)):
        prompt_len = len(prompt_token_ids[i])
        dataset_output_len = len(completion_token_ids[i])
        if force_long_1500:
            dataset_output_len = round(np.random.normal(1500, 200))
        if force_long_out_3x:
            dataset_output_len = dataset_output_len * 3
        tokenized_dataset.append(Request(prompts[i], prompt_len, dataset_output_len, chat_rounds[i]))

    # Filter out too long sequences.
    filtered_dataset: List[Request] = []
    for req in tokenized_dataset:
        prompt, prompt_len, dataset_output_len, chat_rounds = req
        if prompt_len < 4 or dataset_output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len + dataset_output_len > MAX_REQ_TOTAL_TOKENS:
            # Prune too long sequences.
            continue
        if force_long:
            if prompt_len > 2048 or dataset_output_len < 512:
                # Prune too short/long sequences.
                continue
        else:
            if prompt_len > 2048 or prompt_len + dataset_output_len > 4096:
                # Prune too long sequences.
                continue
        filtered_dataset.append(req)

    # Sample the requests.
    print("filtered_dataset", len(filtered_dataset))
    sampled_requests = random.sample(filtered_dataset, num_requests)
    sum_len = 0
    for e in sampled_requests:
        sum_len += e.prompt_len + e.dataset_output_len
    print("requests total tokens (dataset):", sum_len)
    print("requests nums:", len(sampled_requests))
    print("avg chat rounds (N of ask+ans):", sum(x.chat_rounds for x in sampled_requests) / len(sampled_requests))
    return sampled_requests

# request rate
async def get_request(
    input_requests: List[Request],
    request_rate: float,
    follow_request_timestamp: bool = False,
    fixed_interval: bool = False,
) -> AsyncGenerator[Tuple[int, Request], None]:
    input_requests = iter(input_requests)
    t_start = time.time()
    first_request_ts = None
    for i, request in enumerate(input_requests):
        if follow_request_timestamp:
            if first_request_ts is None:
                first_request_ts = request.timestamp
            target_elapsed = request.timestamp - first_request_ts
            now_elapsed = time.time() - t_start
            wait_time = target_elapsed - now_elapsed
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        yield i, request

        if (i+1) % 100 == 0:
            print(f"{i+1} requests sent @ T+{time.time() - t_start:.2f}s")

        if follow_request_timestamp:
            continue

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        if fixed_interval:
            # Fixed (constant) interval between requests.
            interval = 1.0 / request_rate
        else:
            # Sample the request interval from the exponential distribution.
            interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)

# via /generate_stream
async def send_request(
    prompts: List[Dict[str, str]],
    prompt_len: int,
    dataset_output_len: int,
    chat_rounds: int,
    mode: Literal['known_output_len', 'unknown_output_len'],
    i: int,
) -> None:
    global FAILED_REQUESTS
    headers = {'Content-Type': 'application/json', 'Connection': 'keep-alive', "User-Agent": "Benchmark Client"}
    url = f'http://{ADDR}:{PORT}/generate_stream'

    # print("req", prompt_len, output_len)

    if mode == 'known_output_len':
        parameters = dict(
            do_sample = False,
            ignore_eos = True,
            max_new_tokens = dataset_output_len,
        )
    elif mode == 'unknown_output_len':
        parameters = dict(
            do_sample = False,
            # ignore_eos = True,  # FIXME: tmp test
            max_new_tokens = 2048,
        )
    else:
        raise RuntimeError(f"unknown mode: {mode}")

    prompt_str = gen_prompt_from_conversation(prompts)

    req_json = dict(
        inputs = prompt_str,
        parameters = parameters,
    )

    request_start_time = time.time()
    timeout = aiohttp.ClientTimeout(total=24 * 3600, connect=24 * 3600)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                last_time = request_start_time
                async with session.post(url, headers=headers, json=req_json, timeout=timeout) as response:
                    if response.status != 200:
                        raise RuntimeError(f"bad http status: {response.status}")

                    chunks = []
                    latencies = []

                    async for chunk, _ in response.content.iter_chunks():
                        time_now = time.time()
                        chunks.append(chunk)
                        # print(chunk)
                        latencies.append(time_now - last_time)
                        last_time = time_now

                output_len = len(chunks)
                chunks = [json.loads(s.strip()[len('data:'):].strip()) for s in chunks]

                output_str = ''.join(c['token']['text'] for c in chunks)
                # print('_'*10)
                # print("req:", req_json)
                # print("output:", output_str)
                # print("latencies:", ' '.join(f'{x:.5f}' for x in latencies))
                break
    except Exception as e:
        FAILED_REQUESTS += 1
        print(f"request failed: req_id={i}, prompt_len={prompt_len}, dataset_output_len={dataset_output_len}, err={repr(e)}")
        return

    request_end_time = time.time()
    request_latency = request_end_time - request_start_time
    RESULTS.append(Results(
        prompt=prompts,
        prompt_len=prompt_len,
        output=output_str,
        output_len=output_len,
        token_latencys=latencies,
        dataset_output_len=dataset_output_len,
        latency=request_latency, chat_rounds=chat_rounds
    ))

    if len(RESULTS) % 100 == 0:
        time_now = time.time()
        d_time = time_now - _mid_end_times[-1]
        _mid_end_times.append(time_now)
        rate = 100/d_time
        _mid_end_rates.append(rate)
        print(f"{len(RESULTS)} requests completed, current rate {rate:.3f} reqs/s")

# via /v1/chat/completions
# async def send_request(
#     prompts: List[Dict[str, str]],
#     prompt_len: int,
#     dataset_output_len: int,
#     chat_rounds: int,
#     mode: Literal['known_output_len', 'unknown_output_len']
# ) -> None:
#     headers = {'Content-Type': 'application/json', 'Connection': 'keep-alive', "User-Agent": "Benchmark Client"}
#     url = 'http://localhost:8000/v1/chat/completions'

#     # print("req", prompt_len, output_len)

#     if mode == 'known_output_len':
#         parameters = dict(
#             do_sample = False,
#             ignore_eos = True,
#             max_tokens = dataset_output_len,
#         )
#     elif mode == 'unknown_output_len':
#         parameters = dict(
#             do_sample = False,
#             # ignore_eos = True,  # FIXME: tmp test
#             max_tokens = 2048,
#         )
#     else:
#         raise RuntimeError(f"unknown mode: {mode}")

#     req_json = dict(
#         model = '1',
#         stream = True,
#         messages = prompts,
#         **parameters,
#     )

#     request_start_time = time.time()
#     timeout = aiohttp.ClientTimeout(total=3 * 3600)
#     async with aiohttp.ClientSession(timeout=timeout) as session:
#         while True:
#             last_time = request_start_time
#             async with session.post(url, headers=headers, json=req_json, timeout=timeout) as response:
#                 chunks = []
#                 latencies = []
#                 async for chunk, _ in response.content.iter_chunks():
#                     time_now = time.time()
#                     chunks.append(chunk)
#                     print(chunk)
#                     latencies.append(time_now - last_time)
#                     last_time = time_now

#             output_len = len(chunks)
#             chunks = [json.loads(s.strip()[len('data: '):]) for s in chunks]

#             output_str = ''.join(c['choices'][0]['delta']['content'] for c in chunks)
#             print('_'*10)
#             print("req:", req_json)
#             print("output:", output_str)
#             print("latencies:", ' '.join(f'{x:.5f}' for x in latencies))
#             break

#     request_end_time = time.time()
#     request_latency = request_end_time - request_start_time
#     RESULTS.append(Results(
#         prompt=prompts,
#         prompt_len=prompt_len,
#         output=output_str,
#         output_len=output_len,
#         token_latencys=latencies,
#         dataset_output_len=dataset_output_len,
#         latency=request_latency, chat_rounds=chat_rounds
#     ))

#     if len(RESULTS) % 100 == 0:
#         time_now = time.time()
#         d_time = time_now - _mid_times[-1]
#         _mid_times.append(time_now)
#         rate = 100/d_time
#         _mid_rate.append(rate)
#         print(f"{len(RESULTS)} requests completed, current rate {rate:.3f} reqs/s")


async def benchmark(
    input_requests: List[Request],
    request_rate: float,
    mode: Literal['known_output_len', 'unknown_output_len'] = 'unknown_output_len',
    follow_request_timestamp: bool = False,
    fixed_interval: bool = False,
) -> None:
    tasks: List[asyncio.Task] = []
    async for i, request in get_request(input_requests, request_rate, follow_request_timestamp=follow_request_timestamp, fixed_interval=fixed_interval):
        task = asyncio.create_task(
            send_request(
                request.prompts,
                request.prompt_len,
                request.dataset_output_len,
                request.chat_rounds,
                mode,
                i,
            )
        )
        tasks.append(task)
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    global RESULTS, FAILED_REQUESTS
    print(args)
    FAILED_REQUESTS = 0

    os.makedirs(args.dump_dir, exist_ok=True)

    def resolve_dump_path(path: Union[str, None]) -> Union[str, None]:
        if path is None:
            return None
        if os.path.isabs(path):
            return path
        return os.path.join(args.dump_dir, path)

    if not args.use_existing_dump:
        mode = args.mode
        random.seed(args.seed)
        np.random.seed(args.seed)
        if args.dataset_type == 'servegen':
            input_requests = sample_requests_from_servegen(args)
        elif args.dataset_type.startswith('simple.'):
            input_requests = sample_requests_simple(args)
        else:
            tokenizer = get_tokenizer(args.tokenizer, "slow")
            input_requests = sample_requests(args.dataset, args.num_prompts, args.max_round, tokenizer, args)

        if args.use_output_length_record:
            use_output_length_record = resolve_dump_path(args.use_output_length_record)
            print(f"loading output_length from {use_output_length_record}")
            with open(use_output_length_record) as f:
                lens = json.load(f)
            assert len(lens) == len(input_requests)
            for i in range(len(lens)):
                input_requests[i] = input_requests[i]._replace(dataset_output_len=lens[i])

        benchmark_start_time = time.time()
        _mid_end_times.append(benchmark_start_time)
        print(f"\nrunning (mode={mode})...")
        asyncio.run(benchmark(
            input_requests,
            args.request_rate,
            mode=mode,
            follow_request_timestamp=(args.dataset_type == 'servegen'),
            fixed_interval=args.dataset_type.startswith('simple.'),
        ))
        benchmark_end_time = time.time()
        benchmark_time = benchmark_end_time - benchmark_start_time

        # save results
        is_trim = "_trim" if args.trim_bootstrap_and_trailing else ""
        date = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        result_dump_filename = os.path.join(args.dump_dir, f"bench_{date}__n{len(input_requests)}_rate{args.request_rate}{is_trim}.json")
        data = dict(args=vars(args), results=[dataclasses.asdict(r) for r in RESULTS])
        with open(result_dump_filename, 'w') as f:
            json.dump(data, f)
            print("benchmark results saved to", result_dump_filename)

        if args.record_output_length:
            record_output_length = resolve_dump_path(args.record_output_length)
            print(f"writing output length to {record_output_length}")
            lens = [x.output_len for x in RESULTS]
            with open(record_output_length, 'w') as f:
                json.dump(lens, f)
            print(f"writing prompt length to {record_output_length}.prompt.json")
            lens = [x.prompt_len for x in RESULTS]
            with open(record_output_length+'.prompt.json', 'w') as f:
                json.dump(lens, f)

    else:
        # use-existing-dump
        use_existing_dump = resolve_dump_path(args.use_existing_dump)
        with open(use_existing_dump) as f:
            previous_data = json.load(f)
            print(f"use previous dump {use_existing_dump}")
            print(f"args {previous_data['args']}")
            RESULTS = [Results(**x) for x in previous_data['results']]


    if args.trim_bootstrap_and_trailing:
        print("ignore first 200 and last 200 completed requests (--trim-bootstrap-and-trailing)")
        benchmark_time = _mid_end_times[-3] - _mid_end_times[2]
        RESULTS = RESULTS[200:-200]

    # print("calculating output token len ...")
    # real_outputs = [x.output for x in RESULTS]
    # real_outputs_token_ids = tokenizer(real_outputs).input_ids
    # for r, o in zip(RESULTS, real_outputs_token_ids):
    #     r.output_len = len(o)

    prompt_tokens = sum(x.prompt_len for x in RESULTS)
    output_tokens = sum(x.output_len for x in RESULTS)
    dataset_total_tokens = sum(x.prompt_len + x.dataset_output_len for x in RESULTS)
    actual_totol_tokens = sum(x.prompt_len + x.output_len for x in RESULTS)

    print()
    print(f"Number of failed requests: {FAILED_REQUESTS}")
    print(f"Number of successful requests: {len(RESULTS)}")
    print(f"Number of requests for statistic: {len(RESULTS)}")

    if len(RESULTS) == 0:
        print("No successful requests, skip latency/throughput statistics.")
        return

    print()
    print(f"Number of prompt_tokens: {prompt_tokens} (avg {prompt_tokens / len(RESULTS)} tokens/req)")
    print(f"Number of output_tokens: {output_tokens} (avg {output_tokens / len(RESULTS)} tokens/req)")
    print(f"Number of total_tokens: {actual_totol_tokens} (avg {actual_totol_tokens / len(RESULTS)} tokens/req) (for reference: dataset total_tokens {dataset_total_tokens})")

    print()
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {len(RESULTS) / benchmark_time:.2f} requests/s")
    print(f"Overall token throughput: {actual_totol_tokens / benchmark_time:.2f} tokens/s")

    # Compute the latency statistics.
    req_latencies = [x.latency for x in RESULTS]
    avg_latency = np.mean(req_latencies)
    print(f"Average request latency: {avg_latency:.3f} s")
    print(f"    p50: {np.percentile(req_latencies, 50):.3f} s, p90: {np.percentile(req_latencies, 90):.3f} s, p95: {np.percentile(req_latencies, 95):.3f} s, p99: {np.percentile(req_latencies, 99):.3f} s, max: {np.max(req_latencies):.3f} s")

    all_first_token_latency = [x.token_latencys[0] for x in RESULTS]
    avg_first_token_latency = np.mean(all_first_token_latency)
    print(f"Average first token latency: {avg_first_token_latency*1000:.2f} ms")
    print(f"    p50: {np.percentile(all_first_token_latency, 50)*1000:.2f} ms, p90: {np.percentile(all_first_token_latency, 90)*1000:.2f} ms, p95: {np.percentile(all_first_token_latency, 95)*1000:.2f} ms, p99: {np.percentile(all_first_token_latency, 99)*1000:.2f} ms, max: {np.max(all_first_token_latency)*1000:.2f} ms")


    all_per_token_latencies = np.concatenate([x.token_latencys[1:] for x in RESULTS])
    avg_per_token_latency = np.mean(all_per_token_latencies)
    print(f"Average per-token latency (decode): {avg_per_token_latency*1000:.2f} ms", "(old)")
    print(f"    p50: {np.percentile(all_per_token_latencies, 50)*1000:.2f} ms, p90: {np.percentile(all_per_token_latencies, 90)*1000:.2f} ms, p95: {np.percentile(all_per_token_latencies, 95)*1000:.2f} ms, p99: {np.percentile(all_per_token_latencies, 99)*1000:.2f} ms, max: {np.max(all_per_token_latencies)*1000:.2f} ms")

    all_reqmax_per_token_latencies = np.array([max(x.token_latencys[1:]) for x in RESULTS])
    avg_reqmax_per_token_latency = np.mean(all_reqmax_per_token_latencies)
    print(f"Average req-max per-token latency (decode): {avg_reqmax_per_token_latency*1000:.2f} ms")
    print(f"    p50: {np.percentile(all_reqmax_per_token_latencies, 50)*1000:.2f} ms, p75: {np.percentile(all_reqmax_per_token_latencies, 75)*1000:.2f} ms, p90: {np.percentile(all_reqmax_per_token_latencies, 90)*1000:.2f} ms, p95: {np.percentile(all_reqmax_per_token_latencies, 95)*1000:.2f} ms, p99: {np.percentile(all_reqmax_per_token_latencies, 99)*1000:.2f} ms, max: {np.max(all_reqmax_per_token_latencies)*1000:.2f} ms")
    # req_avg_per_token_latencies = [np.mean(x.token_latencys[1:]) for x in RESULTS]
    # avg_req_avg_per_token_latency = np.mean(req_avg_per_token_latencies)
    # print(f"Average of request's average per-token latency (decode): {avg_req_avg_per_token_latency*1000:.2f} ms")
    # print(f"    percentile(avg(X)) p50: {np.percentile(req_avg_per_token_latencies, 50)*1000:.2f} ms, p90: {np.percentile(req_avg_per_token_latencies, 90)*1000:.2f} ms, p95: {np.percentile(req_avg_per_token_latencies, 95)*1000:.2f} ms, p99: {np.percentile(req_avg_per_token_latencies, 99)*1000:.2f} ms")
    # print(f"    avg(percentile(X)) p50: {np.mean([np.percentile(x.token_latencys[1:], 50) for x in RESULTS])*1000:.2f} ms, p90: {np.mean([np.percentile(x.token_latencys[1:], 90) for x in RESULTS])*1000:.2f} ms, p95: {np.mean([np.percentile(x.token_latencys[1:], 95) for x in RESULTS])*1000:.2f} ms, p99: {np.mean([np.percentile(x.token_latencys[1:], 99) for x in RESULTS])*1000:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--dataset-type", type=str, default="sharegpt",
                        help=(
                            "dataset source type: "
                            "sharegpt (need --dataset/--tokenizer), "
                            "servegen (no --dataset needed), "
                            "simple.N (predefined synthetic distributions, no --dataset/--tokenizer needed). "
                            f"Available simple types: {', '.join('simple.' + k for k in SIMPLE_DATASETS)}. "
                            "simple.1=100%% [100,1000]in [50,500]out; "
                            "simple.2=80%% [100,1000]in [50,500]out + 20%% [1000,40000]in [50,500]out. "
                            "simple types use fixed (constant) request intervals instead of Poisson."
                        ))
    parser.add_argument("--servegen-mode", default="m-large", help="ServeGen mode, only work when --dataset-type=servegen, see ServeGen repo for details.")
    parser.add_argument("--addr", type=str, default="127.0.0.1",
                        help="server addr.")
    parser.add_argument("--port", type=str, default="8000",
                        help="server port.")
    parser.add_argument("--dataset", type=str,
                        help="Path to the dataset.")
    parser.add_argument("--tokenizer", type=str,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Request rate (requests per second). If this is inf (or 0), then all requests are sent at time 0. Otherwise, we use Poisson process to synthesize the request arrival times.")
    parser.add_argument("--num-prompts", type=int, default=2000,
                    help="Number of prompts (requests) to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=['known_output_len', 'unknown_output_len'], default='known_output_len')
    parser.add_argument("--dump-dir", type=str, default="./_/", help="directory for benchmark dump files, will be created automatically if not exists")
    parser.add_argument("--bypass-cache", action='store_true', help="add random string to prompt's beginning to bypass cache")
    parser.add_argument("--max-round", type=int, default=99999, help="max chat rounds (1 round = 1 ask + 1 ans) for request")
    parser.add_argument("--record-output-length", help="record request output length to file, for unknown_output_len mode only")
    parser.add_argument("--use-output-length-record", help="use request output length from the record file, for known_output_len mode only, the --num-prompts and --seed must same with the record runs")
    parser.add_argument("--trim-bootstrap-and-trailing", action='store_true', help="ignore first 200 and last 200 completed requests")
    parser.add_argument("--long", action='store_true', help="do not filter long output req in dataset, and add extra prompt to request long answers")
    parser.add_argument("--long-1500", action='store_true', help="set max_new_token to normal(1500, 200), so use with known_output_len mode")
    parser.add_argument("--long-out-3x", action='store_true', help="set max_new_token to 3x of dataset output length, so use with known_output_len mode")
    parser.add_argument("--use-existing-dump", default=None, help="don't run the benchmark, use the existed benckmark dump from previous runs")
    args = parser.parse_args()
    if args.bypass_cache and args.mode == 'unknown_output_len':
        raise UserWarning("use bypass-cache in unknown_output_len mode may break the model's response pattern and cause the output length very different from the dataset output length, which may make the benchmark results less meaningful")
    if args.request_rate == 0:
        print("treat --request-rate 0 as inf")
        args.request_rate = float("inf")
    if args.record_output_length:
        assert args.mode == 'unknown_output_len', "see help"
    if args.use_output_length_record:
        assert args.mode == 'known_output_len', "see help"

    _is_simple = args.dataset_type.startswith('simple.')
    if _is_simple:
        _simple_id = args.dataset_type.split(".", 1)[1]
        if _simple_id not in SIMPLE_DATASETS:
            raise ValueError(
                f"Unknown simple dataset '{args.dataset_type}'. "
                f"Available: {', '.join('simple.' + k for k in SIMPLE_DATASETS)}"
            )

    if args.dataset_type == 'servegen':
        if args.dataset:
            warnings.warn("--dataset is ignored when --dataset-type=servegen")
        if args.tokenizer:
            warnings.warn("--tokenizer is ignored when --dataset-type=servegen")
        if args.num_prompts != 2000:
            warnings.warn("--num-prompts is ignored when --dataset-type=servegen (request count comes from ServeGen workload)")
        if args.max_round != 99999:
            warnings.warn("--max-round is ignored when --dataset-type=servegen")
        if args.long:
            warnings.warn("--long is ignored when --dataset-type=servegen")
        if args.long_1500:
            warnings.warn("--long-1500 is ignored when --dataset-type=servegen")
        if args.long_out_3x:
            warnings.warn("--long-out-3x is ignored when --dataset-type=servegen")
    elif _is_simple:
        if args.dataset:
            warnings.warn(f"--dataset is ignored when --dataset-type={args.dataset_type}")
        if args.tokenizer:
            warnings.warn(f"--tokenizer is ignored when --dataset-type={args.dataset_type}")
        if args.max_round != 99999:
            warnings.warn(f"--max-round is ignored when --dataset-type={args.dataset_type}")
        if args.long:
            warnings.warn(f"--long is ignored when --dataset-type={args.dataset_type}")
        if args.long_1500:
            warnings.warn(f"--long-1500 is ignored when --dataset-type={args.dataset_type}")
        if args.long_out_3x:
            warnings.warn(f"--long-out-3x is ignored when --dataset-type={args.dataset_type}")

    if args.dataset_type not in ('servegen',) and not _is_simple:
        assert args.dataset, "--dataset is required when --dataset-type=sharegpt"
        assert args.tokenizer, "--tokenizer is required when --dataset-type=sharegpt"
    elif args.dataset_type == 'servegen':
        if args.request_rate == float("inf") or args.request_rate <= 0:
            raise ValueError("For --dataset-type servegen, --request-rate must be a finite positive number.")
    if args.dataset_type != 'servegen':
        assert args.num_prompts % 100 == 0, "--num-prompts should be n * 100"
    if args.trim_bootstrap_and_trailing and args.dataset_type != 'servegen':
        assert args.num_prompts >= 500, "bad --num-prompts value for trim, should be >= 500"
    if args.long:
        SYSTEM_PROMPT = SYSTEM_PROMPT + "\n Additionally, this is a very important test which you should answer the question as long as you can, at least say 1000 words for each answer, provide all the details and background knowledges as far as you can."
    if args.long_1500:
        assert args.mode == 'known_output_len', "see help"
    if args.long_out_3x:
        assert args.mode == 'known_output_len', "see help"
    assert not (args.long_1500 and args.long_out_3x), "cannot use both long_1500 and long_out_3x"
    ADDR = args.addr
    PORT = args.port
    main(args)
