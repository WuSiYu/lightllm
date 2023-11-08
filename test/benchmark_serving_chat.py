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
import json
import random
import sys
import time
from typing import AsyncGenerator, Dict, List, Literal, NamedTuple, Tuple, Union

import aiohttp
import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

# from lightllm.server.tokenizer import _FAST_LLAMA_TOKENIZER

from fastchat.model.model_adapter import get_conversation_template
MODEL_TEMPLATE = 'llama-2'
SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

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

def get_tokenizer(
    tokenizer_name: str,
    tokenizer_mode: str = "auto",
    *args,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
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

def sample_requests(
    dataset_path: str,
    num_requests: int,
    max_round: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Request]:   # (prompt, prompt_len, output_len, chat_rounds)
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
        rounds_used = random.randint(1, min(max_round, rounds))
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
        output = conversations[rounds_used*2-1]["value"]
        dataset_.append((prompt_list, output, rounds_used))
    dataset = dataset_

    print("done reading dataset")
    # Tokenize the prompts and completions.
    dataset = random.sample(dataset, num_requests * 3)
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
        tokenized_dataset.append(Request(prompts[i], prompt_len, dataset_output_len, chat_rounds[i]))

    # Filter out too long sequences.
    filtered_dataset: List[Request] = []
    for req in tokenized_dataset:
        prompt, prompt_len, dataset_output_len, chat_rounds = req
        if prompt_len < 4 or dataset_output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 2048 or prompt_len + dataset_output_len > 4096:
            # Prune too long sequences.
            continue
        filtered_dataset.append(req)

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    sum_len = 0
    for e in sampled_requests:
        sum_len += e.prompt_len + e.dataset_output_len
    print("requests total tokens (dataset):", sum_len)
    print("requests nums:", len(sampled_requests))
    print("avg chat rounds (N of ask+ans):", sum(x.chat_rounds for x in sampled_requests) / len(sampled_requests))
    return sampled_requests

async def get_request(
    input_requests: List[Request],
    request_rate: float,
) -> AsyncGenerator[Request, None]:
    input_requests = iter(input_requests)
    for i, request in enumerate(input_requests):

        yield i, request

        if (i+1) % 100 == 0:
            print(f"{i+1} requests sent", time.time())

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
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
    headers = {'Content-Type': 'application/json', 'Connection': 'keep-alive', "User-Agent": "Benchmark Client"}
    url = 'http://localhost:8000/generate_stream'

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
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            last_time = request_start_time
            async with session.post(url, headers=headers, json=req_json, timeout=timeout) as response:
                chunks = []
                latencies = []

                try:
                    async for chunk, _ in response.content.iter_chunks():
                        time_now = time.time()
                        chunks.append(chunk)
                        # print(chunk)
                        latencies.append(time_now - last_time)
                        last_time = time_now
                except aiohttp.client_exceptions.ClientPayloadError as e:
                    tokenizer = get_tokenizer(args.tokenizer, "slow")
                    print("error on req:", i, "chunks:", chunks, "prompt_len:", len(tokenizer([prompt_str]).input_ids[0]), 'e:', e)
                    print("req:", req_json)
                    sys.exit(1)

            output_len = len(chunks)
            chunks = [json.loads(s.strip()[len('data:'):].strip()) for s in chunks]

            output_str = ''.join(c['token']['text'] for c in chunks)
            # print('_'*10)
            # print("req:", req_json)
            # print("output:", output_str)
            # print("latencies:", ' '.join(f'{x:.5f}' for x in latencies))
            break

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
    input_requests: List[Tuple[str, int, int, int]],
    request_rate: float,
    mode: Literal['known_output_len', 'unknown_output_len'] = 'unknown_output_len',
) -> None:
    tasks: List[asyncio.Task] = []
    async for i, request in get_request(input_requests, request_rate):
        prompt, prompt_len, dataset_output_len, chat_rounds = request
        task = asyncio.create_task(send_request(prompt,
                                                prompt_len, dataset_output_len, chat_rounds, mode, i))
        tasks.append(task)
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    global RESULTS
    print(args)
    mode = args.mode
    random.seed(args.seed)
    np.random.seed(args.seed)
    tokenizer = get_tokenizer(args.tokenizer, "slow")
    input_requests = sample_requests(args.dataset, args.num_prompts, args.max_round, tokenizer)

    if args.use_output_length_record:
        print(f"loading output_length from {args.use_output_length_record}")
        with open(args.use_output_length_record) as f:
            lens = json.load(f)
        assert len(lens) == len(input_requests)
        for i in range(len(lens)):
            input_requests[i] = input_requests[i]._replace(output_len=lens[i])

    benchmark_start_time = time.time()
    _mid_end_times.append(benchmark_start_time)
    print(f"\nrunning (mode={mode})...")
    asyncio.run(benchmark(input_requests, args.request_rate, mode=mode))
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time

    if args.trim_bootstrap_and_trailing:
        print("ignore first 200 and last 200 completed requests (--trim-bootstrap-and-trailing)")
        benchmark_time = _mid_end_times[-3] - _mid_end_times[2]
        RESULTS = RESULTS[200:-200]

    print("calculating output token len ...")
    real_outputs = [x.output for x in RESULTS]
    real_outputs_token_ids = tokenizer(real_outputs).input_ids
    for r, o in zip(RESULTS, real_outputs_token_ids):
        r.output_len = len(o)

    prompt_tokens = sum(x.prompt_len for x in RESULTS)
    output_tokens = sum(x.output_len for x in RESULTS)
    dataset_total_tokens = sum(x.prompt_len + x.dataset_output_len for x in RESULTS)
    actual_totol_tokens = sum(x.prompt_len + x.output_len for x in RESULTS)

    print()
    print(f"Number of requests for statistic: {len(RESULTS)}")
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
    print(f"    p50: {np.percentile(req_latencies, 50):.3f} s, p90: {np.percentile(req_latencies, 90):.3f} s, p95: {np.percentile(req_latencies, 95):.3f} s, p99: {np.percentile(req_latencies, 99):.3f} s")

    all_first_token_latency = [x.token_latencys[0] for x in RESULTS]
    avg_first_token_latency = np.mean(all_first_token_latency)
    print(f"Average first token latency: {avg_first_token_latency*1000:.2f} ms")
    print(f"    p50: {np.percentile(all_first_token_latency, 50)*1000:.2f} ms, p90: {np.percentile(all_first_token_latency, 90)*1000:.2f} ms, p95: {np.percentile(all_first_token_latency, 95)*1000:.2f} ms, p99: {np.percentile(all_first_token_latency, 99)*1000:.2f} ms")


    all_per_token_latencies = np.concatenate([x.token_latencys[1:] for x in RESULTS])
    avg_per_token_latency = np.mean(all_per_token_latencies)
    print(f"Average per-token latency (decode): {avg_per_token_latency*1000:.2f} ms")
    print(f"    p50: {np.percentile(all_per_token_latencies, 50)*1000:.2f} ms, p90: {np.percentile(all_per_token_latencies, 90)*1000:.2f} ms, p95: {np.percentile(all_per_token_latencies, 95)*1000:.2f} ms, p99: {np.percentile(all_per_token_latencies, 99)*1000:.2f} ms")


    if args.record_output_length:
        print(f"writing output length to {args.record_output_length}")
        lens = [x.output_len for x in RESULTS]
        with open(args.record_output_length, 'w') as f:
            json.dump(lens, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                             "then all the requests are sent at time 0. "
                             "Otherwise, we use Poisson process to synthesize "
                             "the request arrival times.")
    parser.add_argument("--num-prompts", type=int, default=2000,
                    help="Number of prompts (requests) to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=['known_output_len', 'unknown_output_len'], default='known_output_len')
    parser.add_argument("--max-round", type=int, default=99999, help="max chat rounds (1 round = 1 ask + 1 ans) for request")
    parser.add_argument("--record-output-length", help="record request output length to file, for unknown_output_len mode only")
    parser.add_argument("--use-output-length-record", help="use request output length from the record file, for known_output_len mode only, the --num-prompts and --seed must same with the record runs")
    parser.add_argument("--trim-bootstrap-and-trailing", action='store_true', help="ignore first 200 and last 200 completed requests")
    args = parser.parse_args()
    if args.record_output_length:
        assert args.mode == 'unknown_output_len', "see help"
    if args.use_output_length_record:
        assert args.mode == 'known_output_len', "see help"
    if args.trim_bootstrap_and_trailing:
        assert args.num_prompts >= 500 and args.num_prompts % 100 == 0, "bad --num-prompts value for trim, should be (int_n * 100), which int_n >= 5"
    main(args)
