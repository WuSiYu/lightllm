import argparse
import random
import uuid
import requests
import json
import time


def main():
    parser = argparse.ArgumentParser(description="Send a long text to LLM for summarization")
    parser.add_argument("-f", "--file", help="Path to the text file to summarize")
    parser.add_argument("-l", "--prompt_length", type=int, default=-1, help="Manually generate prompt of this length (Conflicting with [file])")
    parser.add_argument("-t", "--file-repeat-times", type=int, default=1, help="Number of times to repeat the file content (default: 1)")
    parser.add_argument("--host", default=None, help="Server host (default: hostname -i)")
    parser.add_argument("-p", "--port", type=int, default=60011, help="Server port (default: 60011)")
    parser.add_argument("-d", "--max-new-tokens", type=int, default=200, help="Max new tokens (default: 200)")
    args = parser.parse_args()

    assert args.file or args.prompt_length > 0, "Either --file or --prompt-length must be provided"
    assert not (args.file and args.prompt_length > 0), "Cannot specify both --file and --prompt-length"

    if args.file:
        with open(args.file, "r") as f:
            content = f.read()
        content = content * args.file_repeat_times
    else:
        prompt_length = max(0, args.prompt_length - 55)  # 预留系统提示和总结的长度
        content =  str(random.randint(0, 1000000)) + " token" * prompt_length
    nonce = uuid.uuid4().hex
    system_prompt = f"{nonce} ( <-- cache_bypass, ignore it ), You are a helpful assistant for code or document summarization."

    prompt = f"{system_prompt}\n\n请用50字总结以下内容：\n\n{content}\n\n总结："

    if args.host is None:
        import subprocess
        args.host = subprocess.check_output(["hostname", "-i"]).decode().strip()

    url = f"http://{args.host}:{args.port}/generate"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": args.max_new_tokens,
            "frequency_penalty": 1,
        },
    }

    print(f"Sending request to {url}")
    print(f"Prompt length: {len(prompt)} chars")

    start = time.time()
    resp = requests.post(url, json=payload)
    elapsed = time.time() - start

    resp.raise_for_status()
    result = resp.json()

    generated = result["generated_text"]
    if isinstance(generated, list):
        generated = generated[0]

    print(f"\n--- Result ---")
    print(generated)
    print(f"\n--- Stats ---")
    print(f"Prompt tokens: {result.get('prompt_tokens', 'N/A')}")
    print(f"Output tokens: {result.get('count_output_tokens', 'N/A')}")
    print(f"Finish reason: {result.get('finish_reason', 'N/A')}")
    print(f"Time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
