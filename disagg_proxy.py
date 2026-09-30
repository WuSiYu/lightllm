"""
Disaggregated Prefill/Decode Proxy Server for vLLM (P2pNcclConnector).

Routes requests through a two-phase pipeline:
  1. Prefill phase: forward the request to the prefill instance with max_tokens=1
     (triggers KV cache generation and transfer).
  2. Decode phase: forward the original request to the decode instance
     (performs actual token generation using transferred KV cache).

IMPORTANT — P2pNcclConnector request_id protocol:
  The connector parses request_id to discover peer addresses for NCCL
  point-to-point KV transfer. The proxy MUST rewrite every request_id to:
    ___prefill_addr_{prefill_kv_addr}___decode_addr_{decode_kv_addr}_{uuid}
  where kv_addr = {VLLM_HOST_IP}:{kv_port} of each instance.

Usage:
  python3 disagg_proxy.py \\
      --model meta-llama/Llama-3.1-70B-Instruct \\
      --prefill localhost:8100 \\
      --decode  localhost:8200 \\
      --prefill-kv-addr 172.17.0.6:14579 \\
      --decode-kv-addr  172.17.0.6:14590 \\
      --port 8000
"""

import argparse
import asyncio
import itertools
import json
import logging
import sys
import time
import uuid

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = FastAPI(title="vLLM Disagg Proxy (P2pNccl)")


class DisaggProxy:
    def __init__(
        self,
        prefill_instances: list[str],
        decode_instances: list[str],
        prefill_kv_addrs: list[str],
        decode_kv_addrs: list[str],
        model: str,
    ):
        assert len(prefill_instances) == len(prefill_kv_addrs), \
            "--prefill and --prefill-kv-addr must have the same number of entries"
        assert len(decode_instances) == len(decode_kv_addrs), \
            "--decode and --decode-kv-addr must have the same number of entries"

        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.prefill_kv_addrs = prefill_kv_addrs
        self.decode_kv_addrs = decode_kv_addrs

        # Round-robin cyclers — zip http addr with kv addr
        self.prefill_cycler = itertools.cycle(
            list(zip(prefill_instances, prefill_kv_addrs))
        )
        self.decode_cycler = itertools.cycle(
            list(zip(decode_instances, decode_kv_addrs))
        )
        self.model = model
        self.request_count = 0

    def _make_request_id(self, prefill_kv_addr: str, decode_kv_addr: str) -> str:
        """
        Build a request_id that P2pNcclConnector can parse.
        Format: ___prefill_addr_{ip}:{port}___decode_addr_{ip}:{port}_{uuid}
        """
        return (
            f"___prefill_addr_{prefill_kv_addr}"
            f"___decode_addr_{decode_kv_addr}"
            f"_{uuid.uuid4().hex}"
        )

    async def forward_request(self, url: str, data: dict,
                              extra_headers: dict | None = None):
        """Forward a request and yield response chunks."""
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(
                url=url,
                json=data,
                headers=headers,
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    logger.error("Upstream %s returned %d: %s", url, response.status, error_text)
                    raise HTTPException(status_code=response.status, detail=error_text)
                async for chunk in response.content.iter_chunked(1024):
                    yield chunk

    async def handle_completion(self, request: Request, endpoint: str):
        """Handle /v1/completions or /v1/chat/completions."""
        self.request_count += 1
        req_id = self.request_count
        data = await request.json()

        prefill_http, prefill_kv = next(self.prefill_cycler)
        decode_http, decode_kv = next(self.decode_cycler)

        # Build the P2pNccl-compatible request_id and inject via X-Request-Id header
        disagg_request_id = self._make_request_id(prefill_kv, decode_kv)
        rid_headers = {"X-Request-Id": disagg_request_id}

        # Phase 1: Prefill — set max_tokens=1 to trigger KV generation only
        prefill_data = data.copy()
        prefill_data["max_tokens"] = 1
        if "max_completion_tokens" in prefill_data:
            prefill_data["max_completion_tokens"] = 1

        t0 = time.perf_counter()
        logger.info("[req-%d] Prefill → %s%s  (rid=%s)", req_id, prefill_http, endpoint, disagg_request_id[:60])
        try:
            async for _ in self.forward_request(
                f"http://{prefill_http}{endpoint}", prefill_data, rid_headers
            ):
                pass  # consume prefill response (discard the 1-token output)
        except Exception as e:
            logger.error("[req-%d] Prefill failed: %s", req_id, e)
            raise

        t1 = time.perf_counter()
        logger.info("[req-%d] Prefill done in %.2fs, Decode → %s%s",
                     req_id, t1 - t0, decode_http, endpoint)

        # Phase 2: Decode — send original request with the same disagg request_id
        generator = self.forward_request(
            f"http://{decode_http}{endpoint}", data, rid_headers
        )
        return StreamingResponse(content=generator)


proxy: DisaggProxy | None = None


@app.post("/v1/completions")
async def completions(request: Request):
    return await proxy.handle_completion(request, "/v1/completions")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await proxy.handle_completion(request, "/v1/chat/completions")


@app.get("/v1/models")
async def models():
    """Return model info so benchmarking tools can discover the model."""
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": proxy.model,
            "object": "model",
            "owned_by": "vllm-disagg",
        }],
    })


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/status")
async def status():
    return JSONResponse({
        "prefill_instances": proxy.prefill_instances,
        "decode_instances": proxy.decode_instances,
        "total_requests": proxy.request_count,
    })


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Disaggregated Proxy (P2pNccl)")
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument(
        "--prefill", "-p", type=str, nargs="+", required=True,
        help="Prefill HTTP instance(s), e.g. localhost:8100",
    )
    parser.add_argument(
        "--decode", "-d", type=str, nargs="+", required=True,
        help="Decode HTTP instance(s), e.g. localhost:8200",
    )
    parser.add_argument(
        "--prefill-kv-addr", type=str, nargs="+", required=True,
        help="Prefill KV transfer ZMQ addr(s) = VLLM_HOST_IP:kv_port, e.g. 172.17.0.6:14579",
    )
    parser.add_argument(
        "--decode-kv-addr", type=str, nargs="+", required=True,
        help="Decode KV transfer ZMQ addr(s) = VLLM_HOST_IP:kv_port, e.g. 172.17.0.6:14590",
    )
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    proxy = DisaggProxy(
        prefill_instances=args.prefill,
        decode_instances=args.decode,
        prefill_kv_addrs=args.prefill_kv_addr,
        decode_kv_addrs=args.decode_kv_addr,
        model=args.model,
    )
    logger.info("Starting proxy on port %d", args.port)
    logger.info("  Prefill HTTP: %s  KV: %s", args.prefill, args.prefill_kv_addr)
    logger.info("  Decode  HTTP: %s  KV: %s", args.decode, args.decode_kv_addr)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
