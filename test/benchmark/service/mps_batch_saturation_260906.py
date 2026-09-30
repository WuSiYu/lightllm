#!/usr/bin/env python3
"""Inject a mixed length stream for batch-latency measurements.

The worker-side trace is authoritative; this client only keeps the request
mix and send/completion metadata needed to delimit the saturated prefix.
"""
import argparse, asyncio, json, time, uuid
from pathlib import Path
import aiohttp
from transformers import AutoTokenizer
from mps_real_request_saturation_260906 import make_prompt

SHORT = (32, 64, 128, 256, 1024, 2048)
LONG = (4096, 8192, 16384)

async def request(session, url, prompt, length, stream, rows, timeout_s):
    st=time.perf_counter(); first=None; err=None
    try:
        async with session.post(url, json={"inputs":prompt,"parameters":{"do_sample":False,"ignore_eos":True,"max_new_tokens":1}}, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            if resp.status != 200: raise RuntimeError(f"HTTP {resp.status}")
            async for b,_ in resp.content.iter_chunks():
                if first is None and b: first=time.perf_counter()
    except Exception as e: err=repr(e)
    en=time.perf_counter(); rows.append({"length":length,"stream":stream,"start":st,"first":first,"end":en,"latency_s":en-st,"error":err})

async def run(a):
    tok=AutoTokenizer.from_pretrained(a.model_dir,trust_remote_code=True)
    lengths=SHORT if a.mix == "short" else LONG if a.mix == "long" else SHORT+LONG
    cache={l: make_prompt(tok,l,f"batch_{l}_{uuid.uuid4().hex[:8]}")[0] for l in lengths}
    rows=[]; url=a.url.rstrip('/')+'/generate_stream'; tasks=[]; end=time.perf_counter()+a.duration; i=0
    conn=aiohttp.TCPConnector(limit=0,limit_per_host=0)
    async with aiohttp.ClientSession(connector=conn) as s:
        while time.perf_counter()<end:
            l=lengths[i%len(lengths)]; i+=1
            tasks.append(asyncio.create_task(request(s,url,cache[l],l,a.mix,rows,a.timeout)))
            await asyncio.sleep(1.0/a.rate)
        await asyncio.gather(*tasks,return_exceptions=True)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"mix":a.mix,"rate":a.rate,"duration":a.duration,"sent":i,"rows":rows},indent=2))
    print(json.dumps({"output":str(out),"sent":i,"completed":sum(r['error'] is None for r in rows)}))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--url',required=True); p.add_argument('--model-dir',default='/mtc/wusiyu/models/Llama-3.3-70B-Instruct'); p.add_argument('--mix',choices=['short','long','all'],required=True); p.add_argument('--rate',type=float,required=True); p.add_argument('--duration',type=float,default=2); p.add_argument('--timeout',type=float,default=300); p.add_argument('--output',required=True); asyncio.run(run(p.parse_args()))
if __name__=='__main__': main()
