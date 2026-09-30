"""Stdlib-only smoke test for same-host Decode affinity."""

import asyncio
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightllm.server.httpserver_for_pd_master.pd_selector import create_selector
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v14 import (
    FlexTPSelectorV14,
)
from lightllm.server.pd_io_struct import PD_Client_Obj


def _node(port, host, mode, tp=2, gpu_ids="0,1"):
    return PD_Client_Obj(
        node_id=port,
        client_ip_port=f"{host}:{port}",
        mode=mode,
        start_args={"tp": tp, "tp_smt_group_id": "flex0", "tp_smt_gpu_ids": gpu_ids, "host": host},
        instance_generation=f"generation-{port}",
    )


async def _run() -> None:
    selector = create_selector("flex_tp_v14", object())
    assert isinstance(selector, FlexTPSelectorV14)
    prefill = [_node(8000, "host-a", "prefill")]
    decode = [_node(9000, "host-b", "decode", tp=4), _node(9001, "host-a", "decode", tp=4)]
    selector.update_nodes(prefill, decode)

    p_node, d_node = await selector.async_select_p_d_node(
        None, None, None, 8000, time.time(), 1
    )
    assert p_node.client_ip_port == "host-a:8000"
    assert d_node.client_ip_port == "host-a:9001"
    snapshot = selector.scheduler_snapshot()
    assert snapshot["version"] == 14
    assert snapshot["local_decode_affinity_hits"] == 1
    await selector.notify_request_done(p_node, req_id=1)

    fallback_selector = create_selector("flex_tp_v14", object())
    fallback_prefill = [_node(8010, "host-c", "prefill")]
    fallback_decode = [_node(9010, "host-a", "decode", tp=4)]
    fallback_selector.update_nodes(fallback_prefill, fallback_decode)
    fallback_p_node, fallback_d_node = await fallback_selector.async_select_p_d_node(
        None, None, None, 8000, time.time(), 2
    )
    assert fallback_p_node.client_ip_port == "host-c:8010"
    assert fallback_d_node.client_ip_port == "host-a:9010"
    fallback_snapshot = fallback_selector.scheduler_snapshot()
    assert fallback_snapshot["local_decode_affinity_hits"] == 0
    assert fallback_snapshot["remote_decode_fallbacks"] == 1
    await fallback_selector.notify_request_done(fallback_p_node, req_id=2)


if __name__ == "__main__":
    asyncio.run(_run())
    print("V14_STDLIB_SMOKE_OK")
