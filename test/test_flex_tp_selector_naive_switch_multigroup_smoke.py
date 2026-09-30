"""Stdlib-only smoke test for independent naive-switch groups."""

import asyncio
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightllm.server.httpserver_for_pd_master.pd_selector import (
    RoundRobinSelector,
    create_selector,
)
from lightllm.server.pd_io_struct import PD_Client_Obj


def _node(port, host, mode, tp, group_port):
    return PD_Client_Obj(
        node_id=port,
        client_ip_port=f"{host}:{port}",
        mode=mode,
        start_args={
            "tp": tp,
            "host": host,
            "shared_weight": True,
            "shared_weight_master_port_start": group_port,
            "tp_smt_group_id": f"group-{group_port}",
            "tp_smt_gpu_ids": "0,1" if tp == 2 else "0,1,2,3",
        },
        instance_generation=f"generation-{port}",
    )


async def _run() -> None:
    selector = create_selector("flex_tp_naive_switch", object(), flex_tp_threshold=4000)
    prefill = [
        _node(8000, "host-a", "prefill", 2, 1300),
        _node(8001, "host-a", "prefill", 4, 1300),
        _node(8100, "host-b", "prefill", 2, 1400),
        _node(8101, "host-b", "prefill", 4, 1400),
    ]
    decode = [
        _node(9000, "host-a", "decode", 4, 1500),
        _node(9001, "host-b", "decode", 4, 1600),
    ]
    selector.update_nodes(prefill, decode)

    short_p, short_d = await selector.async_select_p_d_node(
        None, None, None, 1000, time.time(), 1
    )
    assert short_p.client_ip_port == "host-a:8000"
    assert short_d.client_ip_port == "host-a:9000"

    long_p, long_d = await asyncio.wait_for(
        selector.async_select_p_d_node(None, None, None, 8000, time.time(), 2),
        timeout=0.2,
    )
    assert long_p.client_ip_port == "host-b:8101"
    assert long_d.client_ip_port == "host-b:9001"

    await selector.notify_request_done(short_p, req_id=1)
    await selector.notify_request_done(long_p, req_id=2)

    default_selector = RoundRobinSelector(object())
    default_selector.update_nodes(
        [_node(8200, "host-a", "prefill", 2, 1700)],
        [_node(9200, "host-b", "decode", 4, 1800)],
    )
    default_p, default_d = default_selector.select_p_d_node(None, None, None)
    assert default_p.client_ip_port == "host-a:8200"
    assert default_d.client_ip_port == "host-b:9200"


if __name__ == "__main__":
    asyncio.run(_run())
    print("NAIVE_SWITCH_MULTIGROUP_STDLIB_SMOKE_OK")
