import time

import pytest

from lightllm.server.httpserver_for_pd_master.pd_selector import create_selector
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v10 import (
    FlexTPSelectorV10,
)
from lightllm.server.pd_io_struct import PD_Client_Obj


def _nodes():
    prefill = [
        PD_Client_Obj(
            node_id=port,
            client_ip_port=f"test:{port}",
            mode="prefill",
            start_args={"tp": tp, "tp_smt_group_id": "flex0", "tp_smt_gpu_ids": gpu_ids},
            instance_generation=f"generation-{port}",
        )
        for port, tp, gpu_ids in (
            (8000, 2, "0,1"),
            (8001, 2, "2,3"),
            (8002, 4, "0,1,2,3"),
        )
    ]
    decode = PD_Client_Obj(
        node_id=9000,
        client_ip_port="test:9000",
        mode="decode",
        start_args={"tp": 4},
        instance_generation="generation-9000",
    )
    return prefill, [decode]


@pytest.mark.asyncio
async def test_v10_hard_route_and_eevdf_state():
    selector = FlexTPSelectorV10(
        object(), slo_ttft=2.0, mps_overlap_slowdown=2.0, overlap_slack_ratio=0.1
    )
    selector.update_nodes(*_nodes())
    long_node, _ = await selector.async_select_p_d_node(None, None, None, 8000, time.time(), 1)
    short_node, _ = await selector.async_select_p_d_node(None, None, None, 256, time.time(), 2)
    assert long_node.start_args["tp"] == 4
    assert short_node.start_args["tp"] == 2
    snapshot = selector.scheduler_snapshot()
    assert snapshot["version"] == 10
    assert snapshot["policy"] == "os_eevdf_deadline_guard"
    assert snapshot["eevdf_admissions"] == 2
    assert snapshot["virtual_time"]["short"] > 0
    assert snapshot["virtual_time"]["long"] > 0
    await selector.notify_request_done(long_node, req_id=1)
    await selector.notify_request_done(short_node, req_id=2)


def test_v10_factory_uses_v10_class():
    selector = create_selector(
        "flex_tp_v10",
        object(),
        flex_tp_slo_ttft=1.0,
        flex_tp_v10_short_weight=1.5,
        flex_tp_v10_long_weight=0.8,
        flex_tp_v10_overlap_slack_ratio=0.2,
    )
    assert type(selector) is FlexTPSelectorV10
    assert selector.short_weight == 1.5
    assert selector.long_weight == 0.8
    assert selector.overlap_slack_ratio == 0.2

