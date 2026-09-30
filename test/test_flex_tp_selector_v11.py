import time

import pytest

from lightllm.server.httpserver_for_pd_master.pd_selector import create_selector
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v11 import (
    FlexTPSelectorV11,
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
async def test_v11_dynamic_route_has_no_length_class_boundary():
    selector = FlexTPSelectorV11(
        object(), slo_ttft=2.0, mps_overlap_slowdown=2.0, routing_cost_weight=0.5
    )
    selector.update_nodes(*_nodes())
    assert selector._class(1) == selector._class(4000) == selector._class(100000)
    assert selector._initial_level(1) == selector._initial_level(100000) == 0

    short_node, _ = await selector.async_select_p_d_node(None, None, None, 256, time.time(), 1)
    await selector.notify_request_done(short_node, req_id=1)
    long_node, _ = await selector.async_select_p_d_node(None, None, None, 8000, time.time(), 2)
    assert short_node.start_args["tp"] == 2
    assert long_node.start_args["tp"] == 4

    snapshot = selector.scheduler_snapshot()
    assert snapshot["version"] == 11
    assert snapshot["policy"] == "os_eevdf_elastic_tp"
    assert snapshot["routing_threshold"] is None
    assert snapshot["dynamic_route_evaluations"] >= 2
    assert snapshot["tp_selection_counts"][2] >= 1
    assert snapshot["tp_selection_counts"][4] >= 1
    await selector.notify_request_done(long_node, req_id=2)


def test_v11_factory_uses_elastic_selector():
    selector = create_selector(
        "flex_tp_v11",
        object(),
        flex_tp_slo_ttft=1.0,
        flex_tp_v11_routing_cost_weight=0.7,
    )
    assert type(selector) is FlexTPSelectorV11
    assert selector.routing_cost_weight == 0.7
