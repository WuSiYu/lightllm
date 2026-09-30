import asyncio
import inspect
import sys
import time

import pytest

from lightllm.server.httpserver_for_pd_master.pd_selector import create_selector
from lightllm.server.httpserver_for_pd_master.pd_selector.pd_selector import PDSelector
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v9 import (
    FlexTPSelectorV9,
)
from lightllm.server.pd_io_struct import PD_Client_Obj


def _nodes():
    prefill = [
        PD_Client_Obj(
            node_id=port,
            client_ip_port=f"test:{port}",
            mode="prefill",
            start_args={
                "tp": tp,
                "tp_smt_group_id": "flex0",
                "tp_smt_gpu_ids": gpu_ids,
            },
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
async def test_v9_hard_route_opens_both_tp_lanes_and_tracks_os_state():
    selector = FlexTPSelectorV9(
        object(),
        slo_ttft=3.0,
        mps_overlap_slowdown=2.0,
        aging_interval_s=0.1,
        interactive_token_limit=1024,
    )
    selector.update_nodes(*_nodes())
    long_node, _ = await selector.async_select_p_d_node(None, None, None, 8000, time.time(), 1)
    short_node, _ = await selector.async_select_p_d_node(None, None, None, 256, time.time(), 2)
    assert long_node.start_args["tp"] == 4
    assert short_node.start_args["tp"] == 2
    snapshot = selector.scheduler_snapshot()
    assert snapshot["policy"] == "os_mlfq_aging_cfs"
    assert snapshot["admitted_by_class"] == {"long": 1, "short": 1}
    assert snapshot["admitted_by_level"]
    assert set(snapshot["class_vruntime"]) == {"short", "long"}
    await selector.notify_request_done(long_node, req_id=1)
    await selector.notify_request_done(short_node, req_id=2)


def test_v9_feedback_levels_and_aging_are_deterministic():
    selector = FlexTPSelectorV9(object(), interactive_token_limit=1024, long_request_threshold=4000)
    assert selector._initial_level(256) == 0
    assert selector._initial_level(2048) == 1
    assert selector._initial_level(8000) == 2
    now = 10.0
    short = selector.pending_type(1, 1, 8000, 100.0, None, 1, 0.0, 2, 2)
    selector._pending[1] = short
    selector._promote_by_aging(now)
    assert short.level == 0
    assert selector.aging_promotions == 2


@pytest.mark.asyncio
async def test_v9_lifecycle_rejects_duplicate_ids_and_stale_generation_messages():
    selector = FlexTPSelectorV9(object(), slo_ttft=3.0)
    selector.update_nodes(*_nodes())
    node, _ = await selector.async_select_p_d_node(None, None, None, 256, time.time(), 41)
    with pytest.raises(ValueError):
        await selector.async_select_p_d_node(None, None, None, 256, time.time(), 41)
    await selector.update_instance_report(
        node.client_ip_port,
        {"queued_requests": 7, "queued_tokens": 700},
        report_seq=2,
    )
    await selector.update_instance_report(
        node.client_ip_port,
        {"queued_requests": 99, "queued_tokens": 9900},
        report_seq=1,
    )
    assert selector.snapshot()[node.client_ip_port]["worker_queued_requests"] == 7
    generation = node.instance_generation
    node.instance_generation = "stale-generation"
    await selector.notify_request_done(node, req_id=41)
    assert selector.snapshot()[node.client_ip_port]["leases"] == 1
    node.instance_generation = generation
    await selector.notify_request_done(node, req_id=41)


def test_v9_factory_and_module_are_independent():
    selector = create_selector("flex_tp_v9", object(), flex_tp_slo_ttft=3.0)
    assert isinstance(selector, FlexTPSelectorV9)
    assert selector.__class__.__bases__ == (PDSelector,)
    source = inspect.getsource(sys.modules[FlexTPSelectorV9.__module__])
    assert not any(f"flex_tp_selector_v{version}" in source for version in ("3", "4", "5", "6", "7", "8"))
