import asyncio
import inspect
import sys
import time

import pytest

from lightllm.server.httpserver_for_pd_master.pd_selector import create_selector
from lightllm.server.httpserver_for_pd_master.pd_selector.pd_selector import PDSelector
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v7 import (
    FlexTPSelectorV7,
)
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v8 import (
    FlexTPSelectorV8,
)
from lightllm.server.pd_io_struct import PD_Client_Obj


def _nodes():
    prefill = []
    for port, tp, gpu_ids in (
        (8000, 2, "0,1"),
        (8001, 2, "2,3"),
        (8002, 4, "0,1,2,3"),
    ):
        prefill.append(
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
        )
    decode = PD_Client_Obj(
        node_id=9000,
        client_ip_port="test:9000",
        mode="decode",
        start_args={"tp": 4},
        instance_generation="generation-9000",
    )
    return prefill, [decode]


@pytest.mark.asyncio
async def test_v7_hard_route_and_online_flow_keeps_both_lanes_open():
    selector = FlexTPSelectorV7(object(), slo_ttft=3.0, mps_overlap_slowdown=2.0)
    selector.update_nodes(*_nodes())
    long_node, _ = await selector.async_select_p_d_node(None, None, None, 8000, time.time(), 1)
    short_node, _ = await selector.async_select_p_d_node(None, None, None, 256, time.time(), 2)
    assert long_node.start_args["tp"] == 4
    assert short_node.start_args["tp"] == 2
    snapshot = selector.scheduler_snapshot()
    assert snapshot["policy"] == "online_dual_price_flow"
    assert snapshot["admitted_by_class"] == {"long": 1, "short": 1}
    await selector.notify_request_done(long_node, req_id=1)
    await selector.notify_request_done(short_node, req_id=2)


@pytest.mark.asyncio
async def test_v8_epoch_admits_short_and_long_without_serial_mode():
    selector = FlexTPSelectorV8(
        object(),
        slo_ttft=3.0,
        epoch_s=0.02,
        epoch_token_budget=16384,
        min_lane_quota_tokens=2048,
    )
    selector.update_nodes(*_nodes())
    long_task = asyncio.create_task(
        selector.async_select_p_d_node(None, None, None, 8000, time.time(), 1)
    )
    short_task = asyncio.create_task(
        selector.async_select_p_d_node(None, None, None, 256, time.time(), 2)
    )
    long_node, _ = await asyncio.wait_for(long_task, timeout=1)
    short_node, _ = await asyncio.wait_for(short_task, timeout=1)
    assert long_node.start_args["tp"] == 4
    assert short_node.start_args["tp"] == 2
    snapshot = selector.scheduler_snapshot()
    assert snapshot["policy"] == "deterministic_weighted_epoch_wavefront"
    assert snapshot["epoch_count"] >= 1
    assert snapshot["admitted_by_lane"] == {"long": 1, "short": 1}
    await selector.notify_request_done(long_node, req_id=1)
    await selector.notify_request_done(short_node, req_id=2)


@pytest.mark.asyncio
async def test_v7_lifecycle_rejects_duplicate_ids_and_stale_control_messages():
    selector = FlexTPSelectorV7(object(), slo_ttft=3.0, mps_overlap_slowdown=2.0)
    selector.update_nodes(*_nodes())
    p_node, _ = await selector.async_select_p_d_node(None, None, None, 256, time.time(), 41)

    try:
        await selector.async_select_p_d_node(None, None, None, 256, time.time(), 41)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate external req_id was accepted")

    await selector.update_instance_report(
        p_node.client_ip_port,
        {"queued_requests": 7, "queued_tokens": 700},
        report_seq=2,
    )
    await selector.update_instance_report(
        p_node.client_ip_port,
        {"queued_requests": 99, "queued_tokens": 9900},
        report_seq=1,
    )
    assert selector.snapshot()[p_node.client_ip_port]["worker_queued_requests"] == 7

    original_generation = p_node.instance_generation
    p_node.instance_generation = "stale-generation"
    await selector.notify_request_done(p_node, req_id=41)
    assert selector.snapshot()[p_node.client_ip_port]["leases"] == 1
    p_node.instance_generation = original_generation
    await selector.notify_request_done(p_node, req_id=41)


@pytest.mark.asyncio
async def test_v8_generation_aware_removal_releases_only_matching_leases():
    selector = FlexTPSelectorV8(object(), slo_ttft=3.0, epoch_s=0.01)
    selector.update_nodes(*_nodes())
    p_node, _ = await selector.async_select_p_d_node(None, None, None, 8000, time.time(), 51)
    selector.notify_node_removed(p_node.client_ip_port, instance_generation="other-generation")
    await asyncio.sleep(0)
    assert selector.snapshot()[p_node.client_ip_port]["leases"] == 1
    selector.notify_node_removed(p_node.client_ip_port, instance_generation=p_node.instance_generation)
    await asyncio.sleep(0)
    assert selector.snapshot()[p_node.client_ip_port]["leases"] == 0


def test_v7_v8_factories_are_independent_classes():
    v7 = create_selector("flex_tp_v7", object(), flex_tp_slo_ttft=3.0)
    v8 = create_selector("flex_tp_v8", object(), flex_tp_slo_ttft=3.0)
    assert isinstance(v7, FlexTPSelectorV7)
    assert isinstance(v8, FlexTPSelectorV8)
    assert type(v7) is not type(v8)
    assert not isinstance(v7, FlexTPSelectorV8)
    assert not isinstance(v8, FlexTPSelectorV7)


def test_v7_v8_have_no_legacy_selector_dependency():
    legacy_names = (
        "flex_tp_selector_v3",
        "flex_tp_selector_v4",
        "flex_tp_selector_v5",
        "flex_tp_selector_v6",
    )
    for selector_cls in (FlexTPSelectorV7, FlexTPSelectorV8):
        assert selector_cls.__bases__ == (PDSelector,)
        source = inspect.getsource(sys.modules[selector_cls.__module__])
        assert not any(name in source for name in legacy_names)
