import asyncio
import time

import pytest

from lightllm.server.httpserver_for_pd_master.pd_selector import create_selector
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v6 import (
    FlexTPSelectorV6,
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


def _selector(*, slo=3.0, slowdown=2.0):
    selector = FlexTPSelectorV6(
        object(),
        slo_ttft=slo,
        long_request_threshold=4000,
        mps_overlap_slowdown=slowdown,
    )
    selector.update_nodes(*_nodes())
    return selector


@pytest.mark.asyncio
async def test_v6_threshold_routing_and_all_mode_default():
    selector = _selector(slowdown=1.6)
    long_node, _ = await selector.async_select_p_d_node(
        None, None, None, 8000, time.time(), 1
    )
    short_node, _ = await selector.async_select_p_d_node(
        None, None, None, 256, time.time(), 2
    )

    assert long_node.start_args["tp"] == 4
    assert short_node.start_args["tp"] == 2
    assert selector.scheduler_snapshot()["admitted_by_mode"]["all"] > 0


@pytest.mark.asyncio
async def test_v6_serializes_only_when_counterfactual_rescues_work():
    selector = _selector(slo=0.9, slowdown=2.0)
    long_node, _ = await selector.async_select_p_d_node(
        None, None, None, 8000, time.time(), 1
    )
    short_task = asyncio.create_task(
        selector.async_select_p_d_node(None, None, None, 256, time.time(), 2)
    )
    await asyncio.sleep(0)

    assert not short_task.done()
    assert selector.scheduler_snapshot()["long_only_decisions"] > 0

    await selector.notify_request_done(long_node, input_token_num=8000, req_id=1)
    short_node, _ = await asyncio.wait_for(short_task, timeout=1)
    assert short_node.start_args["tp"] == 2


@pytest.mark.asyncio
async def test_v6_ordinary_pd_lifecycle_and_generation_cleanup():
    prefill, decode = _nodes()
    nixl_prefill = PD_Client_Obj(
        node_id=8100,
        client_ip_port="test:8100",
        mode="nixl_prefill",
        start_args={"tp": 2, "tp_smt_group_id": "flex0", "tp_smt_gpu_ids": "0,1"},
        instance_generation="nixl-prefill-generation",
    )
    nixl_decode = PD_Client_Obj(
        node_id=9100,
        client_ip_port="test:9100",
        mode="nixl_decode",
        start_args={"tp": 4},
        instance_generation="nixl-decode-generation",
    )
    selector = FlexTPSelectorV6(object(), slo_ttft=3.0)
    selector.update_nodes(prefill + [nixl_prefill], decode + [nixl_decode])

    assert "test:8100" not in selector.instances
    assert [node.mode for node in selector.decode_nodes] == ["decode"]

    long_node, _ = await selector.async_select_p_d_node(
        None, None, None, 5000, time.time(), 77
    )
    with pytest.raises(ValueError, match="already admitted or pending"):
        await selector.async_select_p_d_node(None, None, None, 5000, time.time(), 77)

    await selector.notify_bundle_accepted(long_node, 9, [77])
    await selector.update_instance_report(
        long_node.client_ip_port,
        {"queued_requests": 1},
        report_seq=2,
    )
    await selector.update_instance_report(
        long_node.client_ip_port,
        {"queued_requests": 99},
        report_seq=1,
    )
    assert selector.snapshot()[long_node.client_ip_port]["accepted_bundle_count"] == 1
    assert selector.snapshot()[long_node.client_ip_port]["worker_queued_requests"] == 1

    selector.notify_node_removed(
        long_node.client_ip_port,
        instance_generation=long_node.instance_generation,
    )
    await asyncio.sleep(0)
    assert selector.snapshot()[long_node.client_ip_port]["leases"] == 0


def test_selector_factory_exposes_v6():
    selector = create_selector(
        "flex_tp_v6",
        object(),
        flex_tp_slo_ttft=3.0,
        flex_tp_long_threshold=4000,
        flex_tp_mps_slowdown=1.6,
    )
    assert isinstance(selector, FlexTPSelectorV6)
    assert selector.mps_overlap_slowdown == 1.6
