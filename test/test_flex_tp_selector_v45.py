import asyncio
import time

import pytest

from lightllm.server.httpserver_for_pd_master.pd_selector import create_selector
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v4 import (
    FlexTPSelectorV4,
)
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v5 import (
    FlexTPSelectorV5,
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
                    "shared_weight": "master" if tp == 2 else "slave",
                    "shared_weight_master_port_start": 1300,
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
    return prefill, decode


def _selector(selector_cls, *, slo=30.0):
    selector = selector_cls(
        object(),
        slo_ttft=slo,
        long_request_threshold=4000,
        max_admitted_tokens_per_instance=65536,
        mode_slowdowns={
            (2, (2, 4)): 1.6,
            (4, (2, 4)): 1.6,
            (4, (2, 2, 4)): 1.6,
        },
    )
    prefill, decode = _nodes()
    selector.update_nodes(prefill, [decode])
    return selector


@pytest.mark.asyncio
async def test_v4_long_requests_never_occupy_tp2_when_tp4_exists():
    selector = _selector(FlexTPSelectorV4)
    long_nodes = []
    for req_id in range(1, 5):
        node, _ = await selector.async_select_p_d_node(
            None,
            None,
            None,
            input_token_num=8000,
            arrival_time=time.time(),
            req_id=req_id,
        )
        long_nodes.append(node)

    short_node, _ = await selector.async_select_p_d_node(
        None,
        None,
        None,
        input_token_num=256,
        arrival_time=time.time(),
        req_id=100,
    )

    assert {node.start_args["tp"] for node in long_nodes} == {4}
    assert short_node.start_args["tp"] == 2
    for instance in selector.instances.values():
        if instance.tp_size == 2:
            assert all(lease.seq_len <= 4000 for lease in instance.leases.values())


@pytest.mark.asyncio
async def test_v5_overlaps_short_with_running_long_by_default():
    selector = _selector(FlexTPSelectorV5)
    long_node, _ = await selector.async_select_p_d_node(
        None,
        None,
        None,
        input_token_num=8000,
        arrival_time=time.time(),
        req_id=1,
    )
    short_node, _ = await selector.async_select_p_d_node(
        None,
        None,
        None,
        input_token_num=256,
        arrival_time=time.time(),
        req_id=2,
    )

    assert long_node.start_args["tp"] == 4
    assert short_node.start_args["tp"] == 2
    stats = selector.scheduler_snapshot()
    assert stats["overlap_admissible_evaluations"] > 0


@pytest.mark.asyncio
async def test_v5_temporarily_serializes_when_overlap_misses_new_deadline():
    selector = _selector(FlexTPSelectorV5, slo=0.5)
    short_node, _ = await selector.async_select_p_d_node(
        None,
        None,
        None,
        input_token_num=256,
        arrival_time=time.time(),
        req_id=1,
    )
    long_task = asyncio.create_task(
        selector.async_select_p_d_node(
            None,
            None,
            None,
            input_token_num=4100,
            arrival_time=time.time(),
            req_id=2,
        )
    )
    await asyncio.sleep(0)

    assert not long_task.done()
    assert selector.scheduler_snapshot()["serialization_candidate_evaluations"] > 0

    await selector.notify_request_done(short_node, input_token_num=256, req_id=1)
    long_node, _ = await long_task
    assert long_node.start_args["tp"] == 4


def test_selector_factory_exposes_v4_and_v5():
    kwargs = {"flex_tp_slo_ttft": 3.0, "flex_tp_long_threshold": 4000}
    assert isinstance(create_selector("flex_tp_v4", object(), **kwargs), FlexTPSelectorV4)
    assert isinstance(create_selector("flex_tp_v5", object(), **kwargs), FlexTPSelectorV5)
