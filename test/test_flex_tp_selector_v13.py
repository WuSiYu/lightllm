import time

import pytest

from lightllm.server.httpserver_for_pd_master.pd_selector import create_selector
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v13 import FlexTPSelectorV13
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
        for port, tp, gpu_ids in ((8000, 2, "0,1"), (8001, 2, "2,3"), (8002, 4, "0,1,2,3"))
    ]
    decode = PD_Client_Obj(9000, "test:9000", mode="decode", start_args={"tp": 4})
    return prefill, [decode]


@pytest.mark.asyncio
async def test_v13_factory_and_pressure_spill_snapshot():
    selector = create_selector(
        "flex_tp_v13",
        object(),
        flex_tp_slo_ttft=2.0,
        flex_tp_v13_latency_scale=0.93,
        flex_tp_v13_tp4_pressure_threshold=0.4,
    )
    assert isinstance(selector, FlexTPSelectorV13)
    assert selector.latency_scale == pytest.approx(0.93)
    selector.update_nodes(*_nodes())
    node, _ = await selector.async_select_p_d_node(None, None, None, 8000, time.time(), 1)
    assert node.start_args["tp"] in (2, 4)
    snapshot = selector.scheduler_snapshot()
    assert snapshot["version"] == 13
    assert snapshot["policy"] == "v12_deadline_pressure_spill"
    await selector.notify_request_done(node, req_id=1)
