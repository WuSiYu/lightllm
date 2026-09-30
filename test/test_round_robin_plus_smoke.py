"""Stdlib-only smoke test for same-host Decode affinity."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightllm.server.httpserver_for_pd_master.pd_selector import (
    RoundRobinPlusSelector,
    create_selector,
)
from lightllm.server.pd_io_struct import PD_Client_Obj


def _node(port, host, mode):
    return PD_Client_Obj(
        node_id=port,
        client_ip_port=f"{host}:{port}",
        mode=mode,
        start_args={"tp": 2, "host": host},
    )


def _run() -> None:
    selector = create_selector("round_robin_plus", object())
    assert isinstance(selector, RoundRobinPlusSelector)
    selector.update_nodes(
        [_node(8000, "host-a", "prefill"), _node(8001, "host-b", "prefill")],
        [_node(9000, "host-b", "decode"), _node(9001, "host-a", "decode")],
    )

    first_p, first_d = selector.select_p_d_node(None, None, None)
    second_p, second_d = selector.select_p_d_node(None, None, None)
    assert first_p.client_ip_port == "host-a:8000"
    assert first_d.client_ip_port == "host-a:9001"
    assert second_p.client_ip_port == "host-b:8001"
    assert second_d.client_ip_port == "host-b:9000"


if __name__ == "__main__":
    _run()
    print("ROUND_ROBIN_PLUS_STDLIB_SMOKE_OK")
