"""
Tests for FlexTPSelector drain state machine, focusing on cancellation safety.

Verifies that pending_large_tp / pending_small_tp counters don't leak when
requests are cancelled (e.g. client disconnect), which would otherwise deadlock
the state machine and starve small TP requests.
"""

import asyncio
import time
import pytest
from dataclasses import dataclass, field
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Minimal stubs so we can import FlexTPSelector without the full LightLLM stack
# ---------------------------------------------------------------------------


@dataclass
class _PD_Client_RunStatus:
    total_token_usage_rate: float = 0.0


@dataclass
class PD_Client_Obj:
    node_id: int = 0
    client_ip_port: str = ""
    mode: str = "prefill"
    start_args: dict = field(default_factory=dict)
    websocket: object = None
    run_status: _PD_Client_RunStatus = field(default_factory=_PD_Client_RunStatus)

    def __post_init__(self):
        pass

    def to_llm_url(self):
        return f"http://{self.client_ip_port}/pd_generate_stream"


# Patch the import target before importing the selector module
import sys, types

pd_io_mod = types.ModuleType("lightllm.server.pd_io_struct")
pd_io_mod.PD_Client_Obj = PD_Client_Obj

core_objs_mod = types.ModuleType("lightllm.server.core.objs")
core_objs_mod.SamplingParams = MagicMock

mm_mod = types.ModuleType("lightllm.server.multimodal_params")
mm_mod.MultimodalParams = MagicMock

log_mod = types.ModuleType("lightllm.utils.log_utils")
log_mod.init_logger = lambda name: __import__("logging").getLogger(name)

# Need parent packages too (each must have __path__ to be treated as a package)
import os as _os
_repo = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_pkg_dirs = {
    "lightllm": _os.path.join(_repo, "lightllm"),
    "lightllm.server": _os.path.join(_repo, "lightllm", "server"),
    "lightllm.server.core": _os.path.join(_repo, "lightllm", "server", "core"),
    "lightllm.server.httpserver_for_pd_master": _os.path.join(_repo, "lightllm", "server", "httpserver_for_pd_master"),
    "lightllm.server.httpserver_for_pd_master.pd_selector": _os.path.join(
        _repo, "lightllm", "server", "httpserver_for_pd_master", "pd_selector"
    ),
    "lightllm.utils": _os.path.join(_repo, "lightllm", "utils"),
}
for pkg, pkg_dir in _pkg_dirs.items():
    if pkg not in sys.modules:
        mod = types.ModuleType(pkg)
        mod.__path__ = [pkg_dir]
        sys.modules[pkg] = mod

sys.modules["lightllm.server.pd_io_struct"] = pd_io_mod
sys.modules["lightllm.server.core.objs"] = core_objs_mod
sys.modules["lightllm.server.multimodal_params"] = mm_mod
sys.modules["lightllm.utils.log_utils"] = log_mod


# Stub PDSelector base
class PDSelector:
    def __init__(self, pd_manager):
        self.prefill_nodes = []
        self.decode_nodes = []
        self.pd_manager = pd_manager

    def update_nodes(self, prefill_nodes, decode_nodes):
        self.prefill_nodes = prefill_nodes
        self.decode_nodes = decode_nodes


pd_selector_mod = types.ModuleType("lightllm.server.httpserver_for_pd_master.pd_selector.pd_selector")
pd_selector_mod.PDSelector = PDSelector
sys.modules["lightllm.server.httpserver_for_pd_master.pd_selector.pd_selector"] = pd_selector_mod

# Now import the real module
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector import (
    FlexTPSelector,
    FlexTPGroup,
    LatencyPredictor,
    SLOTracker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(host: str, port: int, tp: int, sw_port: int) -> PD_Client_Obj:
    return PD_Client_Obj(
        node_id=port,
        client_ip_port=f"{host}:{port}",
        mode="prefill",
        start_args={
            "shared_weight": True,
            "shared_weight_master_port_start": sw_port,
            "tp": tp,
            "host": host,
            "pd_node_id": port,
            "pd_decode_rpyc_port": port + 1000,
            "nnodes": 1,
        },
    )


def _make_decode_node() -> PD_Client_Obj:
    return PD_Client_Obj(
        node_id=99,
        client_ip_port="10.0.0.99:8000",
        mode="decode",
        start_args={
            "tp": 1,
            "host": "10.0.0.99",
            "pd_node_id": 99,
            "pd_decode_rpyc_port": 9000,
            "nnodes": 1,
        },
    )


def _build_selector(length_threshold: int = 100, slo_ttft: float = None) -> FlexTPSelector:
    """Build a FlexTPSelector with 1 group: 2 small-TP (tp=2) + 1 large-TP (tp=4)."""
    sel = FlexTPSelector(pd_manager=MagicMock(), length_threshold=length_threshold,
                         slo_ttft=slo_ttft)
    small1 = _make_node("10.0.0.1", 8001, tp=2, sw_port=5000)
    small2 = _make_node("10.0.0.1", 8002, tp=2, sw_port=5000)
    large1 = _make_node("10.0.0.1", 8003, tp=4, sw_port=5000)
    decode = _make_decode_node()
    sel.update_nodes([small1, small2, large1], [decode])
    return sel


def _get_group(sel: FlexTPSelector) -> FlexTPGroup:
    return list(sel.flex_groups.values())[0]


# ---------------------------------------------------------------------------
# Tests: Drain state machine & cancellation safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_large_tp_cancel_during_drain_restores_state():
    """
    Scenario: A large TP request triggers draining, but gets cancelled while
    waiting for small TP inflight to reach zero.
    Expected: pending_large_tp goes back to 0, state returns to small_tp_active.
    """
    sel = _build_selector()
    group = _get_group(sel)

    # Simulate an in-flight small TP request so drain must wait
    small_node = group.small_tp_nodes[0]
    group.add_inflight(small_node.client_ip_port, 50)

    async def cancel_after_yield():
        task = asyncio.current_task()
        asyncio.get_event_loop().call_soon(task.cancel)
        return await sel._dispatch_large_tp(group, group.large_tp_nodes[0], 10000)

    with pytest.raises(asyncio.CancelledError):
        await cancel_after_yield()

    assert group.pending_large_tp == 0, f"pending_large_tp leaked: {group.pending_large_tp}"
    assert group.inflight_large_tp == 0, f"inflight_large_tp leaked: {group.inflight_large_tp}"
    assert group.state == "small_tp_active", f"state stuck: {group.state}"


@pytest.mark.asyncio
async def test_large_tp_cancel_with_another_pending_large():
    """
    Scenario: Two large TP requests pending. One is cancelled.
    Expected: pending_large_tp decrements but state stays draining (other request still waiting).
    """
    sel = _build_selector()
    group = _get_group(sel)

    small_node = group.small_tp_nodes[0]
    group.add_inflight(small_node.client_ip_port, 50)  # block drain

    results = []

    async def large_req(req_id: int):
        try:
            node = await sel._dispatch_large_tp(group, group.large_tp_nodes[0], 10000)
            results.append((req_id, node))
        except asyncio.CancelledError:
            results.append((req_id, "cancelled"))
            raise

    task1 = asyncio.create_task(large_req(1))
    task2 = asyncio.create_task(large_req(2))
    await asyncio.sleep(0)  # let both enter drain wait

    assert group.pending_large_tp == 2

    # Cancel one
    task1.cancel()
    try:
        await task1
    except asyncio.CancelledError:
        pass

    assert group.pending_large_tp == 1, f"pending_large_tp should be 1, got {group.pending_large_tp}"
    assert group.state in ("draining", "large_tp_active")

    # Now complete the drain by removing inflight small TP
    async with group.condition:
        group.remove_inflight(small_node.client_ip_port, 50)
        group.condition.notify_all()

    await task2  # should complete successfully
    assert group.pending_large_tp == 0
    assert group.inflight_large_tp == 1

    # Simulate the large TP request completing
    await sel.notify_request_done(group.large_tp_nodes[0], input_token_num=10000)
    assert group.state == "small_tp_active"
    assert group.inflight_large_tp == 0


@pytest.mark.asyncio
async def test_small_tp_cancel_while_blocked_cleans_pending():
    """
    Scenario: Small TP request blocked waiting for group to become small_tp_active,
    then gets cancelled.
    Expected: pending_small_tp goes back to 0.
    """
    sel = _build_selector()
    group = _get_group(sel)

    # Force state so small TP will block
    group.state = "large_tp_active"
    large_node = group.large_tp_nodes[0]
    group.add_inflight(large_node.client_ip_port, 10000)

    async def cancel_small():
        task = asyncio.current_task()
        asyncio.get_event_loop().call_soon(task.cancel)
        return await sel._dispatch_small_tp(group, group.small_tp_nodes[0], 50,
                                            arrival_time=999999999.0)

    with pytest.raises(asyncio.CancelledError):
        await cancel_small()

    assert group.pending_small_tp == 0, f"pending_small_tp leaked: {group.pending_small_tp}"


@pytest.mark.asyncio
async def test_normal_large_tp_lifecycle():
    """Regression: normal large TP request lifecycle still works correctly."""
    sel = _build_selector()
    group = _get_group(sel)

    assert group.state == "small_tp_active"

    # No inflight small, so large TP should dispatch immediately
    node = await sel._dispatch_large_tp(group, group.large_tp_nodes[0], 10000)
    assert node is not None
    assert group.state == "large_tp_active"
    assert group.inflight_large_tp == 1
    assert group.pending_large_tp == 0

    # Complete the large TP request
    await sel.notify_request_done(node, input_token_num=10000)
    assert group.state == "small_tp_active"
    assert group.inflight_large_tp == 0


@pytest.mark.asyncio
async def test_normal_small_tp_lifecycle():
    """Regression: normal small TP request lifecycle still works correctly."""
    sel = _build_selector()
    group = _get_group(sel)

    node = await sel._dispatch_small_tp(group, group.small_tp_nodes[0], 50)
    assert node is not None
    assert group.inflight_small_tp == 1

    await sel.notify_request_done(node, input_token_num=50)
    assert group.inflight_small_tp == 0


@pytest.mark.asyncio
async def test_drain_then_resume_full_cycle():
    """Full cycle: small active → drain → large active → back to small."""
    sel = _build_selector()
    group = _get_group(sel)

    # 1. Dispatch a small TP request
    small_node = await sel._dispatch_small_tp(group, group.small_tp_nodes[0], 50)
    assert group.inflight_small_tp == 1
    assert group.state == "small_tp_active"

    # 2. Large TP request arrives — triggers drain
    large_task = asyncio.create_task(
        sel._dispatch_large_tp(group, group.large_tp_nodes[0], 10000)
    )
    await asyncio.sleep(0)  # let it enter drain wait
    assert group.state == "draining"
    assert group.pending_large_tp == 1

    # 3. Complete the small TP request — unblocks drain
    await sel.notify_request_done(small_node, input_token_num=50)
    large_node = await large_task
    assert group.state == "large_tp_active"
    assert group.inflight_large_tp == 1
    assert group.pending_large_tp == 0

    # 4. Complete large TP — back to small
    await sel.notify_request_done(large_node, input_token_num=10000)
    assert group.state == "small_tp_active"
    assert group.inflight_large_tp == 0


@pytest.mark.asyncio
async def test_drain_allows_early_arrival_small_tp():
    """
    Scenario: A long request triggers drain, but a short request that arrived
    earlier (lower timestamp) should still be dispatched during draining.
    """
    sel = _build_selector()
    group = _get_group(sel)

    # 1. Dispatch a small TP request to create inflight
    small_node = await sel._dispatch_small_tp(group, group.small_tp_nodes[0], 50,
                                              arrival_time=1.0)
    assert group.inflight_small_tp == 1

    # 2. Large TP request arrives at T=3.0 — triggers drain
    large_task = asyncio.create_task(
        sel._dispatch_large_tp(group, group.large_tp_nodes[0], 10000, arrival_time=3.0)
    )
    await asyncio.sleep(0)
    assert group.state == "draining"
    assert group.drain_start_time == 3.0

    # 3. A short request that arrived at T=2.0 (before the long request) should be allowed
    early_node = await sel._dispatch_small_tp(group, group.small_tp_nodes[1], 60,
                                              arrival_time=2.0)
    assert early_node is not None
    assert group.inflight_small_tp == 2

    # 4. A short request that arrived at T=4.0 (after drain) should be blocked
    late_task = asyncio.create_task(
        sel._dispatch_small_tp(group, group.small_tp_nodes[0], 70, arrival_time=4.0)
    )
    await asyncio.sleep(0)
    assert group.pending_small_tp == 1  # blocked

    # 5. Complete both small TP requests — drain proceeds
    await sel.notify_request_done(small_node, input_token_num=50)
    await sel.notify_request_done(early_node, input_token_num=60)

    # Large TP should now proceed
    large_node = await large_task
    assert group.state == "large_tp_active"

    # 6. Complete large TP — small_tp_active restored, late request unblocked
    await sel.notify_request_done(large_node, input_token_num=10000)
    assert group.state == "small_tp_active"
    assert group.drain_start_time is None

    late_node = await late_task
    assert late_node is not None
    assert group.pending_small_tp == 0


# ---------------------------------------------------------------------------
# LatencyPredictor & SLOTracker Tests
# ---------------------------------------------------------------------------


def test_latency_predictor_default():
    """LatencyPredictor uses default constants correctly: C[tp]*seq_len + overhead."""
    pred = LatencyPredictor()
    overhead = 0.1  # predict_execution_time 中的固定 overhead
    # TP4, 1000 tokens -> 0.00012 * 1000 + 0.1 = 0.22s
    t = pred.predict_execution_time(tp_size=4, seq_len=1000)
    assert abs(t - (0.00012 * 1000 + overhead)) < 1e-9

    # TP2, 5000 tokens -> 0.00018 * 5000 + 0.1 = 1.0s
    t = pred.predict_execution_time(tp_size=2, seq_len=5000)
    assert abs(t - (0.00018 * 5000 + overhead)) < 1e-9


def test_latency_predictor_custom_constants():
    """LatencyPredictor accepts custom constants."""
    pred = LatencyPredictor(constants={2: 0.0001})
    overhead = 0.1
    t = pred.predict_execution_time(tp_size=2, seq_len=10000)
    assert abs(t - (0.0001 * 10000 + overhead)) < 1e-9
    # Default for TP4 still works
    t4 = pred.predict_execution_time(tp_size=4, seq_len=1000)
    assert abs(t4 - (0.00012 * 1000 + overhead)) < 1e-9


def test_latency_predictor_unknown_tp_extrapolates():
    """LatencyPredictor extrapolates for unknown TP sizes."""
    pred = LatencyPredictor()
    overhead = 0.1
    # TP16 not in defaults, should extrapolate from TP8 (0.00008)
    # c = 0.00008 * (8 / 16) = 0.00004
    t = pred.predict_execution_time(tp_size=16, seq_len=1000)
    assert abs(t - (0.00004 * 1000 + overhead)) < 1e-9


def test_slo_tracker_record_and_stats():
    """SLOTracker records predicted and actual TTFT and computes stats."""
    tracker = SLOTracker()

    tracker.record_predicted(tp_size=2, seq_len=1000, queue_time=0.01,
                             exec_time=0.18, predicted_ttft=0.19)
    tracker.record_predicted(tp_size=2, seq_len=2000, queue_time=0.02,
                             exec_time=0.36, predicted_ttft=0.38)

    stats = tracker.get_stats(tp_size=2)
    assert stats["count_predicted"] == 2
    assert stats["count_actual"] == 0
    assert abs(stats["avg_predicted"] - 0.285) < 1e-9

    # Record actuals
    tracker.record_actual(tp_size=2, actual_ttft=0.20)
    tracker.record_actual(tp_size=2, actual_ttft=0.40)

    stats = tracker.get_stats(tp_size=2)
    assert stats["count_actual"] == 2
    assert abs(stats["avg_actual"] - 0.30) < 1e-9


@pytest.mark.asyncio
async def test_predicted_ttft_logged_on_select():
    """async_select_p_d_node computes predicted TTFT and records in slo_tracker."""
    sel = _build_selector()

    # Small TP request: 50 tokens, arrival at T=100.0
    p_node, d_node = await sel.async_select_p_d_node(
        prompt="test", sampling_params=None, multimodal_params=None,
        input_token_num=50, arrival_time=100.0,
    )
    assert p_node is not None
    stats = sel.slo_tracker.get_stats()
    assert stats["count_predicted"] == 1
    assert stats["avg_predicted"] > 0

    # Complete the small TP request before sending a large one
    await sel.notify_request_done(p_node, input_token_num=50, actual_ttft=0.05)
    tp_size_small = p_node.start_args.get("tp", 1)
    tp_stats = sel.slo_tracker.get_stats(tp_size=tp_size_small)
    assert tp_stats["count_actual"] == 1

    # Large TP request: 200 tokens (above threshold=100), no inflight small TP blocking
    p_node2, _ = await sel.async_select_p_d_node(
        prompt="test", sampling_params=None, multimodal_params=None,
        input_token_num=200, arrival_time=100.0,
    )
    stats = sel.slo_tracker.get_stats()
    assert stats["count_predicted"] == 2

    # Notify large with actual TTFT
    await sel.notify_request_done(p_node2, input_token_num=200, actual_ttft=0.10)
    tp_size_large = p_node2.start_args.get("tp", 1)
    tp_stats_large = sel.slo_tracker.get_stats(tp_size=tp_size_large)
    assert tp_stats_large["count_actual"] >= 1


def test_estimate_node_wait_time_empty():
    """No inflight → wait time = 0."""
    group = FlexTPGroup("test")
    pred = LatencyPredictor()
    assert group.estimate_node_wait_time("node1", tp_size=2, predictor=pred) == 0.0


def test_estimate_node_wait_time_with_inflight():
    """Wait time accounts for inflight requests' predicted exec time minus elapsed."""
    group = FlexTPGroup("test")
    pred = LatencyPredictor()

    # Simulate: node just finished (last_done_time ≈ now), 2 inflight requests
    now = time.time()
    group.node_last_done_time["node1"] = now
    group.node_inflight_seq_lens["node1"] = [1000, 2000]

    wait = group.estimate_node_wait_time("node1", tp_size=2, predictor=pred)
    # Expected: exec(1000) + exec(2000) ≈ (0.00018*1000+0.1) + (0.00018*2000+0.1) = 0.28 + 0.46 = 0.74s
    # Since last_done_time ≈ now, wait ≈ total_exec
    expected_total = (pred.predict_execution_time(2, 1000)
                      + pred.predict_execution_time(2, 2000))
    assert abs(wait - expected_total) < 0.05  # within 50ms tolerance for time.time() drift


def test_estimate_node_wait_time_elapsed():
    """If time has passed since last_done, wait time decreases."""
    group = FlexTPGroup("test")
    pred = LatencyPredictor()

    # last_done was 0.5s ago, one inflight with exec_time ~0.28s → should already be done
    group.node_last_done_time["node1"] = time.time() - 0.5
    group.node_inflight_seq_lens["node1"] = [1000]

    wait = group.estimate_node_wait_time("node1", tp_size=2, predictor=pred)
    exec_time = pred.predict_execution_time(2, 1000)  # ~0.28s
    # 0.28 - 0.5 < 0 → clamped to 0
    assert wait == 0.0 or wait < 0.01  # should be ~0


@pytest.mark.asyncio
async def test_inflight_seq_lens_tracking():
    """add_inflight / remove_inflight correctly maintain per-request seq_lens."""
    sel = _build_selector()
    group = _get_group(sel)

    # Dispatch two small TP requests via _dispatch_small_tp
    n1 = await sel._dispatch_small_tp(group, group.small_tp_nodes[0], 100)
    n2 = await sel._dispatch_small_tp(group, group.small_tp_nodes[1], 200)

    # Both should appear in seq_lens
    total_seq_lens = []
    for node_key in group.node_inflight_seq_lens:
        total_seq_lens.extend(group.node_inflight_seq_lens[node_key])
    assert sorted(total_seq_lens) == [100, 200]

    # Remove one
    await sel.notify_request_done(n1, input_token_num=100)
    total_seq_lens = []
    for node_key in group.node_inflight_seq_lens:
        total_seq_lens.extend(group.node_inflight_seq_lens[node_key])
    assert total_seq_lens == [200]

    # last_done_time should be set
    assert group.node_last_done_time.get(n1.client_ip_port, 0) > 0


# ---------------------------------------------------------------------------
# SLO-based decision tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slo_mode_picks_smallest_tp_meeting_slo():
    """SLO mode should pick the smallest TP that meets the TTFT target."""
    # Use a generous SLO so small TP (tp=2) can satisfy it
    sel = _build_selector(slo_ttft=100.0)  # 100s SLO — trivially met
    group = _get_group(sel)

    p_node, d_node = await sel.async_select_p_d_node(
        prompt="test", sampling_params=None, multimodal_params=None,
        input_token_num=5000, arrival_time=time.time(),
    )
    # Should pick small TP (tp=2) since it meets SLO and is smaller
    assert p_node.start_args["tp"] == 2


@pytest.mark.asyncio
async def test_slo_mode_falls_back_to_min_ttft():
    """When no TP meets SLO, pick the one with lowest predicted TTFT."""
    # Use an impossibly tight SLO
    sel = _build_selector(slo_ttft=0.001)  # 1ms — impossible to meet
    group = _get_group(sel)

    p_node, d_node = await sel.async_select_p_d_node(
        prompt="test", sampling_params=None, multimodal_params=None,
        input_token_num=50000, arrival_time=time.time(),
    )
    # For very long sequences, large TP (tp=4) should have lower exec time
    # and thus lower TTFT, so it should be selected
    assert p_node.start_args["tp"] == 4
