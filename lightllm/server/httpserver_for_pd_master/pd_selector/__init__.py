from .pd_selector import (
    PDSelector,
    RandomSelector,
    RoundRobinSelector,
    RoundRobinPlusSelector,
    AdaptiveLoadSelector,
)
from .flex_tp_selector import FlexTPSelector
from .flex_tp_selector_naive import FlexTPNaiveSelector
from .flex_tp_selector_naive_switch import FlexTPNaiveSwitchSelector
from .flex_tp_selector_static_2node import FlexTPStatic2NodeSelector
from .flex_tp_selector_static_plus_2node import FlexTPStaticPlus2NodeSelector
from .flex_tp_selector_v2 import FlexTPSelectorV2
from .flex_tp_selector_v3 import FlexTPSelectorV3
from .flex_tp_selector_v4 import FlexTPSelectorV4
from .flex_tp_selector_v5 import FlexTPSelectorV5
from .flex_tp_selector_v6 import FlexTPSelectorV6
from .flex_tp_selector_v7 import FlexTPSelectorV7
from .flex_tp_selector_v8 import FlexTPSelectorV8
from .flex_tp_selector_v9 import FlexTPSelectorV9
from .flex_tp_selector_v10 import FlexTPSelectorV10
from .flex_tp_selector_v11 import FlexTPSelectorV11
from .flex_tp_selector_v12 import FlexTPSelectorV12
from .flex_tp_selector_v13 import FlexTPSelectorV13
from .flex_tp_selector_v14 import FlexTPSelectorV14


def create_selector(selector_type: str, pd_manager, **kwargs) -> PDSelector:
    if selector_type == "random":
        return RandomSelector(pd_manager)
    elif selector_type == "round_robin":
        return RoundRobinSelector(pd_manager)
    elif selector_type == "round_robin_plus":
        return RoundRobinPlusSelector(pd_manager)
    elif selector_type == "adaptive_load":
        return AdaptiveLoadSelector(pd_manager)
    elif selector_type == "flex_tp":
        length_threshold = kwargs.get("flex_tp_threshold", 8000)
        slo_ttft = kwargs.get("flex_tp_slo_ttft", None)
        return FlexTPSelector(pd_manager, length_threshold=length_threshold, slo_ttft=slo_ttft)
    elif selector_type == "flex_tp_naive":
        length_threshold = kwargs.get("flex_tp_threshold", 8000)
        return FlexTPNaiveSelector(pd_manager, length_threshold=length_threshold)
    elif selector_type == "flex_tp_naive_switch":
        length_threshold = kwargs.get("flex_tp_threshold", 8000)
        return FlexTPNaiveSwitchSelector(pd_manager, length_threshold=length_threshold)
    elif selector_type == "flex_tp_static_2node":
        length_threshold = kwargs.get("flex_tp_threshold", 8000)
        return FlexTPStatic2NodeSelector(pd_manager, length_threshold=length_threshold)
    elif selector_type == "flex_tp_static_plus_2node":
        length_threshold = kwargs.get("flex_tp_threshold", 8000)
        overload_tokens = kwargs.get("flex_tp_overload_tokens", 20000)
        spill_ratio = kwargs.get("flex_tp_spill_ratio", 2.0)
        return FlexTPStaticPlus2NodeSelector(
            pd_manager, length_threshold=length_threshold,
            overload_tokens=overload_tokens, spill_ratio=spill_ratio,
        )
    elif selector_type == "flex_tp_v2":
        slo_ttft = kwargs.get("flex_tp_slo_ttft", 5.0)
        return FlexTPSelectorV2(pd_manager, slo_ttft=slo_ttft)
    elif selector_type == "flex_tp_v3":
        return FlexTPSelectorV3(
            pd_manager,
            slo_ttft=kwargs.get("flex_tp_slo_ttft", 5.0),
            batch_token_cap=kwargs.get("flex_tp_bundle_token_cap", 8192),
            batch_token_trigger=kwargs.get("flex_tp_bundle_token_trigger", 4096),
            max_inflight_per_instance=kwargs.get("flex_tp_max_inflight", 64),
            max_admitted_tokens_per_instance=kwargs.get("flex_tp_instance_token_credit", 16384),
            batch_window_s=float(kwargs.get("flex_tp_bundle_window_ms", 20.0)) / 1000.0,
            prediction_margin_s=kwargs.get("flex_tp_prediction_margin", 0.08),
            overload_policy=kwargs.get("flex_tp_overload_policy", "best_effort"),
        )
    elif selector_type in ("flex_tp_v4", "flex_tp_v5"):
        selector_cls = FlexTPSelectorV4 if selector_type == "flex_tp_v4" else FlexTPSelectorV5
        return selector_cls(
            pd_manager,
            slo_ttft=kwargs.get("flex_tp_slo_ttft", 5.0),
            long_request_threshold=kwargs.get("flex_tp_long_threshold", 4000),
            batch_token_cap=kwargs.get("flex_tp_bundle_token_cap", 8192),
            batch_token_trigger=kwargs.get("flex_tp_bundle_token_trigger", 4096),
            max_inflight_per_instance=kwargs.get("flex_tp_max_inflight", 64),
            max_admitted_tokens_per_instance=kwargs.get("flex_tp_instance_token_credit", 16384),
            batch_window_s=float(kwargs.get("flex_tp_bundle_window_ms", 20.0)) / 1000.0,
            prediction_margin_s=kwargs.get("flex_tp_prediction_margin", 0.08),
            overload_policy=kwargs.get("flex_tp_overload_policy", "best_effort"),
        )
    elif selector_type == "flex_tp_v6":
        return FlexTPSelectorV6(
            pd_manager,
            slo_ttft=kwargs.get("flex_tp_slo_ttft", 5.0),
            long_request_threshold=kwargs.get("flex_tp_long_threshold", 4000),
            batch_token_cap=kwargs.get("flex_tp_bundle_token_cap", 8192),
            batch_token_trigger=kwargs.get("flex_tp_bundle_token_trigger", 4096),
            max_inflight_per_instance=kwargs.get("flex_tp_max_inflight", 64),
            max_admitted_tokens_per_instance=kwargs.get("flex_tp_instance_token_credit", 16384),
            batch_window_s=float(kwargs.get("flex_tp_bundle_window_ms", 20.0)) / 1000.0,
            prediction_margin_s=kwargs.get("flex_tp_prediction_margin", 0.08),
            mps_overlap_slowdown=kwargs.get("flex_tp_mps_slowdown", 2.0),
            overload_policy=kwargs.get("flex_tp_overload_policy", "best_effort"),
        )
    elif selector_type == "flex_tp_v7":
        return FlexTPSelectorV7(
            pd_manager,
            slo_ttft=kwargs.get("flex_tp_slo_ttft", 5.0),
            long_request_threshold=kwargs.get("flex_tp_long_threshold", 4000),
            batch_token_cap=kwargs.get("flex_tp_bundle_token_cap", 8192),
            batch_token_trigger=kwargs.get("flex_tp_bundle_token_trigger", 4096),
            max_inflight_per_instance=kwargs.get("flex_tp_max_inflight", 64),
            max_admitted_tokens_per_instance=kwargs.get("flex_tp_instance_token_credit", 16384),
            batch_window_s=float(kwargs.get("flex_tp_bundle_window_ms", 20.0)) / 1000.0,
            prediction_margin_s=kwargs.get("flex_tp_prediction_margin", 0.08),
            mps_overlap_slowdown=kwargs.get("flex_tp_mps_slowdown", 2.0),
            class_quantum_tokens=kwargs.get("flex_tp_v7_class_quantum_tokens", 4096),
            conflict_price_weight=kwargs.get("flex_tp_v7_conflict_price_weight", 0.35),
            urgency_weight=kwargs.get("flex_tp_v7_urgency_weight", 2.0),
            overload_policy=kwargs.get("flex_tp_overload_policy", "best_effort"),
        )
    elif selector_type == "flex_tp_v8":
        return FlexTPSelectorV8(
            pd_manager,
            slo_ttft=kwargs.get("flex_tp_slo_ttft", 5.0),
            long_request_threshold=kwargs.get("flex_tp_long_threshold", 4000),
            batch_token_cap=kwargs.get("flex_tp_bundle_token_cap", 8192),
            batch_token_trigger=kwargs.get("flex_tp_bundle_token_trigger", 4096),
            max_inflight_per_instance=kwargs.get("flex_tp_max_inflight", 64),
            max_admitted_tokens_per_instance=kwargs.get("flex_tp_instance_token_credit", 16384),
            batch_window_s=float(kwargs.get("flex_tp_bundle_window_ms", 20.0)) / 1000.0,
            prediction_margin_s=kwargs.get("flex_tp_prediction_margin", 0.08),
            mps_overlap_slowdown=kwargs.get("flex_tp_mps_slowdown", 2.0),
            epoch_s=float(kwargs.get("flex_tp_v8_epoch_ms", 50.0)) / 1000.0,
            epoch_token_budget=kwargs.get("flex_tp_v8_epoch_token_budget", 16384),
            min_lane_quota_tokens=kwargs.get("flex_tp_v8_min_lane_quota", 2048),
            deadline_pressure_weight=kwargs.get("flex_tp_v8_deadline_pressure_weight", 2.0),
            overload_policy=kwargs.get("flex_tp_overload_policy", "best_effort"),
        )
    elif selector_type == "flex_tp_v9":
        return FlexTPSelectorV9(
            pd_manager,
            slo_ttft=kwargs.get("flex_tp_slo_ttft", 5.0),
            long_request_threshold=kwargs.get("flex_tp_long_threshold", 4000),
            batch_token_cap=kwargs.get("flex_tp_bundle_token_cap", 8192),
            batch_token_trigger=kwargs.get("flex_tp_bundle_token_trigger", 4096),
            max_inflight_per_instance=kwargs.get("flex_tp_max_inflight", 64),
            max_admitted_tokens_per_instance=kwargs.get("flex_tp_instance_token_credit", 16384),
            batch_window_s=float(kwargs.get("flex_tp_bundle_window_ms", 20.0)) / 1000.0,
            prediction_margin_s=kwargs.get("flex_tp_prediction_margin", 0.08),
            replan_interval_s=kwargs.get("flex_tp_replan_interval", 0.05),
            mps_overlap_slowdown=kwargs.get("flex_tp_mps_slowdown", 2.0),
            aging_interval_s=kwargs.get("flex_tp_v9_aging_interval_ms", 150.0) / 1000.0,
            interactive_token_limit=kwargs.get("flex_tp_v9_interactive_tokens", 1024),
            short_class_weight=kwargs.get("flex_tp_v9_short_weight", 1.0),
            long_class_weight=kwargs.get("flex_tp_v9_long_weight", 1.0),
            overload_policy=kwargs.get("flex_tp_overload_policy", "best_effort"),
        )
    elif selector_type == "flex_tp_v10":
        return FlexTPSelectorV10(
            pd_manager,
            slo_ttft=kwargs.get("flex_tp_slo_ttft", 5.0),
            long_request_threshold=kwargs.get("flex_tp_long_threshold", 4000),
            batch_token_cap=kwargs.get("flex_tp_bundle_token_cap", 8192),
            batch_token_trigger=kwargs.get("flex_tp_bundle_token_trigger", 4096),
            max_inflight_per_instance=kwargs.get("flex_tp_max_inflight", 64),
            max_admitted_tokens_per_instance=kwargs.get("flex_tp_instance_token_credit", 16384),
            batch_window_s=float(kwargs.get("flex_tp_bundle_window_ms", 20.0)) / 1000.0,
            prediction_margin_s=kwargs.get("flex_tp_prediction_margin", 0.08),
            replan_interval_s=kwargs.get("flex_tp_replan_interval", 0.05),
            mps_overlap_slowdown=kwargs.get("flex_tp_mps_slowdown", 2.0),
            short_weight=kwargs.get("flex_tp_v10_short_weight", 1.0),
            long_weight=kwargs.get("flex_tp_v10_long_weight", 1.0),
            slack_weight=kwargs.get("flex_tp_v10_slack_weight", 1.0),
            overlap_slack_ratio=kwargs.get("flex_tp_v10_overlap_slack_ratio", 0.10),
            overload_policy=kwargs.get("flex_tp_overload_policy", "best_effort"),
        )
    elif selector_type == "flex_tp_v11":
        return FlexTPSelectorV11(
            pd_manager,
            slo_ttft=kwargs.get("flex_tp_slo_ttft", 5.0),
            long_request_threshold=kwargs.get("flex_tp_long_threshold", 4000),
            batch_token_cap=kwargs.get("flex_tp_bundle_token_cap", 8192),
            batch_token_trigger=kwargs.get("flex_tp_bundle_token_trigger", 4096),
            max_inflight_per_instance=kwargs.get("flex_tp_max_inflight", 64),
            max_admitted_tokens_per_instance=kwargs.get("flex_tp_instance_token_credit", 16384),
            batch_window_s=float(kwargs.get("flex_tp_bundle_window_ms", 20.0)) / 1000.0,
            prediction_margin_s=kwargs.get("flex_tp_prediction_margin", 0.08),
            replan_interval_s=kwargs.get("flex_tp_replan_interval", 0.05),
            mps_overlap_slowdown=kwargs.get("flex_tp_mps_slowdown", 2.0),
            short_weight=kwargs.get("flex_tp_v11_short_weight", 1.0),
            long_weight=kwargs.get("flex_tp_v11_long_weight", 1.0),
            slack_weight=kwargs.get("flex_tp_v11_slack_weight", 1.0),
            overlap_slack_ratio=kwargs.get("flex_tp_v11_overlap_slack_ratio", 0.10),
            routing_cost_weight=kwargs.get("flex_tp_v11_routing_cost_weight", 0.50),
            overload_policy=kwargs.get("flex_tp_overload_policy", "best_effort"),
        )
    elif selector_type == "flex_tp_v12":
        return FlexTPSelectorV12(
            pd_manager,
            slo_ttft=kwargs.get("flex_tp_slo_ttft", 5.0),
            long_request_threshold=kwargs.get("flex_tp_long_threshold", 4000),
            batch_token_cap=kwargs.get("flex_tp_bundle_token_cap", 8192),
            batch_token_trigger=kwargs.get("flex_tp_bundle_token_trigger", 4096),
            max_inflight_per_instance=kwargs.get("flex_tp_max_inflight", 64),
            max_admitted_tokens_per_instance=kwargs.get("flex_tp_instance_token_credit", 16384),
            batch_window_s=float(kwargs.get("flex_tp_bundle_window_ms", 20.0)) / 1000.0,
            prediction_margin_s=kwargs.get("flex_tp_prediction_margin", 0.08),
            replan_interval_s=kwargs.get("flex_tp_replan_interval", 0.05),
            mps_overlap_slowdown=kwargs.get("flex_tp_mps_slowdown", 2.0),
            request_utility_tokens=kwargs.get("flex_tp_v6_request_utility_tokens", 2000.0),
            routing_cost_weight=kwargs.get("flex_tp_v12_routing_cost_weight", 2.0),
            tp4_service_ratio_limit=kwargs.get("flex_tp_v12_tp4_service_ratio_limit", 0.60),
            overload_policy=kwargs.get("flex_tp_overload_policy", "best_effort"),
        )
    elif selector_type == "flex_tp_v13":
        return FlexTPSelectorV13(
            pd_manager,
            slo_ttft=kwargs.get("flex_tp_slo_ttft", 5.0),
            long_request_threshold=kwargs.get("flex_tp_v13_long_threshold", 12000),
            batch_token_cap=kwargs.get("flex_tp_bundle_token_cap", 8192),
            batch_token_trigger=kwargs.get("flex_tp_bundle_token_trigger", 4096),
            max_inflight_per_instance=kwargs.get("flex_tp_max_inflight", 64),
            max_admitted_tokens_per_instance=kwargs.get("flex_tp_instance_token_credit", 16384),
            batch_window_s=float(kwargs.get("flex_tp_bundle_window_ms", 20.0)) / 1000.0,
            prediction_margin_s=kwargs.get("flex_tp_prediction_margin", 0.08),
            replan_interval_s=kwargs.get("flex_tp_replan_interval", 0.05),
            mps_overlap_slowdown=kwargs.get("flex_tp_mps_slowdown", 2.0),
            request_utility_tokens=kwargs.get("flex_tp_v6_request_utility_tokens", 2000.0),
            latency_scale=kwargs.get("flex_tp_v13_latency_scale", 1.0),
            routing_cost_weight=kwargs.get("flex_tp_v13_routing_cost_weight", 2.0),
            tp4_service_ratio_limit=kwargs.get("flex_tp_v13_tp4_service_ratio_limit", 0.60),
            tp4_pressure_threshold=kwargs.get("flex_tp_v13_tp4_pressure_threshold", 1.0),
            overload_policy=kwargs.get("flex_tp_overload_policy", "best_effort"),
        )
    elif selector_type == "flex_tp_v14":
        return FlexTPSelectorV14(
            pd_manager,
            slo_ttft=kwargs.get("flex_tp_slo_ttft", 5.0),
            long_request_threshold=kwargs.get("flex_tp_v14_long_threshold", 12000),
            batch_token_cap=kwargs.get("flex_tp_bundle_token_cap", 8192),
            batch_token_trigger=kwargs.get("flex_tp_bundle_token_trigger", 4096),
            max_inflight_per_instance=kwargs.get("flex_tp_max_inflight", 64),
            max_admitted_tokens_per_instance=kwargs.get("flex_tp_instance_token_credit", 16384),
            batch_window_s=float(kwargs.get("flex_tp_bundle_window_ms", 20.0)) / 1000.0,
            prediction_margin_s=kwargs.get("flex_tp_prediction_margin", 0.08),
            replan_interval_s=kwargs.get("flex_tp_replan_interval", 0.05),
            mps_overlap_slowdown=kwargs.get("flex_tp_mps_slowdown", 2.0),
            request_utility_tokens=kwargs.get("flex_tp_v6_request_utility_tokens", 2000.0),
            latency_scale=kwargs.get("flex_tp_v14_latency_scale", 1.0),
            routing_cost_weight=kwargs.get("flex_tp_v14_routing_cost_weight", 2.0),
            tp4_service_ratio_limit=kwargs.get("flex_tp_v14_tp4_service_ratio_limit", 0.60),
            tp4_pressure_threshold=kwargs.get("flex_tp_v14_tp4_pressure_threshold", 1.0),
            decode_locality_enabled=kwargs.get("flex_tp_v14_decode_locality", True),
            overload_policy=kwargs.get("flex_tp_overload_policy", "best_effort"),
        )
    else:
        raise ValueError(f"Invalid selector type: {selector_type}")
