from .pd_selector import PDSelector, RandomSelector, RoundRobinSelector, AdaptiveLoadSelector
from .flex_tp_selector import FlexTPSelector
from .flex_tp_selector_naive import FlexTPNaiveSelector
from .flex_tp_selector_v2 import FlexTPSelectorV2


def create_selector(selector_type: str, pd_manager, **kwargs) -> PDSelector:
    if selector_type == "random":
        return RandomSelector(pd_manager)
    elif selector_type == "round_robin":
        return RoundRobinSelector(pd_manager)
    elif selector_type == "adaptive_load":
        return AdaptiveLoadSelector(pd_manager)
    elif selector_type == "flex_tp":
        length_threshold = kwargs.get("flex_tp_threshold", 8000)
        slo_ttft = kwargs.get("flex_tp_slo_ttft", None)
        return FlexTPSelector(pd_manager, length_threshold=length_threshold, slo_ttft=slo_ttft)
    elif selector_type == "flex_tp_naive":
        length_threshold = kwargs.get("flex_tp_threshold", 8000)
        return FlexTPNaiveSelector(pd_manager, length_threshold=length_threshold)
    elif selector_type == "flex_tp_v2":
        slo_ttft = kwargs.get("flex_tp_slo_ttft", 5.0)
        latency_constants = kwargs.get("flex_tp_latency_constants", None)
        return FlexTPSelectorV2(pd_manager, slo_ttft=slo_ttft, latency_constants=latency_constants)
    else:
        raise ValueError(f"Invalid selector type: {selector_type}")
