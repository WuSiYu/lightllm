"""
Flex TP Naive Selector: 无状态切换的简化版 Flex TP 调度器。

仅根据请求长度立即分配到对应 TP 配置的节点，不使用 drain 状态机，
因此同一组 GPU 上的不同 TP 实例可能同时处理请求（需要外部保证不冲突，或接受冲突）。

保留同一 TP 配置内多实例的负载均衡（按在途 token 数最少选择）。
"""

import random
import time
from typing import Union, List, Tuple, Dict, Optional

from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.utils.log_utils import init_logger
from .pd_selector import PDSelector

logger = init_logger(__name__)


class FlexTPNaiveGroup:
    """一组共享同一批 GPU 的不同 TP 大小的 prefill 节点（无状态机）"""

    def __init__(self, group_id: str):
        self.group_id: str = group_id
        self.small_tp_nodes: List[PD_Client_Obj] = []
        self.large_tp_nodes: List[PD_Client_Obj] = []
        self.small_tp_size: int = 0
        self.large_tp_size: int = 0

        # 每个节点的在途请求数和 token 数: node client_ip_port -> count
        self.node_inflight_requests: Dict[str, int] = {}
        self.node_inflight_tokens: Dict[str, int] = {}

    def _sum_nodes(self, nodes: List[PD_Client_Obj], data: Dict[str, int]) -> int:
        return sum(data.get(n.client_ip_port, 0) for n in nodes)

    @property
    def inflight_small_tp(self) -> int:
        return self._sum_nodes(self.small_tp_nodes, self.node_inflight_requests)

    @property
    def inflight_large_tp(self) -> int:
        return self._sum_nodes(self.large_tp_nodes, self.node_inflight_requests)

    def add_inflight(self, node_key: str, token_num: int):
        self.node_inflight_requests[node_key] = self.node_inflight_requests.get(node_key, 0) + 1
        self.node_inflight_tokens[node_key] = self.node_inflight_tokens.get(node_key, 0) + token_num

    def remove_inflight(self, node_key: str, token_num: int):
        new_req = max(0, self.node_inflight_requests.get(node_key, 0) - 1)
        self.node_inflight_requests[node_key] = new_req
        if new_req == 0:
            self.node_inflight_tokens[node_key] = 0
        else:
            self.node_inflight_tokens[node_key] = max(0, self.node_inflight_tokens.get(node_key, 0) - token_num)

    def __repr__(self):
        small_details = ", ".join(
            f"{n.client_ip_port}={self.node_inflight_requests.get(n.client_ip_port, 0)}"
            f"({self.node_inflight_tokens.get(n.client_ip_port, 0)}tok)"
            for n in self.small_tp_nodes
        )
        large_details = ", ".join(
            f"{n.client_ip_port}={self.node_inflight_requests.get(n.client_ip_port, 0)}"
            f"({self.node_inflight_tokens.get(n.client_ip_port, 0)}tok)"
            for n in self.large_tp_nodes
        )
        return (
            f"FlexTPNaiveGroup(id={self.group_id}, "
            f"small_tp={self.small_tp_size}x{len(self.small_tp_nodes)} [{small_details}], "
            f"large_tp={self.large_tp_size}x{len(self.large_tp_nodes)} [{large_details}])"
        )


class FlexTPNaiveSelector(PDSelector):
    """
    Naive Flex TP 调度选择器：无状态切换，仅根据请求长度立即分配。

    - input_token_num > length_threshold → 大 TP 节点
    - 否则 → 小 TP 节点
    - 同一 TP 配置多实例间按在途 token 数最少做负载均衡
    """

    def __init__(self, pd_manager, length_threshold: int = 8000):
        super().__init__(pd_manager)
        self.length_threshold: int = length_threshold
        # group_id -> FlexTPNaiveGroup
        self.flex_groups: Dict[str, FlexTPNaiveGroup] = {}
        # node client_ip_port -> FlexTPNaiveGroup
        self.node_to_group: Dict[str, FlexTPNaiveGroup] = {}
        # node client_ip_port -> bool (True=large TP)
        self.node_is_large_tp: Dict[str, bool] = {}
        # 不属于任何 flex group 的普通 prefill 节点
        self.ungrouped_prefill_nodes: List[PD_Client_Obj] = []
        # decode 节点轮询索引
        self._decode_rr_index: int = 0

    def update_nodes(self, prefill_nodes, decode_nodes):
        super().update_nodes(prefill_nodes, decode_nodes)
        self._rebuild_flex_groups()

    def _rebuild_flex_groups(self):
        """根据注册的 prefill 节点信息，自动构建 FlexTPNaiveGroup"""
        groups_map: Dict[str, List[PD_Client_Obj]] = {}
        ungrouped = []

        for node in self.prefill_nodes:
            sa = node.start_args if isinstance(node.start_args, dict) else {}
            shared_weight = sa.get("shared_weight")
            port_start = sa.get("shared_weight_master_port_start")

            if shared_weight and port_start is not None:
                host = node.client_ip_port.split(":")[0]
                group_id = f"{host}:{port_start}"
                groups_map.setdefault(group_id, []).append(node)
            else:
                ungrouped.append(node)

        new_flex_groups: Dict[str, FlexTPNaiveGroup] = {}
        new_node_to_group: Dict[str, FlexTPNaiveGroup] = {}
        new_node_is_large_tp: Dict[str, bool] = {}

        for group_id, nodes in groups_map.items():
            tp_sizes = set(n.start_args.get("tp", 1) for n in nodes)
            if len(tp_sizes) < 2:
                ungrouped.extend(nodes)
                continue

            min_tp = min(tp_sizes)
            max_tp = max(tp_sizes)

            old_group = self.flex_groups.get(group_id)
            if old_group:
                group = old_group
                group.small_tp_nodes = []
                group.large_tp_nodes = []
            else:
                group = FlexTPNaiveGroup(group_id=group_id)

            group.small_tp_size = min_tp
            group.large_tp_size = max_tp

            for n in nodes:
                tp = n.start_args.get("tp", 1)
                if tp == min_tp:
                    group.small_tp_nodes.append(n)
                    new_node_is_large_tp[n.client_ip_port] = False
                else:
                    group.large_tp_nodes.append(n)
                    new_node_is_large_tp[n.client_ip_port] = True
                new_node_to_group[n.client_ip_port] = group

            new_flex_groups[group_id] = group
            logger.info(
                f"FlexTP naive group [{group_id}]: small_tp={min_tp} ({len(group.small_tp_nodes)} nodes), "
                f"large_tp={max_tp} ({len(group.large_tp_nodes)} nodes)"
            )

        # 保留已移除节点的映射（有 inflight 请求时）
        for old_key, old_group in self.node_to_group.items():
            if old_key not in new_node_to_group:
                if old_group.node_inflight_requests.get(old_key, 0) > 0:
                    new_node_to_group[old_key] = old_group
                    new_node_is_large_tp[old_key] = self.node_is_large_tp.get(old_key, False)

        self.flex_groups = new_flex_groups
        self.node_to_group = new_node_to_group
        self.node_is_large_tp = new_node_is_large_tp
        self.ungrouped_prefill_nodes = ungrouped

        if ungrouped:
            logger.info(f"FlexTP naive: {len(ungrouped)} ungrouped prefill node(s)")

    def _pick_least_loaded(self, group: FlexTPNaiveGroup, nodes: List[PD_Client_Obj]) -> PD_Client_Obj:
        """选择负载最低（在途 token 数最少）的节点"""
        return min(nodes, key=lambda n: group.node_inflight_tokens.get(n.client_ip_port, 0))

    def _pick_decode_node(self) -> PD_Client_Obj:
        """轮询方式选择 decode 节点"""
        self._decode_rr_index = self._decode_rr_index % len(self.decode_nodes)
        d_node = self.decode_nodes[self._decode_rr_index]
        self._decode_rr_index += 1
        return d_node

    async def async_select_p_d_node(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        input_token_num: Optional[int] = None,
        arrival_time: Optional[float] = None,
        req_id: Optional[int] = None,
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        if not self.prefill_nodes or not self.decode_nodes:
            raise RuntimeError(
                f"FlexTPNaiveSelector: req_id={req_id} no available nodes "
                f"(prefill={len(self.prefill_nodes)}, decode={len(self.decode_nodes)})"
            )

        d_node = self._pick_decode_node()

        if not self.flex_groups:
            p_node = random.choice(self.prefill_nodes)
            return p_node, d_node

        token_num = input_token_num or 0
        use_large_tp = token_num > self.length_threshold

        # 选择最优 group 和节点
        best_group: Optional[FlexTPNaiveGroup] = None
        best_node: Optional[PD_Client_Obj] = None
        best_load = float("inf")

        for group in self.flex_groups.values():
            nodes = group.large_tp_nodes if use_large_tp else group.small_tp_nodes
            if not nodes:
                continue
            candidate = self._pick_least_loaded(group, nodes)
            load = group.node_inflight_tokens.get(candidate.client_ip_port, 0)
            if load < best_load:
                best_load = load
                best_group = group
                best_node = candidate

        if best_group is None or best_node is None:
            # 目标 TP 无可用节点，回退到另一种 TP
            for group in self.flex_groups.values():
                nodes = group.small_tp_nodes if use_large_tp else group.large_tp_nodes
                if not nodes:
                    continue
                candidate = self._pick_least_loaded(group, nodes)
                load = group.node_inflight_tokens.get(candidate.client_ip_port, 0)
                if load < best_load:
                    best_load = load
                    best_group = group
                    best_node = candidate

        if best_group is None or best_node is None:
            p_node = random.choice(self.prefill_nodes)
            return p_node, d_node

        best_group.add_inflight(best_node.client_ip_port, token_num)
        tp_type = "large" if self.node_is_large_tp.get(best_node.client_ip_port, False) else "small"
        tp_size = best_node.start_args.get("tp", 1) if isinstance(best_node.start_args, dict) else 1

        logger.info(
            f"FlexTP naive select: req_id={req_id}, input_tokens={token_num}, "
            f"tp={tp_size} ({tp_type}), node={best_node.client_ip_port}, "
            f"group={best_group.group_id}, inflight_tokens={best_load + token_num}"
        )

        return best_node, d_node

    async def notify_request_done(self, p_node: PD_Client_Obj, input_token_num: int = 0,
                                  actual_ttft: Optional[float] = None,
                                  req_id: Optional[int] = None):
        if p_node is None:
            return
        group = self.node_to_group.get(p_node.client_ip_port)
        if group is None:
            return
        group.remove_inflight(p_node.client_ip_port, input_token_num)

    def select_p_d_node(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        raise NotImplementedError(
            "FlexTPNaiveSelector requires async_select_p_d_node. "
            "Ensure PDManager.select_p_d_node is using the async path."
        )
