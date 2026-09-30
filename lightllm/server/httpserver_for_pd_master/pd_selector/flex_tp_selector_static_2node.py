"""
Flex TP Static 2-Node Selector: 双主机静态 TP 分区调度器（临时实验用）。

与 naive 版本不同，这里不在同一主机上混跑多种 TP，而是按主机静态分区：
  - 第一个主机：仅使用 tp2（小 TP）实例，服务短请求
  - 第二个主机：仅使用 tp4（大 TP）实例，服务长请求

调度逻辑：
  - input_token_num > length_threshold → 大 TP 池（第二主机 tp4）
  - 否则 → 小 TP 池（第一主机 tp2）
  - 同一池内多实例间按在途 token 数最少做负载均衡
  - decode 节点优先选择与选中 p 节点同主机的实例

这是用于模拟“按长度静态 TP 分区”基线的临时实现，没有任何状态机/动态切换。
"""

import random
from typing import Union, List, Tuple, Dict, Optional

from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.utils.log_utils import init_logger
from .pd_selector import PDSelector

logger = init_logger(__name__)


class FlexTPStatic2NodeSelector(PDSelector):
    """
    双主机静态 TP 分区选择器：

    - 全局按 TP 大小把 prefill 节点分成小 TP 池和大 TP 池（min_tp / max_tp）
    - input_token_num > length_threshold → 大 TP 池，否则 → 小 TP 池
    - 池内按在途 token 数最少做负载均衡
    - 目标池为空时回退到另一个池
    """

    def __init__(self, pd_manager, length_threshold: int = 8000):
        super().__init__(pd_manager)
        self.length_threshold: int = length_threshold
        # 静态分区后的两个池
        self.small_tp_nodes: List[PD_Client_Obj] = []
        self.large_tp_nodes: List[PD_Client_Obj] = []
        self.small_tp_size: int = 0
        self.large_tp_size: int = 0
        # node client_ip_port -> 在途 token 数 / 请求数
        self.node_inflight_tokens: Dict[str, int] = {}
        self.node_inflight_requests: Dict[str, int] = {}
        # decode 节点轮询索引
        self._decode_rr_index: int = 0

    def update_nodes(self, prefill_nodes, decode_nodes):
        super().update_nodes(prefill_nodes, decode_nodes)
        self._rebuild_partition()

    def _rebuild_partition(self):
        """根据注册的 prefill 节点，按 TP 大小静态分区为小/大 TP 两个池。"""
        small_nodes: List[PD_Client_Obj] = []
        large_nodes: List[PD_Client_Obj] = []

        def _tp_of(node) -> int:
            sa = node.start_args if isinstance(node.start_args, dict) else {}
            return sa.get("tp", 1)

        tp_sizes = set(_tp_of(n) for n in self.prefill_nodes)

        if len(tp_sizes) >= 2:
            min_tp = min(tp_sizes)
            max_tp = max(tp_sizes)
            for n in self.prefill_nodes:
                if _tp_of(n) == min_tp:
                    small_nodes.append(n)
                elif _tp_of(n) == max_tp:
                    large_nodes.append(n)
                # 中间 TP 大小在静态 2node 实验中忽略
        elif len(tp_sizes) == 1:
            # 只有一种 TP，全部放进小 TP 池
            min_tp = max_tp = next(iter(tp_sizes))
            small_nodes = list(self.prefill_nodes)
        else:
            min_tp = max_tp = 0

        self.small_tp_nodes = small_nodes
        self.large_tp_nodes = large_nodes
        self.small_tp_size = min_tp
        self.large_tp_size = max_tp

        def _hosts(nodes):
            return sorted(set(n.client_ip_port.split(":")[0] for n in nodes))

        logger.info(
            f"FlexTP static-2node partition: "
            f"small_tp={min_tp} ({len(small_nodes)} nodes on {_hosts(small_nodes)}), "
            f"large_tp={max_tp} ({len(large_nodes)} nodes on {_hosts(large_nodes)})"
        )

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

    def _pick_least_loaded(self, nodes: List[PD_Client_Obj]) -> PD_Client_Obj:
        """选择负载最低（在途 token 数最少）的节点"""
        return min(nodes, key=lambda n: self.node_inflight_tokens.get(n.client_ip_port, 0))

    def _pick_decode_node(self, p_node: Optional[PD_Client_Obj] = None) -> PD_Client_Obj:
        """选择 decode 节点：优先选择与选中的 p 节点在同一主机上的 decode 节点，
        若无同主机 decode 节点则回退到全局轮询。"""
        candidates = self.decode_nodes
        if p_node is not None:
            p_host = p_node.client_ip_port.split(":")[0]
            same_host = [d for d in self.decode_nodes if d.client_ip_port.split(":")[0] == p_host]
            if same_host:
                candidates = same_host

        self._decode_rr_index = self._decode_rr_index % len(candidates)
        d_node = candidates[self._decode_rr_index]
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
                f"FlexTPStatic2NodeSelector: req_id={req_id} no available nodes "
                f"(prefill={len(self.prefill_nodes)}, decode={len(self.decode_nodes)})"
            )

        token_num = input_token_num or 0
        use_large_tp = token_num > self.length_threshold

        # 按长度选择目标池，目标池为空时回退到另一个池
        primary = self.large_tp_nodes if use_large_tp else self.small_tp_nodes
        fallback = self.small_tp_nodes if use_large_tp else self.large_tp_nodes
        nodes = primary if primary else fallback

        if not nodes:
            p_node = random.choice(self.prefill_nodes)
            return p_node, self._pick_decode_node(p_node)

        best_node = self._pick_least_loaded(nodes)
        best_load = self.node_inflight_tokens.get(best_node.client_ip_port, 0)
        self.add_inflight(best_node.client_ip_port, token_num)

        is_large = best_node in self.large_tp_nodes
        tp_type = "large" if is_large else "small"
        tp_size = best_node.start_args.get("tp", 1) if isinstance(best_node.start_args, dict) else 1

        logger.info(
            f"FlexTP static-2node select: req_id={req_id}, input_tokens={token_num}, "
            f"tp={tp_size} ({tp_type}), node={best_node.client_ip_port}, "
            f"inflight_tokens={best_load + token_num}"
        )

        return best_node, self._pick_decode_node(best_node)

    async def notify_request_done(self, p_node: PD_Client_Obj, input_token_num: int = 0,
                                  actual_ttft: Optional[float] = None,
                                  req_id: Optional[int] = None):
        if p_node is None:
            return
        self.remove_inflight(p_node.client_ip_port, input_token_num)

    def select_p_d_node(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        raise NotImplementedError(
            "FlexTPStatic2NodeSelector requires async_select_p_d_node. "
            "Ensure PDManager.select_p_d_node is using the async path."
        )
