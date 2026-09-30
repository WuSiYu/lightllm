"""
Flex TP Static-Plus 2-Node Selector: 带溢出阀门的双主机静态 TP 分区调度器。

在 static_2node 的基础上增加一个“溢出阀门”：
  - 仍然优先根据请求长度选择主池（短→小 TP，长→大 TP）
  - 但当主池中最空的目标实例已经过载，而另一池中最空的实例明显更轻时，
    把该请求溢出（spill）到另一池，以削峰、避免主池排队恶化。

溢出判定（均以 selector 估计的「在途 token 数」为压力指标，比较各池最空节点）：
  load_primary >= overload_tokens  且  load_other * spill_ratio < load_primary
则溢出到 other 池的最空节点；否则正常落在 primary 池的最空节点。

这是用于「静态 TP 分区 + 轻量负载溢出」基线的临时实现，无状态机/动态切换。
"""

import random
from typing import Union, List, Tuple, Dict, Optional

from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.utils.log_utils import init_logger
from .pd_selector import PDSelector

logger = init_logger(__name__)


class FlexTPStaticPlus2NodeSelector(PDSelector):
    """
    双主机静态 TP 分区 + 溢出阀门选择器：

    - 全局按 TP 大小把 prefill 节点分成小 TP 池和大 TP 池（min_tp / max_tp）
    - input_token_num > length_threshold → 大 TP 池为主池，否则 → 小 TP 池为主池
    - 主池目标节点过载且另一池明显更空时，溢出到另一池
    - 池内/溢出落点均为「在途 token 数最少」的节点
    """

    def __init__(self, pd_manager, length_threshold: int = 8000,
                 overload_tokens: int = 20000, spill_ratio: float = 2.0):
        super().__init__(pd_manager)
        self.length_threshold: int = length_threshold
        # 溢出阀门参数
        self.overload_tokens: int = overload_tokens
        self.spill_ratio: float = spill_ratio
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
            f"FlexTP static-plus-2node partition: "
            f"small_tp={min_tp} ({len(small_nodes)} nodes on {_hosts(small_nodes)}), "
            f"large_tp={max_tp} ({len(large_nodes)} nodes on {_hosts(large_nodes)}), "
            f"overload_tokens={self.overload_tokens}, spill_ratio={self.spill_ratio}"
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
                f"FlexTPStaticPlus2NodeSelector: req_id={req_id} no available nodes "
                f"(prefill={len(self.prefill_nodes)}, decode={len(self.decode_nodes)})"
            )

        token_num = input_token_num or 0
        use_large_tp = token_num > self.length_threshold

        primary_pool = self.large_tp_nodes if use_large_tp else self.small_tp_nodes
        other_pool = self.small_tp_nodes if use_large_tp else self.large_tp_nodes

        # 主池为空则直接退到另一池（与 static_2node 一致）
        if not primary_pool:
            primary_pool, other_pool = other_pool, []

        if not primary_pool:
            p_node = random.choice(self.prefill_nodes)
            return p_node, self._pick_decode_node(p_node)

        cand_p = self._pick_least_loaded(primary_pool)
        load_p = self.node_inflight_tokens.get(cand_p.client_ip_port, 0)

        best_node = cand_p
        spilled = False
        # 溢出判定：主池过载 且 另一池明显更空
        if other_pool:
            cand_o = self._pick_least_loaded(other_pool)
            load_o = self.node_inflight_tokens.get(cand_o.client_ip_port, 0)
            if load_p >= self.overload_tokens and load_o * self.spill_ratio < load_p:
                best_node = cand_o
                spilled = True

        best_load = self.node_inflight_tokens.get(best_node.client_ip_port, 0)
        self.add_inflight(best_node.client_ip_port, token_num)

        is_large = best_node in self.large_tp_nodes
        tp_type = "large" if is_large else "small"
        tp_size = best_node.start_args.get("tp", 1) if isinstance(best_node.start_args, dict) else 1

        logger.info(
            f"FlexTP static-plus-2node select: req_id={req_id}, input_tokens={token_num}, "
            f"tp={tp_size} ({tp_type}), node={best_node.client_ip_port}, "
            f"spilled={spilled}, inflight_tokens={best_load + token_num}"
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
            "FlexTPStaticPlus2NodeSelector requires async_select_p_d_node. "
            "Ensure PDManager.select_p_d_node is using the async path."
        )
