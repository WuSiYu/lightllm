"""
Flex TP Selector: 支持不同 TP 数的 prefill 节点共享同一组 GPU（通过 --shared_weight）。

基于 PD master 的请求级排他（drain 状态机）。

调度流程（三步式）：
1. 预调度：对每种 TP 配置进行假设调度，预测 TTFT 并记录决策的具体实例
2. Best-effort 决策：选择满足 SLO 的最小 TP 配置；若都不满足则选全局 TTFT 最小的
3. 执行：按预调度记录的 group + node 执行 drain 状态机和 inflight 记录

背景： 在PD分离模式中，Flex TP (--shared_weight) 是一种在同一组 GPU 上部署不同 TP 大小 prefill 节点的方式，
适用于请求长度分布广泛的场景。通过在同一组GPU上运行多组不同TP大小的节点（例如2xTP2 + TP4），
并使用权重共享避免额外显存开销，可以在一组GPU上无缝切换prefill TP数，
让短请求以TP2 + DP2处理提高吞吐，而长请求则使用TP4降低延迟。
"""

import asyncio
import random
import time
import collections
from typing import Union, List, Tuple, Dict, Optional

from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.utils.log_utils import init_logger
from .pd_selector import PDSelector

logger = init_logger(__name__)


# ===========================================================================
#  延迟预测与 SLO 跟踪
# ===========================================================================


class LatencyPredictor:
    """简单的 prefill 延迟预测器：latency = C[tp] * seq_len（单位：秒）"""

    # 默认常数（秒/token），可通过 set_constant 覆盖
    DEFAULT_CONSTANTS: Dict[int, float] = {
        1: 0.00024,   # TP1: ~0.24 ms/tok
        2: 0.00018,   # TP2: ~0.18 ms/tok
        4: 0.00012,   # TP4: ~0.12 ms/tok
        8: 0.00008,   # TP8: ~0.08 ms/tok
    }

    def __init__(self, constants: Optional[Dict[int, float]] = None):
        self._constants: Dict[int, float] = dict(self.DEFAULT_CONSTANTS)
        if constants:
            self._constants.update(constants)

    def set_constant(self, tp_size: int, seconds_per_token: float):
        self._constants[tp_size] = seconds_per_token

    def predict_execution_time(self, tp_size: int, seq_len: int) -> float:
        """预测 prefill 执行时间（秒）"""
        c = self._constants.get(tp_size)
        if c is None:
            # 简单外推：取已知最大 TP 的常数 * (已知最大TP / tp_size)
            known = max(self._constants.keys())
            c = self._constants[known] * (known / tp_size)
        return c * seq_len + 0.1  # 加上 overhead


class SLOTracker:
    """轻量级 SLO 跟踪器，记录预测 TTFT 与实际 TTFT 的统计信息"""

    def __init__(self, window_size: int = 1000):
        self._window_size = window_size
        # 请求级记录：(predicted_ttft, actual_ttft_or_None, tp_size, seq_len)
        self._records: collections.deque = collections.deque(maxlen=window_size)
        # 按 TP 分组的统计
        self._per_tp_predicted: Dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=window_size)
        )
        self._per_tp_actual: Dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=window_size)
        )

    def record_predicted(self, tp_size: int, seq_len: int, queue_time: float,
                         exec_time: float, predicted_ttft: float):
        self._records.append({
            "tp": tp_size, "seq_len": seq_len,
            "queue_time": queue_time, "exec_time": exec_time,
            "predicted_ttft": predicted_ttft, "actual_ttft": None,
            "timestamp": time.time(),
        })
        self._per_tp_predicted[tp_size].append(predicted_ttft)

    def record_actual(self, tp_size: int, actual_ttft: float):
        self._per_tp_actual[tp_size].append(actual_ttft)
        # 回填最近同 TP 的未填记录
        for rec in reversed(self._records):
            if rec["tp"] == tp_size and rec["actual_ttft"] is None:
                rec["actual_ttft"] = actual_ttft
                break

    def get_stats(self, tp_size: Optional[int] = None) -> Dict:
        """获取指定 TP（或全局）的预测/实际 TTFT 统计"""
        if tp_size is not None:
            pred = list(self._per_tp_predicted.get(tp_size, []))
            actual = list(self._per_tp_actual.get(tp_size, []))
        else:
            pred = [r["predicted_ttft"] for r in self._records]
            actual = [r["actual_ttft"] for r in self._records if r["actual_ttft"] is not None]
        return {
            "count_predicted": len(pred),
            "count_actual": len(actual),
            "avg_predicted": sum(pred) / len(pred) if pred else 0.0,
            "avg_actual": sum(actual) / len(actual) if actual else 0.0,
            "max_predicted": max(pred) if pred else 0.0,
            "max_actual": max(actual) if actual else 0.0,
        }


# ===========================================================================
#  FlexTPGroup & FlexTPSelector
# ===========================================================================


class FlexTPGroup:
    """表示一组共享同一批 GPU 的不同 TP 大小的 prefill 节点（drain 模式）"""

    def __init__(self, group_id: str):
        self.group_id: str = group_id
        self.small_tp_nodes: List[PD_Client_Obj] = []
        self.large_tp_nodes: List[PD_Client_Obj] = []
        self.small_tp_size: int = 0
        self.large_tp_size: int = 0

        # 状态机: "small_tp_active" -> "draining" -> "large_tp_active" -> "small_tp_active"
        self.state: str = "small_tp_active"

        # 每个节点的在途请求数和 token 数: node client_ip_port -> count
        self.node_inflight_requests: Dict[str, int] = {}
        self.node_inflight_tokens: Dict[str, int] = {}
        # 每个节点的在途请求各自的 seq_len: node client_ip_port -> [seq_len, ...]
        self.node_inflight_seq_lens: Dict[str, List[int]] = {}
        # 每个节点上次请求完成的时间戳: node client_ip_port -> timestamp
        self.node_last_done_time: Dict[str, float] = {}
        # 等待执行的大 TP 请求数（已决定走大 TP 但还在等 drain 完成）
        self.pending_large_tp: int = 0
        # 等待执行的小 TP 请求数（所有 group 都被大 TP 占据时排队等待）
        self.pending_small_tp: int = 0

        # drain 发起时长请求的到达时间戳，用于判断短请求是否早于 drain
        self.drain_start_time: Optional[float] = None

        # 用于状态变更通知的 asyncio.Condition
        self.condition: asyncio.Condition = asyncio.Condition()

    def _sum_nodes(self, nodes: List[PD_Client_Obj], data: Dict[str, int]) -> int:
        return sum(data.get(n.client_ip_port, 0) for n in nodes)

    @property
    def inflight_small_tp(self) -> int:
        return self._sum_nodes(self.small_tp_nodes, self.node_inflight_requests)

    @property
    def inflight_large_tp(self) -> int:
        return self._sum_nodes(self.large_tp_nodes, self.node_inflight_requests)

    @property
    def inflight_tokens_small_tp(self) -> int:
        return self._sum_nodes(self.small_tp_nodes, self.node_inflight_tokens)

    @property
    def inflight_tokens_large_tp(self) -> int:
        return self._sum_nodes(self.large_tp_nodes, self.node_inflight_tokens)

    def add_inflight(self, node_key: str, token_num: int):
        self.node_inflight_requests[node_key] = self.node_inflight_requests.get(node_key, 0) + 1
        self.node_inflight_tokens[node_key] = self.node_inflight_tokens.get(node_key, 0) + token_num
        self.node_inflight_seq_lens.setdefault(node_key, []).append(token_num)

    def remove_inflight(self, node_key: str, token_num: int):
        new_req = max(0, self.node_inflight_requests.get(node_key, 0) - 1)
        self.node_inflight_requests[node_key] = new_req
        if new_req == 0:
            # 请求数归零时强制清零 token 数，防止累积漂移
            self.node_inflight_tokens[node_key] = 0
            self.node_inflight_seq_lens[node_key] = []
        else:
            self.node_inflight_tokens[node_key] = max(0, self.node_inflight_tokens.get(node_key, 0) - token_num)
            # 移除一个匹配的 seq_len（FIFO 顺序，最早的先完成）
            seq_lens = self.node_inflight_seq_lens.get(node_key, [])
            if token_num in seq_lens:
                seq_lens.remove(token_num)
        self.node_last_done_time[node_key] = time.time()

    def estimate_node_wait_time(self, node_key: str, tp_size: int, predictor: "LatencyPredictor") -> float:
        """估算节点上已有 inflight 请求全部执行完毕还需要的时间。

        模型：所有 inflight 请求从 last_done_time 起顺序执行，
        estimated_all_done = last_done_time + sum(exec_time(seq_len_i))，
        wait = max(0, estimated_all_done - now)。
        """
        seq_lens = self.node_inflight_seq_lens.get(node_key, [])
        if not seq_lens:
            return 0.0
        total_remaining_exec = sum(predictor.predict_execution_time(tp_size, s) for s in seq_lens)
        # 未有完成记录时，以当前时间作为起始（请求从现在开始执行）
        last_done = self.node_last_done_time.get(node_key) or time.time()
        estimated_all_done = last_done + total_remaining_exec
        return max(0.0, estimated_all_done - time.time())

    def estimate_drain_wait(self, predictor: "LatencyPredictor") -> float:
        """估算 drain 完成所需时间（所有 small TP inflight 请求执行完毕）。
        取所有 small TP 节点中最大的剩余等待时间（drain 要等最慢的节点完成）。"""
        max_wait = 0.0
        for node in self.small_tp_nodes:
            wait = self.estimate_node_wait_time(node.client_ip_port, self.small_tp_size, predictor)
            max_wait = max(max_wait, wait)
        return max_wait

    def estimate_large_tp_remaining(self, predictor: "LatencyPredictor") -> float:
        """估算所有 large TP inflight 请求执行完毕所需时间。"""
        max_wait = 0.0
        for node in self.large_tp_nodes:
            wait = self.estimate_node_wait_time(node.client_ip_port, self.large_tp_size, predictor)
            max_wait = max(max_wait, wait)
        return max_wait

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
            f"FlexTPGroup(id={self.group_id}, state={self.state}, "
            f"small_tp={self.small_tp_size}x{len(self.small_tp_nodes)} [{small_details}], "
            f"large_tp={self.large_tp_size}x{len(self.large_tp_nodes)} [{large_details}], "
            f"pending_s={self.pending_small_tp}, pending_l={self.pending_large_tp})"
        )


class FlexTPSelector(PDSelector):
    """
    Flex TP 调度选择器（请求级排他，drain 状态机）。

    调度策略（三步式）：
    1. 预调度：对每种 TP 配置，假设调度并预测 TTFT，记录最优 group + node
    2. 决策：
       - slo_ttft 模式（slo_ttft is not None）：选择满足 SLO 的最小 TP 配置；
         若都不满足则选全局 TTFT 最小的配置
       - length_threshold 模式（slo_ttft is None）：input_token_num > threshold → 大 TP
    3. 执行：按预调度记录的 group + node 执行 drain 状态机并记录 inflight
    """

    def __init__(self, pd_manager, length_threshold: int = 8000,
                 slo_ttft: Optional[float] = None,
                 tp_latency_constants: Optional[Dict[int, float]] = None):
        super().__init__(pd_manager)
        self.length_threshold: int = length_threshold
        # SLO TTFT 目标（秒）；设置后覆盖 length_threshold 逻辑
        self.slo_ttft: Optional[float] = slo_ttft
        # group_id -> FlexTPGroup
        self.flex_groups: Dict[str, FlexTPGroup] = {}
        # node client_ip_port -> FlexTPGroup
        self.node_to_group: Dict[str, FlexTPGroup] = {}
        # node client_ip_port -> bool (True=large TP, False=small TP)
        self.node_is_large_tp: Dict[str, bool] = {}
        # 不属于任何 flex group 的普通 prefill 节点
        self.ungrouped_prefill_nodes: List[PD_Client_Obj] = []
        # decode 节点轮询索引
        self._decode_rr_index: int = 0
        # 延迟预测与 SLO 跟踪
        self.latency_predictor = LatencyPredictor(tp_latency_constants)
        self.slo_tracker = SLOTracker()

    def update_nodes(self, prefill_nodes, decode_nodes):
        super().update_nodes(prefill_nodes, decode_nodes)
        self._rebuild_flex_groups()

    def _rebuild_flex_groups(self):
        """根据注册的 prefill 节点信息，自动构建 FlexTPGroup"""
        # 按 (host_ip, shared_weight_master_port_start) 分组
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

        new_flex_groups: Dict[str, FlexTPGroup] = {}
        new_node_to_group: Dict[str, FlexTPGroup] = {}
        new_node_is_large_tp: Dict[str, bool] = {}

        for group_id, nodes in groups_map.items():
            # 获取该组所有 TP 大小
            tp_sizes = set(n.start_args.get("tp", 1) for n in nodes)
            if len(tp_sizes) < 2:
                # 只有一种 TP 大小，不形成 flex 组
                ungrouped.extend(nodes)
                continue

            min_tp = min(tp_sizes)
            max_tp = max(tp_sizes)

            # 复用已有的 group 对象以保留运行时状态（inflight 计数等）
            old_group = self.flex_groups.get(group_id)
            if old_group:
                group = old_group
                group.small_tp_nodes = []
                group.large_tp_nodes = []
            else:
                group = FlexTPGroup(group_id=group_id)

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
                f"FlexTP group [{group_id}]: small_tp={min_tp} ({len(group.small_tp_nodes)} nodes), "
                f"large_tp={max_tp} ({len(group.large_tp_nodes)} nodes)"
            )

        # 保留已移除节点的映射，直到该节点的 inflight 请求归零，
        # 避免 notify_request_done 找不到节点导致计数泄漏和死锁。
        for old_key, old_group in self.node_to_group.items():
            if old_key not in new_node_to_group:
                if old_group.node_inflight_requests.get(old_key, 0) > 0:
                    new_node_to_group[old_key] = old_group
                    new_node_is_large_tp[old_key] = self.node_is_large_tp.get(old_key, False)
                    logger.warning(
                        f"FlexTP: preserving stale mapping for {old_key} "
                        f"(group {old_group.group_id}, node_inflight="
                        f"{old_group.node_inflight_requests.get(old_key, 0)})"
                    )

        self.flex_groups = new_flex_groups
        self.node_to_group = new_node_to_group
        self.node_is_large_tp = new_node_is_large_tp
        self.ungrouped_prefill_nodes = ungrouped

        if ungrouped:
            logger.info(f"FlexTP: {len(ungrouped)} ungrouped prefill node(s)")

    # ---- 第一步：预调度（TTFT 预测 + 决策记录） ----

    def _predict_ttft_for_config(
        self, tp_type: str, token_num: int, arrival_time: Optional[float]
    ) -> Optional[Dict]:
        """预测将请求调度到指定 TP 类型（"small" / "large"）时的 TTFT。

        遍历所有 flex group，找到预测 TTFT 最低的 group 和节点。
        纯只读方法，不修改任何状态。

        Returns:
            dict with keys: tp_size, tp_type, ttft, queue_time, state_wait, node_wait,
                            exec_time, group_id, group_state, best_node,
                            group (FlexTPGroup), node (PD_Client_Obj)
            or None if no suitable group available.
        """
        predictor = self.latency_predictor
        best: Optional[Dict] = None

        for group in self.flex_groups.values():
            if tp_type == "small":
                tp_size = group.small_tp_size
                nodes = group.small_tp_nodes
            else:
                tp_size = group.large_tp_size
                nodes = group.large_tp_nodes

            if not nodes:
                continue

            exec_time = predictor.predict_execution_time(tp_size, token_num)
            queue_time = (time.time() - arrival_time) if arrival_time is not None else 0.0
            state_wait = 0.0
            node_wait = 0.0
            state_desc = group.state

            if tp_type == "small":
                if group.state == "small_tp_active":
                    best_node = self._pick_least_loaded(group, nodes)
                    node_wait = group.estimate_node_wait_time(
                        best_node.client_ip_port, tp_size, predictor
                    )
                elif group.state == "draining":
                    if (arrival_time is not None
                            and group.drain_start_time is not None
                            and arrival_time < group.drain_start_time):
                        best_node = self._pick_least_loaded(group, nodes)
                        node_wait = group.estimate_node_wait_time(
                            best_node.client_ip_port, tp_size, predictor
                        )
                        state_desc = "draining(early)"
                    else:
                        drain_remaining = group.estimate_drain_wait(predictor)
                        large_remaining = group.estimate_large_tp_remaining(predictor)
                        pending_large_exec = 0.0
                        if group.pending_large_tp > 0:
                            pending_large_exec = group.pending_large_tp * predictor.predict_execution_time(
                                group.large_tp_size, self.length_threshold
                            )
                        state_wait = drain_remaining + large_remaining + pending_large_exec
                        best_node = self._pick_least_loaded(group, nodes)
                        node_wait = 0.0
                        state_desc = "draining(late)"
                elif group.state == "large_tp_active":
                    large_remaining = group.estimate_large_tp_remaining(predictor)
                    pending_large_exec = 0.0
                    if group.pending_large_tp > 0:
                        pending_large_exec = group.pending_large_tp * predictor.predict_execution_time(
                            group.large_tp_size, self.length_threshold
                        )
                    state_wait = large_remaining + pending_large_exec
                    best_node = self._pick_least_loaded(group, nodes)
                    node_wait = 0.0
                else:
                    continue
            else:  # large
                if group.state == "large_tp_active":
                    best_node = self._pick_least_loaded(group, nodes)
                    node_wait = group.estimate_node_wait_time(
                        best_node.client_ip_port, tp_size, predictor
                    )
                elif group.state in ("small_tp_active", "draining"):
                    state_wait = group.estimate_drain_wait(predictor)
                    best_node = self._pick_least_loaded(group, nodes)
                    if group.pending_large_tp > 0:
                        node_wait = group.pending_large_tp * exec_time / max(len(nodes), 1)
                    else:
                        node_wait = 0.0
                else:
                    continue

            total_ttft = queue_time + state_wait + node_wait + exec_time
            candidate = {
                "tp_size": tp_size,
                "tp_type": tp_type,
                "ttft": total_ttft,
                "queue_time": queue_time,
                "state_wait": state_wait,
                "node_wait": node_wait,
                "exec_time": exec_time,
                "group_id": group.group_id,
                "group_state": state_desc,
                "best_node": best_node.client_ip_port,
                # 保留对象引用供第三步直接使用
                "group": group,
                "node": best_node,
            }
            if best is None or total_ttft < best["ttft"]:
                best = candidate

        return best

    # ---- 第二步：Best-effort 决策 ----

    def _select_best_config(
        self, predictions: List[Optional[Dict]], req_id: Optional[int]
    ) -> Dict:
        """从预调度结果中选择最优 TP 配置。

        slo_ttft 模式：选满足 SLO 的最小 TP（吞吐优先），都不满足则选 TTFT 最小的。
        length_threshold 模式：已在调用前决定 use_large_tp，此方法不会被调用。
        """
        valid = [p for p in predictions if p is not None]
        if not valid:
            raise RuntimeError(f"FlexTPSelector: req_id={req_id}, no valid TP config predictions")

        # 按 tp_size 升序排列（小 TP 优先 = 吞吐优先）
        valid.sort(key=lambda p: p["tp_size"])

        # 找满足 SLO 的最小 TP
        for p in valid:
            if p["ttft"] <= self.slo_ttft:
                logger.info(
                    f"FlexTP decision: req_id={req_id}, chose tp={p['tp_size']} "
                    f"(ttft={p['ttft'] * 1000:.1f}ms <= slo={self.slo_ttft * 1000:.1f}ms, "
                    f"smallest tp meeting SLO)"
                )
                return p

        # 都不满足 SLO，选最小TP
        best = min(valid, key=lambda p: p["tp_size"])
        logger.info(
            f"FlexTP decision: req_id={req_id}, no tp meets SLO={self.slo_ttft * 1000:.1f}ms, "
            f"chose tp={best['tp_size']} (ttft={best['ttft'] * 1000:.1f}ms, smallest tp)"
        )
        return best

    # ---- 第三步：执行调度（drain 状态机 + inflight 记录） ----

    async def _dispatch_to_node(
        self, decision: Dict, input_token_num: int,
        arrival_time: Optional[float] = None,
        req_id: Optional[int] = None,
    ) -> PD_Client_Obj:
        """按预调度决策执行：drain 状态机等待 + inflight 记录。"""
        group: FlexTPGroup = decision["group"]
        p_node: PD_Client_Obj = decision["node"]
        tp_type: str = decision["tp_type"]

        if tp_type == "large":
            return await self._dispatch_large_tp(group, p_node, input_token_num,
                                                 arrival_time=arrival_time, req_id=req_id)
        else:
            return await self._dispatch_small_tp(group, p_node, input_token_num,
                                                 arrival_time=arrival_time, req_id=req_id)

    async def _dispatch_large_tp(
        self, group: FlexTPGroup, p_node: PD_Client_Obj, input_token_num: int,
        arrival_time: Optional[float] = None, req_id: Optional[int] = None,
    ) -> PD_Client_Obj:
        """大 TP 调度执行：触发 drain 并等待小 TP 清零。"""
        async with group.condition:
            group.pending_large_tp += 1
            try:
                if group.state == "small_tp_active":
                    group.state = "draining"
                    group.drain_start_time = arrival_time or time.time()
                    logger.info(
                        f"FlexTP group [{group.group_id}]: req_id={req_id}, "
                        f"draining small TP (inflight={group.inflight_small_tp})"
                    )

                while group.inflight_small_tp > 0:
                    await group.condition.wait()

                group.state = "large_tp_active"
                # drain 后重新选择负载最低节点（drain 期间可能有变化）
                p_node = self._pick_least_loaded(group, group.large_tp_nodes)
                group.add_inflight(p_node.client_ip_port, input_token_num)
                group.pending_large_tp -= 1
            except BaseException:
                group.pending_large_tp -= 1
                if group.pending_large_tp == 0 and group.inflight_large_tp == 0:
                    group.state = "small_tp_active"
                    group.drain_start_time = None
                    group.condition.notify_all()
                    logger.info(
                        f"FlexTP group [{group.group_id}]: req_id={req_id}, "
                        f"large TP request cancelled during drain, resuming small TP"
                    )
                else:
                    logger.info(
                        f"FlexTP group [{group.group_id}]: req_id={req_id}, "
                        f"large TP request cancelled during drain "
                        f"(pending_l={group.pending_large_tp}, inflight_l={group.inflight_large_tp})"
                    )
                raise

        logger.info(
            f"FlexTP group [{group.group_id}]: req_id={req_id}, "
            f"large TP request dispatched to {p_node.client_ip_port} "
            f"(inflight_l={group.inflight_large_tp}, tokens_l={group.inflight_tokens_large_tp})"
        )
        return p_node

    async def _dispatch_small_tp(
        self, group: FlexTPGroup, p_node: PD_Client_Obj, input_token_num: int,
        arrival_time: Optional[float] = None, req_id: Optional[int] = None,
    ) -> PD_Client_Obj:
        """小 TP 调度执行：处理 draining/large_tp_active 阻塞等待。"""
        if group.state == "small_tp_active":
            pass  # 直接可用
        elif (group.state == "draining"
              and arrival_time is not None
              and group.drain_start_time is not None
              and arrival_time < group.drain_start_time):
            # 早到请求，允许放行
            logger.info(
                f"FlexTP: req_id={req_id}, small TP request (arrival={arrival_time:.6f}) allowed into draining "
                f"group [{group.group_id}] (drain_start={group.drain_start_time:.6f})"
            )
        else:
            # 需要等待 group 恢复到 small_tp_active
            group.pending_small_tp += 1
            logger.warning(
                f"FlexTP: req_id={req_id}, small TP request blocked waiting for "
                f"group [{group.group_id}] (state={group.state}, pending_small={group.pending_small_tp})"
            )
            try:
                async with group.condition:
                    while group.state != "small_tp_active":
                        await group.condition.wait()
            finally:
                group.pending_small_tp -= 1

        # 阻塞等待后重新选择负载最低节点
        p_node = self._pick_least_loaded(group, group.small_tp_nodes)
        group.add_inflight(p_node.client_ip_port, input_token_num)
        return p_node

    # ---- 核心调度接口 ----

    async def async_select_p_d_node(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        input_token_num: Optional[int] = None,
        arrival_time: Optional[float] = None,
        req_id: Optional[int] = None,
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        """异步版本的节点选择，支持排他等待"""
        if not self.prefill_nodes or not self.decode_nodes:
            raise RuntimeError(
                f"FlexTPSelector: req_id={req_id} no available nodes "
                f"(prefill={len(self.prefill_nodes)}, decode={len(self.decode_nodes)})"
            )

        d_node = self._pick_decode_node()

        if not self.flex_groups:
            p_node = random.choice(self.prefill_nodes)
            return p_node, d_node

        token_num = input_token_num or 0

        logger.info(
            f"FlexTPSelector: req_id={req_id}, input_tokens={token_num}, arrival_time={arrival_time:.6f}, "
            f"starting TP selection among {len(self.flex_groups)} flex groups"
        )

        logger.info(f"FlexTP group status: req_id={req_id}, {self.flex_groups}")

        # ---- 第一步：预调度，预测每种 TP 配置的 TTFT ----
        small_pred = self._predict_ttft_for_config("small", token_num, arrival_time)
        large_pred = self._predict_ttft_for_config("large", token_num, arrival_time)

        def _fmt(pred: Optional[Dict]) -> str:
            if pred is None:
                return "N/A (no nodes)"
            return (
                f"queue={pred['queue_time'] * 1000:.1f}ms + "
                f"state_wait={pred['state_wait'] * 1000:.1f}ms + "
                f"node_wait={pred['node_wait'] * 1000:.1f}ms + "
                f"exec={pred['exec_time'] * 1000:.1f}ms = "
                f"{pred['ttft'] * 1000:.1f}ms  "
                f"[group={pred['group_id']}, state={pred['group_state']}, "
                f"node={pred['best_node']}]"
            )

        small_ttft = small_pred["ttft"] if small_pred else float("inf")
        large_ttft = large_pred["ttft"] if large_pred else float("inf")
        small_tp = small_pred["tp_size"] if small_pred else "?"
        large_tp = large_pred["tp_size"] if large_pred else "?"
        delta = small_ttft - large_ttft

        logger.info(
            f"FlexTP TTFT prediction: req_id={req_id}, input_tokens={token_num}\n"
            f"  small_tp(tp={small_tp}): {_fmt(small_pred)}\n"
            f"  large_tp(tp={large_tp}): {_fmt(large_pred)}\n"
            f"  delta(small-large): {delta * 1000:+.1f}ms "
            f"({'large_tp faster' if delta > 0 else 'small_tp faster'})"
        )

        # ---- 第二步：决策 ----
        if self.slo_ttft is not None:
            # SLO 模式：best-effort 选择满足 SLO 的最小 TP
            decision = self._select_best_config([small_pred, large_pred], req_id)
        else:
            # length_threshold 模式：与原逻辑一致
            use_large_tp = input_token_num is not None and input_token_num > self.length_threshold
            if use_large_tp:
                decision = large_pred if large_pred is not None else small_pred
            else:
                decision = small_pred if small_pred is not None else large_pred
            if decision is None:
                raise RuntimeError(f"FlexTPSelector: req_id={req_id}, no valid TP config")

        # ---- 第三步：按决策执行 ----
        p_node = await self._dispatch_to_node(decision, token_num,
                                              arrival_time=arrival_time, req_id=req_id)

        # SLO 跟踪记录
        tp_size = self._get_tp_size(p_node)
        queue_time = (time.time() - arrival_time) if arrival_time is not None else 0.0
        exec_time = self.latency_predictor.predict_execution_time(tp_size, token_num)
        all_inflight_wait = self.node_to_group.get(p_node.client_ip_port)
        if all_inflight_wait is not None:
            all_inflight_wait = all_inflight_wait.estimate_node_wait_time(
                p_node.client_ip_port, tp_size, self.latency_predictor
            )
        else:
            all_inflight_wait = 0.0
        prefill_wait_time = max(0.0, all_inflight_wait - exec_time)
        predicted_ttft = queue_time + prefill_wait_time + exec_time

        self.slo_tracker.record_predicted(tp_size, token_num, queue_time, exec_time, predicted_ttft)
        logger.info(
            f"FlexTP select: req_id={req_id}, input_tokens={input_token_num}, tp={tp_size}, "
            f"p_node={p_node.client_ip_port}, "
            f"predicted_ttft={predicted_ttft * 1000:.1f}ms "
            f"(queue={queue_time * 1000:.1f}ms + prefill_wait={prefill_wait_time * 1000:.1f}ms "
            f"+ exec={exec_time * 1000:.1f}ms)"
        )
        return p_node, d_node

    async def notify_request_done(self, p_node: PD_Client_Obj, input_token_num: int = 0,
                                  actual_ttft: Optional[float] = None,
                                  req_id: Optional[int] = None):
        """请求完成后调用，更新在途计数和 token 数并触发状态切换。
        actual_ttft: 实际 TTFT（秒），用于 SLO 跟踪校准。"""
        if p_node is None:
            return

        group = self.node_to_group.get(p_node.client_ip_port)
        if group is None:
            return

        is_large = self.node_is_large_tp.get(p_node.client_ip_port, False)

        # 记录实际 TTFT 用于 SLO 跟踪
        if actual_ttft is not None:
            tp_size = self._get_tp_size(p_node)
            self.slo_tracker.record_actual(tp_size, actual_ttft)
            logger.info(
                f"FlexTP SLO: req_id={req_id}, actual_ttft={actual_ttft * 1000:.1f}ms, tp={tp_size}, "
                f"node={p_node.client_ip_port}"
            )

        async with group.condition:
            group.remove_inflight(p_node.client_ip_port, input_token_num)
            if is_large:
                if group.inflight_large_tp == 0 and group.pending_large_tp == 0:
                    group.state = "small_tp_active"
                    group.drain_start_time = None
                    group.condition.notify_all()
                    logger.info(f"FlexTP group [{group.group_id}]: req_id={req_id}, large TP done, resuming small TP")
            else:
                if group.inflight_small_tp == 0 and group.state == "draining":
                    group.condition.notify_all()
                    logger.info(f"FlexTP group [{group.group_id}]: req_id={req_id}, small TP drained (inflight=0)")


    # ---- 辅助方法 ----

    def _pick_decode_node(self) -> PD_Client_Obj:
        """轮询方式选择 decode 节点"""
        self._decode_rr_index = self._decode_rr_index % len(self.decode_nodes)
        d_node = self.decode_nodes[self._decode_rr_index]
        self._decode_rr_index += 1
        return d_node

    @staticmethod
    def _get_tp_size(node: PD_Client_Obj) -> int:
        """从节点的 start_args 中获取 TP 大小"""
        sa = node.start_args
        return sa.get("tp", 1) if isinstance(sa, dict) else 1

    def _pick_least_loaded(self, group: FlexTPGroup, nodes: List[PD_Client_Obj]) -> PD_Client_Obj:
        """选择负载最低（在途 token 数最少）的节点"""
        return min(nodes, key=lambda n: group.node_inflight_tokens.get(n.client_ip_port, 0))

    def select_p_d_node(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        """同步回退方法，flex TP 模式下不应使用"""
        raise NotImplementedError(
            "FlexTPSelector requires async_select_p_d_node. "
            "Ensure PDManager.select_p_d_node is using the async path."
        )
