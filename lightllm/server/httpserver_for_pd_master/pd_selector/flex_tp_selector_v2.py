"""
Flex TP Selector V2: 无 drain 状态机，所有 TP 实例常驻。

核心设计：
- 所有 TP 实例始终运行，大 TP 请求到达时立即派发
- 通过 contention budget 对小 TP 请求做流控，保证大 TP 请求的 SLO
- MPS 下 GPU 吞吐守恒：每个小请求对大请求的 slowdown = 该小请求的独占执行时间

调度流程：
1. TP 选择（Phase 1）：从小到大遍历候选 TP，预测 TTFT，选满足 SLO 的最小 TP
2. 大 TP 派发（Phase 2）：直接派发，注册 active large-TP instance 并记录 budget
3. 小 TP 准入控制（Phase 3）：检查 contention budget，budget 不足则 block 等待大 TP 完成
"""

import asyncio
import random
import time
import collections
from typing import Literal, Union, List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.utils.log_utils import init_logger
from .pd_selector import PDSelector

logger = init_logger(__name__)


# ===========================================================================
#  Latency Model V2: 4-constant model per TP
# ===========================================================================


class LatencyModelV2:
    """
    每种 TP 配置独立拟合 4 常数 (a, b, c, d)：

        latency({s_i}, tp) = max(a * sum(s_i)/tp + b * sum(s_i^2)/tp, c) + d

    对单个请求（一个 seq_len）简化为：
        latency(seq_len, tp) = max(a * seq_len/tp + b * seq_len^2/tp, c) + d
    """

    # 默认常数 (a, b, c, d)，可通过 set_constants 覆盖
    DEFAULT_CONSTANTS: Dict[int, Tuple[float, float, float, float]] = {
        1: (0.00024, 1e-8, 0.005, 0.005),
        2: (0.00018, 8e-9, 0.004, 0.004),
        4: (0.00012, 5e-9, 0.003, 0.003),
        8: (0.00008, 3e-9, 0.002, 0.002),
    }

    def __init__(self, constants: Optional[Dict[int, Tuple[float, float, float, float]]] = None):
        self._constants: Dict[int, Tuple[float, float, float, float]] = dict(self.DEFAULT_CONSTANTS)
        if constants:
            self._constants.update(constants)

    def set_constants(self, tp_size: int, a: float, b: float, c: float, d: float):
        self._constants[tp_size] = (a, b, c, d)

    def _get_constants(self, tp_size: int) -> Tuple[float, float, float, float]:
        if tp_size in self._constants:
            return self._constants[tp_size]
        # 简单外推：取已知最大 TP 的常数，按比例缩放 a 和 b
        known = max(self._constants.keys())
        a0, b0, c0, d0 = self._constants[known]
        ratio = known / tp_size
        return (a0 * ratio, b0 * ratio, c0, d0)

    def predict_exclusive(self, seq_len: int, tp_size: int) -> float:
        """预测单个请求在独占 GPU 下的 prefill 执行时间（秒）"""
        a, b, c, d = self._get_constants(tp_size)
        compute = a * seq_len / tp_size + b * (seq_len ** 2) / tp_size
        return max(compute, c) + d

    def predict_batch(self, seq_lens: List[int], tp_size: int) -> float:
        """预测一批请求在独占 GPU 下的总 prefill 执行时间（秒）
        latency = max(a * sum(s)/tp + b * sum(s^2)/tp, c) + d
        """
        if not seq_lens:
            return 0.0
        a, b, c, d = self._get_constants(tp_size)
        sum_s = sum(seq_lens)
        sum_s2 = sum(s * s for s in seq_lens)
        compute = a * sum_s / tp_size + b * sum_s2 / tp_size
        return max(compute, c) + d


# ===========================================================================
#  Active Large-TP Instance Tracker
# ===========================================================================


@dataclass
class ActiveLargeTPInstance:
    """跟踪一个正在执行的大 TP 请求"""
    req_id: Optional[int]
    tp_size: int
    gpu_set: frozenset  # 该实例占用的 GPU 集合
    group_id: str
    exclusive_latency: float  # 独占执行时间
    budget: float  # contention budget = T_slo - exclusive_latency
    budget_remaining: float  # 剩余 budget
    start_time: float  # 开始执行的墙钟时间
    node_key: str  # 节点标识

    def wall_clock_remaining(self) -> float:
        """预计剩余墙钟执行时间。
        remain(I) = start_time + exclusive_latency + (budget - budget_remaining) - now
        简化：预计完成时间 = start_time + exclusive_latency + 已消耗 budget
        """
        budget_consumed = self.budget - self.budget_remaining
        estimated_finish = self.start_time + self.exclusive_latency + budget_consumed
        return max(0.0, estimated_finish - time.time())


# ===========================================================================
#  FlexTPGroupV2: 无状态机的 GPU 组
# ===========================================================================


class FlexTPGroupV2:
    """一组共享同一批 GPU 的不同 TP 大小的 prefill 节点（V2: 无 drain 状态机）"""

    def __init__(self, group_id: str):
        self.group_id: str = group_id
        # tp_size -> [PD_Client_Obj, ...] 所有 TP 配置的节点
        self.tp_nodes: Dict[int, List[PD_Client_Obj]] = {}
        # 该组所有 TP 大小，排序后存储
        self.tp_sizes: List[int] = []
        # 该组的 GPU 集合 (由节点的 gpu_ids 决定)
        self.gpu_set: frozenset = frozenset()

        # 每个节点的在途请求数
        self.node_inflight_requests: Dict[str, int] = {}

        # 队列剩余时间估计，由于受contention影响，记录exclusive等价延迟和时间基线（上一次warp），并在large_tp更新时进行warp
        self.node_queue_remains_exclusive: Dict[str, float] = {}  # 节点上排队请求的 exclusive predicted_latency 之和
        self.node_queue_last_warp_time: Dict[str, float] = {}  # 上一次 warp 的时间戳

        # 活跃大 TP 实例列表
        self.active_large_tp: List[ActiveLargeTPInstance] = []

        # 大 TP 完成事件：用于唤醒被 block 的小 TP 请求
        self.large_tp_done_event: asyncio.Event = asyncio.Event()

        # 用于保护共享状态的锁
        self.lock: asyncio.Lock = asyncio.Lock()

    @property
    def min_tp(self) -> int:
        return self.tp_sizes[0] if self.tp_sizes else 0

    def add_inflight(self, node_key: str, predicted_latency: float):
        current_inflight = self.node_inflight_requests.get(node_key, 0)
        if not current_inflight:
            # 第一个请求到达，初始化 warp 时间基线
            self.node_queue_last_warp_time[node_key] = time.time()
        self.node_inflight_requests[node_key] = current_inflight + 1
        self.node_queue_remains_exclusive[node_key] = self.node_queue_remains_exclusive.get(node_key, 0.0) + predicted_latency

    def remove_inflight(self, node_key: str, predicted_latency: float):
        new_req = max(0, self.node_inflight_requests.get(node_key, 0) - 1)
        self.node_inflight_requests[node_key] = new_req
        if new_req == 0:
            self.node_queue_remains_exclusive[node_key] = 0.0
        else:
            self.node_queue_remains_exclusive[node_key] = max(0.0, self.node_queue_remains_exclusive.get(node_key, 0.0) - predicted_latency)
        self.node_queue_last_warp_time[node_key] = time.time()

    def get_queue_delay(self, node_key: str) -> float:
        """获取节点上排队请求的 predicted_latency 之和"""
        if self.node_inflight_requests.get(node_key, 0) == 0:
            logger.info(f"Node {node_key} queue delay: = 0s (empty)")
            return 0.0
        t_old_remain_exclusive = self.node_queue_remains_exclusive.get(node_key, 0.0)
        n_contention = self.get_current_contention()
        t_old_remain = t_old_remain_exclusive * n_contention
        t_last_warp = self.node_queue_last_warp_time.get(node_key, 0.0)
        t_remain = max(0.0, t_old_remain - (time.time() - t_last_warp))
        logger.info(f"Node {node_key} queue delay: = {t_remain:.3f}s (exclusive remain={t_old_remain_exclusive:.3f}s, contention={n_contention}, t_old_remain={t_old_remain:.3f}s, last warp={time.time() - t_last_warp:.3f}s)")
        return t_remain

    def warp_all_queue_delays(self):
        """当大 TP 状态发生变化时，更新所有节点的 queue delay 基线"""
        for node_key in self.node_inflight_requests.keys():
            if self.node_inflight_requests[node_key] > 0:
                wall_clock_passed = time.time() - self.node_queue_last_warp_time.get(node_key, 0.0)
                exclusive_reduction = wall_clock_passed / self.get_current_contention()  # 粗略估计 exclusive 已经减少的部分
                self.node_queue_last_warp_time[node_key] = time.time()
                self.node_queue_remains_exclusive[node_key] = max(0.0, self.node_queue_remains_exclusive.get(node_key, 0.0) - exclusive_reduction)
                logger.info(f"Warping node {node_key} queue delay baseline at {wall_clock_passed:.3f}s since last warp, reducing exclusive remain by {exclusive_reduction:.3f}s, new exclusive remain={self.node_queue_remains_exclusive[node_key]:.3f}s")

    def get_current_contention(self, exclude_tp_size: Optional[int] = None) -> int:
        """获取 GPU 上当前活跃进程数"""
        # TODO: 此处为近似，目前无法获得准确映射关系
        return sum(any(self.node_inflight_requests.get(n.client_ip_port, 0) for n in nodes)
                   for ts, nodes in self.tp_nodes.items() if ts != exclude_tp_size)

    def get_min_budget_for_gpu_set(self, exclude_req_id: Optional[int] = None) -> float:
        """获取该组 GPU 上所有活跃大 TP 实例的最低剩余 budget"""
        if not self.active_large_tp:
            return float("inf")
        min_budget = float("inf")
        for inst in self.active_large_tp:
            if exclude_req_id is not None and inst.req_id == exclude_req_id:
                continue
            min_budget = min(min_budget, inst.budget_remaining)
        return min_budget

    def register_active_large_tp(self, instance: ActiveLargeTPInstance):
        self.warp_all_queue_delays()
        if instance not in self.active_large_tp:
            self.active_large_tp.append(instance)
        # 清除 event，表示有活跃大 TP
        self.large_tp_done_event.clear()

    def unregister_active_large_tp(self, req_id: Optional[int], node_key: str):
        self.warp_all_queue_delays()
        self.active_large_tp = [
            inst for inst in self.active_large_tp
            if not (inst.req_id == req_id and inst.node_key == node_key)
        ]
        if not self.active_large_tp:
            # 所有大 TP 完成，唤醒等待的小 TP 请求
            self.large_tp_done_event.set()

    def deduct_budget(self, delta: float):
        """从所有活跃大 TP 实例扣减 budget"""
        for inst in self.active_large_tp:
            inst.budget_remaining -= delta

    def __repr__(self):
        tp_details = []
        for tp in self.tp_sizes:
            nodes = self.tp_nodes.get(tp, [])
            node_info = ", ".join(
                f"{n.client_ip_port}={self.node_inflight_requests.get(n.client_ip_port, 0)}"
                f"(qd={self.node_queue_delay.get(n.client_ip_port, 0.0) * 1000:.1f}ms)"
                for n in nodes
            )
            tp_details.append(f"tp{tp}x{len(nodes)}[{node_info}]")
        active = len(self.active_large_tp)
        return (
            f"FlexTPGroupV2(id={self.group_id}, {', '.join(tp_details)}, "
            f"active_large={active})"
        )


# ===========================================================================
#  FlexTPSelectorV2
# ===========================================================================


class FlexTPSelectorV2(PDSelector):
    """
    Flex TP 调度选择器 V2（无 drain 状态机，contention budget 流控）。

    - 所有 TP 实例常驻运行
    - 大 TP 请求立即派发，通过 contention budget 限制小 TP 请求的并发
    - MPS 下 GPU 吞吐守恒：小请求对大请求的 slowdown = 小请求独占执行时间
    """
    SLO_IMMPOSIBLE_FALLBACK: Literal['best_effort', 'max_throughput'] = 'max_throughput'  # 当所有 TP 都无法满足 SLO 时的 fallback 策略

    def __init__(self, pd_manager, slo_ttft: float = 5.0,
                 latency_constants: Optional[Dict[int, Tuple[float, float, float, float]]] = None):
        super().__init__(pd_manager)
        self.slo_ttft: float = slo_ttft  # T_slo (秒)
        # group_id -> FlexTPGroupV2
        self.flex_groups: Dict[str, FlexTPGroupV2] = {}
        # node client_ip_port -> FlexTPGroupV2
        self.node_to_group: Dict[str, FlexTPGroupV2] = {}
        # node client_ip_port -> tp_size
        self.node_tp_size: Dict[str, int] = {}
        # 不属于任何 flex group 的普通 prefill 节点
        self.ungrouped_prefill_nodes: List[PD_Client_Obj] = []
        # decode 节点轮询索引
        self._decode_rr_index: int = 0
        # 延迟模型
        self.latency_model = LatencyModelV2(latency_constants)
        # 每个请求的 predicted_latency 记录，用于 remove_inflight 时准确扣减
        # req_id -> (node_key, predicted_latency, tp_size, group)
        self._req_dispatch_info: Dict[int, Tuple[str, float, int, FlexTPGroupV2]] = {}

    def update_nodes(self, prefill_nodes, decode_nodes):
        super().update_nodes(prefill_nodes, decode_nodes)
        self._rebuild_flex_groups()

    def _rebuild_flex_groups(self):
        """根据注册的 prefill 节点信息，自动构建 FlexTPGroupV2"""
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

        new_flex_groups: Dict[str, FlexTPGroupV2] = {}
        new_node_to_group: Dict[str, FlexTPGroupV2] = {}
        new_node_tp_size: Dict[str, int] = {}

        for group_id, nodes in groups_map.items():
            tp_sizes = set(n.start_args.get("tp", 1) for n in nodes)
            if len(tp_sizes) < 2:
                ungrouped.extend(nodes)
                continue

            # 复用已有 group 保留运行时状态
            old_group = self.flex_groups.get(group_id)
            if old_group:
                group = old_group
                group.tp_nodes = {}
            else:
                group = FlexTPGroupV2(group_id=group_id)

            group.tp_sizes = sorted(tp_sizes)

            for n in nodes:
                tp = n.start_args.get("tp", 1)
                group.tp_nodes.setdefault(tp, []).append(n)
                new_node_to_group[n.client_ip_port] = group
                new_node_tp_size[n.client_ip_port] = tp

            new_flex_groups[group_id] = group
            tp_info = ", ".join(
                f"tp{tp}={len(group.tp_nodes.get(tp, []))}nodes"
                for tp in group.tp_sizes
            )
            logger.info(f"FlexTP V2 group [{group_id}]: {tp_info}")

        # 保留已移除节点的映射（有 inflight 请求时）
        for old_key, old_group in self.node_to_group.items():
            if old_key not in new_node_to_group:
                if old_group.node_inflight_requests.get(old_key, 0) > 0:
                    new_node_to_group[old_key] = old_group
                    new_node_tp_size[old_key] = self.node_tp_size.get(old_key, 1)

        self.flex_groups = new_flex_groups
        self.node_to_group = new_node_to_group
        self.node_tp_size = new_node_tp_size
        self.ungrouped_prefill_nodes = ungrouped

        if ungrouped:
            logger.info(f"FlexTP V2: {len(ungrouped)} ungrouped prefill node(s)")

    # ---- Phase 1: TP 选择 ----

    def _select_tp(self, seq_len: int) -> Tuple[int, FlexTPGroupV2, PD_Client_Obj]:
        """
        从小到大遍历候选 TP 组，选满足 SLO 的最小 TP。

        对每种 TP：
        1. exclusive = predict_exclusive(seq_len, tp)
        2. n = 该 tp 组 GPU 上当前活跃进程数
        3. exec_time = exclusive * max(n, 1)  (MPS 平分资源)
        4. predicted_ttft = exec_time + queue_delay
        5. 如果 predicted_ttft <= T_slo → 选这个 tp

        TP 选择是 one-shot decision，后续不重新评估。
        """
        best_tp: Optional[int] = None
        best_group: Optional[FlexTPGroupV2] = None
        best_node: Optional[PD_Client_Obj] = None
        best_ttft: float = float("inf")
        fallback_tp: Optional[int] = None
        fallback_group: Optional[FlexTPGroupV2] = None
        fallback_node: Optional[PD_Client_Obj] = None
        fallback_ttft: float = float("inf")
        minimal_tp: Optional[int] = None
        minimal_group: Optional[FlexTPGroupV2] = None
        minimal_node: Optional[PD_Client_Obj] = None
        minimal_ttft: float = float("inf")

        # 收集所有可用 TP 大小（跨所有 group），从小到大
        all_tp_sizes = sorted(set(
            tp for group in self.flex_groups.values() for tp in group.tp_sizes
        ))

        logger.info(f"FlexTP V2 SELECT_TP: START")
        for tp_size in all_tp_sizes:
            exclusive = self.latency_model.predict_exclusive(seq_len, tp_size)
            logger.info(f"FlexTP V2 SELECT_TP: TP={tp_size}")
            logger.info(f"FlexTP V2 SELECT_TP: predicted exclusive latency={exclusive * 1000:.1f}ms")

            # 遍历所有含该 TP 的 group，选 queue_delay 最小的实例
            for group in self.flex_groups.values():
                nodes = group.tp_nodes.get(tp_size)
                if not nodes:
                    continue
                logger.info(f"FlexTP V2 SELECT_TP: Checking group nodes: {[n.client_ip_port for n in nodes]}")

                # 选 queue_delay 最小的节点
                candidate_node = min(
                    nodes, key=lambda n: group.get_queue_delay(n.client_ip_port)
                )
                node_key = candidate_node.client_ip_port

                logger.info(f"FlexTP V2 SELECT_TP: Selected candidate node: {node_key}")

                n = 1 + group.get_current_contention(exclude_tp_size=tp_size)
                exec_time = exclusive * n
                queue_delay = group.get_queue_delay(node_key)
                predicted_ttft = exec_time + queue_delay

                logger.info(
                    f"FlexTP V2 SELECT_TP: TP={tp_size}, node={node_key}, n={n}, exec_time={exec_time * 1000:.1f}ms, "
                    f"queue_delay={queue_delay * 1000:.1f}ms, predicted_ttft={predicted_ttft * 1000:.1f}ms"
                )

                if not minimal_tp:
                    minimal_tp = tp_size
                    minimal_group = group
                    minimal_node = candidate_node
                    minimal_ttft = predicted_ttft

                # 记录为 fallback（全局最小 TTFT）
                if predicted_ttft < fallback_ttft:
                    fallback_ttft = predicted_ttft
                    fallback_tp = tp_size
                    fallback_group = group
                    fallback_node = candidate_node

                if predicted_ttft <= self.slo_ttft:
                    if best_tp is None or tp_size < best_tp or (tp_size == best_tp and predicted_ttft < best_ttft):
                        best_tp = tp_size
                        best_group = group
                        best_node = candidate_node
                        best_ttft = predicted_ttft
                    # 找到满足 SLO 的最小 TP，break 内层 group 循环
                    break

            # 如果这个 TP 大小已经满足 SLO，不再看更大的 TP
            if best_tp == tp_size:
                break

        if best_tp is not None:
            logger.info(
                f"FlexTP V2 TP select: tp={best_tp}, group={best_group.group_id}, "
                f"node={best_node.client_ip_port}, predicted_ttft={best_ttft * 1000:.1f}ms "
                f"(<= slo={self.slo_ttft * 1000:.1f}ms)"
            )
            return best_tp, best_group, best_node

        # 没有满足 SLO 的 TP
        # 策略1: fallback（最小 TTFT）
        if self.SLO_IMMPOSIBLE_FALLBACK == 'best_effort':
            logger.warning(
                f"FlexTP V2 TP select: no TP meets SLO={self.slo_ttft * 1000:.1f}ms, "
                f"fallback tp={fallback_tp}, group={fallback_group.group_id}, "
                f"node={fallback_node.client_ip_port}, predicted_ttft={fallback_ttft * 1000:.1f}ms"
            )
            return fallback_tp, fallback_group, fallback_node

        # 策略2: 直接选最小 TP（不考虑 predicted_ttft），优化吞吐
        elif self.SLO_IMMPOSIBLE_FALLBACK == 'max_throughput':
            logger.warning(
                f"FlexTP V2 TP select: no TP meets SLO={self.slo_ttft * 1000:.1f}ms, "
                f"selecting smallest tp={minimal_tp} for better throughput, "
                f"group={minimal_group.group_id}, node={minimal_node.client_ip_port}, "
                f"predicted_ttft={minimal_ttft * 1000:.1f}ms"
            )
            return minimal_tp, minimal_group, minimal_node
        else:
            raise ValueError(f"Invalid SLO_IMMPOSIBLE_FALLBACK strategy: {self.SLO_IMMPOSIBLE_FALLBACK}")



    # ---- Phase 2 & 3: 派发 ----

    async def _dispatch(
        self, tp_size: int, group: FlexTPGroupV2, p_node: PD_Client_Obj,
        seq_len: int, req_id: Optional[int] = None,
    ) -> PD_Client_Obj:
        """派发请求到选定的 TP 配置。

        大 TP (非最小 TP): 直接派发，注册 active large-TP instance
        小 TP (最小 TP): 准入控制，检查 contention budget
        """
        is_min_tp = (tp_size == group.min_tp)

        if not is_min_tp:
            return await self._dispatch_large_tp(tp_size, group, p_node, seq_len, req_id)
        else:
            return await self._dispatch_small_tp(tp_size, group, p_node, seq_len, req_id)

    async def _dispatch_large_tp(
        self, tp_size: int, group: FlexTPGroupV2, p_node: PD_Client_Obj,
        seq_len: int, req_id: Optional[int] = None,
    ) -> PD_Client_Obj:
        """Phase 2: 大 TP 请求直接派发。

        1. budget = T_slo - exclusive_latency
        2. 注册为 active large-TP instance
        3. 直接派发
        """
        exclusive = self.latency_model.predict_exclusive(seq_len, tp_size)
        budget = self.slo_ttft - exclusive

        async with group.lock:
            instance = ActiveLargeTPInstance(
                req_id=req_id,
                tp_size=tp_size,
                gpu_set=group.gpu_set,
                group_id=group.group_id,
                exclusive_latency=exclusive,
                budget=budget,
                budget_remaining=budget,
                start_time=time.time(),
                node_key=p_node.client_ip_port,
            )
            group.register_active_large_tp(instance)
            group.add_inflight(p_node.client_ip_port, exclusive)

        # 记录 dispatch info 用于 notify_request_done
        if req_id is not None:
            self._req_dispatch_info[req_id] = (p_node.client_ip_port, exclusive, tp_size, group)

        logger.info(
            f"FlexTP V2 dispatch large: req_id={req_id}, tp={tp_size}, "
            f"node={p_node.client_ip_port}, group={group.group_id}, "
            f"exclusive={exclusive * 1000:.1f}ms, budget={budget * 1000:.1f}ms"
        )
        return p_node

    async def _dispatch_small_tp(
        self, tp_size: int, group: FlexTPGroupV2, p_node: PD_Client_Obj,
        seq_len: int, req_id: Optional[int] = None,
    ) -> PD_Client_Obj:
        """Phase 3: 小 TP 请求准入控制。

        如果没有活跃大 TP 实例，直接派发（退化为无流控）。
        否则：
        1. delta = min(exclusive_latency, min(remain(I) for 所有活跃大 TP I))
        2. 如果 delta <= min_budget → 准入，扣 delta
        3. 如果 delta > min_budget → block，await 大 TP 完成后重试
        """
        exclusive = self.latency_model.predict_exclusive(seq_len, tp_size)

        while True:
            async with group.lock:
                if not group.active_large_tp:
                    # 没有活跃大 TP，直接派发
                    # 重新选 queue_delay 最小的节点
                    nodes = group.tp_nodes.get(tp_size, [])
                    if nodes:
                        p_node = min(nodes, key=lambda n: group.get_queue_delay(n.client_ip_port))
                    group.add_inflight(p_node.client_ip_port, exclusive)
                    if req_id is not None:
                        self._req_dispatch_info[req_id] = (p_node.client_ip_port, exclusive, tp_size, group)
                    logger.info(
                        f"FlexTP V2 dispatch small (no contention): req_id={req_id}, tp={tp_size}, "
                        f"node={p_node.client_ip_port}, group={group.group_id}"
                    )
                    return p_node

                # 有活跃大 TP，计算 delta 和 min_budget
                min_remain = min(inst.wall_clock_remaining() for inst in group.active_large_tp)
                delta = min(exclusive, min_remain)
                min_budget = group.get_min_budget_for_gpu_set()

                if delta <= min_budget:
                    # 准入：扣 delta
                    group.deduct_budget(delta)
                    # 重新选节点
                    nodes = group.tp_nodes.get(tp_size, [])
                    if nodes:
                        p_node = min(nodes, key=lambda n: group.get_queue_delay(n.client_ip_port))
                    group.add_inflight(p_node.client_ip_port, exclusive)
                    if req_id is not None:
                        self._req_dispatch_info[req_id] = (p_node.client_ip_port, exclusive, tp_size, group)
                    logger.info(
                        f"FlexTP V2 dispatch small (admitted): req_id={req_id}, tp={tp_size}, "
                        f"node={p_node.client_ip_port}, delta={delta * 1000:.1f}ms, "
                        f"min_budget={min_budget * 1000:.1f}ms, group={group.group_id}"
                    )
                    return p_node

                # Budget 不足，需要 block
                logger.info(
                    f"FlexTP V2 small TP blocked: req_id={req_id}, tp={tp_size}, "
                    f"delta={delta * 1000:.1f}ms > min_budget={min_budget * 1000:.1f}ms, "
                    f"waiting for large TP completion in group={group.group_id}"
                )

            # 释放锁后 await 大 TP 完成事件
            group.large_tp_done_event.clear()
            await group.large_tp_done_event.wait()
            # 大 TP 完成后重试准入检查

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
        """异步节点选择（V2：无 drain 状态机）"""
        if not self.prefill_nodes or not self.decode_nodes:
            raise RuntimeError(
                f"FlexTPSelectorV2: req_id={req_id} no available nodes "
                f"(prefill={len(self.prefill_nodes)}, decode={len(self.decode_nodes)})"
            )

        d_node = self._pick_decode_node()

        if not self.flex_groups:
            p_node = random.choice(self.prefill_nodes)
            return p_node, d_node

        seq_len = input_token_num or 0

        logger.info(
            f"FlexTPSelectorV2: req_id={req_id}, seq_len={seq_len}, "
            f"groups={len(self.flex_groups)}, slo={self.slo_ttft * 1000:.1f}ms"
        )

        # Phase 1: TP 选择 (one-shot, 不重评估)
        tp_size, group, p_node = self._select_tp(seq_len)

        # Phase 2/3: 派发
        p_node = await self._dispatch(tp_size, group, p_node, seq_len, req_id)

        return p_node, d_node

    async def notify_request_done(self, p_node: PD_Client_Obj, input_token_num: int = 0,
                                  actual_ttft: Optional[float] = None,
                                  req_id: Optional[int] = None):
        """请求完成后调用，更新 inflight 并释放大 TP 资源"""
        if p_node is None:
            return

        group = self.node_to_group.get(p_node.client_ip_port)
        if group is None:
            return

        node_key = p_node.client_ip_port
        tp_size = self.node_tp_size.get(node_key, 1)

        # 获取并清除 dispatch info
        dispatch_info = self._req_dispatch_info.pop(req_id, None) if req_id is not None else None
        if dispatch_info is not None:
            node_key_d, predicted_latency, tp_size_d, group_d = dispatch_info
            # 使用 dispatch 时的信息
            async with group.lock:
                group.remove_inflight(node_key_d, predicted_latency)
        else:
            # Fallback: 使用当前信息
            predicted_latency = self.latency_model.predict_exclusive(input_token_num, tp_size)
            async with group.lock:
                group.remove_inflight(node_key, predicted_latency)

        is_min_tp = (tp_size == group.min_tp)

        if not is_min_tp:
            # 大 TP 完成：移除 active instance，fire 完成事件
            async with group.lock:
                group.unregister_active_large_tp(req_id, node_key)
            logger.info(
                f"FlexTP V2 large TP done: req_id={req_id}, tp={tp_size}, "
                f"node={node_key}, group={group.group_id}, "
                f"remaining_active={len(group.active_large_tp)}"
            )
        else:
            logger.info(
                f"FlexTP V2 small TP done: req_id={req_id}, tp={tp_size}, "
                f"node={node_key}, group={group.group_id}"
            )

        if actual_ttft is not None:
            logger.info(
                f"FlexTP V2 actual TTFT: req_id={req_id}, actual={actual_ttft * 1000:.1f}ms, "
                f"tp={tp_size}, node={node_key}"
            )

    # ---- 辅助方法 ----

    def _pick_decode_node(self) -> PD_Client_Obj:
        """轮询方式选择 decode 节点"""
        self._decode_rr_index = self._decode_rr_index % len(self.decode_nodes)
        d_node = self.decode_nodes[self._decode_rr_index]
        self._decode_rr_index += 1
        return d_node

    def select_p_d_node(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        raise NotImplementedError(
            "FlexTPSelectorV2 requires async_select_p_d_node. "
            "Ensure PDManager.select_p_d_node is using the async path."
        )
