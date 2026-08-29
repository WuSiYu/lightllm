"""TP-SMT 集群级 Router 调度器。

一、文件定位
==========

本文件是对旧 FlexTP 调度器的完整替代实现。为便于在 LightLLM 中直接覆盖，
仍保留 ``FlexTPSelectorV2`` 类名以及 ``async_select_p_d_node``、
``notify_request_done``、``select_p_d_node`` 等原有入口；调度算法本身已经重写，
不再使用旧设计中的大小 TP 二分、一次性 TP 选择、drain 状态机或 contention
budget。代码中的 ``FlexTP`` 名称仅用于接口兼容，不代表沿用旧机制。

二、系统模型与控制边界
====================

同一模型可以常驻多个不同 TP 度数的 Prefill instance。例如 4 张 GPU 上可以
同时存在两个互不重叠的 TP2 instance，以及一个覆盖全部 GPU 的 TP4 instance。
这些 instance 只共享只读权重，除此之外完全独立：各自拥有进程、通信域、
batch、KV 状态与执行队列。Router 不假设它们能够交换中间状态或协同 kernel。

Router 只控制三个动作：

1. 请求被派发到哪个具体 Prefill instance；
2. 请求在什么时刻离开 Router pending queue；
3. 哪些 GPU 重叠的 instance 可以同时处于运行状态。

请求一旦派发便不可迁移、不可改变 TP、不可抢占。当前每个 instance 只允许
一个由 Router 管理的在途 Prefill 请求，这使 Router 能准确知道 instance
何时可重新使用，也避免请求进入不可见的 instance 内部深队列。

三、拓扑表示
==========

每个 Prefill 节点通过 ``start_args`` 描述拓扑：

* ``tp``：该 instance 的 TP 度数；
* ``tp_smt_group_id``：共享同一模型权重的一组候选 instance；
* ``tp_smt_gpu_ids``：该 instance 实际占用的物理 GPU 编号。

节点的唯一键使用 ``client_ip_port``。GPU 键由主机地址与 GPU 编号共同组成，
因此只有物理 GPU 集合相交的 instance 才互相干扰。旧字段
``flex_tp_group_id`` 或 ``shared_weight_master_port_start`` 只用于兼容拓扑发现。
缺少 ``tp_smt_gpu_ids`` 时，代码会保守地认为同组 instance 全部重叠；这可以
避免低估干扰，但调度质量会下降，不能作为正式实验配置。

只有包含至少两种 TP 度数的组才进入 TP-SMT 调度。普通单 TP 节点保持旧系统
兜底行为；显式标记为 TP-SMT、但尚未组成完整拓扑的节点会失败关闭，避免在
系统启动阶段将尚未注册或仍在执行的 instance 误判为空闲。

四、请求与时间模型
================

请求到达后先留在 Router 的 ``_pending`` 中，不立即进入某个 instance。
``deadline = arrival_time + slo_ttft``，两者均为 ``time.time()`` 时钟上的绝对
秒数。若调用方没有传 ``arrival_time``，则使用 Router 收到请求的当前时间。

延迟模型先预测请求在某个 TP instance 上独占执行所需的秒数：

``exclusive = max(a * L / TP + b * L^2 / TP, c) + d``

运行中的 ``remaining_work`` 也以“独占等价秒”为单位。若当前并发模式对该
instance 的 slowdown 为 ``s``，经过 ``dt`` 秒后其工作量减少 ``dt / s``。
默认 slowdown 是目标 instance 所占任一 GPU 上的最大并发进程数，属于保守
启动模型；``mode_slowdowns`` 可用离线测量结果覆盖特定 TP 并发模式。

五、一次调度事件的完整流程
========================

请求到达、Prefill 完成、节点拓扑变化或短周期重规划都会触发调度。所有共享
状态都在 ``_state_lock`` 内读取和修改，一次事件依次执行：

1. 按上一并发模式和经过时间，更新所有运行请求的 ``remaining_work``；
2. 清理已取消的 pending，并处理无可用拓扑或明确不可行的请求；
3. 对当前运行集合做一次基线模拟，得到不派发新请求时各请求的完成时间；
4. 枚举每个 pending 请求与每个空闲具体 instance 的组合；
5. 对每个组合模拟完整的非抢占执行序列，直到所有请求完成；
6. 做双向可行性检查：新请求必须满足自身 deadline，同时不能使已经接纳的
   请求新增超时，或进一步恶化一个已经预计超时的请求；
7. 对可行候选按 deadline、FIFO、增量干扰、GPU 时间和完成时间排序；
8. 在锁内原子写入 instance reservation 和 running 状态，再唤醒请求协程。

调度循环会继续构造下一项派发，直到没有安全候选或没有空闲 instance。因此
单次事件可以形成一个小型贪心 dispatch bundle，但不会把同一个 instance
同时卖给两个请求。

六、TP4 独占区间与防饥饿
======================

考虑 ``TP2-A + TP2-B`` 正在运行，而一个长 TP4 请求到达。若 TP4 与两个 TP2
同时运行无法满足 SLO，TP4 会继续留在 Router。预测器会计算自然的阶段序列：

``TP4+TP2-A+TP2-B -> TP4+剩余的一个 TP2 -> TP4 独占``。

当某个等待请求在空系统中可行、但当前没有安全派发位置时，Router 会为它选
一个预计最早完成的未来 placement。优先级更低的新请求不得进入与该 placement
重叠的 GPU；不重叠的 instance 仍可继续工作。这样只阻止会破坏未来独占区间
的 refill，而不会冻结整个 Group，也不会错误限制彼此独立的 instance。

七、完成、失败和取消
==================

``async_select_p_d_node`` 返回意味着 Router 已经预留了 Prefill instance，
不意味着 worker 一定成功接收请求。上层必须遵循以下生命周期：

* Prefill 槽位真正可复用时调用 ``notify_request_done``；
* 派发异常、worker 拒绝或派发后的请求取消时调用 ``notify_request_failed``；
* selector 协程在返回前被取消时，代码自身会回滚尚未交付的 reservation。

完成时间预测归零不能替代权威回调。若回调尚未到达，instance 仍保持 busy，
且新的重叠请求不会获准进入。这个规则避免预测误差造成资源重叠；相应地，
远程实现必须保证失败路径回调，节点失联时还需要由健康检查触发失败清理。

八、过载策略和当前边界
====================

``reject`` 会拒绝已经过期或即使在空系统中也无法满足 deadline 的请求；
``best_effort`` 会在不继续伤害已接纳请求的前提下选择最快候选。暂时无法安全
派发、但空系统可行的请求只是等待，不会被当作过载。论文中若要声明“无解通常
意味着过载”，仍需增加基于 deadline window 的容量 Oracle，本实现没有把这项
待验证结论写成正确性假设。

当前实现还依赖以下远程工作：用目标模型和 GPU 重新拟合独占延迟及并发模式；
由 Router 作为这些 Prefill instance 的唯一派发入口；确认 ``arrival_time``
时钟口径；为派发失败和节点失联补齐回调或 lease/watchdog。以上条件不满足时，
调度器可能保守等待，但不会主动猜测 instance 已经释放。
"""

from __future__ import annotations

import asyncio
import collections
import itertools
import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Literal, Optional, Sequence, Set, Tuple, Union

from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.utils.log_utils import init_logger

from .pd_selector import PDSelector


logger = init_logger(__name__)


RequestKey = int
ModeSignature = Tuple[int, ...]


class LatencyModelV2:
    """独占延迟模型与可替换的并发减速模型。

    ``DEFAULT_CONSTANTS[tp] = (a, b, c, d)`` 用于计算单请求独占延迟。
    ``mode_slowdowns[(target_tp, signature)]`` 覆盖某种并发模式下目标 TP 的
    slowdown。``signature`` 是与目标实例共享至少一张 GPU 的运行实例 TP
    多重集合，并包含目标自身。例如 TP4 与两个 TP2 同时运行时，TP4 的签名是
    ``(2, 2, 4)``。

    默认常数只用于让代码可运行；正式部署必须按模型、GPU、输入长度和 TP
    重新拟合。若没有并发模式实测值，则使用每张 GPU 最大重叠进程数作为
    equal-share 近似，而不是声称 MPS 一定线性平分算力。
    """

    DEFAULT_CONSTANTS: Dict[int, Tuple[float, float, float, float]] = {
        1: (0.00024, 1e-8, 0.05, 0.05),
        2: (2.018352e-04, 6.048427e-09, 2.341091e-02, 6.794939e-02),
        # Refit from ttft_sweep_v8_tp4.csv (42-29990 prompt tokens).  The old
        # long-only v7 fit imposed a false ~408 ms floor on short TP4 requests;
        # the measured short-request plateau in v8 is ~94 ms.
        4: (2.655797e-04, 2.330274e-09, 5.651746e-02, 3.767872e-02),
        8: (0.00008, 3e-9, 0.05, 0.05),
    }

    def __init__(
        self,
        constants: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
        mode_slowdowns: Optional[Dict[Tuple[int, ModeSignature], float]] = None,
    ) -> None:
        self._constants = dict(self.DEFAULT_CONSTANTS)
        if constants:
            self._constants.update(constants)
        self._mode_slowdowns = dict(mode_slowdowns or {})

    def set_constants(self, tp_size: int, a: float, b: float, c: float, d: float) -> None:
        """覆盖一种 TP 度数的独占延迟常数。"""
        self._constants[tp_size] = (a, b, c, d)

    def set_mode_slowdown(
        self,
        target_tp: int,
        mode_signature: Sequence[int],
        slowdown: float,
    ) -> None:
        """登记离线测得的并发减速因子；小于 1 的值按 1 处理。"""
        self._mode_slowdowns[(target_tp, tuple(sorted(mode_signature)))] = max(1.0, slowdown)

    def _get_constants(self, tp_size: int) -> Tuple[float, float, float, float]:
        """读取目标 TP 常数；缺失时保守提示并使用数值上最近的已知 TP。"""
        if tp_size in self._constants:
            return self._constants[tp_size]
        known = min(self._constants, key=lambda configured: abs(configured - tp_size))
        logger.warning("TP-SMT 缺少 TP%d 延迟常数，暂用 TP%d 常数且不缩放", tp_size, known)
        return self._constants[known]

    def predict_exclusive(self, seq_len: int, tp_size: int) -> float:
        """返回请求在目标 TP instance 上无干扰执行的预测秒数。"""
        a, b, c, d = self._get_constants(tp_size)
        compute = a * seq_len / tp_size + b * seq_len * seq_len / tp_size
        return max(compute, c) + d

    def predict_batch(self, seq_lens: List[int], tp_size: int) -> float:
        """兼容旧 V2 接口，预测一个 Prefill batch 的独占执行时间。"""
        if not seq_lens:
            return 0.0
        a, b, c, d = self._get_constants(tp_size)
        total_tokens = sum(seq_lens)
        squared_tokens = sum(seq_len * seq_len for seq_len in seq_lens)
        compute = a * total_tokens / tp_size + b * squared_tokens / tp_size
        return max(compute, c) + d

    def predict_queue(self, seq_lens: List[int], tp_size: int) -> float:
        """兼容旧 V2 接口，按旧版 6000-token 组 batch 规则估算队列。"""
        batch_threshold = 6000
        batch: List[int] = []
        batch_tokens = 0
        total_time = 0.0
        for seq_len in seq_lens:
            batch.append(seq_len)
            batch_tokens += seq_len
            if batch_tokens >= batch_threshold:
                total_time += self.predict_batch(batch, tp_size)
                batch = []
                batch_tokens = 0
        if batch:
            total_time += self.predict_batch(batch, tp_size)
        return total_time

    @staticmethod
    def _overlaps(left: "InstanceState", right: "InstanceState") -> bool:
        """两个 instance 至少共享一张物理 GPU 时返回真。"""
        return bool(left.gpu_set.intersection(right.gpu_set))

    def mode_signature(
        self,
        target: "InstanceState",
        busy_instances: Iterable["InstanceState"],
    ) -> ModeSignature:
        """构造目标实例看到的并发模式键。

        不共享 GPU 的繁忙实例不会进入签名。当前键只编码 TP 多重集合，无法
        区分具有相同 TP 组成、但 GPU 放置不同的复杂拓扑；需要时应扩展配置键。
        """
        relevant: List[int] = []
        target_is_present = False
        for inst in busy_instances:
            if inst.node_key == target.node_key:
                target_is_present = True
            if inst.node_key == target.node_key or self._overlaps(target, inst):
                relevant.append(inst.tp_size)
        if not target_is_present:
            relevant.append(target.tp_size)
        return tuple(sorted(relevant))

    def slowdown_factor(
        self,
        target: "InstanceState",
        busy_instances: Sequence["InstanceState"],
    ) -> float:
        """返回目标实例在给定并发集合下的墙钟减速倍数。

        优先读取实测覆盖值。没有覆盖值时，计算目标占用的每张 GPU 上有多少
        活跃实例，并取最大值。例如 TP4 覆盖 0-3，两个 TP2 分别覆盖 0-1 和
        2-3，则三者同时运行时 TP4、TP2-A、TP2-B 的默认 slowdown 都是 2。
        """
        signature = self.mode_signature(target, busy_instances)
        override = self._mode_slowdowns.get((target.tp_size, signature))
        if override is not None:
            return max(1.0, override)

        max_processes_per_gpu = 1
        for gpu in target.gpu_set:
            count = sum(1 for inst in busy_instances if gpu in inst.gpu_set)
            max_processes_per_gpu = max(max_processes_per_gpu, count)
        return float(max_processes_per_gpu)


@dataclass
class RunningRequest:
    """已经原子绑定到某个 Prefill instance 的请求。

    ``deadline`` 和 ``predicted_finish`` 是绝对时间；``remaining_work`` 是独占
    等价秒；``last_progress_time`` 是最近一次扣减工作量的时间。预测工作量归零
    后只设置 ``awaiting_completion_callback``，权威完成回调到达前不释放资源。
    """

    internal_id: RequestKey
    external_req_id: Optional[int]
    deadline: float
    instance_key: str
    remaining_work: float
    predicted_finish: float
    last_progress_time: float
    awaiting_completion_callback: bool = False


@dataclass
class InstanceState:
    """一个常驻 Prefill instance 的 Router 侧状态。

    ``node_key`` 标识具体 LightLLM 节点；``group_id`` 只表示共享权重和候选
    关系；真正的干扰关系由 ``gpu_set`` 是否相交决定。``available=False`` 表示
    节点已经从可派发拓扑移除，但若仍有 ``running``，状态会保留到失败或完成
    回调，以免拓扑刷新误释放正在执行的资源。
    """

    node: PD_Client_Obj
    node_key: str
    group_id: str
    tp_size: int
    gpu_set: FrozenSet[str]
    placement_is_explicit: bool
    available: bool = True
    running: Optional[RunningRequest] = None

    @property
    def busy(self) -> bool:
        """是否已有一个 Router 管理的在途 Prefill 请求。"""
        return self.running is not None


@dataclass
class GroupState:
    """同一主机上共享权重、且具有多种 TP 度数的 instance 集合。

    Group 用于拓扑组织和兼容旧集成，不是隔离或调度单位。两个同组 instance
    若 GPU 不相交，可以独立派发；跨组 instance 若 GPU 键相交，仍会被判定冲突。
    """

    group_id: str
    instances: Dict[str, InstanceState] = field(default_factory=dict)

    @property
    def tp_sizes(self) -> List[int]:
        """兼容旧 ``FlexTPGroupV2.tp_sizes`` 的只读拓扑视图。"""
        return sorted({instance.tp_size for instance in self.instances.values() if instance.available})

    @property
    def tp_nodes(self) -> Dict[int, List[PD_Client_Obj]]:
        """兼容旧 ``tp -> nodes`` 映射；不可派发的历史实例不包含在内。"""
        nodes: Dict[int, List[PD_Client_Obj]] = collections.defaultdict(list)
        for instance in self.instances.values():
            if instance.available:
                nodes[instance.tp_size].append(instance.node)
        return dict(nodes)

    @property
    def min_tp(self) -> int:
        """兼容旧组对象的最小 TP 属性。"""
        return self.tp_sizes[0] if self.tp_sizes else 0

    @property
    def gpu_set(self) -> FrozenSet[str]:
        """返回组内可用实例的 GPU 并集，兼容旧诊断字段。"""
        return frozenset(
            gpu
            for instance in self.instances.values()
            if instance.available
            for gpu in instance.gpu_set
        )

    @property
    def node_inflight_requests(self) -> Dict[str, int]:
        """以 0/1 reservation 数提供旧版 inflight 只读视图。"""
        return {
            node_key: int(instance.busy)
            for node_key, instance in self.instances.items()
        }


@dataclass
class PendingRequest:
    """尚未绑定 Prefill instance、仍由 Router 持有的请求。

    ``future`` 在成功 commit 后返回 ``(prefill_node, decode_node)``；被拒绝或
    拓扑失效时携带异常结束。``enqueue_order`` 用于相同 deadline 下保持 FIFO。
    """

    internal_id: RequestKey
    external_req_id: Optional[int]
    seq_len: int
    deadline: float
    future: asyncio.Future
    enqueue_order: int


@dataclass
class CandidateEvaluation:
    """一个 ``(pending request, concrete instance)`` 候选的模拟结果。

    ``finish_times`` 包含接纳候选后所有运行请求及候选请求的预测完成时间。
    ``candidate_meets_deadline`` 检查新请求，``protects_existing`` 检查旧请求；
    两者同时为真才是可行候选。``incremental_delay`` 衡量相对基线新增的总延迟，
    ``gpu_time_cost`` 用独占时长乘 GPU 数近似资源成本。
    """

    pending: PendingRequest
    instance: InstanceState
    candidate_finish: float
    finish_times: Dict[RequestKey, float]
    candidate_meets_deadline: bool
    protects_existing: bool
    incremental_delay: float
    gpu_time_cost: float

    @property
    def feasible(self) -> bool:
        """候选自身和所有已接纳请求是否同时满足保护条件。"""
        return self.candidate_meets_deadline and self.protects_existing

    @property
    def score(self) -> Tuple[float, int, float, float, float, int, str]:
        """正常可行候选的确定性排序键；先保证 deadline 和 FIFO。"""
        return (
            self.pending.deadline,
            self.pending.enqueue_order,
            self.incremental_delay,
            self.gpu_time_cost,
            self.candidate_finish,
            self.instance.tp_size,
            self.instance.node_key,
        )

    @property
    def best_effort_score(self) -> Tuple[float, float, float, int, str]:
        """尽力服务候选排序键；优先选择预计最快完成的 placement。"""
        return (
            self.candidate_finish,
            self.incremental_delay,
            self.gpu_time_cost,
            self.instance.tp_size,
            self.instance.node_key,
        )


class FlexTPSelectorV2(PDSelector):
    """TP-SMT Router；类名保留用于兼容旧配置。

    参数：

    * ``slo_ttft``：从请求到达到 Prefill 槽位完成的目标秒数；
    * ``latency_constants``：按 TP 覆盖独占延迟模型的 ``(a, b, c, d)``；
    * ``mode_slowdowns``：按 ``(目标 TP, 并发模式签名)`` 覆盖 slowdown；
    * ``replan_interval_s``：有 pending 但暂不可派发时的重试周期；
    * ``prediction_tolerance_s``：浮点误差和轻微预测误差容忍量；
    * ``overload_policy``：选择 ``best_effort`` 或 ``reject``。

    关键不变量：

    * 每个 ``InstanceState`` 最多关联一个 ``RunningRequest``；
    * ``_pending`` 与 ``_running`` 中同一 internal id 不会同时存在；
    * 非空 external req_id 在 pending/running 中唯一；
    * instance reservation、索引更新和 Future 唤醒在同一把锁内原子完成；
    * 只有完成/失败回调或返回前取消才能释放 running reservation；
    * ``available=False`` 的 instance 不再接收新请求，但原有运行状态仍被保护。
    """

    def __init__(
        self,
        pd_manager,
        slo_ttft: Optional[float] = 5.0,
        *,
        latency_constants: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
        mode_slowdowns: Optional[Dict[Tuple[int, ModeSignature], float]] = None,
        replan_interval_s: float = 0.05,
        prediction_tolerance_s: float = 0.005,
        overload_policy: Literal["best_effort", "reject"] = "best_effort",
    ) -> None:
        super().__init__(pd_manager)
        # PDManager 总是传 flex_tp_slo_ttft；CLI 未配置时该值是 None。
        # 旧版构造签名允许省略该参数，因此 None 也应退化到相同的 5 秒默认值。
        self.slo_ttft = 5.0 if slo_ttft is None else float(slo_ttft)
        if not math.isfinite(self.slo_ttft) or self.slo_ttft <= 0:
            raise ValueError(f"slo_ttft 必须是正的有限秒数，实际为 {slo_ttft!r}")
        self.replan_interval_s = max(0.005, float(replan_interval_s))
        self.prediction_tolerance_s = max(0.0, float(prediction_tolerance_s))
        if overload_policy not in ("best_effort", "reject"):
            raise ValueError(f"未知 overload_policy: {overload_policy}")
        self.overload_policy = overload_policy

        self.latency_model = LatencyModelV2(
            constants=latency_constants,
            mode_slowdowns=mode_slowdowns,
        )

        self.flex_groups: Dict[str, GroupState] = {}
        self.instances: Dict[str, InstanceState] = {}
        # 以下三个字段保留，便于旧集成代码读取拓扑；调度决策只使用 instances。
        self.node_to_group: Dict[str, GroupState] = {}
        self.node_tp_size: Dict[str, int] = {}
        self.ungrouped_prefill_nodes: List[PD_Client_Obj] = []

        self._pending: "collections.OrderedDict[RequestKey, PendingRequest]" = collections.OrderedDict()
        self._running: Dict[RequestKey, RunningRequest] = {}
        self._external_to_internal: Dict[int, RequestKey] = {}
        self._request_counter = itertools.count(1)
        self._enqueue_counter = itertools.count(1)

        self._state_lock = asyncio.Lock()
        self._owner_loop: Optional[asyncio.AbstractEventLoop] = None
        self._decode_rr_index = 0
        self._replan_task: Optional[asyncio.Task] = None

    # 拓扑发现

    @staticmethod
    def _node_start_args(node: PD_Client_Obj) -> Dict:
        """安全读取节点启动参数；非字典值按空配置处理。"""
        return node.start_args if isinstance(node.start_args, dict) else {}

    @staticmethod
    def _node_host(node: PD_Client_Obj) -> str:
        """从 ``client_ip_port`` 提取主机部分，用于命名组和物理 GPU。"""
        return str(node.client_ip_port).split(":", 1)[0]

    def _group_id_for_node(self, node: PD_Client_Obj) -> Optional[str]:
        """返回带主机前缀的组键，优先使用 TP-SMT 显式配置。

        主机前缀防止不同机器复用相同 group id 时被错误合并。旧 FlexTP 和
        shared-weight 字段仅作为迁移兼容路径；正式 TP-SMT 配置应使用显式字段。
        """
        args = self._node_start_args(node)
        host = self._node_host(node)
        explicit = args.get("tp_smt_group_id") or args.get("flex_tp_group_id")
        if explicit is not None:
            return f"{host}:{explicit}"

        shared_port = args.get("shared_weight_master_port_start")
        if args.get("shared_weight") and shared_port is not None:
            return f"{host}:{shared_port}"
        return None

    @staticmethod
    def _parse_gpu_values(raw_value) -> List[str]:
        """将列表、整数或逗号/空白分隔字符串统一转换为 GPU id 列表。"""
        if raw_value is None:
            return []
        if isinstance(raw_value, (list, tuple, set, frozenset)):
            return [str(value).strip() for value in raw_value if str(value).strip()]
        if isinstance(raw_value, int):
            return [str(raw_value)]
        if isinstance(raw_value, str):
            return [token for token in re.split(r"[,;\s]+", raw_value.strip()) if token]
        return []

    def _gpu_set_for_node(
        self,
        node: PD_Client_Obj,
        group_id: str,
    ) -> Tuple[FrozenSet[str], bool]:
        """返回全局唯一 GPU 集合，以及该集合是否来自显式配置。

        没有物理 GPU 信息时使用组级占位键，使所有同组 instance 互相重叠。
        这是宁可少并发也不低估干扰的安全退化，不应替代真实拓扑元数据。
        """
        args = self._node_start_args(node)
        host = self._node_host(node)
        gpu_ids = self._parse_gpu_values(args.get("tp_smt_gpu_ids"))
        if gpu_ids:
            return frozenset(f"{host}/gpu:{gpu_id}" for gpu_id in gpu_ids), True

        # 未提供物理 GPU 放置时按组内全重叠处理，避免低估干扰。
        return frozenset({f"unknown-placement:{group_id}"}), False

    def update_nodes(self, prefill_nodes, decode_nodes) -> None:
        """接收 PDManager 的节点快照，重建拓扑并触发重调度。

        asyncio 锁只能在所属事件循环中安全使用。若节点发现线程从其他线程或
        事件循环调用本方法，则把整个更新投递到 selector 首次请求所在的循环。
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if self._owner_loop is not None and current_loop is not self._owner_loop:
            self._owner_loop.call_soon_threadsafe(self.update_nodes, prefill_nodes, decode_nodes)
            return
        super().update_nodes(prefill_nodes, decode_nodes)
        self._rebuild_topology()
        self._fail_unserviceable_pending()
        self._request_reschedule()

    def _fail_unserviceable_pending(self) -> None:
        """在已无 TP-SMT Prefill 或 Decode 时终止 pending，避免永久挂起。

        running 不在这里猜测性清理：节点可能只是暂时从发现结果中消失，只有
        完成、失败或健康检查产生的失败回调才有权释放已经派发的 reservation。
        """
        if self.decode_nodes and any(instance.available for instance in self.instances.values()):
            return
        reason = "Decode 节点为空" if not self.decode_nodes else "TP-SMT 拓扑已无可用实例"
        for internal_id, pending in list(self._pending.items()):
            self._pending.pop(internal_id, None)
            if not pending.future.done():
                pending.future.set_exception(RuntimeError(f"{reason}；请由上层重试请求"))

    def _rebuild_topology(self) -> None:
        """从当前 Prefill 节点快照重建 Group 与具体 instance。

        重建分三步：先发现节点并筛选至少含两种 TP 的合法组；再复用旧
        ``InstanceState`` 以保留 running 状态；最后生成只读兼容索引。忙实例
        若消失或配置变化，会标为不可派发并保留旧放置，直至权威回调到达。
        空闲且已消失的实例可以直接删除。
        """
        discovered: List[Tuple[PD_Client_Obj, str, int, FrozenSet[str], bool]] = []
        group_tp_sizes: Dict[str, Set[int]] = collections.defaultdict(set)
        ungrouped: List[PD_Client_Obj] = []

        for node in self.prefill_nodes:
            group_id = self._group_id_for_node(node)
            if group_id is None:
                ungrouped.append(node)
                continue
            args = self._node_start_args(node)
            tp_size = int(args.get("tp", 1))
            gpu_set, explicit = self._gpu_set_for_node(node, group_id)
            discovered.append((node, group_id, tp_size, gpu_set, explicit))
            group_tp_sizes[group_id].add(tp_size)

        eligible_groups = {
            group_id for group_id, tp_sizes in group_tp_sizes.items() if len(tp_sizes) >= 2
        }

        for instance in self.instances.values():
            instance.available = False

        for node, group_id, tp_size, gpu_set, explicit in discovered:
            if group_id not in eligible_groups:
                ungrouped.append(node)
                continue
            node_key = str(node.client_ip_port)
            instance = self.instances.get(node_key)
            if instance is None:
                instance = InstanceState(
                    node=node,
                    node_key=node_key,
                    group_id=group_id,
                    tp_size=tp_size,
                    gpu_set=gpu_set,
                    placement_is_explicit=explicit,
                )
                self.instances[node_key] = instance
            elif instance.busy and (
                instance.group_id != group_id
                or instance.tp_size != tp_size
                or instance.gpu_set != gpu_set
            ):
                logger.warning(
                    "TP-SMT 忙节点 %s 的拓扑发生变化，完成前保留派发时拓扑",
                    node_key,
                )
                instance.node = node
                instance.available = False
                continue
            else:
                instance.node = node
                instance.group_id = group_id
                instance.tp_size = tp_size
                instance.gpu_set = gpu_set
                instance.placement_is_explicit = explicit
            instance.available = True

        for node_key, instance in list(self.instances.items()):
            if not instance.available and not instance.busy:
                del self.instances[node_key]

        groups: Dict[str, GroupState] = {}
        for instance in self.instances.values():
            group = groups.setdefault(instance.group_id, GroupState(instance.group_id))
            group.instances[instance.node_key] = instance
        self.flex_groups = groups
        self.node_to_group = {
            node_key: groups[instance.group_id]
            for node_key, instance in self.instances.items()
            if instance.group_id in groups
        }
        self.node_tp_size = {
            node_key: instance.tp_size for node_key, instance in self.instances.items()
        }
        self.ungrouped_prefill_nodes = ungrouped

        for group in self.flex_groups.values():
            description = ", ".join(
                f"tp{inst.tp_size}@{inst.node_key}[{','.join(sorted(inst.gpu_set))}]"
                for inst in sorted(group.instances.values(), key=lambda item: (item.tp_size, item.node_key))
            )
            if any(not inst.placement_is_explicit for inst in group.instances.values()):
                logger.warning(
                    "TP-SMT 组 %s 缺少显式 GPU 放置，按组内全重叠处理",
                    group.group_id,
                )
            logger.info("TP-SMT 组 %s: %s", group.group_id, description)

    def _rebuild_flex_groups(self) -> None:
        """保留旧 V2 的私有入口名称，供现有诊断和测试代码调用。"""
        self._rebuild_topology()

    # 运行进度与并发模式模拟

    def _busy_instances(self) -> List[InstanceState]:
        """返回所有仍持有 Router reservation 的 instance。"""
        return [instance for instance in self.instances.values() if instance.busy]

    def _update_progress_locked(self, now: float) -> None:
        """按当前并发模式把墙钟时间折算为独占等价工作量。

        本方法在每个调度事件开始时调用。所有实例先基于同一份 busy snapshot
        计算 slowdown，因此更新顺序不会改变本时间段的并发模式。工作量即使
        预测到零也不释放 instance，而是等待 worker 的权威完成或失败回调。
        调用方必须持有 ``_state_lock``。
        """
        busy_instances = self._busy_instances()
        if not busy_instances:
            return

        for instance in busy_instances:
            running = instance.running
            if running is None:
                continue
            elapsed = max(0.0, now - running.last_progress_time)
            factor = self.latency_model.slowdown_factor(instance, busy_instances)
            running.remaining_work = max(0.0, running.remaining_work - elapsed / factor)
            if running.remaining_work <= 1e-9:
                # 预测归零不等于权威完成；回调前禁止新的重叠准入。
                running.awaiting_completion_callback = True
            running.last_progress_time = now

    def _simulate_finish_times_locked(
        self,
        now: float,
        extra_pending: Optional[PendingRequest] = None,
        extra_instance: Optional[InstanceState] = None,
    ) -> Dict[RequestKey, float]:
        """模拟从 ``now`` 开始的完整非抢占执行序列。

        默认只模拟当前 running；传入 ``extra_pending`` 和空闲
        ``extra_instance`` 时，模拟“现在接纳该候选”的结果。每一轮先计算当前
        mode 下各请求完成剩余工作所需的墙钟时间，推进到最早完成事件，删除
        同时完成的请求，再用新的活跃集合重算 slowdown。返回值是 internal id
        到绝对完成时间的映射。

        该事件推进方式能够显式表达 ``TP4+TP2+TP2 -> TP4+TP2 -> TP4``，而
        不是用单个静态 slowdown 乘完整执行时间。调用方必须持锁，且候选实例
        必须在真实状态中空闲；模拟过程本身不会修改 Router 状态。
        """
        remaining: Dict[RequestKey, float] = {}
        job_instances: Dict[RequestKey, InstanceState] = {}

        for instance in self._busy_instances():
            if instance.running is None:
                continue
            remaining[instance.running.internal_id] = max(1e-9, instance.running.remaining_work)
            job_instances[instance.running.internal_id] = instance

        if extra_pending is not None:
            if extra_instance is None or extra_instance.busy:
                raise ValueError("候选模拟要求一个空闲的具体实例")
            remaining[extra_pending.internal_id] = self.latency_model.predict_exclusive(
                extra_pending.seq_len,
                extra_instance.tp_size,
            )
            job_instances[extra_pending.internal_id] = extra_instance

        finish_times: Dict[RequestKey, float] = {}
        simulated_now = now

        while remaining:
            active_instances = [job_instances[request_id] for request_id in remaining]
            wall_times: Dict[RequestKey, float] = {}
            factors: Dict[RequestKey, float] = {}
            for request_id, work in remaining.items():
                factor = self.latency_model.slowdown_factor(
                    job_instances[request_id],
                    active_instances,
                )
                factors[request_id] = factor
                wall_times[request_id] = work * factor

            delta = min(wall_times.values())
            if not math.isfinite(delta) or delta < 0:
                raise RuntimeError("TP-SMT 并发模式模拟产生了非法结果")
            simulated_now += delta

            completed: List[RequestKey] = []
            for request_id in list(remaining):
                remaining[request_id] = max(
                    0.0,
                    remaining[request_id] - delta / factors[request_id],
                )
                if remaining[request_id] <= 1e-9:
                    finish_times[request_id] = simulated_now
                    completed.append(request_id)
            for request_id in completed:
                del remaining[request_id]

        return finish_times

    def _evaluate_candidate_locked(
        self,
        now: float,
        pending: PendingRequest,
        instance: InstanceState,
        baseline_finish: Dict[RequestKey, float],
    ) -> CandidateEvaluation:
        """评估立即把一个 pending 请求放到一个空闲 instance 的影响。

        ``baseline_finish`` 是不接纳候选时当前 running 的完成时间。候选模拟后
        同时检查两类约束：候选自身是否在 deadline 前完成，以及每个已接纳请求
        是否仍受保护。对于基线已预计超时的旧请求，只允许保持原预测，不允许
        新候选继续增加其延迟。增量干扰是所有旧请求相对基线增加的完成时间和。

        若重叠实例的预测工作量已经归零、但权威回调尚未到达，则直接返回不可行
        候选，不能仅凭模型猜测 GPU 已释放。调用方必须持有 ``_state_lock``。
        """
        for busy_instance in self._busy_instances():
            running = busy_instance.running
            if (
                running is not None
                and running.awaiting_completion_callback
                and self.latency_model._overlaps(instance, busy_instance)
            ):
                return CandidateEvaluation(
                    pending=pending,
                    instance=instance,
                    candidate_finish=float("inf"),
                    finish_times=baseline_finish,
                    candidate_meets_deadline=False,
                    protects_existing=False,
                    incremental_delay=float("inf"),
                    gpu_time_cost=float("inf"),
                )

        finish_times = self._simulate_finish_times_locked(
            now,
            extra_pending=pending,
            extra_instance=instance,
        )
        candidate_finish = finish_times[pending.internal_id]
        tolerance = self.prediction_tolerance_s
        candidate_meets_deadline = candidate_finish <= pending.deadline + tolerance

        protects_existing = True
        incremental_delay = 0.0
        for running in self._running.values():
            before = baseline_finish.get(running.internal_id, now)
            after = finish_times.get(running.internal_id, before)
            incremental_delay += max(0.0, after - before)

            # 已预计超时的运行请求不阻塞全局，但新请求不能进一步推迟它。
            allowed_finish = (
                running.deadline + tolerance
                if before <= running.deadline + tolerance
                else before + tolerance
            )
            if after > allowed_finish:
                protects_existing = False

        exclusive = self.latency_model.predict_exclusive(pending.seq_len, instance.tp_size)
        gpu_count = max(instance.tp_size, len(instance.gpu_set))
        gpu_time_cost = exclusive * gpu_count
        return CandidateEvaluation(
            pending=pending,
            instance=instance,
            candidate_finish=candidate_finish,
            finish_times=finish_times,
            candidate_meets_deadline=candidate_meets_deadline,
            protects_existing=protects_existing,
            incremental_delay=incremental_delay,
            gpu_time_cost=gpu_time_cost,
        )

    # Router 原子调度

    @staticmethod
    def _pending_priority(pending: PendingRequest) -> Tuple[float, int]:
        """返回 EDF + FIFO 优先级；元组越小，调度优先级越高。"""
        return pending.deadline, pending.enqueue_order

    def _can_meet_alone_locked(self, now: float, pending: PendingRequest) -> bool:
        """判断请求此刻在任一可用 instance 上独占运行是否仍能按时完成。

        这是一个乐观的局部测试，只区分“空系统仍不可能”和“受当前运行工作
        阻塞”；它不是集群容量或 deadline-window 过载证明。
        """
        return any(
            now + self.latency_model.predict_exclusive(pending.seq_len, instance.tp_size)
            <= pending.deadline + self.prediction_tolerance_s
            for instance in self.instances.values()
            if instance.available
        )

    def _drop_finished_pending_locked(self) -> None:
        """移除 Future 已取消或已由其他路径结束的 pending 条目。"""
        for internal_id, pending in list(self._pending.items()):
            if pending.future.done():
                self._pending.pop(internal_id, None)

    def _reject_definitely_infeasible_locked(self, now: float) -> None:
        """在 reject 策略下，拒绝过期或空系统也不可行的请求。"""
        if self.overload_policy != "reject":
            return
        for internal_id, pending in list(self._pending.items()):
            if pending.deadline > now and self._can_meet_alone_locked(now, pending):
                continue
            self._pending.pop(internal_id, None)
            if not pending.future.done():
                pending.future.set_exception(
                    RuntimeError("TP-SMT 拒绝请求：当前模型下不存在满足截止时间的放置")
                )
            logger.warning("TP-SMT 拒绝 req=%s len=%d", pending.external_req_id, pending.seq_len)

    def _blocked_waiters_locked(
        self,
        now: float,
        evaluations: Sequence[CandidateEvaluation],
        feasible: Sequence[CandidateEvaluation],
        baseline_finish: Dict[RequestKey, float],
    ) -> List[Tuple[Tuple[float, int], InstanceState]]:
        """为暂不可派发但空系统可行的早到请求预留未来放置位置。

        这里解决的是 refill 饥饿：早到长请求可能需要等待某个重叠实例结束后
        获得独占区间；若每当 TP2 空闲就继续派发短请求，覆盖全部 GPU 的 TP4
        可能永远等不到可行 mode。

        对每个当前无可行立即派发候选的早到请求，先找出它独占可满足 deadline
        的 instance，再选择预测最早完成的一个作为软 reservation。若该 instance
        当前空闲，使用候选模拟的完成时间；若它忙，使用现有请求基线完成时间加
        新请求独占时长估算。返回 ``(请求优先级, 预留 instance)`` 列表。

        reservation 不会修改 instance.running，也不保证未来一定采用该位置；
        它只用于本轮阻止更低优先级请求 refill 冲突 GPU。下次事件会重新计算。
        """
        feasible_ids = {evaluation.pending.internal_id for evaluation in feasible}
        blocked: List[Tuple[Tuple[float, int], InstanceState]] = []
        for pending in sorted(self._pending.values(), key=self._pending_priority):
            if pending.internal_id in feasible_ids or not self._can_meet_alone_locked(now, pending):
                continue
            placements = tuple(
                instance
                for instance in self.instances.values()
                if instance.available
                and now + self.latency_model.predict_exclusive(pending.seq_len, instance.tp_size)
                <= pending.deadline + self.prediction_tolerance_s
            )
            if placements:
                candidate_finish = {
                    evaluation.instance.node_key: evaluation.candidate_finish
                    for evaluation in evaluations
                    if evaluation.pending.internal_id == pending.internal_id
                }

                def reserved_finish(instance: InstanceState) -> float:
                    """估算 waiter 在该未来 placement 上完成的绝对时间。"""
                    if instance.node_key in candidate_finish:
                        return candidate_finish[instance.node_key]
                    if instance.running is None:
                        ready = now
                    else:
                        ready = baseline_finish.get(instance.running.internal_id, float("inf"))
                    return ready + self.latency_model.predict_exclusive(
                        pending.seq_len, instance.tp_size
                    )

                placement = min(
                    placements,
                    key=lambda instance: (
                        reserved_finish(instance),
                        instance.tp_size,
                        instance.node_key,
                    ),
                )
                blocked.append((self._pending_priority(pending), placement))
        return blocked

    def _waiter_allows_candidate(
        self,
        evaluation: CandidateEvaluation,
        blocked: Sequence[Tuple[Tuple[float, int], InstanceState]],
    ) -> bool:
        """判断候选是否会占用更早 blocked waiter 预留的冲突 GPU。

        只比较物理 GPU 重叠，而不是整个 Group，因此无关 instance 仍可派发。
        同优先级由 enqueue_order 决定，保证相同 deadline 下的 FIFO 保护。
        """
        priority = self._pending_priority(evaluation.pending)
        return not any(
            blocked_priority < priority
            and self.latency_model._overlaps(evaluation.instance, placement)
            for blocked_priority, placement in blocked
        )

    def _commit_candidate_locked(
        self,
        now: float,
        evaluation: CandidateEvaluation,
    ) -> bool:
        """原子提交一个已评估候选并唤醒等待的请求协程。

        提交顺序是：从 pending 移除、创建 ``RunningRequest``、占用 instance、
        更新 internal/external id 索引、刷新所有受影响请求的预测完成时间，最后
        才给 Future 写入 ``(Prefill, Decode)``。因此调用方一旦恢复执行，就能
        观察到完整 reservation。若 Future 在锁内已被取消，立即回滚所有状态。

        返回 ``True`` 表示成功交付；``False`` 表示 Future/Decode 状态变化导致
        提交失败。调用方必须持有 ``_state_lock``，且 evaluation.instance 空闲。
        """
        pending = evaluation.pending
        instance = evaluation.instance
        if pending.future.done() or not self.decode_nodes:
            self._pending.pop(pending.internal_id, None)
            if not pending.future.done():
                pending.future.set_exception(RuntimeError("TP-SMT 没有可用的 Decode 节点"))
            return False

        self._pending.pop(pending.internal_id, None)

        exclusive = self.latency_model.predict_exclusive(pending.seq_len, instance.tp_size)
        running = RunningRequest(
            internal_id=pending.internal_id,
            external_req_id=pending.external_req_id,
            deadline=pending.deadline,
            instance_key=instance.node_key,
            remaining_work=exclusive,
            predicted_finish=evaluation.candidate_finish,
            last_progress_time=now,
        )
        instance.running = running
        self._running[running.internal_id] = running
        if running.external_req_id is not None:
            self._external_to_internal[running.external_req_id] = running.internal_id

        for request_id, predicted_finish in evaluation.finish_times.items():
            existing = self._running.get(request_id)
            if existing is not None:
                existing.predicted_finish = predicted_finish

        # 锁内先写入预留状态，再唤醒 Future；调用方会在事件循环下一拍恢复。
        try:
            pending.future.set_result((instance.node, self._pick_decode_node()))
        except asyncio.InvalidStateError:
            instance.running = None
            self._running.pop(running.internal_id, None)
            if running.external_req_id is not None:
                self._external_to_internal.pop(running.external_req_id, None)
            return False

        logger.info(
            "TP-SMT 派发 req=%s len=%d -> group=%s tp=%d node=%s "
            "预计完成=%.1fms deadline=%.1fms 增量干扰=%.1fms",
            pending.external_req_id,
            pending.seq_len,
            instance.group_id,
            instance.tp_size,
            instance.node_key,
            max(0.0, evaluation.candidate_finish - now) * 1000,
            max(0.0, pending.deadline - now) * 1000,
            evaluation.incremental_delay * 1000,
        )
        return True

    def _schedule_locked(self, now: float) -> None:
        """执行一次完整调度，并贪心构造可原子接纳的派发组合。

        每轮先固定当前真实状态，计算不接纳新请求时的 baseline，然后枚举全部
        ``pending x idle instance`` 候选并做双向可行性过滤。正常候选排序键是：

        ``(deadline, FIFO, incremental_delay, gpu_time, finish, tp, node)``。

        选中并 commit 后，running 集合已经变化，因此重新计算下一轮候选，不复用
        旧预测。如果没有正常可行候选：``reject`` 保持可恢复请求等待；
        ``best_effort`` 只处理已经过期或空系统也不可行的请求，并仍要求不进一步
        伤害已接纳请求。最后若仍有 pending，启动短周期 timer 等待状态推进。

        本方法是共享调度状态的唯一主要写入口之一，调用方必须持锁。
        """
        self._update_progress_locked(now)
        self._drop_finished_pending_locked()
        self._reject_definitely_infeasible_locked(now)
        self._fail_unserviceable_pending()

        while self._pending:
            idle_instances = [
                instance
                for instance in self.instances.values()
                if instance.available and not instance.busy
            ]
            if not idle_instances:
                break

            baseline_finish = self._simulate_finish_times_locked(now)
            evaluations: List[CandidateEvaluation] = []
            for pending in sorted(
                self._pending.values(),
                key=lambda request: (request.deadline, request.enqueue_order),
            ):
                for instance in idle_instances:
                    evaluations.append(
                        self._evaluate_candidate_locked(
                            now,
                            pending,
                            instance,
                            baseline_finish,
                        )
                    )

            feasible = [evaluation for evaluation in evaluations if evaluation.feasible]
            blocked = self._blocked_waiters_locked(
                now,
                evaluations,
                feasible,
                baseline_finish,
            )
            feasible = [
                evaluation
                for evaluation in feasible
                if self._waiter_allows_candidate(evaluation, blocked)
            ]
            if feasible:
                chosen = min(feasible, key=lambda evaluation: evaluation.score)
                self._commit_candidate_locked(now, chosen)
                continue

            # 暂时不可行则继续排队；仅过期或空系统仍不可行时触发尽力服务。
            if self.overload_policy == "reject":
                break
            best_effort_candidates: List[CandidateEvaluation] = []
            for pending in self._pending.values():
                own_evaluations = [
                    evaluation
                    for evaluation in evaluations
                    if evaluation.pending.internal_id == pending.internal_id
                ]
                if not own_evaluations:
                    continue
                if pending.deadline <= now or not self._can_meet_alone_locked(now, pending):
                    best_effort_candidates.extend(
                        evaluation
                        for evaluation in own_evaluations
                        if evaluation.protects_existing
                        and self._waiter_allows_candidate(evaluation, blocked)
                    )

            if not best_effort_candidates:
                break

            chosen = min(
                best_effort_candidates,
                key=lambda evaluation: evaluation.best_effort_score,
            )
            logger.warning(
                "TP-SMT 尽力服务 req=%s len=%d 预计超时=%.1fms",
                chosen.pending.external_req_id,
                chosen.pending.seq_len,
                max(0.0, chosen.candidate_finish - chosen.pending.deadline) * 1000,
            )
            self._commit_candidate_locked(now, chosen)

        if self._pending:
            self._ensure_replan_timer_locked()

    async def _reschedule(self) -> None:
        """获取状态锁并以当前墙钟时间触发一次调度。"""
        async with self._state_lock:
            self._schedule_locked(time.time())

    def _request_reschedule(self) -> None:
        """在同步回调中尽力安排异步重调度；没有运行事件循环时直接返回。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._reschedule())

    def _ensure_replan_timer_locked(self) -> None:
        """保证最多只有一个 pending 重规划定时器。"""
        if self._replan_task is not None and not self._replan_task.done():
            return
        self._replan_task = asyncio.create_task(self._replan_after_delay())

    async def _replan_after_delay(self) -> None:
        """等待固定短周期后重算进度和候选；完成前清除 timer 引用。"""
        await asyncio.sleep(self.replan_interval_s)
        self._replan_task = None
        await self._reschedule()

    # LightLLM 兼容接口

    def _pick_decode_node(self) -> PD_Client_Obj:
        """以简单轮询选择 Decode；TP-SMT 只重新设计 Prefill admission。"""
        self._decode_rr_index %= len(self.decode_nodes)
        node = self.decode_nodes[self._decode_rr_index]
        self._decode_rr_index += 1
        return node

    async def async_select_p_d_node(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        input_token_num: Optional[int] = None,
        arrival_time: Optional[float] = None,
        req_id: Optional[int] = None,
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        """将请求加入 Router，并等待一个原子提交的 Prefill/Decode 选择。

        ``input_token_num`` 是当前延迟模型使用的输入长度。``arrival_time`` 必须
        与 ``time.time()`` 同源且单位为秒；省略时以进入本方法的时间为到达时间。
        ``req_id`` 若非空，必须在 pending 和 running 中唯一。

        合法 TP-SMT 拓扑存在时，本方法创建 ``PendingRequest``，在锁内触发调度，
        然后 await 其 Future。返回前，目标 Prefill instance 已被记为 running；
        上层派发失败时必须调用 ``notify_request_failed``。如果等待协程在返回前
        被取消，异常继续向上传播，同时本类回滚 pending 或未交付 reservation。

        没有任何 TP-SMT 配置的普通部署沿用随机 Prefill 兜底；出现显式 TP-SMT
        标记但拓扑不完整时抛出异常，不让启动阶段流量绕过 Router 状态管理。
        """
        if not self.prefill_nodes or not self.decode_nodes:
            raise RuntimeError(
                f"FlexTPSelectorV2: req_id={req_id} 没有可用节点 "
                f"(prefill={len(self.prefill_nodes)}, decode={len(self.decode_nodes)})"
            )

        if not self.flex_groups:
            # 显式 TP-SMT 节点尚未组成完整拓扑时失败关闭，避免把忙实例当空闲实例。
            if any(
                self._node_start_args(node).get("tp_smt_group_id") is not None
                for node in self.prefill_nodes
            ):
                raise RuntimeError("TP-SMT 拓扑尚未完整注册")
            # 普通或旧式单 TP 部署继续沿用随机 Prefill 兜底。
            return random.choice(self.prefill_nodes), self._pick_decode_node()

        now = time.time()
        request_arrival = now if arrival_time is None else float(arrival_time)
        deadline = request_arrival + self.slo_ttft
        sequence_length = int(input_token_num or 0)
        internal_id = next(self._request_counter)
        loop = asyncio.get_running_loop()
        if self._owner_loop is None:
            self._owner_loop = loop
        elif self._owner_loop is not loop:
            raise RuntimeError("TP-SMT selector 不能跨事件循环使用")
        future = loop.create_future()

        pending = PendingRequest(
            internal_id=internal_id,
            external_req_id=req_id,
            seq_len=sequence_length,
            deadline=deadline,
            future=future,
            enqueue_order=next(self._enqueue_counter),
        )

        async with self._state_lock:
            if req_id is not None and (
                req_id in self._external_to_internal
                or any(item.external_req_id == req_id for item in self._pending.values())
            ):
                raise ValueError(f"TP-SMT req_id={req_id} 重复")
            self._pending[internal_id] = pending
            logger.info(
                "TP-SMT 入队 req=%s len=%d 剩余deadline=%.1fms pending=%d",
                req_id,
                sequence_length,
                max(0.0, deadline - now) * 1000,
                len(self._pending),
            )
            self._schedule_locked(now)

        try:
            return await future
        except asyncio.CancelledError:
            await self._cancel_selection_internal(internal_id)
            raise

    async def _cancel_selection_internal(self, internal_id: RequestKey) -> None:
        """清理在 selector 返回前被取消的请求。

        请求可能仍在 pending，也可能刚刚 commit 并唤醒 Future、但调用协程尚未
        取得返回值。两种情况都在锁内清理。调用方已经取得节点后的取消不经过
        本路径，必须由上层调用 ``notify_request_failed``。
        """
        async with self._state_lock:
            pending = self._pending.pop(internal_id, None)
            if pending is not None and not pending.future.done():
                pending.future.cancel()
            running = self._running.pop(internal_id, None)
            if running is not None:
                if running.external_req_id is not None:
                    self._external_to_internal.pop(running.external_req_id, None)
                instance = self.instances.get(running.instance_key)
                if instance is not None and instance.running is running:
                    instance.running = None
                logger.info("TP-SMT 回滚未交付的预留 req=%s", running.external_req_id)
            self._schedule_locked(time.time())

    async def notify_request_done(
        self,
        p_node: Optional[PD_Client_Obj],
        input_token_num: int = 0,
        actual_ttft: Optional[float] = None,
        req_id: Optional[int] = None,
    ) -> None:
        """在 Prefill 槽位真正可复用时释放 Router reservation。

        这里的“完成”指本调度器管理的 Prefill 阶段完成，不是整条生成请求完成。
        ``actual_ttft`` 当前只用于日志，不会在线修改延迟模型；在线校准需要先确认
        它是否包含 Router 等待时间以及准确的计时起点。
        """
        await self._finish_request(
            p_node=p_node,
            req_id=req_id,
            actual_ttft=actual_ttft,
            outcome="done",
        )

    async def notify_request_failed(
        self,
        p_node: Optional[PD_Client_Obj],
        req_id: Optional[int] = None,
    ) -> None:
        """在派发失败、worker 拒绝或派发后取消时释放 reservation。

        该回调与正常完成走同一清理路径，但 ``outcome`` 用于区分日志。任何已经
        从 selector 取得 Prefill 节点却未开始/未完成 Prefill 的异常路径都必须
        调用本方法，否则 instance 会按设计继续保持 busy。
        """
        await self._finish_request(
            p_node=p_node,
            req_id=req_id,
            actual_ttft=None,
            outcome="failed",
        )

    async def _finish_request(
        self,
        p_node: Optional[PD_Client_Obj],
        req_id: Optional[int],
        actual_ttft: Optional[float],
        outcome: str,
    ) -> None:
        """完成和失败回调共用的幂等式状态清理入口。

        首选用 external req_id 查 internal id，再从 ``_running`` 和 instance 上
        同时解除 reservation。只有调用方明确没有 req_id 时才允许按节点兜底，
        防止过期或错误 req_id 清掉恰好运行在同节点上的无关请求。未知回调只记
        警告，不破坏现有状态。清理后若节点曾在 busy 期间退出拓扑，则重建拓扑，
        最后立即调度 pending，使刚释放的资源可以在同一事件中重新使用。
        """
        now = time.time()
        async with self._state_lock:
            self._update_progress_locked(now)
            internal_id = self._external_to_internal.pop(req_id, None) if req_id is not None else None
            running = self._running.pop(internal_id, None) if internal_id is not None else None

            # 仅当回调没有 req_id 时才按节点兜底；错误 req_id 不能清掉同节点其他请求。
            if running is None and req_id is None and p_node is not None:
                instance = self.instances.get(str(p_node.client_ip_port))
                if instance is not None and instance.running is not None:
                    running = instance.running
                    internal_id = running.internal_id
                    self._running.pop(internal_id, None)
                    if running.external_req_id is not None:
                        self._external_to_internal.pop(running.external_req_id, None)

            if running is None:
                node_key = str(p_node.client_ip_port) if p_node is not None else None
                if node_key in self.instances or req_id in self._external_to_internal:
                    logger.warning("TP-SMT 收到未知请求的完成回调 req=%s outcome=%s", req_id, outcome)
                self._schedule_locked(now)
            else:
                instance = self.instances.get(running.instance_key)
                if instance is not None and instance.running is running:
                    instance.running = None

                if instance is not None and not instance.available:
                    self._rebuild_topology()

                logger.info(
                    "TP-SMT 完成 req=%s outcome=%s tp=%s node=%s actual_ttft_ms=%s",
                    req_id,
                    outcome,
                    instance.tp_size if instance is not None else "unknown",
                    running.instance_key,
                    f"{actual_ttft * 1000:.1f}" if actual_ttft is not None else "n/a",
                )
                self._schedule_locked(now)

    def select_p_d_node(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        """拒绝同步入口，因为 TP-SMT 可能需要在 Router pending queue 中等待。"""
        raise NotImplementedError(
            "FlexTPSelectorV2 仅支持 async_select_p_d_node；请让 PDManager 使用异步入口"
        )

    def snapshot(self) -> Dict:
        """返回便于日志和测试读取的浅层状态快照。

        快照包含 pending 的请求 id、长度和 deadline，以及每个 instance 的组、
        TP、GPU、可用性、运行请求和预测完成时间。它不持锁，也不返回内部对象；
        生产监控若要求强一致视图，应新增异步加锁版本。
        """
        return {
            "pending": [
                {
                    "req_id": request.external_req_id,
                    "seq_len": request.seq_len,
                    "deadline": request.deadline,
                }
                for request in self._pending.values()
            ],
            "instances": {
                node_key: {
                    "group_id": instance.group_id,
                    "tp": instance.tp_size,
                    "gpu_set": sorted(instance.gpu_set),
                    "available": instance.available,
                    "running_req_id": (
                        instance.running.external_req_id if instance.running is not None else None
                    ),
                    "predicted_finish": (
                        instance.running.predicted_finish if instance.running is not None else None
                    ),
                }
                for node_key, instance in self.instances.items()
            },
        }
