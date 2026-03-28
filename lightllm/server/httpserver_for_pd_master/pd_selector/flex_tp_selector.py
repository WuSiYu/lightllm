"""
Flex TP Selector: 调度逻辑，用于支持不同 TP 数的 prefill 节点共享同一组 GPU（通过 --shared_weight）。

核心功能：
1. 自动识别共享 GPU 的 TP 组（FlexTPGroup）
2. 基于请求长度的选择性调度：长请求 -> 大 TP，短请求 -> 小 TP
3. 同 GPU 上多 TP 组的排他与抢占：默认运行小 TP，大 TP 请求到达时排空小 TP 后切换

背景： 在PD分离模式中，Flex TP (--shared_weight) 是一种在同一组 GPU 上部署不同 TP 大小 prefill 节点的方式，适用于请求长度分布广泛的场景。通过在同一组GPU上运行多组不同TP大小的节点（例如2xTP2 + TP4），并使用权重共享避免额外显存开销，可以在一组GPU上无缝切换prefill TP数，让短请求以TP2 + DP2处理提高吞吐，而长请求则使用TP4降低延迟，从而提高资源利用率和响应速度。
"""

import asyncio
import random
from typing import Union, List, Tuple, Dict, Optional

from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.utils.log_utils import init_logger
from .pd_selector import PDSelector

logger = init_logger(__name__)


class FlexTPGroup:
    """表示一组共享同一批 GPU 的不同 TP 大小的 prefill 节点"""

    def __init__(self, group_id: str):
        self.group_id: str = group_id
        self.small_tp_nodes: List[PD_Client_Obj] = []
        self.large_tp_nodes: List[PD_Client_Obj] = []
        self.small_tp_size: int = 0
        self.large_tp_size: int = 0

        # 状态机: "small_tp_active" -> "draining" -> "large_tp_active" -> "small_tp_active"
        self.state: str = "small_tp_active"

        # 在途请求计数
        self.inflight_small_tp: int = 0
        self.inflight_large_tp: int = 0
        # 等待执行的大 TP 请求数（已决定走大 TP 但还在等 drain 完成）
        self.pending_large_tp: int = 0
        # 等待执行的小 TP 请求数（所有 group 都被大 TP 占据时排队等待）
        self.pending_small_tp: int = 0

        # 用于状态变更通知的 asyncio.Condition
        self.condition: asyncio.Condition = asyncio.Condition()

    def __repr__(self):
        return (
            f"FlexTPGroup(id={self.group_id}, state={self.state}, "
            f"small_tp={self.small_tp_size}x{len(self.small_tp_nodes)}, "
            f"large_tp={self.large_tp_size}x{len(self.large_tp_nodes)}, "
            f"inflight_s={self.inflight_small_tp}, inflight_l={self.inflight_large_tp}, "
            f"pending_s={self.pending_small_tp}, pending_l={self.pending_large_tp})"
        )


class FlexTPSelector(PDSelector):
    """
    Flex TP 调度选择器。

    调度策略：
    - input_token_num > length_threshold 的请求路由到大 TP 节点
    - 其余请求路由到小 TP 节点
    - 同一 FlexTPGroup 内，大小 TP 互斥执行：
      - 默认状态为 small_tp_active，小 TP 节点正常接收请求
      - 当有大 TP 请求到达时，进入 draining 状态，不再调度新的小 TP 请求
      - 等待所有在途小 TP 请求完成后，进入 large_tp_active 状态
      - 大 TP 请求全部完成后，恢复 small_tp_active 状态
    """

    def __init__(self, pd_manager, length_threshold: int = 8000):
        super().__init__(pd_manager)
        self.length_threshold: int = length_threshold
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

        # 保留已移除节点的映射，直到其所属 group 的 inflight 请求归零，
        # 避免 notify_request_done 找不到节点导致计数泄漏和死锁。
        for old_key, old_group in self.node_to_group.items():
            if old_key not in new_node_to_group:
                if old_group.inflight_small_tp > 0 or old_group.inflight_large_tp > 0:
                    new_node_to_group[old_key] = old_group
                    new_node_is_large_tp[old_key] = self.node_is_large_tp.get(old_key, False)
                    logger.warning(
                        f"FlexTP: preserving stale mapping for {old_key} "
                        f"(group {old_group.group_id}: {old_group})"
                    )

        self.flex_groups = new_flex_groups
        self.node_to_group = new_node_to_group
        self.node_is_large_tp = new_node_is_large_tp
        self.ungrouped_prefill_nodes = ungrouped

        if ungrouped:
            logger.info(f"FlexTP: {len(ungrouped)} ungrouped prefill node(s)")

    # ---- 核心调度接口 ----

    async def async_select_p_d_node(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        input_token_num: Optional[int] = None,
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        """异步版本的节点选择，支持排他等待"""
        if not self.prefill_nodes or not self.decode_nodes:
            raise RuntimeError(
                f"FlexTPSelector: no available nodes "
                f"(prefill={len(self.prefill_nodes)}, decode={len(self.decode_nodes)})"
            )

        d_node = self._pick_decode_node()
        use_large_tp = input_token_num is not None and input_token_num > self.length_threshold

        if not self.flex_groups:
            # 没有 flex TP 组，回退到普通选择
            p_node = random.choice(self.prefill_nodes)
            return p_node, d_node

        if use_large_tp:
            p_node = await self._select_large_tp()
        else:
            p_node = await self._select_small_tp()

        logger.info(
            f"FlexTP select: input_tokens={input_token_num}, use_large_tp={use_large_tp}, "
            f"p_node={p_node.client_ip_port}"
        )
        return p_node, d_node

    async def _select_large_tp(self) -> PD_Client_Obj:
        """选择大 TP 节点，必要时触发排空小 TP 并等待"""
        # 选择一个有大 TP 节点的 group
        groups_with_large = [g for g in self.flex_groups.values() if g.large_tp_nodes]
        if not groups_with_large:
            # 没有大 TP 节点可用，回退到任意 prefill 节点
            logger.warning("FlexTP: no large TP nodes available, falling back")
            return random.choice(self.prefill_nodes)

        # 选择 pending 最小的 group
        group = min(groups_with_large, key=lambda g: g.pending_large_tp)

        async with group.condition:
            group.pending_large_tp += 1

            # 如果当前是 small_tp_active，发起排空
            if group.state == "small_tp_active":
                group.state = "draining"
                logger.info(f"FlexTP group [{group.group_id}]: draining small TP (inflight={group.inflight_small_tp})")

            # 等待所有小 TP 在途请求完成
            while group.inflight_small_tp > 0:
                await group.condition.wait()

            # 排空完成，切换到大 TP 运行态
            group.state = "large_tp_active"
            group.inflight_large_tp += 1
            group.pending_large_tp -= 1

        p_node = self._pick_from_nodes(group.large_tp_nodes)
        logger.info(
            f"FlexTP group [{group.group_id}]: large TP request dispatched to {p_node.client_ip_port} "
            f"(inflight_l={group.inflight_large_tp})"
        )
        return p_node

    async def _select_small_tp(self) -> PD_Client_Obj:
        """选择小 TP 节点，如果当前组正在排空或大 TP 运行中则等待"""
        # 优先选择处于 small_tp_active 状态的 group
        groups_with_small = [g for g in self.flex_groups.values() if g.small_tp_nodes]
        if not groups_with_small:
            # 没有小 TP 节点，尝试使用 ungrouped 节点
            if self.ungrouped_prefill_nodes:
                return random.choice(self.ungrouped_prefill_nodes)
            return random.choice(self.prefill_nodes)

        # 找一个处于 small_tp_active 的 group
        active_groups = [g for g in groups_with_small if g.state == "small_tp_active"]
        if active_groups:
            group = random.choice(active_groups)
        else:
            # 所有 group 都在 draining 或 large_tp_active，选择最快恢复的 group 等待
            group = min(groups_with_small, key=lambda g: g.pending_large_tp)
            group.pending_small_tp += 1
            logger.warning(
                f"FlexTP: all groups busy, small TP request blocked waiting for "
                f"group [{group.group_id}] (pending_small={group.pending_small_tp}, "
                f"pending_large={group.pending_large_tp}, "
                f"inflight_large={group.inflight_large_tp}). "
                f"Continuous large TP traffic may starve small TP requests."
            )
            async with group.condition:
                while group.state != "small_tp_active":
                    await group.condition.wait()
            group.pending_small_tp -= 1

        # 直接增加计数（在 condition 外面即可，因为 asyncio 是单线程的）
        group.inflight_small_tp += 1
        p_node = self._pick_from_nodes(group.small_tp_nodes)
        return p_node

    async def notify_request_done(self, p_node: PD_Client_Obj):
        """请求完成后调用，更新在途计数并触发状态切换"""
        if p_node is None:
            return

        group = self.node_to_group.get(p_node.client_ip_port)
        if group is None:
            return

        is_large = self.node_is_large_tp.get(p_node.client_ip_port, False)

        async with group.condition:
            if is_large:
                group.inflight_large_tp = max(0, group.inflight_large_tp - 1)
                if group.inflight_large_tp == 0 and group.pending_large_tp == 0:
                    # 所有大 TP 请求完成，恢复小 TP 运行
                    group.state = "small_tp_active"
                    group.condition.notify_all()
                    logger.info(f"FlexTP group [{group.group_id}]: large TP done, resuming small TP")
            else:
                group.inflight_small_tp = max(0, group.inflight_small_tp - 1)
                if group.inflight_small_tp == 0 and group.state == "draining":
                    # 小 TP 排空完成，通知等待中的大 TP 请求
                    group.condition.notify_all()
                    logger.info(f"FlexTP group [{group.group_id}]: small TP drained (inflight=0)")

            logger.debug(f"FlexTP group status: {group}")

    # ---- 辅助方法 ----

    def _pick_decode_node(self) -> PD_Client_Obj:
        """轮询方式选择 decode 节点"""
        self._decode_rr_index = self._decode_rr_index % len(self.decode_nodes)
        d_node = self.decode_nodes[self._decode_rr_index]
        self._decode_rr_index += 1
        return d_node

    def _pick_from_nodes(self, nodes: List[PD_Client_Obj]) -> PD_Client_Obj:
        """轮询方式从给定节点列表中选择一个"""
        return random.choice(nodes)

    def select_p_d_node(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        """同步回退方法，flex TP 模式下不应使用"""
        raise NotImplementedError(
            "FlexTPSelector requires async_select_p_d_node. "
            "Ensure PDManager.select_p_d_node is using the async path."
        )
