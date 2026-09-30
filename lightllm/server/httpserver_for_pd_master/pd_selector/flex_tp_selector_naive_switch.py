"""Naive FlexTP selector with exclusive class switching.

Requests are still classified by the naive length threshold, but TP2 and TP4
classes never have in-flight requests at the same time.  A request for the
other class waits at the PD master until the current class drains, then the
selector switches the active class.
"""

import asyncio
from typing import Dict, Optional

from .flex_tp_selector_naive import FlexTPNaiveSelector


class FlexTPNaiveSwitchSelector(FlexTPNaiveSelector):
    def __init__(self, pd_manager, length_threshold: int = 8000):
        super().__init__(pd_manager, length_threshold=length_threshold)
        self._switch_cv = asyncio.Condition()
        self._active_large_by_group: Dict[str, Optional[bool]] = {}

    def update_nodes(self, prefill_nodes, decode_nodes):
        super().update_nodes(prefill_nodes, decode_nodes)
        self._active_large_by_group = {
            group_id: self._active_large_by_group.get(group_id)
            for group_id in self.flex_groups
        }

    def _inflight_class(self, group_id: str, is_large: bool) -> int:
        group = self.flex_groups.get(group_id)
        if group is None:
            return 0
        nodes = group.large_tp_nodes if is_large else group.small_tp_nodes
        return sum(group.node_inflight_requests.get(n.client_ip_port, 0) for n in nodes)

    @staticmethod
    def _class_nodes(group, is_large: bool):
        return group.large_tp_nodes if is_large else group.small_tp_nodes

    def _candidate_groups(self, is_large: bool):
        return [
            group
            for group in self.flex_groups.values()
            if self._class_nodes(group, is_large)
        ]

    def _group_is_available(self, group, is_large: bool) -> bool:
        group_id = group.group_id
        active_large = self._active_large_by_group.get(group_id)
        if active_large is None or active_large == is_large:
            return True
        if self._inflight_class(group_id, bool(active_large)) == 0:
            self._active_large_by_group[group_id] = None
            return True
        return False

    async def async_select_p_d_node(self, *args, **kwargs):
        input_token_num = kwargs.get("input_token_num")
        if input_token_num is None and len(args) >= 4:
            input_token_num = args[3]
        desired_large = (input_token_num or 0) > self.length_threshold

        async with self._switch_cv:
            if not self.prefill_nodes or not self.decode_nodes:
                raise RuntimeError(
                    f"FlexTPNaiveSwitchSelector: req_id={kwargs.get('req_id')} no available nodes "
                    f"(prefill={len(self.prefill_nodes)}, decode={len(self.decode_nodes)})"
                )
            if not self.flex_groups:
                return await super().async_select_p_d_node(*args, **kwargs)

            while True:
                actual_large = desired_large
                target_groups = self._candidate_groups(actual_large)
                if not target_groups:
                    actual_large = not desired_large
                    target_groups = self._candidate_groups(actual_large)

                candidates = []
                for group in target_groups:
                    if not self._group_is_available(group, actual_large):
                        continue
                    node = self._pick_least_loaded(group, self._class_nodes(group, actual_large))
                    load = group.node_inflight_tokens.get(node.client_ip_port, 0)
                    candidates.append((load, group.group_id, group, node))

                if candidates:
                    _, _, group, node = min(candidates)
                    group_id = group.group_id
                    self._active_large_by_group[group_id] = actual_large
                    token_num = input_token_num or 0
                    group.add_inflight(node.client_ip_port, token_num)
                    return node, self._pick_decode_node(node)

                await self._switch_cv.wait()

    async def notify_request_done(self, *args, **kwargs):
        await super().notify_request_done(*args, **kwargs)
        async with self._switch_cv:
            p_node = args[0] if args else kwargs.get("p_node")
            if p_node is None:
                return
            group = self.node_to_group.get(p_node.client_ip_port)
            if group is None:
                return
            group_id = group.group_id
            active_large = self._active_large_by_group.get(group_id)
            if active_large is not None and self._inflight_class(group_id, bool(active_large)) == 0:
                self._active_large_by_group[group_id] = None
                self._switch_cv.notify_all()
