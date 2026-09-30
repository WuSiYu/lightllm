"""V14: V13 Prefill scheduling with same-host Decode affinity."""

from __future__ import annotations

from typing import Set

from .flex_tp_selector_v13 import FlexTPSelectorV13
from .flex_tp_selector_v6 import V6InstanceState


class FlexTPSelectorV14(FlexTPSelectorV13):
    """Keep V13 Prefill routing and prefer Decode on the Prefill host."""

    is_flex_tp_v14 = True
    is_flex_tp_v13 = False

    def __init__(self, pd_manager, *, decode_locality_enabled: bool = True, **kwargs) -> None:
        super().__init__(pd_manager, **kwargs)
        self.decode_locality_enabled = bool(decode_locality_enabled)
        self.local_decode_affinity_hits = 0
        self.remote_decode_fallbacks = 0

    @staticmethod
    def _node_hosts(node) -> Set[str]:
        hosts = set()
        client_ip_port = str(getattr(node, "client_ip_port", ""))
        if client_ip_port:
            hosts.add(client_ip_port.split(":", 1)[0])
        start_args = getattr(node, "start_args", None)
        if isinstance(start_args, dict):
            advertised_host = start_args.get("host")
            if advertised_host:
                hosts.add(str(advertised_host))
        return {host for host in hosts if host}

    def _decode_candidates_for_prefill(self, prefill_node):
        if not self.decode_nodes:
            raise RuntimeError("no Decode node is registered")
        if not self.decode_locality_enabled:
            return self.decode_nodes, False

        prefill_hosts = self._node_hosts(prefill_node)
        local_nodes = [
            node
            for node in self.decode_nodes
            if prefill_hosts.intersection(self._node_hosts(node))
        ]
        if local_nodes:
            return local_nodes, True
        return self.decode_nodes, False

    def _pick_decode_node(self):
        prefill_node = getattr(self, "_decode_affinity_prefill", None)
        if prefill_node is None:
            return super()._pick_decode_node()

        candidates, local = self._decode_candidates_for_prefill(prefill_node)
        index = self._decode_rr_index % len(candidates)
        self._decode_rr_index += 1
        if local:
            self.local_decode_affinity_hits += 1
        else:
            self.remote_decode_fallbacks += 1
        return candidates[index]

    def _admit(self, now: float, pending, state: V6InstanceState, mode) -> None:
        self._decode_affinity_prefill = state.node
        try:
            return super()._admit(now, pending, state, mode)
        finally:
            self._decode_affinity_prefill = None

    def scheduler_snapshot(self):
        snapshot = super().scheduler_snapshot()
        snapshot.update(
            {
                "version": 14,
                "policy": "v13_prefill_same_host_decode_affinity",
                "decode_locality_enabled": self.decode_locality_enabled,
                "local_decode_affinity_hits": self.local_decode_affinity_hits,
                "remote_decode_fallbacks": self.remote_decode_fallbacks,
            }
        )
        return snapshot
