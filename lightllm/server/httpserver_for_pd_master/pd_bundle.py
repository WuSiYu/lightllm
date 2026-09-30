"""Master-side normal-PD Prefill bundle dispatcher.

Selector admission and websocket transmission are intentionally separate.  A
selector lease bounds how much work a Prefill instance may own; this dispatcher
coalesces already-admitted requests for the same concrete instance and sends
    one atomic ``REQ_BUNDLE`` envelope.  Each item carries both its block id
    and original lease id so completion cannot release a neighboring request.
    The worker still performs final KV-safe
batch packing in its local Router.
"""

from __future__ import annotations

import asyncio
import itertools
import pickle
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional

from lightllm.server.pd_io_struct import ObjType, PD_Client_Obj


@dataclass
class PrefillBundleItem:
    group_request_id: int
    lease_request_id: int
    prompt: object
    sampling_params: object
    multimodal_params: object
    input_token_num: int
    sent_future: asyncio.Future
    send_state: str = "queued"
    on_send_failed: Optional[Callable[[BaseException], Awaitable[None]]] = None


class PrefillBundleDispatchCancelled(Exception):
    """Cancellation raced with a bundle send.

    ``sent`` distinguishes a queued item that was safely removed from one
    already handed to the worker websocket. This is a regular exception
    internally because asyncio tasks normalize all ``CancelledError``
    subclasses; the master converts it back to normal cancellation after
    recording the send state.
    """

    def __init__(self, *, sent: bool) -> None:
        super().__init__()
        self.sent = bool(sent)


class PrefillBundleDispatcher:
    """Coalesce normal-PD requests per Prefill websocket.

    ``batch_window_s`` is a maximum wait, not a fixed sleep on every request.
    A token/request trigger flushes immediately.  A per-node send lock avoids
    concurrent websocket writes from independent HTTP request coroutines.
    """

    def __init__(
        self,
        *,
        batch_window_s: float = 0.020,
        token_trigger: int = 4096,
        max_bundle_tokens: int = 8192,
        max_bundle_requests: int = 64,
    ) -> None:
        self.batch_window_s = max(0.0, float(batch_window_s))
        self.token_trigger = max(1, int(token_trigger))
        self.max_bundle_tokens = max(1, int(max_bundle_tokens))
        self.max_bundle_requests = max(1, int(max_bundle_requests))
        self._queues: Dict[str, List[PrefillBundleItem]] = {}
        self._flush_tasks: Dict[str, asyncio.Task] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._bundle_counter = itertools.count(1)
        self._state_lock = asyncio.Lock()

    def _node_key(self, p_node: PD_Client_Obj) -> str:
        generation = getattr(p_node, "instance_generation", None)
        return f"{p_node.client_ip_port}#{generation or 'legacy'}"

    async def enqueue(
        self,
        p_node: PD_Client_Obj,
        group_request_id: int,
        prompt,
        sampling_params,
        multimodal_params,
        input_token_num: int = 0,
        lease_request_id: Optional[int] = None,
        on_send_failed: Optional[Callable[[BaseException], Awaitable[None]]] = None,
    ) -> None:
        loop = asyncio.get_running_loop()
        item = PrefillBundleItem(
            group_request_id=group_request_id,
            lease_request_id=group_request_id if lease_request_id is None else int(lease_request_id),
            prompt=prompt,
            sampling_params=sampling_params,
            multimodal_params=multimodal_params,
            input_token_num=max(0, int(input_token_num or 0)),
            sent_future=loop.create_future(),
            on_send_failed=on_send_failed,
        )
        node_key = self._node_key(p_node)
        async with self._state_lock:
            queue = self._queues.setdefault(node_key, [])
            queue.append(item)
            token_sum = sum(e.input_token_num for e in queue)
            should_flush = (
                self.batch_window_s <= 0
                or len(queue) >= self.max_bundle_requests
                or token_sum >= self.token_trigger
            )
            if should_flush:
                old_task = self._flush_tasks.pop(node_key, None)
                if old_task is not None:
                    old_task.cancel()
                asyncio.create_task(self._flush_node(p_node))
            elif node_key not in self._flush_tasks:
                self._flush_tasks[node_key] = asyncio.create_task(self._delayed_flush(p_node))
        try:
            await item.sent_future
        except asyncio.CancelledError:
            # If the request disappears during the coalescing window, remove
            # it before a delayed flush can send work for a lease that the
            # master has already failed/aborted.
            async with self._state_lock:
                sent = item.send_state in {"sending", "sent"}
                queue = self._queues.get(node_key)
                if queue is not None:
                    try:
                        queue.remove(item)
                    except ValueError:
                        pass  # _flush_node already owns the item.
                    if not queue:
                        self._queues.pop(node_key, None)
                        flush_task = self._flush_tasks.pop(node_key, None)
                        if flush_task is not None and flush_task is not asyncio.current_task():
                            flush_task.cancel()
            if not item.sent_future.done():
                item.sent_future.cancel()
            if sent:
                raise PrefillBundleDispatchCancelled(sent=True) from None
            raise

    async def _delayed_flush(self, p_node: PD_Client_Obj) -> None:
        try:
            await asyncio.sleep(self.batch_window_s)
            await self._flush_node(p_node)
        except asyncio.CancelledError:
            return

    async def _flush_node(self, p_node: PD_Client_Obj) -> None:
        node_key = self._node_key(p_node)
        async with self._state_lock:
            task = self._flush_tasks.get(node_key)
            current = asyncio.current_task()
            if task is not None and task is not current:
                self._flush_tasks.pop(node_key, None)
            queue = self._queues.pop(node_key, [])
            items = []
            token_sum = 0
            while queue and len(items) < self.max_bundle_requests:
                candidate = queue[0]
                candidate_tokens = candidate.input_token_num
                if items and token_sum + candidate_tokens > self.max_bundle_tokens:
                    break
                items.append(queue.pop(0))
                token_sum += candidate_tokens
            if task is current:
                self._flush_tasks.pop(node_key, None)
            if queue:
                self._queues[node_key] = queue
                if node_key not in self._flush_tasks:
                    self._flush_tasks[node_key] = asyncio.create_task(self._delayed_flush(p_node))
            # Mark ownership before releasing the state lock. A caller may be
            # cancelled immediately after this point while the per-node send
            # lock is still waiting on an earlier bundle.
            for item in items:
                item.send_state = "sending"
        if not items:
            return

        bundle_id = next(self._bundle_counter)
        payload = [
            (
                item.group_request_id,
                item.lease_request_id,
                item.input_token_num,
                item.prompt,
                item.sampling_params,
                item.multimodal_params,
            )
            for item in items
        ]
        try:
            lock = self._locks.setdefault(node_key, asyncio.Lock())
            async with lock:
                await p_node.websocket.send_bytes(
                    pickle.dumps(
                        (ObjType.REQ_BUNDLE, getattr(p_node, "instance_generation", None), bundle_id, payload),
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                )
        except BaseException as exc:
            async with self._state_lock:
                for item in items:
                    item.send_state = "failed"
            for item in items:
                if item.on_send_failed is not None:
                    asyncio.create_task(item.on_send_failed(exc))
            for item in items:
                if not item.sent_future.done():
                    item.sent_future.set_exception(exc)
            return
        async with self._state_lock:
            for item in items:
                item.send_state = "sent"
        for item in items:
            if not item.sent_future.done():
                item.sent_future.set_result(None)
