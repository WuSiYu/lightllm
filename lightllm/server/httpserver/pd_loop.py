import asyncio
import pickle
import websockets
import ujson as json
import socket
import httpx
import base64
import weakref
import time
import uuid
from typing import Dict, Optional, Union, List, Set
from websockets import ClientConnection
from lightllm.server.pd_io_struct import NodeRole, ObjType
from lightllm.server.httpserver.async_queue import AsyncQueue
from lightllm.utils.net_utils import get_hostname_ip
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_lightllm_websocket_max_message_size
from lightllm.server.httpserver.manager import HttpServerManager
from ..pd_io_struct import PD_Master_Obj
from lightllm.server.core.objs import StartArgs
from lightllm.server.core.objs import SamplingParams
from lightllm.utils.error_utils import NixlPrefillNodeStopGenToken

logger = init_logger(__name__)


async def timer_log(manager: HttpServerManager):
    while True:
        await asyncio.sleep(30)
        manager.first_time_costs.print_log("mean first cost")
        manager.per_token_costs.print_log("mean per token cost")
    return


async def pd_handle_loop(manager: HttpServerManager):
    assert manager.args.host not in ["127.0.0.1", "localhost"], "pd mode must specify host ip"
    if manager.args.host in ["0.0.0.0"]:
        manager.host_ip = get_hostname_ip()
    else:
        manager.host_ip = manager.args.host

    asyncio.create_task(timer_log(manager))

    id_to_handle_task: Dict[int, asyncio.Task] = {}

    while True:
        try:
            id_to_pd_master_obj = await _get_pd_master_objs(manager.args)
            logger.info(f"get pd_master_objs {id_to_pd_master_obj}")

            if id_to_pd_master_obj is not None:
                for node_id, pd_master_obj in id_to_handle_task.items():
                    if node_id not in id_to_pd_master_obj:
                        id_to_handle_task[node_id].cancel()
                        id_to_handle_task.pop(node_id, None)
                        logger.info(f"pd_handle_task {pd_master_obj} cancelled")

                for node_id, pd_master_obj in id_to_pd_master_obj.items():
                    if node_id not in id_to_handle_task:
                        id_to_handle_task[node_id] = asyncio.create_task(_pd_handle_task(manager, pd_master_obj))

            await asyncio.sleep(30)

        except Exception as e:
            logger.exception(str(e))
            await asyncio.sleep(10)


async def _pd_handle_task(manager: HttpServerManager, pd_master_obj: PD_Master_Obj):
    """
    pd_handle_loop 主要负责与 pd master 进行注册连接，然后接收pd master发来的请求，然后
    将推理结果转发给 pd master进行处理。
    """
    # 创建转发队列
    forwarding_queue = AsyncQueue()

    while True:
        forwarding_tokens_task = None
        lifecycle_task = None
        report_task = None
        worker_tasks: Set[asyncio.Task] = set()
        try:
            uri = f"ws://{pd_master_obj.host_ip_port}/pd_register"
            async with websockets.connect(
                uri, max_size=get_lightllm_websocket_max_message_size(), max_queue=(2048 * 1024, 2048 * 1023)  # 关键修改
            ) as websocket:

                sock = websocket.transport.get_extra_info("socket")
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                # Do not mutate the process-wide StartArgs while preparing a
                # registration; another worker connection may use it too.
                args_dict = dict(vars(manager.args))
                args_dict["host"] = manager.host_ip
                instance_generation = uuid.uuid4().hex
                # 发送注册信息
                regist_json = {
                    "node_id": manager.args.pd_node_id,
                    "client_ip_port": f"{manager.host_ip}:{manager.args.port}",
                    "mode": manager.pd_mode.value,
                    "start_args": args_dict,
                    "instance_generation": instance_generation,
                }

                await websocket.send(json.dumps(regist_json))
                logger.info(f"Sent registration JSON: {regist_json}")

                send_lock = asyncio.Lock()
                lifecycle_queue = AsyncQueue()
                active_group_ids: Set[int] = set()
                active_group_token_counts: Dict[int, int] = {}
                normal_pd_bundle = manager.pd_mode == NodeRole.P

                # 转发任务。生命周期事件与 token 包使用同一把锁，保证消息不会交错。
                forwarding_tokens_task = asyncio.create_task(
                    _up_tokens_to_pd_master(
                        forwarding_queue,
                        websocket,
                        send_lock=send_lock,
                        bundle_client_ip_port=regist_json["client_ip_port"],
                        instance_generation=instance_generation,
                    )
                )
                if normal_pd_bundle:
                    lifecycle_task = asyncio.create_task(
                        _up_pd_lifecycle_to_master(
                            lifecycle_queue, websocket, manager, bundle_client_ip_port=regist_json["client_ip_port"],
                            send_lock=send_lock, active_group_ids=active_group_ids,
                        )
                    )
                    report_task = asyncio.create_task(
                        _up_pd_instance_report(
                            websocket, manager, bundle_client_ip_port=regist_json["client_ip_port"],
                            send_lock=send_lock, active_group_ids=active_group_ids,
                            active_group_token_counts=active_group_token_counts,
                            instance_generation=instance_generation,
                        )
                    )

                group_req_id_to_event: Dict[int, asyncio.Event] = weakref.WeakValueDictionary()
                # 接收 pd master 发来的请求，并推理后，将生成的token转发回pd master。
                while True:
                    recv_bytes = await websocket.recv()
                    obj = pickle.loads(recv_bytes)
                    if obj[0] == ObjType.REQ:
                        prompt, sampling_params, multimodal_params = obj[1]
                        group_req_id = sampling_params.group_request_id
                        nixl_pd_event = asyncio.Event()
                        group_req_id_to_event[group_req_id] = nixl_pd_event
                        asyncio.create_task(
                            _pd_process_generate(
                                manager=manager,
                                prompt=prompt,
                                sampling_params=sampling_params,
                                multimodal_params=multimodal_params,
                                forwarding_queue=forwarding_queue,
                                nixl_pd_upload_websocket=websocket,
                                nixl_pd_event=nixl_pd_event,
                            )
                        )
                    elif obj[0] == ObjType.REQ_BUNDLE:
                        # V3 sends (type, generation, bundle_id, payload);
                        # older dispatchers used (type, bundle_id, payload).
                        if len(obj) == 4:
                            _, bundle_generation, bundle_id, bundle_payload = obj
                        else:
                            _, bundle_id, bundle_payload = obj
                            bundle_generation = instance_generation

                        rejected = []
                        reason = ""
                        if not manager.pd_mode.is_P():
                            reason = "worker is not ordinary Prefill"
                            rejected = [
                                (item[0], item[1] if len(item) in (5, 6) else item[0])
                                for item in bundle_payload
                                if len(item) in (4, 5, 6)
                            ]
                        elif bundle_generation != instance_generation:
                            logger.warning(
                                "reject stale REQ_BUNDLE generation=%s current=%s bundle=%s",
                                bundle_generation,
                                instance_generation,
                                bundle_id,
                            )
                            reason = "stale instance generation"
                            rejected = [
                                (item[0], item[1] if len(item) in (5, 6) else item[0])
                                for item in bundle_payload
                                if len(item) in (4, 5, 6)
                            ]
                        if not manager.pd_mode.is_P() or bundle_generation != instance_generation:
                            async with send_lock:
                                await websocket.send(
                                    pickle.dumps(
                                        (
                                            ObjType.BUNDLE_REJECTED,
                                            regist_json["client_ip_port"],
                                            instance_generation,
                                            bundle_id,
                                            rejected,
                                            reason,
                                        )
                                    )
                                )
                            continue

                        for item in bundle_payload:
                            if len(item) == 6:
                                (
                                    group_req_id,
                                    lease_request_id,
                                    input_token_num,
                                    prompt,
                                    sampling_params,
                                    multimodal_params,
                                ) = item
                            elif len(item) == 5:
                                group_req_id, lease_request_id, prompt, sampling_params, multimodal_params = item
                                input_token_num = 0
                            elif len(item) == 4:
                                group_req_id, prompt, sampling_params, multimodal_params = item
                                lease_request_id = group_req_id
                                input_token_num = 0
                            else:
                                logger.error("invalid REQ_BUNDLE item: %r", item)
                                continue
                            task = asyncio.create_task(
                                _pd_process_generate(
                                    manager=manager,
                                    prompt=prompt,
                                    sampling_params=sampling_params,
                                    multimodal_params=multimodal_params,
                                    forwarding_queue=forwarding_queue,
                                    nixl_pd_upload_websocket=None,
                                    nixl_pd_event=None,
                                    lifecycle_queue=lifecycle_queue,
                                    bundle_id=bundle_id,
                                    instance_generation=instance_generation,
                                    lease_request_id=lease_request_id,
                                    active_group_ids=active_group_ids,
                                    active_group_token_counts=active_group_token_counts,
                                    input_token_num=input_token_num,
                                )
                            )
                            worker_tasks.add(task)
                            task.add_done_callback(worker_tasks.discard)
                    elif obj[0] == ObjType.ABORT:
                        group_req_id = obj[1]
                        logger.warning(f"recv cmd aborted req id {group_req_id}")
                        if not (await manager.abort(group_req_id)):

                            async def delayed_abort_task(group_req_id, retry_count):
                                for _ in range(retry_count):
                                    await asyncio.sleep(5.0)
                                    if await manager.abort(group_req_id):
                                        break

                            asyncio.create_task(delayed_abort_task(group_req_id=group_req_id, retry_count=4))

                    elif obj[0] == ObjType.NIXL_REQ_DECODE_NODE_INFO:
                        _, group_req_id, decode_node_info = obj
                        nixl_pd_event = group_req_id_to_event.pop(group_req_id, None)
                        if nixl_pd_event is None:
                            logger.error(f"error in find nixl_pd_event, info: {obj}")
                            continue
                        nixl_pd_event.decode_node_info = decode_node_info
                        nixl_pd_event.set()
                    else:
                        logger.error(f"recevie error obj {str(obj)}")

        except asyncio.CancelledError:
            # 如果任务被取消，则退出循环
            logger.warning(f"forwarding_tokens_task {pd_master_obj} cancelled")
            if forwarding_tokens_task is not None:
                forwarding_tokens_task.cancel()
            if lifecycle_task is not None:
                lifecycle_task.cancel()
            if report_task is not None:
                report_task.cancel()
            for task in worker_tasks:
                task.cancel()
            return

        except Exception as e:
            logger.error("connetion to pd_master has error")
            logger.exception(str(e))
            if forwarding_tokens_task is not None:
                forwarding_tokens_task.cancel()
            if lifecycle_task is not None:
                lifecycle_task.cancel()
            if report_task is not None:
                report_task.cancel()
            for task in worker_tasks:
                task.cancel()
            await asyncio.sleep(10)
            await forwarding_queue.get_all_data()
            logger.info("reconnection to pd_master")


async def _get_pd_master_objs(args: StartArgs) -> Optional[Dict[int, PD_Master_Obj]]:
    """
    get_pd_master_objs 主要负责从 pd master 获取所有的pd master对象。
    """
    use_config_server = args.config_server_host and args.config_server_port

    # 如果不使用config_server服务来发现所有的 pd_master, 则需要使用启动参数中的
    # --pd_master_ip 和--pd_master_port 设置的唯一pd_master来进行连接, 其默认
    # node_id 为 0
    if not use_config_server:
        ans = dict()
        ans[0] = PD_Master_Obj(node_id=0, host_ip_port=f"{args.pd_master_ip}:{args.pd_master_port}")
        return ans

    # 使用 config_server 服务来发现所有的 pd_master 节点。
    uri = f"ws://{args.config_server_host}:{args.config_server_port}/registered_objects"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(uri)
            if response.status_code == 200:
                base64data = response.json()["data"]
                id_to_pd_master_obj = pickle.loads(base64.b64decode(base64data))
                return id_to_pd_master_obj
            else:
                logger.error(f"get pd_master_objs error {response.status_code}")
                return None
    except Exception as e:
        logger.exception(str(e))
        await asyncio.sleep(10)
        return None


# 触发推理的task
async def _pd_process_generate(
    manager: HttpServerManager,
    prompt: Union[str, List[int]],
    sampling_params: SamplingParams,
    multimodal_params: Dict,
    forwarding_queue: AsyncQueue,
    nixl_pd_upload_websocket: ClientConnection,
    nixl_pd_event: asyncio.Event,
    lifecycle_queue: Optional[AsyncQueue] = None,
    bundle_id: Optional[int] = None,
    lease_request_id: Optional[int] = None,
    active_group_ids: Optional[Set[int]] = None,
    instance_generation: Optional[str] = None,
    active_group_token_counts: Optional[Dict[int, int]] = None,
    input_token_num: int = 0,
):
    worker_start_time = time.time()
    lifecycle_terminal_sent = False
    base_group_req_id = sampling_params.group_request_id

    async def lifecycle_callback(kind: str, group_req_id: int, reason: Optional[object]):
        nonlocal lifecycle_terminal_sent
        if lifecycle_queue is not None and bundle_id is not None:
            await lifecycle_queue.put(
                (
                    kind,
                    instance_generation,
                    bundle_id,
                    group_req_id,
                    lease_request_id if lease_request_id is not None else group_req_id,
                    reason or "",
                )
            )
            if kind == "accepted" and active_group_ids is not None:
                active_group_ids.add(group_req_id)
                if active_group_token_counts is not None:
                    active_group_token_counts[group_req_id] = max(0, int(input_token_num or 0))
            elif kind in {"failed", "finished"}:
                lifecycle_terminal_sent = True
                if active_group_ids is not None:
                    active_group_ids.discard(group_req_id)
                if active_group_token_counts is not None:
                    active_group_token_counts.pop(group_req_id, None)

    try:
        async for sub_req_id, request_output, metadata, finish_status in manager.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            multimodal_params=multimodal_params,
            request=None,
            nixl_pd_upload_websocket=nixl_pd_upload_websocket,
            nixl_pd_event=nixl_pd_event,
            pd_lifecycle_callback=lifecycle_callback if lifecycle_queue is not None else None,
        ):
            # p d 模式下，将 token 数据放入到转发队列中, 请求id 小于0的请求是health探测请求，不用转发。
            is_health_check_req = sub_req_id < 0
            if not is_health_check_req:
                metadata["node_mode"] = manager.args.run_mode
                if lifecycle_queue is not None and bundle_id is not None and not lifecycle_terminal_sent:
                    # The first generated token is the normal-PD authority that
                    # the Prefill work has completed.  It is sent separately
                    # from TOKEN_PACKS so lease release does not depend on
                    # websocket queueing order.
                    await lifecycle_callback(
                        "finished",
                        base_group_req_id,
                        {
                            "actual_ttft": time.time() - worker_start_time,
                            "input_token_num": metadata.get("prompt_tokens", 0),
                        },
                    )
                await forwarding_queue.put((sub_req_id, request_output, metadata, finish_status))
                if active_group_ids is not None:
                    active_group_ids.discard(base_group_req_id)
                if active_group_token_counts is not None:
                    active_group_token_counts.pop(base_group_req_id, None)
    except asyncio.CancelledError:
        raise
    except NixlPrefillNodeStopGenToken as e:
        logger.info(f"nixl prefill node stop gen token for group_request_id {e.group_request_id}")
    except BaseException as e:
        logger.error(str(e))
    finally:
        # A client abort can make the local generator finish without raising.
        # Tell the master so a bundle lease cannot remain admitted forever.
        if (
            lifecycle_queue is not None
            and bundle_id is not None
            and not lifecycle_terminal_sent
        ):
            try:
                await lifecycle_callback(
                    "failed",
                    base_group_req_id,
                    "worker generation ended before Prefill finished",
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("failed to report terminal Prefill lifecycle state")


# 转发token的task
async def _up_tokens_to_pd_master(
    forwarding_queue: AsyncQueue,
    websocket: ClientConnection,
    send_lock: Optional[asyncio.Lock] = None,
    bundle_client_ip_port: Optional[str] = None,
    instance_generation: Optional[str] = None,
):
    while True:
        handle_list = await forwarding_queue.wait_to_get_all_data()

        if handle_list:
            load_info: dict = _get_load_info()
            if bundle_client_ip_port is None and instance_generation is None:
                obj = (ObjType.TOKEN_PACKS, handle_list, load_info)
            else:
                obj = (ObjType.TOKEN_PACKS, handle_list, load_info, bundle_client_ip_port, instance_generation)
            payload = pickle.dumps(obj)
            if send_lock is None:
                await websocket.send(payload)
            else:
                async with send_lock:
                    await websocket.send(payload)


async def _up_pd_lifecycle_to_master(
    lifecycle_queue: AsyncQueue,
    websocket: ClientConnection,
    manager: HttpServerManager,
    bundle_client_ip_port: str,
    send_lock: asyncio.Lock,
    active_group_ids: Set[int],
    instance_generation: Optional[str] = None,
):
    while True:
        events = await lifecycle_queue.wait_to_get_all_data()
        for event in events:
            if len(event) == 6:
                kind, instance_generation, bundle_id, group_req_id, lease_request_id, detail = event
            else:
                kind, bundle_id, group_req_id, lease_request_id, detail = event
                instance_generation = None
            if kind == "accepted":
                obj = (ObjType.BUNDLE_ACCEPTED, bundle_client_ip_port, instance_generation, bundle_id, [lease_request_id])
            elif kind == "failed":
                active_group_ids.discard(group_req_id)
                obj = (ObjType.PREFILL_FAILED, bundle_client_ip_port, instance_generation,
                       group_req_id, lease_request_id, detail)
            elif kind == "finished":
                active_group_ids.discard(group_req_id)
                obj = (ObjType.PREFILL_FINISHED, bundle_client_ip_port, instance_generation,
                       group_req_id, lease_request_id, detail)
            else:
                logger.warning("unknown normal-PD lifecycle event: %s", kind)
                continue
            async with send_lock:
                await websocket.send(pickle.dumps(obj))


async def _up_pd_instance_report(
    websocket: ClientConnection,
    manager: HttpServerManager,
    bundle_client_ip_port: str,
    send_lock: asyncio.Lock,
    active_group_ids: Set[int],
    instance_generation: Optional[str] = None,
    active_group_token_counts: Optional[Dict[int, int]] = None,
):
    """Report worker load plus the requests admitted through the bundle path."""
    report_seq = 0
    while True:
        await asyncio.sleep(0.2)
        try:
            report_seq += 1
            report = _get_load_info()
            report["queued_requests"] = len(active_group_ids)
            report["queued_group_ids"] = sorted(active_group_ids)
            report["queued_tokens"] = sum((active_group_token_counts or {}).values())
            report["running_requests"] = len(active_group_ids)
            report["running_group_ids"] = sorted(active_group_ids)
            report["running_tokens"] = sum((active_group_token_counts or {}).values())
            async with send_lock:
                await websocket.send(
                    pickle.dumps(
                        (ObjType.INSTANCE_REPORT, bundle_client_ip_port, instance_generation, report_seq, report)
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("failed to send normal-PD instance report")


# 获取节点负载信息
def _get_load_info() -> dict:

    from lightllm.server.api_http import g_objs

    assert g_objs.shared_token_load is not None, "shared_token_load is not initialized"
    args = g_objs.args
    dp_size_in_node = max(1, args.dp // args.nnodes)

    # 获取当前每个 dp 的负载，数值含义为当前的 token 总容量使用率， 上报给 PD_Master 用于做
    # 调度决策。
    current_load = [
        float(g_objs.shared_token_load.get_dynamic_max_load(dp_index)) for dp_index in range(dp_size_in_node)
    ]
    mean_node_load = sum(current_load) / len(current_load)
    load_info = {
        "total_token_usage_rate": mean_node_load,
        "client_ip_port": f"{g_objs.httpserver_manager.host_ip}:{g_objs.args.port}",
    }
    return load_info
