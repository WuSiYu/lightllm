import torch
import rpyc
import threading
import socket
from rpyc.utils.server import ThreadedServer
from typing import Dict, Any, Tuple, Optional


# helper: singleton decorator
def singleton(cls):
    instances = {}
    _lock = threading.Lock()

    def get_instance(*args, **kwargs):
        if cls not in instances:
            with _lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance


class TensorShareService(rpyc.Service):
    """
    RPyC Service exposed to clients.
    It accesses the global registry of the TensorServer.
    """
    def __init__(self, registry: Dict):
        self._registry = registry

    def on_connect(self, conn):
        pass

    def on_disconnect(self, conn):
        pass

    def exposed_get_tensor_handle(self, tensor_id: str) -> Optional[Dict]:
        """
        Returns the IPC handle and metadata for a tensor.
        NOTE: We do NOT return the tensor object itself to avoid pickling the data.
        """
        if tensor_id not in self._registry:
            return None

        item = self._registry[tensor_id]
        tensor = item['tensor']
        user_meta = item['meta']

        # Ensure tensor is in contiguous memory for safe sharing
        if not tensor.is_contiguous():
            # In a real app, you might want to warn here or handle it differently
            # because making it contiguous creates a copy.
            # Ideally, register contiguous tensors only.
            pass

        # Get the UntypedStorage (raw memory)
        storage = tensor.untyped_storage()

        # Determine strict size info
        # share_cuda_() returns a handle that can be pickled
        ipc_handle = storage._share_cuda_()

        # We pack all necessary info to reconstruct the tensor view on the client
        transport_data = {
            'ipc_handle': ipc_handle,
            'device': tensor.device.index, # GPU Index
            'shape': tuple(tensor.shape),
            'stride': tuple(tensor.stride()),
            'dtype': tensor.dtype,
            'storage_offset': tensor.storage_offset(),
            'storage_size_bytes': storage.nbytes(),
            'storage_size_elements': storage.size(),
            'user_meta': user_meta # User defined metadata (rank, etc.)
        }
        return transport_data


@singleton
class TensorServer:
    def __init__(self):
        self._registry = {} # Holds references to tensors: {id: {'tensor': t, 'meta': m}}
        self._server_thread = None
        self._rpyc_server = None
        self._running = False

    def __bool__(self):
        return self._running

    def start(self, port: int):
        """
        Starts the RPyC server in a background thread.
        """
        host = '127.0.0.1'  # only local connections make sense for tensor sharing

        if self._running:
            print(f"[TensorServer] Already running.")
            return

        self._running = True

        # Factory to pass the registry to the service
        class BoundService(TensorShareService):
            def __init__(self_service, conn):
                super().__init__(self._registry)

        self._rpyc_server = ThreadedServer(
            BoundService,
            port=port,
            hostname=host,
            protocol_config={"allow_public_attrs": True, "allow_all_attrs": True}
        )

        self._server_thread = threading.Thread(target=self._rpyc_server.start, daemon=True)
        self._server_thread.start()
        print(f"[TensorServer] Started on {host}:{port}")

    def register(self, tensor_id: str, tensor: torch.Tensor, meta: Dict[str, Any] = None):
        """
        Register a tensor to be shared.
        IMPORTANT: The tensor must remain in scope (alive) in this process
        as long as clients are using it. This registry keeps the reference.
        """
        if not tensor.is_cuda:
            raise ValueError("TensorServer currently only supports CUDA tensors for IPC sharing.")

        # Ensure contiguous memory layout for simpler reconstruction
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        self._registry[tensor_id] = {
            'tensor': tensor,
            'meta': meta or {}
        }
        print(f"[TensorServer] Registered tensor '{tensor_id}' {tensor.shape} on {tensor.device}")

    def stop(self):
        if self._rpyc_server:
            self._rpyc_server.close()
            self._running = False
            print("[TensorServer] Stopped.")


@singleton
class TensorClient:
    def __init__(self):
        self._conn = None

    def __bool__(self):
        return self._conn is not None

    def connect(self, port: int):
        host = '127.0.0.1'  # only local connections make sense for tensor sharing

        if self._conn and not self._conn.closed:
            return

        # Increase generic timeout/max header for large metadata if necessary
        self._conn = rpyc.connect(
            host,
            port,
            config={"allow_public_attrs": True, "allow_all_attrs": True}
        )
        print(f"[TensorClient] Connected to {host}:{port}")

    def get_tensor(self, tensor_id: str) -> Tuple[Optional[torch.Tensor], Optional[Dict]]:
        """
        Fetches the tensor handle from server and reconstructs the tensor
        on this process's GPU context (Zero-Copy).
        """
        if not self._conn or self._conn.closed:
            raise ConnectionError("Client not connected. Call connect() first.")

        # 1. Fetch handle and metadata via RPyC
        # We use value() to force RPyC to bring the dict data over by value immediately
        remote_data = self._conn.root.get_tensor_handle(tensor_id)

        if remote_data is None:
            return None, None

        # Convert netref to local dict if necessary (usually handled by rpyc auto-proxy,
        # but creating a local copy is safer for the reconstruction step)
        if hasattr(remote_data, '__dict__') or isinstance(remote_data, object):
            # If it's a netref, accessing items fetches them
            pass

        ipc_handle = remote_data['ipc_handle']
        device_idx = remote_data['device']
        shape = remote_data['shape']
        stride = remote_data['stride']
        dtype = remote_data['dtype']
        storage_offset = remote_data['storage_offset']
        storage_size_elements = remote_data['storage_size_elements']
        user_meta = remote_data['user_meta']

        # 2. Reconstruct the Storage from the IPC handle
        # This attaches to the existing GPU memory allocation
        device = torch.device(f"cuda:{device_idx}")

        # PyTorch Internal API to rebuild storage from IPC handle
        # format: _new_shared_cuda(device, handle, size_in_elements, view_offset_bytes=0)
        # Note: API requirements vary slightly by PyTorch version, this is standard for 1.13+ / 2.0+
        new_storage = torch.UntypedStorage._new_shared_cuda(
            device,
            ipc_handle,
            storage_size_elements
        )

        # 3. Create the Tensor View
        # We create an empty tensor and set its data to point to the shared storage
        shared_tensor = torch.tensor([], dtype=dtype, device=device)
        shared_tensor.set_(new_storage, storage_offset, shape, stride)

        return shared_tensor, user_meta

    def close(self):
        if self._conn:
            self._conn.close()
