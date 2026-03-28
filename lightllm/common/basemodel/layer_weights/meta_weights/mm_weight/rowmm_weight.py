import torch
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import MMWeightTpl, BMMWeightTpl
from lightllm.common.quantization import Quantcfg
from lightllm.common.quantization.no_quant import NoQuantization
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.common.quantization.quantize_method import QuantizationMethod
from typing import Dict, List, Optional, Union
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size
from .mm_slicer import get_row_slice_mixin

class ROWMMWeight(MMWeightTpl):
    def __init__(
        self,
        in_dim: int,
        out_dims: Optional[Union[int, List[int]]],
        weight_names: Union[str, List[str]],
        data_type: torch.dtype,
        bias_names: Optional[Union[str, List[str]]] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        self.tp_rank_ = tp_rank if tp_rank is not None else get_current_rank_in_dp()
        self.tp_world_size_ = tp_world_size if tp_world_size is not None else get_dp_world_size()
        out_dims = [self._get_tp_dim(out_dim) for out_dim in out_dims]
        super().__init__(
            in_dim=in_dim,
            out_dims=out_dims,
            weight_names=weight_names,
            bias_names=bias_names,
            data_type=data_type,
            quant_method=quant_method,
            tp_rank=self.tp_rank_,
            tp_world_size=self.tp_world_size_,
        )
        self.param_slicer = get_row_slice_mixin(
            self.quant_method.method_name, tp_rank=self.tp_rank_, tp_world_size=self.tp_world_size_
        )

    def _to_gpu_device(self):
        super()._to_gpu_device()

        from lightllm.common.basemodel.layer_weights.meta_weights.shared_weight import TensorClient, TensorServer
        if TensorClient():
            original_shape = self.mm_param.weight.shape
            weight, meta = TensorClient().get_tensor_blocking(self.weight_names[0])

            assert self.tp_world_size_ % meta["tp_world_size"] == 0, \
                f"shared_weight tp_world_size not align: master={meta['tp_world_size']} vs slave={self.tp_world_size_}"

            if meta["tp_world_size"] == self.tp_world_size_:
                # Same TP: zero-copy, use master's weight directly
                assert meta["tp_rank"] == self.tp_rank_
                self.mm_param.weight = weight
            else:
                # Flex TP: slave has finer partition, need to slice master's weight
                assert meta["tp_world_size"] < self.tp_world_size_
                tp_sub_partation_world_size = self.tp_world_size_ // meta["tp_world_size"]
                assert self.tp_rank_ // tp_sub_partation_world_size == meta["tp_rank"], \
                    f"shared_weight master tp_rank mismatch: wanted {self.tp_rank_ // tp_sub_partation_world_size}, got {meta['tp_rank']}"

                tp_sub_partation_rank = self.tp_rank_ % tp_sub_partation_world_size
                weight_comb_numbers = len(self.weight_names)

                # ROWMMWeight partitions output dim (dim 0 for NoQuant layout (out, in))
                # For quantized layout (in, out), transpose first to get (out, in)
                transposed = not isinstance(self.quant_method, NoQuantization)
                if transposed:
                    weight = weight.transpose(0, 1)

                per_comb_out = weight.shape[0] // weight_comb_numbers
                tp_slice_size = per_comb_out // tp_sub_partation_world_size
                start = tp_slice_size * tp_sub_partation_rank
                end = tp_slice_size * (tp_sub_partation_rank + 1)

                if weight_comb_numbers == 1:
                    # Single weight: row slice is contiguous, zero-copy
                    weight = weight[start:end]
                else:
                    # Multi weight (e.g. gate_up_proj): 3D non-contiguous view, no copy needed
                    weight = weight.view(weight_comb_numbers, per_comb_out, -1)
                    weight = weight[:, start:end]  # (N, tp_slice, in_dim)
                    if transposed:
                        # Quantized: need contiguous reshape for quant apply path
                        weight = weight.contiguous().reshape(-1, weight.shape[-1])
                    else:
                        # NoQuant: permute to (N, in_dim, tp_slice) for mm() 3D path
                        weight = weight.permute(0, 2, 1)

                if transposed:
                    weight = weight.transpose(0, 1)

                self.mm_param.weight = weight

            # Rebuild mm_param_list to point to new weight (release references to old weight)
            if self.mm_param.weight.ndim == 3:
                # 3D non-contiguous (NoQuant multi-weight flex TP)
                assert self.mm_param.weight.numel() == original_shape[0] * original_shape[1], \
                    f"shared weight numel mismatch: expected {original_shape[0] * original_shape[1]}, got {self.mm_param.weight.numel()}"
                for i, wp in enumerate(self.mm_param_list):
                    wp.weight = self.mm_param.weight[i]
                    wp.load_ok = [True, True, True]
            else:
                assert self.mm_param.weight.shape == original_shape, \
                    f"shared weight shape mismatch: expected {original_shape}, got {self.mm_param.weight.shape}"
                new_weights = torch.split(self.mm_param.weight, self.out_dims, dim=-2)
                for i, wp in enumerate(self.mm_param_list):
                    wp.weight = new_weights[i]
                    wp.load_ok = [True, True, True]

        elif TensorServer():
            meta = dict(tp_rank=self.tp_rank_, tp_world_size=self.tp_world_size_)
            TensorServer().register(self.weight_names[0], self.mm_param.weight, meta)


class KVROWNMMWeight(MMWeightTpl):
    def __init__(
        self,
        in_dim: int,
        kv_head_num: int,
        head_dim: int,
        weight_names: Union[str, List[str]],
        data_type: torch.dtype,
        bias_names: Optional[Union[str, List[str]]] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        self.tp_rank_ = tp_rank if tp_rank is not None else get_current_rank_in_dp()
        self.tp_world_size_ = tp_world_size if tp_world_size is not None else get_dp_world_size()
        self.repeat_times = 1
        assert kv_head_num % self.tp_world_size_ == 0 or self.tp_world_size_ % kv_head_num == 0, (
            f"kv_head_num must be divisible by tp_world_size_ or "
            f"tp_world_size_ must be divisible by kv_head_num, "
            f"but found: {kv_head_num} % {self.tp_world_size_}"
        )
        kv_hidden_size = self._get_tp_padded_head_num(kv_head_num) * head_dim
        out_dims = [kv_hidden_size, kv_hidden_size]
        super().__init__(
            in_dim=in_dim,
            out_dims=out_dims,
            weight_names=weight_names,
            data_type=data_type,
            bias_names=bias_names,
            quant_method=quant_method,
            tp_rank=self.tp_rank_,
            tp_world_size=self.tp_world_size_,
        )
        self.param_slicer = get_row_slice_mixin(
            self.quant_method.method_name,
            tp_rank=self.tp_rank_,
            tp_world_size=self.tp_world_size_,
            repeat_times=self.repeat_times,
        )

    def _get_tp_padded_head_num(self, head_num: int):
        if head_num % self.tp_world_size_ == 0:
            return head_num // self.tp_world_size_
        elif self.tp_world_size_ % head_num == 0:
            self.repeat_times = self.tp_world_size_ // head_num
            return self.repeat_times * head_num // self.tp_world_size_
        else:
            raise ValueError(
                f"head_num must be divisible by tp_world_size_ or "
                f"tp_world_size_ must be divisible by head_num, "
                f"but found: {head_num} % {self.tp_world_size_}"
            )

    def _to_gpu_device(self):
        # KVROWNMMWeight has 2 combined weights (K, V) with equal sizes,
        # same ROW partition logic as ROWMMWeight
        super()._to_gpu_device()

        from lightllm.common.basemodel.layer_weights.meta_weights.shared_weight import TensorClient, TensorServer
        if TensorClient():
            original_shape = self.mm_param.weight.shape
            weight, meta = TensorClient().get_tensor_blocking(self.weight_names[0])

            assert self.tp_world_size_ % meta["tp_world_size"] == 0, \
                f"shared_weight tp_world_size not align: master={meta['tp_world_size']} vs slave={self.tp_world_size_}"

            if meta["tp_world_size"] == self.tp_world_size_:
                assert meta["tp_rank"] == self.tp_rank_
                self.mm_param.weight = weight
            else:
                assert meta["tp_world_size"] < self.tp_world_size_
                tp_sub_partation_world_size = self.tp_world_size_ // meta["tp_world_size"]
                assert self.tp_rank_ // tp_sub_partation_world_size == meta["tp_rank"]

                tp_sub_partation_rank = self.tp_rank_ % tp_sub_partation_world_size
                weight_comb_numbers = len(self.weight_names)

                transposed = not isinstance(self.quant_method, NoQuantization)
                if transposed:
                    weight = weight.transpose(0, 1)

                per_comb_out = weight.shape[0] // weight_comb_numbers
                tp_slice_size = per_comb_out // tp_sub_partation_world_size
                start = tp_slice_size * tp_sub_partation_rank
                end = tp_slice_size * (tp_sub_partation_rank + 1)

                # Multi weight (K, V): 3D non-contiguous view, no copy needed
                weight = weight.view(weight_comb_numbers, per_comb_out, -1)
                weight = weight[:, start:end]  # (N, tp_slice, in_dim)
                if transposed:
                    # Quantized: need contiguous reshape for quant apply path
                    weight = weight.contiguous().reshape(-1, weight.shape[-1])
                else:
                    # NoQuant: permute to (N, in_dim, tp_slice) for mm() 3D path
                    weight = weight.permute(0, 2, 1)

                if transposed:
                    weight = weight.transpose(0, 1)

                self.mm_param.weight = weight

            # Rebuild mm_param_list to point to new weight (release references to old weight)
            if self.mm_param.weight.ndim == 3:
                assert self.mm_param.weight.numel() == original_shape[0] * original_shape[1], \
                    f"shared weight numel mismatch: expected {original_shape[0] * original_shape[1]}, got {self.mm_param.weight.numel()}"
                for i, wp in enumerate(self.mm_param_list):
                    wp.weight = self.mm_param.weight[i]
                    wp.load_ok = [True, True, True]
            else:
                assert self.mm_param.weight.shape == original_shape, \
                    f"shared weight shape mismatch: expected {original_shape}, got {self.mm_param.weight.shape}"
                new_weights = torch.split(self.mm_param.weight, self.out_dims, dim=-2)
                for i, wp in enumerate(self.mm_param_list):
                    wp.weight = new_weights[i]
                    wp.load_ok = [True, True, True]

        elif TensorServer():
            meta = dict(tp_rank=self.tp_rank_, tp_world_size=self.tp_world_size_)
            TensorServer().register(self.weight_names[0], self.mm_param.weight, meta)


class QKVROWNMMWeight(MMWeightTpl):
    def __init__(
        self,
        in_dim: int,
        q_head_num: int,
        kv_head_num: int,
        head_dim: int,
        weight_names: Union[str, List[str]],
        data_type: torch.dtype,
        bias_names: Optional[Union[str, List[str]]] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        self.tp_rank_ = tp_rank if tp_rank is not None else get_current_rank_in_dp()
        self.tp_world_size_ = tp_world_size if tp_world_size is not None else get_dp_world_size()
        self.q_repeat_times = 1
        self.kv_repeat_times = 1
        assert q_head_num % self.tp_world_size_ == 0, (
            f"q_head_num must be divisible by tp_world_size_, " f"but found: {q_head_num} % {self.tp_world_size_}"
        )
        assert kv_head_num % self.tp_world_size_ == 0 or self.tp_world_size_ % kv_head_num == 0, (
            f"kv_head_num must be divisible by tp_world_size_ or "
            f"tp_world_size_ must be divisible by kv_head_num, "
            f"but found: {kv_head_num} % {self.tp_world_size_}"
        )
        q_hidden_size = (q_head_num // self.tp_world_size_) * head_dim
        kv_hidden_size = self._get_tp_padded_head_num(kv_head_num) * head_dim
        out_dims = [q_hidden_size, kv_hidden_size, kv_hidden_size]
        super().__init__(
            in_dim=in_dim,
            out_dims=out_dims,
            weight_names=weight_names,
            data_type=data_type,
            bias_names=bias_names,
            quant_method=quant_method,
            tp_rank=self.tp_rank_,
            tp_world_size=self.tp_world_size_,
        )
        self.q_param_slicer = get_row_slice_mixin(
            self.quant_method.method_name,
            tp_rank=self.tp_rank_,
            tp_world_size=self.tp_world_size_,
            repeat_times=self.q_repeat_times,
        )
        self.kv_param_slicer = get_row_slice_mixin(
            self.quant_method.method_name,
            tp_rank=self.tp_rank_,
            tp_world_size=self.tp_world_size_,
            repeat_times=self.kv_repeat_times,
        )

    def _get_param_slicer(self, sub_child_index: int):
        """
        sub_child_index:
            0 -> q
            1 -> k
            2 -> v
        q 使用 q_param_slicer, k / v 使用 kv_param_slicer.
        """
        if sub_child_index == 0:
            return self.q_param_slicer
        else:
            return self.kv_param_slicer

    def _get_tp_padded_head_num(self, head_num: int):
        if head_num % self.tp_world_size_ == 0:
            return head_num // self.tp_world_size_
        elif self.tp_world_size_ % head_num == 0:
            self.kv_repeat_times = self.tp_world_size_ // head_num
            return self.kv_repeat_times * head_num // self.tp_world_size_
        else:
            raise ValueError(
                f"head_num must be divisible by tp_world_size_ or "
                f"tp_world_size_ must be divisible by head_num, "
                f"but found: {head_num} % {self.tp_world_size_}"
            )

    def _to_gpu_device(self):
        # QKVROWNMMWeight has 3 combined weights (Q, K, V) with potentially different sizes,
        # need to slice each sub-weight individually
        super()._to_gpu_device()

        from lightllm.common.basemodel.layer_weights.meta_weights.shared_weight import TensorClient, TensorServer
        if TensorClient():
            original_shape = self.mm_param.weight.shape
            weight, meta = TensorClient().get_tensor_blocking(self.weight_names[0])

            assert self.tp_world_size_ % meta["tp_world_size"] == 0, \
                f"shared_weight tp_world_size not align: master={meta['tp_world_size']} vs slave={self.tp_world_size_}"

            if meta["tp_world_size"] == self.tp_world_size_:
                assert meta["tp_rank"] == self.tp_rank_
                self.mm_param.weight = weight
            else:
                assert meta["tp_world_size"] < self.tp_world_size_
                tp_sub_partation_world_size = self.tp_world_size_ // meta["tp_world_size"]
                assert self.tp_rank_ // tp_sub_partation_world_size == meta["tp_rank"]

                tp_sub_partation_rank = self.tp_rank_ % tp_sub_partation_world_size

                transposed = not isinstance(self.quant_method, NoQuantization)
                if transposed:
                    weight = weight.transpose(0, 1)

                # Non-uniform out_dims (Q, K, V may differ): compute master's out_dims
                # master_out_dims[i] = slave_out_dims[i] * tp_sub_partation_world_size
                master_out_dims = [d * tp_sub_partation_world_size for d in self.out_dims]
                sub_weights = torch.split(weight, master_out_dims, dim=0)
                sliced = []
                for sub_w, slave_dim in zip(sub_weights, self.out_dims):
                    start = slave_dim * tp_sub_partation_rank
                    end = slave_dim * (tp_sub_partation_rank + 1)
                    sliced.append(sub_w[start:end])
                weight = torch.cat(sliced, dim=0)

                if transposed:
                    weight = weight.transpose(0, 1)

                self.mm_param.weight = weight

            assert self.mm_param.weight.shape == original_shape, \
                f"shared weight shape mismatch: expected {original_shape}, got {self.mm_param.weight.shape}"

            new_weights = torch.split(self.mm_param.weight, self.out_dims, dim=-2)
            for i, wp in enumerate(self.mm_param_list):
                wp.weight = new_weights[i]
                wp.load_ok = [True, True, True]

        elif TensorServer():
            meta = dict(tp_rank=self.tp_rank_, tp_world_size=self.tp_world_size_)
            TensorServer().register(self.weight_names[0], self.mm_param.weight, meta)


class ROWBMMWeight(BMMWeightTpl):
    def __init__(
        self,
        dim0: int,
        dim1: int,
        dim2: int,
        weight_names: Union[str, List[str]],
        data_type: torch.dtype,
        bias_names: Optional[Union[str, List[str]]] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        self.tp_rank_ = tp_rank if tp_rank is not None else get_current_rank_in_dp()
        self.tp_world_size_ = tp_world_size if tp_world_size is not None else get_dp_world_size()
        assert (
            dim0 % self.tp_world_size_ == 0
        ), f"dim0 of bmm must be divisible by tp_world_size_, but found: {dim0} % {self.tp_world_size_}"
        dim0 = dim0 // self.tp_world_size_
        super().__init__(
            dim0=dim0,
            dim1=dim1,
            dim2=dim2,
            weight_names=weight_names,
            bias_names=bias_names,
            data_type=data_type,
            quant_method=quant_method,
            tp_rank=self.tp_rank_,
            tp_world_size=self.tp_world_size_,
        )
        self.param_slicer = get_row_slice_mixin(
            quant_method_name="none", tp_rank=self.tp_rank_, tp_world_size=self.tp_world_size_
        )
