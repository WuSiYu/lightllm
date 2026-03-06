import torch
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import (
    MMWeightTpl,
)
from lightllm.common.quantization import Quantcfg
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.common.quantization.quantize_method import QuantizationMethod
from typing import Dict, List, Optional, Union
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size
from .mm_slicer import get_col_slice_mixin


class COLMMWeight(MMWeightTpl):
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
        in_dim = self._get_tp_dim(in_dim)
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
        self.param_slicer = get_col_slice_mixin(
            self.quant_method.method_name, tp_rank=self.tp_rank_, tp_world_size=self.tp_world_size_
        )

    def _to_gpu_device(self):
        super()._to_gpu_device()

        from lightllm.common.basemodel.layer_weights.meta_weights.shared_weight import TensorClient, TensorServer
        if TensorClient():
            from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.rowmm_weight import buf_print
            print = buf_print()

            print("\n\n\n\nCOLMMWeight _to_gpu_device")
            print(f"[TensorClient].{get_current_device_id()} local weight {self.weight_names}[0] to replace ({self.mm_param.weight.shape}): {self.mm_param.weight}")

            weight, meta = TensorClient().get_tensor(self.weight_names[0])    # replace weight with shared weight to free GPU memory
            original_shape = self.mm_param.weight.shape
            remote_original_shape = weight.shape
            print(f"[TensorClient].{get_current_device_id()} loaded weight: {self.weight_names[0]} tp_rank: {meta['tp_rank']} tp_world_size: {meta['tp_world_size']}")
            print(f"[TensorClient].{get_current_device_id()} (ours: tp_rank: {self.tp_rank_} tp_world_size: {self.tp_world_size_})")
            assert meta["tp_world_size"] < self.tp_world_size_ and self.tp_world_size_ % meta["tp_world_size"] == 0, \
                f"[TensorClient].{get_current_device_id()} shared_weight tp_world_size not align: master={meta['tp_world_size']} vs slave={self.tp_world_size_}"
            tp_sub_partation_world_size = self.tp_world_size_ // meta["tp_world_size"]
            assert self.tp_rank_ // tp_sub_partation_world_size == meta["tp_rank"], \
                f"[TensorClient].{get_current_device_id()} shared_weight master tp_rank not correct: wanted {self.tp_rank_ // tp_sub_partation_world_size}, master has {meta['tp_rank']}"

            tp_sub_partation_rank = self.tp_rank_ % tp_sub_partation_world_size
            transposed = self.quant_method is None  # FIXME: make it more robust
            weight_comb_numbers = len(self.weight_names)
            print(f"[TensorClient].{get_current_device_id()} {weight_comb_numbers=}, {transposed=}")
            if transposed:
                weight = weight.transpose(0, 1)

            tp_sub_partation_size = weight.shape[1] // weight_comb_numbers // tp_sub_partation_world_size
            print(f"[TensorClient].{get_current_device_id()} 0 {weight.shape=}")
            weight = weight.view(weight.shape[0], weight_comb_numbers, -1)
            print(f"[TensorClient].{get_current_device_id()} 1 {weight.shape=}")
            weight = weight[:, :, tp_sub_partation_size * tp_sub_partation_rank : tp_sub_partation_size * (tp_sub_partation_rank + 1)]
            print(f"[TensorClient].{get_current_device_id()} 2 {weight.shape=}")
            weight = weight.view(weight.shape[0], -1)

            if transposed:
                weight = weight.transpose(0, 1)
            self.mm_param.weight = weight   # replace it
            print(f"[TensorClient].{get_current_device_id()} COLMMWeight {tp_sub_partation_rank=}, {tp_sub_partation_size=}, shape {remote_original_shape} -> {self.mm_param.weight.shape}")

            print(f"[TensorClient].{get_current_device_id()} replaced shared weight {self.weight_names}[0] ({self.mm_param.weight.shape}): {self.mm_param.weight}")
            print(f"[TensorClient].{get_current_device_id()} {original_shape=}, {self.mm_param.weight.shape=}")
            print(f"---\n")
            print = print.flush()
            assert self.mm_param.weight.shape == original_shape or self.mm_param.weight.shape == (weight_comb_numbers, original_shape[0], original_shape[1] // weight_comb_numbers), \
                f"[TensorClient].{get_current_device_id()} shared weight shape mismatch: original {original_shape} vs shared {self.mm_param.weight.shape}"


        else:
            if TensorServer():
                meta = dict(tp_rank=self.tp_rank_, tp_world_size=self.tp_world_size_)
                TensorServer().register(self.weight_names[0], self.mm_param.weight, meta)

