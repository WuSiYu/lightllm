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
            self.mm_param.weight, meta = TensorClient().get_tensor(self.weight_names[0])    # replace weight with shared weight to free GPU memory
            print(f"[TensorClient] loaded weight: {self.weight_names[0]} tp_rank: {meta['tp_rank']} tp_world_size: {meta['tp_world_size']}")
            print(f"(ours: tp_rank: {self.tp_rank_} tp_world_size: {self.tp_world_size_})")
            assert meta["tp_world_size"] < self.tp_world_size_ and self.tp_world_size_ % meta["tp_world_size"] == 0, \
                f"shared_weight tp_world_size not align: master={meta['tp_world_size']} vs slave={self.tp_world_size_}"
            tp_sub_partation_world_size = self.tp_world_size_ // meta["tp_world_size"]
            assert self.tp_rank_ // tp_sub_partation_world_size == meta["tp_rank"], \
                f"shared_weight master tp_rank not correct: wanted {self.tp_rank_ // tp_sub_partation_world_size}, master has {meta['tp_rank']}"

            tp_sub_partation_rank = self.tp_rank_ % tp_sub_partation_world_size
            tp_sub_partation_size = self.mm_param.weight.shape[1] // tp_sub_partation_world_size
            self.mm_param.weight = self.mm_param.weight[:, tp_sub_partation_size * tp_sub_partation_rank : tp_sub_partation_size * (tp_sub_partation_rank + 1)]

        else:
            if TensorServer():
                meta = dict(tp_rank=self.tp_rank_, tp_world_size=self.tp_world_size_)
                TensorServer().register(self.weight_names[0], self.mm_param.weight, meta)

