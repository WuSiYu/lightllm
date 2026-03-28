import torch
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import (
    MMWeightTpl,
)
from lightllm.common.quantization import Quantcfg
from lightllm.common.quantization.no_quant import NoQuantization
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

                # COLMMWeight partitions input dim (dim 1 for NoQuant layout (out, in))
                # For quantized layout (in, out), transpose first to get (out, in)
                transposed = not isinstance(self.quant_method, NoQuantization)
                if transposed:
                    weight = weight.transpose(0, 1)

                # Slice along input dimension (dim 1)
                in_dim = weight.shape[1]
                tp_slice_size = in_dim // tp_sub_partation_world_size
                start = tp_slice_size * tp_sub_partation_rank
                end = tp_slice_size * (tp_sub_partation_rank + 1)

                # Column slice: non-contiguous 2D view, torch.mm supports non-contiguous tensors
                weight = weight[:, start:end]

                if transposed:
                    weight = weight.transpose(0, 1)

                self.mm_param.weight = weight

            assert self.mm_param.weight.shape == original_shape, \
                f"shared weight shape mismatch: expected {original_shape}, got {self.mm_param.weight.shape}"

            # Rebuild mm_param_list to point to new weight (release references to old weight)
            new_weights = torch.split(self.mm_param.weight, self.out_dims, dim=-2)
            for i, wp in enumerate(self.mm_param_list):
                wp.weight = new_weights[i]
                wp.load_ok = [True, True, True]

        elif TensorServer():
            meta = dict(tp_rank=self.tp_rank_, tp_world_size=self.tp_world_size_)
            TensorServer().register(self.weight_names[0], self.mm_param.weight, meta)

