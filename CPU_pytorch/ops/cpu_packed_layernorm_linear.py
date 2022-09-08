from time import time
from typing import Optional
from torch import Tensor
import torch
import torch.nn as nn
from .CPUPackedLayernormLinearBuilder import CPUPackedLayernormLinearBuilder
import torch.utils.benchmark as benchmark
import torch.nn.functional as F
from utils import time_counter

cpu_packed_lin_op_dict = {}


class CPUPackedLayernormLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, id, input, norm_weight, norm_bias, bias=None):
        return cpu_packed_lin_op_dict[id].linear_forward(
            id, input, norm_weight, norm_bias, bias
        )

    @staticmethod
    def backward(ctx, id, out_grad):
        return cpu_packed_lin_op_dict[id].linear_backward(id, out_grad)


class CPUPackedLayernormLinear(nn.Linear):
    packed_linear_id = 0

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_tensor: Tensor,
        bias_tensor: Optional[Tensor],
        norm_weight: Tensor,
        norm_bias: Tensor,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert weight_tensor.shape == self.weight.shape
        assert not bias or bias_tensor.shape == self.bias.shape
        self.packed_lin_id = CPUPackedLayernormLinear.packed_linear_id
        CPUPackedLayernormLinear.packed_linear_id = (
            CPUPackedLayernormLinear.packed_linear_id + 1
        )
        self.bias = bias_tensor
        self.norm_weight = norm_weight
        self.norm_bias = norm_bias
        cpu_packed_lin_op = CPUPackedLayernormLinearBuilder().load(False)

        print("Compile OK! ")

        cpu_packed_lin_op.create_linear(
            self.packed_lin_id, in_features, out_features, bias, True
        )

        print("Create OK! ")

        cpu_packed_lin_op.pack_weight(self.packed_lin_id, weight_tensor)

        print("Pack OK! ")

        cpu_packed_lin_op_dict[self.packed_lin_id] = cpu_packed_lin_op

        print("Init Done! ")

    def __del__(self):
        cpu_packed_lin_op_dict[self.packed_lin_id].destroy_linear(self.packed_lin_id)
        del cpu_packed_lin_op_dict[self.packed_lin_id]

    def forward(self, input: Tensor) -> Tensor:
        if (
            not self.training or not torch.is_grad_enabled()
        ) and self.weight.device == torch.device("cpu"):
            return CPUPackedLayernormLinearFunction.apply(
                self.packed_lin_id,
                input,
                self.norm_weight,
                self.norm_bias,
                self.bias,
            )
        return F.linear(input, self.weight, self.bias)
