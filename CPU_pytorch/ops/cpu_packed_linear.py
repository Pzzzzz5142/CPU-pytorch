from time import time
from typing import Optional
from torch import Tensor
import torch
import torch.nn as nn
from .CPUPackedLinearBuilder import CPUPackedLinearBuilder
import torch.utils.benchmark as benchmark
import torch.nn.functional as F
from utils import time_counter

cpu_packed_lin_op_dict = {}


class CPUPackedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, id, input, weight, bias=None):
        return cpu_packed_lin_op_dict[id].linear_forward(id, input, bias)

    @staticmethod
    def backward(ctx, id, out_grad):
        return cpu_packed_lin_op_dict[id].linear_backward(id, out_grad)


class CPUPackedLinear(nn.Linear):
    packed_linear_id = 0

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_tensor: Tensor,
        bias_tensor: Optional[Tensor],
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert weight_tensor.shape == self.weight.shape
        assert not bias or bias_tensor.shape == self.bias.shape
        self.packed_lin_id = CPUPackedLinear.packed_linear_id
        CPUPackedLinear.packed_linear_id = CPUPackedLinear.packed_linear_id + 1
        self.bias = bias_tensor
        cpu_packed_lin_op = CPUPackedLinearBuilder().load(True)

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
            return CPUPackedLinearFunction.apply(
                self.packed_lin_id, input, self.weight, self.bias
            )
        return F.linear(input, self.weight, self.bias)
