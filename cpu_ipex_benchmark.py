import torch
from CPU_pytorch.ops import CPUPackedLinear
from CPU_pytorch.ops import CPUPackedLayernormLinear
import torch.utils.benchmark as benchmark
import torch.nn as nn

N_FEATURE = 2048


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(N_FEATURE, N_FEATURE)
        self.layernorm = nn.LayerNorm(N_FEATURE)
        self.linear1 = nn.Linear(N_FEATURE, 30000)

    def forward(self, input):
        return self.linear1(self.linear(input))


class MyModel(nn.Module):
    def __init__(self, weight1, bias1, weight2, bias2, weight3, bias3) -> None:
        super().__init__()
        self.linear = CPUPackedLinear(N_FEATURE, N_FEATURE, weight1, bias1)
        self.linear1 = CPUPackedLayernormLinear(
            N_FEATURE, 30000, weight2, bias2, weight3, bias3
        )

    def forward(self, input):
        return self.linear1(self.linear(input))


net = Model()
my_net = MyModel(
    net.linear.weight,
    net.linear.bias,
    net.linear1.weight,
    net.linear1.bias,
    net.layernorm.weight,
    net.layernorm.bias,
)
a = torch.empty(6, N_FEATURE)
b = torch.empty(6, N_FEATURE)

import intel_extension_for_pytorch as ipex

net = net.to(memory_format=torch.channels_last)
net = ipex.optimize(net)
b = b.to(memory_format=torch.channels_last)

t0 = benchmark.Timer(stmt="a(b)", globals={"a": net, "b": b})
t1 = benchmark.Timer(stmt="d(b)", globals={"d": my_net, "b": a})

print(t0.timeit(10))
print(t1.timeit(10))
