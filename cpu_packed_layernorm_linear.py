from turtle import forward
import torch
from CPU_pytorch.ops import CPUPackedLinear
from CPU_pytorch.ops import CPUPackedLayernormLinear
import torch.utils.benchmark as benchmark
import torch.nn as nn


class RefModule(nn.Module):
    def __init__(self, in_features, out_features, bias) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_features, out_features, bias)
        self.layernorm = nn.LayerNorm(out_features)

    def forward(self, x):
        return self.layernorm(self.l1(x))


class FusedModule(nn.Module):
    def __init__(self, in_features, out_features, bias, weight: RefModule) -> None:
        super().__init__()
        self.l1 = CPUPackedLayernormLinear(
            in_features,
            out_features,
            weight.l1.weight,
            weight.l1.bias,
            weight.layernorm.weight,
            weight.layernorm.bias,
            bias,
        )

    def forward(self, x):
        return self.l1(x)


with torch.no_grad():
    ref = RefModule(6, 5, True)
    test = FusedModule(6, 5, True, ref)
    input_tensor = torch.empty(1, 6, 6)
    print(
        f"input: {input_tensor}\ntest output: {test(input_tensor)}\nref output: {ref(input_tensor)}"
    )
    assert test(input_tensor).allclose(ref(input_tensor), atol=1e-7, equal_nan=True)
    print("Init test OK! ")
    for in_f in range(7, 13):
        for out_f in range(7, 13):
            for b_a in range(2):
                in_f_n = 2**in_f
                out_f_n = 2**out_f
                print(in_f_n, out_f_n, b_a == 0)
                a = RefModule(in_f_n, out_f_n, b_a == 0)
                d = FusedModule(in_f_n, out_f_n, b_a == 0, a)
                b = torch.empty((1, 6, in_f_n))
                r1 = a(b)
                r2 = d(b)
                # assert r1.allclose(r2, atol=1e-7, equal_nan=True)
                t0 = benchmark.Timer(stmt="a(b)", globals={"a": a, "b": b})
                t1 = benchmark.Timer(stmt="d(b)", globals={"d": d, "b": b})
                print(t0.timeit(10))
                print(t1.timeit(10))
                print()
