from time import time
from torch import Tensor
import torch
import torch.nn as nn
from CPULinearBuilder import CPULinearBuilder
import torch.utils.benchmark as benchmark


class CPULinear(nn.Linear):
    linear_id = 0

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.lin_id = CPULinear.linear_id
        CPULinear.linear_id = CPULinear.linear_id + 1
        self.cpu_lin_op = CPULinearBuilder().load(False)

        self.cpu_lin_op.create_linear(
            self.lin_id, in_features, out_features, bias, False
        )

    def __del__(self):
        self.cpu_lin_op.destroy_linear(self.lin_id)

    def forward(self, input: Tensor) -> Tensor:
        return self.cpu_lin_op.linear_forward(input, self.weight, self.bias)


if __name__ == "__main__":
    with torch.no_grad():
        for in_f in range(7, 13):
            for out_f in range(7, 13):
                for b_a in range(2):
                    in_f_n = 2**in_f
                    out_f_n = 2**out_f
                    a = CPULinear(in_f_n, out_f_n, bias=b_a == 0)
                    b = torch.empty((100, in_f_n))
                    d = nn.Linear(in_f_n, out_f_n, bias=b_a == 0)
                    d.weight = a.weight
                    d.bias = a.bias
                    r1 = a(b)
                    r2 = d(b)
                    assert r1.allclose(r2, atol=1e-7)
                    t0 = benchmark.Timer(stmt="a(b)", globals={"a": a, "b": b})
                    t1 = benchmark.Timer(stmt="d(b)", globals={"d": d, "b": b})
                    print(in_f_n, out_f_n, b_a == 0)
                    print(t0.timeit(1000))
                    print(t1.timeit(1000))
                    print()
