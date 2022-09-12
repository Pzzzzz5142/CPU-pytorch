import torch
from CPU_pytorch.ops import CPUPackedLinear
import torch.utils.benchmark as benchmark
import torch.nn as nn

with torch.no_grad():
    ref = nn.Linear(512, 128)
    test = CPUPackedLinear(512, 128, ref.weight, ref.bias)
    input_tensor = torch.empty(1, 6, 512)
    print(
        f"input: {input_tensor}\nweight: {ref.weight.t()}\ntest output: {test(input_tensor)}\nref output: {ref(input_tensor)}"
    )
    assert test(input_tensor).allclose(ref(input_tensor), atol=1e-7, equal_nan=True)
    print("Init test OK! ")
    for in_f in range(7, 13):
        for out_f in range(7, 13):
            for b_a in range(2):
                in_f_n = 2**in_f
                out_f_n = 2**out_f
                print(in_f_n, out_f_n, b_a == 0)
                d = nn.Linear(in_f_n, out_f_n, bias=b_a == 0)
                a = CPUPackedLinear(in_f_n, out_f_n, d.weight, d.bias, bias=b_a == 0)
                b = torch.empty((1, 6, in_f_n))
                r1 = a(b)
                r2 = d(b)
                #assert r1.allclose(r2, atol=1e-7, equal_nan=True)
                t0 = benchmark.Timer(stmt="a(b)", globals={"a": a, "b": b},num_threads=6)
                t1 = benchmark.Timer(stmt="d(b)", globals={"d": d, "b": b},num_threads=6)
                print(t0.timeit(10))
                print(t1.timeit(10))
                print()
