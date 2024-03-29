import torch
from CPU_pytorch.ops import CPULinear
import torch.utils.benchmark as benchmark
import torch.nn as nn

with torch.no_grad():
    test = CPULinear(4, 3)
    ref = nn.Linear(4, 3)
    ref.weight = test.weight
    ref.bias = test.bias
    input_tensor = torch.empty(6, 4)
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
                a = CPULinear(in_f_n, out_f_n, bias=b_a == 0)
                b = torch.empty((in_f_n, in_f_n))
                d = nn.Linear(in_f_n, out_f_n, bias=b_a == 0)
                d.weight = a.weight
                d.bias = a.bias
                r1 = a(b)
                r2 = d(b)
                assert r1.allclose(r2, atol=1e-7, equal_nan=True)
                t0 = benchmark.Timer(stmt="a(b)", globals={"a": a, "b": b})
                t1 = benchmark.Timer(stmt="d(b)", globals={"d": d, "b": b})
                print(t0.timeit(10))
                print(t1.timeit(10))
                if torch.cuda.is_available():
                    b = b.to("cuda")
                    d = d.to("cuda")
                    t2 = benchmark.Timer(stmt="d(b)", globals={"d": d, "b": b})
                    print(t2.timeit(1000))
                print()
