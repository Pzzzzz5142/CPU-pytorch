import torch
from CPU_pytorch.ops import CPULinear
import torch.utils.benchmark as benchmark
import torch.nn as nn

with torch.no_grad():
    for in_f in range(7, 13):
        for out_f in range(7, 13):
            for b_a in range(2):
                in_f_n = 2 ** in_f
                out_f_n = 2 ** out_f
                a = CPULinear(in_f_n, out_f_n, bias=b_a == 0)
                b = torch.empty((100, 100, in_f_n))
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
                if torch.cuda.is_available():
                    b = b.to("cuda")
                    d = d.to("cuda")
                    t2 = benchmark.Timer(stmt="d(b)", globals={"d": d, "b": b})
                    print(t2.timeit(1000))
                print()
