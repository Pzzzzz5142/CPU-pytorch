from turtle import forward
import torch
import numpy as np
import torch.nn as nn
from cpu_adam import DeepSpeedCPUAdam


def show_weight(l):
    print()
    print("================================")
    for i in l.parameters():
        print(i)
        print(i.grad)
    print("================================")
    print()


torch.random.manual_seed(42)

x_data = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])

y_data = torch.Tensor([0, 1, 2, 2])


class M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l = nn.Linear(2, 5)
        self.l1 = nn.Linear(5, 1)

    def forward(self, input):
        a = self.l(input)
        a = self.l1(a)
        return a

class ModedM(nn.Mudule):
    def __init__(self)->None:
        super().__init__()
        self.l=nn.Linear(2,5)

m = M()

# optim = torch.optim.AdamW(m.parameters(), lr=1, betas=(0.9, 0.5), eps=0)
optim = DeepSpeedCPUAdam(
    m.parameters(), lr=1, betas=(0.9, 0.5), eps=0, weight_decay=0.01
)
show_weight(m)

a = m(x_data)

loss = (a - y_data).sum()

loss.backward()

show_weight(m)

optim.step()

show_weight(m)
