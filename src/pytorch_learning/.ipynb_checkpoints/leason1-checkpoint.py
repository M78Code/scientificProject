import torch
import time
from torch import autograd

print(torch.__version__)
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())

a = torch.randn(10000, 1000)
b = torch.randn(1000, 2000)

t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
print(a.device, t1 - t0, c.norm(2))
# gpu加速，苹果下是mps, nvidia下是cuda
device = torch.device('mps')
a = a.to(device)
b = b.to(device)

t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))

t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, "{:.9f}".format(t2 - t0), c.norm(2))

# 2，pytorch自动求导功能
x = torch.tensor(1.)
a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
c = torch.tensor(3., requires_grad=True)
y = pow(a, 2) * x + b * x + c
print("before:", a.grad, b.grad, c.grad)
grads = autograd.grad(y, [a, b, c])
print("after1:", grads[0], grads[1], grads[2])
