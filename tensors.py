import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device)
# print(tens.requires_grad)

emp = torch.empty(size=(3, 3))
zero = torch.zeros((3, 3))
rand = torch.rand((3, 3))  # for a uniform distribution
one = torch.ones((3, 3))
eye = torch.eye(5, 5)
ara = torch.arange(start=0, end=5, step=1)
lin = torch.linspace(start=0.1, end=1, steps=10)

norm = torch.empty(size=(1, 5)).normal_(mean=1, std=2)
uni = torch.empty(size=(1, 5)).uniform_(0, 10)
dia = torch.diag(torch.ones(3))

# convert tensors to other types
tens = torch.arange(4).bool()
# print(tens)

# convert tensor to array and vice-versa
import numpy as np

arr = np.zeros((5, 5))
tens = torch.from_numpy(arr)
arr = tens.numpy()

# print(tens, '\n', arr)

# math and comparison operations
tens1 = torch.tensor([1, 2, 4])
tens2 = torch.tensor([9, 10, 11])

sum = torch.empty(3)
torch.add(tens1, tens2, out=sum)

sum = torch.add(tens1, tens2)
sum = tens1 + tens2

diff = tens1 - tens2

div = torch.true_divide(tens1, tens2)  # divides element wise, each element of tens1 with corresponding element of tens2

# inplace operations
tens = torch.zeros(3)
tens.add_(tens1)  # a function followed by '_' is always an inplace function
# print(tens)
tens += tens1  # inplace
tens = tens + tens1  # not inplace

exp = tens1.pow(2)
# print(exp)

# comparison
com = tens1 > 0  # element wise
# print(com)

# matrix multiplication
ten = torch.rand((2, 5))
sor = torch.rand((5, 4))

out = torch.mm(ten, sor)  # 2x4
out = ten.mm(sor)
# print(out.shape)

# matrix exponentiation
exp = torch.rand(5, 5)
# print(exp.matrix_power(3))

# element wise multiplication
out = tens1 * tens2

# dot product
out = torch.dot(tens1, tens2)

# batch matrix multiplication
batch = 32
n, m, p = 10, 20, 30

tens1 = torch.rand(batch, n, m)  # 32x10x20
tens2 = torch.rand(batch, m, p)  # 32x20x30
out = torch.bmm(tens1, tens2)  # 32x10x30

# broadcasting
tens1 = torch.rand((5, 5))
tens2 = torch.rand((1, 5))  # it replicates row wise such that the dimensions match

diff = tens1 - tens2
elem_pow = tens1 ** tens2

# print(diff, "\n", elem_pow)

tens1 = torch.tensor([1, 4, 4, 6, 7])
tens2 = torch.tensor([9, 10, 11, 12, 13])
sum = torch.sum(tens1, dim=0)  # axis 0 is the direction of rows

vals, inds = torch.max(tens1, dim=0)  # returns value along with index, tens1.max(dim=0)
vals, inds = torch.min(tens1, dim=0)

abs_ = torch.abs(tens1)
ind = torch.argmax(tens1, dim=0)  # returns index

mean_ = torch.mean(tens1.float(), dim=0)  # requires elements of tensor to be float
eq_ = torch.eq(tens1, tens2)  # checks element wise equality
vals, inds = torch.sort(tens1, dim=0, descending=False)  # returns sorted values and their indices

out = torch.clamp(tens1, min=2, max=5)  # values smaller than min are changed to min, same for max

boo = torch.tensor([2, -1, 4, 0], dtype=torch.bool)  # tensor([ True, True, True, False])
out = torch.any(boo)  # returns True if any value is True
out = torch.all(boo)  # returns True if all values are True

