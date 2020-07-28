import torch
from torch.utils.checkpoint import checkpoint


def custom_enc(module):
    def custom_forward(*inputs):
        output = module(inputs[0])
        return output[0]

    return custom_forward

m = torch.nn.Linear(4,3)
x = torch.randn(10,4, requires_grad=True)

z1 = checkpoint(
    custom_enc(m),
    x)

print(z1.requires_grad)

z2 = m(x)
print(z2.requires_grad)