import torch

try:
    import cupy
except BaseException as e:
    cupy = None

assert cupy is not None

print(cupy)
import spikingjelly
print(spikingjelly)
from spikingjelly.clock_driven.neuron import MultiStepLIFNode


lif = MultiStepLIFNode(backend='cupy').to('cuda:0')
x = torch.randn(10, 3, 224, 224).to('cuda:0')
out = lif(x)
print(out.shape)
