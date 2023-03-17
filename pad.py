import torch
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential
from cldm.cldm import RisingWeight, TimestepRisingWeight
import torch.nn as nn

q = TimestepRisingWeight(nn.Conv1d(4, 4, 1))

print(q)
print(hasattr(q, 'weight_factor'))
for layer in q:
    print(layer)
    print('weight_factor', hasattr(layer, 'weight_factor'))
    print('weight', hasattr(layer, 'weight'))
    print('bias', hasattr(layer, 'bias'))

