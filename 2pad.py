import wandb
import torch

q = torch.rand(8, 3, 512, 512)

im = wandb.Image(q)

p = torch.cat([q, q, q])
im2 = wandb.Image(p)
