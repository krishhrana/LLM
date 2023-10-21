import math

from model import MultiHeadAttention, DecoderLM
from torchsummaryX import summary
import torch
from train import sequential_batch_sampler


max_lr = 0.1
min_lr = 0.01
warmup = 5
t = 7
adjusted_t = t - warmup
print(adjusted_t)
num_training_steps = 10

print(0.5 * ((max_lr - min_lr) * math.cos(math.pi * (2 * adjusted_t / num_training_steps)) + (max_lr + min_lr)))

a = torch.arange(20, dtype=torch.long)
val_loader = sequential_batch_sampler(a, batch_size=2, seq_len=3, device='cpu')
for i in val_loader:
    print(i)
