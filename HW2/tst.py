import math

import tiktoken

from model import MultiHeadAttention, DecoderLM
from torchsummaryX import summary
import torch
from train import sequential_batch_sampler
from generate import generate


'''max_lr = 0.1
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
    print(i)'''

tokenizer = tiktoken.get_encoding('gpt2')
model = DecoderLM(n_vocab=tokenizer.n_vocab, n_positions=128, n_layer=1, n_embd=8, n_head=1)
a = generate(model,
             tokenizer=tokenizer,
             prefixes = ['a', 'bb', 'cc', 'dddd'],
             batch_size = 2,
             device = 'cpu')
print(len(a))
print(a)
