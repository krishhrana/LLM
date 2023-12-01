import torch
import torch.nn.functional as F
import tiktoken
from contextlib import nullcontext
import math
import numpy as np
import sys
sys.path.append('HW3/llms-class-hw-3-main/src/train')
from model import DecoderLM
import argparse
from omegaconf import OmegaConf



device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = tiktoken.get_encoding('gpt2')
autocast = (torch.autocast(device, dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32), ) 
            if device == "cuda" 
            else nullcontext()
    )


def compute_language_modeling_loss(input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
    logits = logits[:, :-1, :]
    B, T, C = logits.shape
    logits = logits.reshape(B*T, C)
    labels = input_ids[:, 1:].reshape(B*T)
    loss = F.cross_entropy(logits, labels)
    return loss

def get_perplexity(model, input_ids):
    with autocast:
        logits = model(input_ids) # [1, T, n_vocab]
    loss = compute_language_modeling_loss(input_ids=input_ids, logits=logits)
    print(loss)
    ppl = np.exp(loss.item())
    return ppl

def process_input(input_text):
    input_ids = tokenizer.encode(input_text)
    input_ids = torch.tensor(input_ids, device = device).reshape(1, -1)
    return input_ids


parser = argparse.ArgumentParser()
parser.add_argument(
        "--config",
        type=OmegaConf.load,
        required=True,
        help="the yaml config file used for model training"
        )

args = parser.parse_args()
config = args.config
model_path = 'HW2/raw_model/model.pt'


inputs = [
    '联合国生物多样性保护公约 (UNCBD) / 竹藤与生物多样 更多',
    'La isla Jekyll es hermosa y no puedo esperar a volver para explorarla un poco más.', 
    '4129095867', 
    'my@email.address', 
    'School districts in New York, Pennsylvania California, Washington and other states said they were bracing for the supply shortages, which are expected to last into early 2024.'
]

model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
model.load_state_dict(torch.load(model_path))

print(f'Loaded model from {model_path}')

ppl = [get_perplexity(model, process_input(i)) for i in inputs]
print(ppl)
