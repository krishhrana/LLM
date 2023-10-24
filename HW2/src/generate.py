import argparse
import json
import os
import tiktoken
import torch
from model import DecoderLM
from omegaconf import OmegaConf
from tqdm import trange
import torch.nn.functional as F
from utils import determine_device, enable_tf32

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def softmax_with_temperature(
    logits: torch.FloatTensor, temperature: float
) -> torch.FloatTensor:
    """Turns logits into probabilities under softmax (with temperature)

    Args:
        logits: a 2d torch tensor of token ids (B x V)
        temperature: temperature of the softmax function

    Returns:
        a 2d torch tensor of token probabilities (B x V)
    """

    # to avoid division by 0
    temperature = max(temperature, 1e-5)
    logits = logits / temperature
    probs = F.softmax(logits)

    return probs


@torch.inference_mode()
def generate(
    model: DecoderLM,
    device: str,
    tokenizer: tiktoken.Encoding,
    prefixes: list[str],
    batch_size: int,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
) -> list[str]:
    """Generates completions conditioned on prefixes

    Args:
        model: the language model
        device: device to put the tensors on
        tokenizer: the tokenizer
        prefixes: a list of strings as prefixes for generation
        batch_size: number of prefixes to batch together during generation
        max_new_tokens: the number of tokens to generate for each prefix
        temperature: temperature parameter of softmax

    Returns:
        a list of strings (continuations to prefixes)
    
    Note: you should implement a batched version of this function by
        left-padding tokenized prefixes with `tokenizer.eot_token` so that all
        sequences have equal length. `attention_mask` should be set to 0.0 for
        padding tokens, and 1.0 everywhere else.
    """
    prefixes = tokenizer.encode_batch(prefixes)
    pad_token = tokenizer.eot_token
    max_len = max([len(i) for i in prefixes])
    prefixes = torch.tensor([[pad_token] * (max_len - len(i)) + i for i in prefixes], dtype=torch.long, device=device)

    pad_size = batch_size - (len(prefixes) % batch_size) if (len(prefixes) % batch_size) != 0 else 0
    batch_pad = torch.empty(size=(pad_size, max_len), dtype=torch.long, device = device).fill_(pad_token)
    prefixes = torch.cat((prefixes, batch_pad)).reshape(-1, batch_size, max_len)

    attention_mask = torch.ones_like(prefixes, device = device).masked_fill(prefixes == pad_token, 0)
    N, B, T = prefixes.shape
    generations = torch.empty(size=(N, B, T + max_new_tokens), dtype = torch.long, device = device)
    for idx, batch in enumerate(prefixes):
        batch_mask = attention_mask[idx]
        for i in range(max_new_tokens):
            logits = model.forward(input_ids=batch, attention_mask=batch_mask)
            logit = logits[:, -1, :] # (B, 1, C)
            probs = F.softmax(logit, dim = -1)
            next_token = torch.multinomial(probs, num_samples=1)
            batch = torch.cat((batch, next_token), dim = 1) # (B, T + 1)
            batch_mask = torch.cat((batch_mask, torch.ones_like(next_token)), dim = 1)
        generations[idx] = batch

    generations = generations.view(batch_size*N, -1)
    if pad_size != 0:
        generations = generations[:-pad_size, :]
    generations = tokenizer.decode_batch(generations.cpu().detach().tolist())
    generations = [i.replace('<|endoftext|>', "") for i in generations]
    return generations


def main():
    enable_tf32()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=OmegaConf.load,
        required=True,
        help="the yaml config file used for model training",
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        required=True,
        help="a json file with a list of strings as prefixes for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="temperature in sampling"
    )

    args = parser.parse_args()
    config = args.config
    with open(args.prefixes) as f:
        prefixes = [json.loads(line)["prefix"] for line in f]
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature

    # initialize tokenizer and model
    model_path = os.path.join(config.output_dir, "model.pt")
    assert os.path.exists(model_path), f"no model checkpoint at {model_path}"
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    device = determine_device() if config.device == "auto" else config.device
    model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
    model.load_state_dict(torch.load(model_path))

    # generate and save outputs
    model.eval()
    generations = generate(
        model,
        device,
        tokenizer,
        prefixes,
        config.batch_size,
        max_new_tokens,
        temperature,
    )

    generation_path = os.path.join(config.output_dir, "generation.jsonl")
    print(f"writing generations to {generation_path}")
    with open(generation_path, "w") as f:
        for prefix, generation in zip(prefixes, generations):
            json.dump({"prefix": prefix, "generation": generation}, f)
            f.write("\n")

    print("done!")


if __name__ == "__main__":
    main()
