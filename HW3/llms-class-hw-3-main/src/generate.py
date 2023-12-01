import argparse
import json
import os
import torch 
import torch.nn.functional as F
from train.model import DecoderLM
from omegaconf import OmegaConf
import tiktoken
import sys
sys.path.append('/home/ubuntu/LLM/HW3/llms-class-hw-3-main/src')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def softmax_with_temperature(logits: torch.FloatTensor, temperature: float) -> torch.FloatTensor:
    # to avoid division by 0
    temperature = max(temperature, 1e-5)
    logits = logits / temperature
    probs = F.softmax(logits)

    return probs

@torch.inference_mode()
def generate(
    model: DecoderLM,
    seq_len: str, 
    device: str,
    tokenizer: tiktoken.Encoding,
    prefixes: list[str],
    batch_size: int,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
) -> list[str]:
    
    prefixes = tokenizer.encode_batch(prefixes)
    # truncation of larger seq lengths
    prefixes = [i[:seq_len] for i in prefixes]
    pad_token = tokenizer.eot_token
    max_len = max([len(i) for i in prefixes])
    print(f"MAX LEN{max_len}")
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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefixes",
        type=str,
        required=True,
        help="a jsonl file with a list of strings as prefixes for generation",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="original",
        help="Either the string 'original' or the string 'cleaned'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getcwd(),
        help="Directory where to save outputs.",
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
    parser.add_argument(
        "--config",
        type=OmegaConf.load,
        required=True,
        help="the yaml config file used for model training",
    )

    args = parser.parse_args()
    # load prefixes
    with open(args.prefixes) as f:
        prefixes = [json.loads(line)["prefixes"] for line in f]
    print(f"PREFIXES LENGTH: {len(prefixes)}")

    # Get metadata
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    model_version = args.model_version
    output_dir = args.output_dir
    config = args.config
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)

    # Load Models
    if model_version == "original":
        model_path = os.path.join(config.output_dir, "model.pt")
        assert os.path.exists(model_path), f"no model checkpoint at {model_path}"
        print(f"LOADING MODEL FROM: {model_path}")

        model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        generations = generate(model, config.model_config.n_positions, device, tokenizer, prefixes, config.batch_size, max_new_tokens, temperature)


    elif model_version == "cleaned":
        model_path = os.path.join(config.output_dir, "model.pt")
        assert os.path.exists(model_path), f"no model checkpoint at {model_path}"
        print(f"LOADING MODEL FROM: {model_path}")

        model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        generations = generate(model, config.model_config.n_positions, device, tokenizer, prefixes, config.batch_size, max_new_tokens, temperature)

    else:
        raise ValueError("Invalid model version.")

    generation_path = os.path.join(output_dir, "generation_raw.jsonl")
    print(f"writing generations to {generation_path}")
    with open(generation_path, "w") as f:
        for prefix, generation in zip(prefixes, generations):
            json.dump({"prefix": prefix, "generation": generation}, f)
            f.write("\n")

    print("done!")


if __name__ == "__main__":
    main()
