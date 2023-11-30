import argparse
import json
import os
import torch 
from train.model import DecoderLM
from omegaconf import OmegaConf
import tiktoken

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# YOUR CODE HERE
# You may add additional imports or helper funcitons here.


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
    with open(args.prefixes) as f:
        prefixes = [json.loads(line)["prefix"] for line in f]

    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    model_version = args.model_version
    output_dir = args.output_dir
    config = args.config
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    model_path = os.path.join(config.output_dir, "model.pt")

    if model_version == "original":
        # YOUR CODE HERE
        # Implement inference using your model trained on the original data here.
        # You may copy and paste from HW2.
        generations = []
    elif model_version == "cleaned":
        model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
        model.load_state_dict(torch.load(model_path))

        # YOUR CODE HERE
        # Implement inference using your model trained on the cleaned data here.
        # You may copy and paste from HW2.
        generations = []
    else:
        raise ValueError("Invalid model version.")

    generation_path = os.path.join(output_dir, "generation.jsonl")
    print(f"writing generations to {generation_path}")
    with open(generation_path, "w") as f:
        for prefix, generation in zip(prefixes, generations):
            json.dump({"prefix": prefix, "generation": generation}, f)
            f.write("\n")

    print("done!")


if __name__ == "__main__":
    main()
