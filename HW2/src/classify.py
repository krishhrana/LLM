import argparse
import os
import tiktoken
import torch
from datasets import load_dataset
from model import DecoderLM
from omegaconf import OmegaConf
from tqdm import trange
from utils import determine_device, enable_tf32

device = 'cpu'

YELP_TEMPLATE = "Here is a yelp review.\n{text}\nThis review is"
YELP_LABEL_MAP = {0: " negative", 1: " positive"}


@torch.inference_mode()
def score(
    model: DecoderLM,
    device: str,
    tokenizer: tiktoken.Encoding,
    texts: list[str],
    batch_size: int,
) -> torch.FloatTensor:
    """Scores all possible next tokens for the given texts

    Args:
        model: the language model
        device: device to put the tensors on
        tokenizer: the tokenizer
        texts: a list of strings for scoring
        batch_size: number of instances to score during one forward pass

    Returns:
        Logits corresponding to next token probabilities (B x V).

    
    Note: you should implement a batched version of this function by
        left-padding tokenized instances with `tokenizer.eot_token` so that all
        sequences have equal length. `attention_mask` should be set to 0.0 for
        padding tokens, and 1.0 everywhere else.
    """
    texts = tokenizer.encode_batch(texts)
    pad_token = tokenizer.eot_token
    max_len = max([len(i) for i in texts])
    texts = torch.tensor([[pad_token] * (max_len - len(i)) + i for i in texts], dtype=torch.long, device = device)

    pad_size = batch_size - (len(texts) % batch_size) if (len(texts) % batch_size) != 0 else 0
    batch_pad = torch.empty(size=(pad_size, max_len), dtype=torch.long, device = device).fill_(pad_token)
    texts = torch.cat((texts, batch_pad)).reshape(-1, batch_size, max_len)

    attention_mask = torch.ones_like(texts, device = device).masked_fill(texts == pad_token, 0)
    N, B, T = texts.shape
    next_token_logits = torch.empty(size=(N, B, tokenizer.n_vocab))

    for idx, batch in enumerate(texts):
        batch_mask = attention_mask[idx]
        logits = model.forward(input_ids=batch, attention_mask=batch_mask)
        logit = logits[:, -1, :]
        next_token_logits[idx] = logit

    next_token_logits = next_token_logits.reshape(N * batch_size, tokenizer.n_vocab)
    if pad_size != 0:
        next_token_logits = next_token_logits[:-pad_size, :]
    return next_token_logits


def classify_binary_sentiment(
    logits: torch.FloatTensor,
    tokens_of_interest: list[int],
    calibrate: bool = False,
) -> list[int]:
    """
    Args:
        logits: torch tensor corresponding to next token probabilities (B x V)
        tokens_of_interest: the indices for the tokens corresponding to negative
          and positive labels
        calibrate: when calibration is true, set the threshold according to your
          proposed calibration strategy in Question 3.6
    Returns:
        A list of predictions with length B, an element should be 0 if the
          negative class is more likely and 1 if the positive class is more
          likely.
    """

    probs = logits[:, tokens_of_interest].softmax(1)

    if calibrate:
        threshold = 0.7
    else:
        threshold = 0.5
    
    predictions = torch.argmax(probs, dim = -1)
    return predictions.cpu().detach().tolist()


def main():
    enable_tf32()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=OmegaConf.load,
        required=True,
        help="the yaml config file used for model training",
    )
    parser.add_argument("--calibrate", action="store_true")

    args = parser.parse_args()
    config = args.config

    # initialize tokenizer and model
    model_path = os.path.join(config.output_dir, "model.pt")
    assert os.path.exists(model_path), f"no model checkpoint at {model_path}"
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    device = determine_device() if config.device == "auto" else config.device
    model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location = 'cpu'))

    dataset = load_dataset("yelp_polarity")
    test_subset = (
        dataset["test"]
        .filter(
            lambda instance: len(
                tokenizer.encode(YELP_TEMPLATE.format(text=instance["text"]))
            )
            <= model.n_positions
        )
        .shuffle(seed=42)[:1000]
    )
    texts = [YELP_TEMPLATE.format(text=text) for text in test_subset["text"]]
    negative_token_id = tokenizer.encode_single_token(YELP_LABEL_MAP[0])
    positive_token_id = tokenizer.encode_single_token(YELP_LABEL_MAP[1])

    model.eval()
    logits = score(
        model,
        device,
        tokenizer,
        texts,
        config.batch_size,
    )

    predictions = classify_binary_sentiment(
        logits, [negative_token_id, positive_token_id], calibrate=args.calibrate
    )

    acc = sum(
        1 if pred == label else 0
        for pred, label in zip(predictions, test_subset["label"])
    ) / len(predictions)
    print(f"accuracy on yelp: {acc * 100:.1f}")


if __name__ == "__main__":
    main()
