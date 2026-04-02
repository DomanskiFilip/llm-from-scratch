"""
WHAT THIS FILE DOES
-------------------
Loads a trained checkpoint and generates code completions from text prompts
Supports interactive mode (REPL) and single-shot command-line generation

SAMPLING STRATEGIES EXPLAINED
------------------------------

GREEDY DECODING  (temperature → 0, top_k = 1)
    Always picks the single most probable next token.  Deterministic but
    often produces repetitive, "safe" output because the model exploits the
    same high-confidence patterns every time.

TEMPERATURE SAMPLING
    Divides all logits by a temperature scalar T before applying softmax:
        P(token) ∝ exp(logit / T)
    T < 1 (e.g. 0.7): sharpens the distribution → more confident, less varied
    T = 1.0         : unmodified model probabilities
    T > 1 (e.g. 1.5): flattens the distribution → more random, creative

TOP-K SAMPLING
    After temperature scaling, retain only the K most probable tokens and
    redistribute probability mass among them (set the rest to -inf before
    softmax).  Prevents the model from sampling very unlikely tokens that
    can derail generation (e.g. a random punctuation character mid-function)
    K = 40 is a common default (Fan et al. 2018, arXiv:1805.04833)

TOP-P (NUCLEUS) SAMPLING
    Instead of a fixed K, keep the smallest set of tokens whose cumulative
    probability ≥ p (e.g. p = 0.9).  Adapts the effective vocabulary size
    based on the model's confidence: when the model is very confident
    (one token has prob 0.99), top-p picks just that token; when uncertain
    it allows more variety.  Holtzman et al. (2020), arXiv:1904.09751

REPETITION PENALTY
    Divides the logit of any token that has already appeared in the context
    by a penalty factor > 1.  Helps prevent degenerate repetition loops
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent))

from model import CodingLM, LMConfig
from train import get_device

from src.config import Config

TOKENISER_JSON = Path("tokeniser") / "qwen_style.json"


# Load model
def load_model(ckpt_path: Path, device: torch.device) -> tuple:
    """Return (model, tokeniser)."""
    from tokenizers import Tokenizer

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt["config"]
    model_cfg = LMConfig(
        **{k: v for k, v in cfg_dict.items() if k in LMConfig.__dataclass_fields__}
    )
    model = CodingLM(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tokeniser = Tokenizer.from_file(str(TOKENISER_JSON))
    eot_id = tokeniser.token_to_id("<|endoftext|>") or 0

    print(f"Loaded model ({model.count_parameters():,} params) from {ckpt_path}")
    return model, tokeniser, eot_id


# Sampling with top-k + top-p + repetition penalty
@torch.no_grad()
def generate(
    model: CodingLM,
    prompt_ids: torch.Tensor,
    max_new: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    rep_penalty: float = 1.1,
) -> torch.Tensor:
    """
    Generate `max_new` tokens continuing `prompt_ids`.

    Args:
        model       : trained CodingLM in eval mode
        prompt_ids  : [1, prompt_len] int64 tensor
        max_new     : max tokens to generate
        temperature : sampling temperature (see module docstring)
        top_k       : top-k filtering (0 = disabled)
        top_p       : nucleus sampling threshold (1.0 = disabled)
        rep_penalty : repetition penalty factor (1.0 = disabled)

    Returns:
        [1, prompt_len + max_new] int64 tensor
    """
    ids = prompt_ids.clone()
    device = ids.device
    hidden = None

    # Warm-up: process all but the last prompt token to build hidden state
    if ids.size(1) > 1:
        _, hidden = model(ids[:, :-1], hidden)

    current = ids[:, -1:]  # [1, 1]

    for _ in range(max_new):
        logits, hidden = model(current, hidden)
        logits = logits[:, -1, :].float()  # [1, vocab_size]

        # Repetition penalty — penalise tokens already in the prompt+generation
        if rep_penalty != 1.0:
            generated_ids = ids[0].tolist()
            for prev_id in set(generated_ids):
                if logits[0, prev_id] > 0:
                    logits[0, prev_id] /= rep_penalty
                else:
                    logits[0, prev_id] *= rep_penalty

        # Temperature
        logits /= max(temperature, 1e-8)

        # Top-k filtering
        if top_k > 0:
            top_vals = torch.topk(logits, min(top_k, logits.size(-1))).values
            min_val = top_vals[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < min_val, float("-inf"))

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative prob above threshold (shift by 1 to keep the
            # token that pushes us over threshold)
            remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
            # Scatter back to original ordering
            logits.scatter_(1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # [1, 1]
        ids = torch.cat([ids, next_id], dim=1)
        current = next_id

    return ids


# Encode / decode helpers
def encode_prompt(tokeniser, text: str, device: torch.device) -> torch.Tensor:
    """Tokenise `text` and return a [1, T] int64 tensor."""
    ids = tokeniser.encode(text).ids
    return torch.tensor([ids], dtype=torch.long, device=device)


def decode_ids(tokeniser, ids: torch.Tensor) -> str:
    """Decode a [1, T] tensor back to a string."""
    return tokeniser.decode(ids[0].tolist(), skip_special_tokens=False)


# Interactive REPL
REPL_HELP = """
Commands:
  /temp  <float>    Set sampling temperature  (default 0.2)
  /topk  <int>      Set top-k                 (default 5)
  /topp  <float>    Set top-p                 (default 0.95)
  /rep   <float>    Set repetition penalty    (default 1.1)
  /len   <int>      Set max new tokens        (default 200)
  /quit             Exit
  <any other text>  Generate a completion
"""


def interactive_loop(model, tokeniser, eot_id, device: torch.device) -> None:
    print("\n=== CodingLM Interactive Mode ===")
    print(REPL_HELP)

    params = dict(max_new=200, temperature=0.2, top_k=5, top_p=0.95, rep_penalty=1.1)

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue

        if prompt.startswith("/quit"):
            break
        elif prompt.startswith("/temp"):
            params["temperature"] = float(prompt.split()[1])
            print(f"  temperature = {params['temperature']}")
        elif prompt.startswith("/topk"):
            params["top_k"] = int(prompt.split()[1])
            print(f"  top_k = {params['top_k']}")
        elif prompt.startswith("/topp"):
            params["top_p"] = float(prompt.split()[1])
            print(f"  top_p = {params['top_p']}")
        elif prompt.startswith("/rep"):
            params["rep_penalty"] = float(prompt.split()[1])
            print(f"  rep_penalty = {params['rep_penalty']}")
        elif prompt.startswith("/len"):
            params["max_new"] = int(prompt.split()[1])
            print(f"  max_new = {params['max_new']}")
        else:
            prompt_ids = encode_prompt(tokeniser, prompt, device)
            output_ids = generate(model, prompt_ids, **params)
            # Decode only the newly generated portion
            new_ids = output_ids[:, prompt_ids.size(1) :]
            completion = decode_ids(tokeniser, new_ids)
            # Stop at <|endoftext|> if present
            eot_str = tokeniser.id_to_token(eot_id) or ""
            if eot_str in completion:
                completion = completion.split(eot_str)[0]
            print(completion)
            print()


# Entry point
def main(config: Config) -> None:
    parser = argparse.ArgumentParser(
        description="Generate code with a trained CodingLM"
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text (omit for interactive mode)",
    )
    parser.add_argument("--max-new", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--rep-penalty", type=float, default=1.1)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    model, tokeniser, eot_id = load_model(Path(args.ckpt), device)

    if args.prompt:
        # Single-shot mode
        prompt_ids = encode_prompt(tokeniser, args.prompt, device)
        output_ids = generate(
            model,
            prompt_ids,
            max_new=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            rep_penalty=args.rep_penalty,
        )
        new_ids = output_ids[:, prompt_ids.size(1) :]
        completion = decode_ids(tokeniser, new_ids)
        eot_str = tokeniser.id_to_token(eot_id) or ""
        if eot_str in completion:
            completion = completion.split(eot_str)[0]
        print(args.prompt + completion)
    else:
        interactive_loop(model, tokeniser, eot_id, device)


if __name__ == "__main__":
    config = Config()
    main(config)
