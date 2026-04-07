"""
Loads a trained checkpoint to generate text completions via a REPL interface or single-shot prompts.

Flags:
--ckpt [path]: Path to the specific model checkpoint file to load.

--prompt "[text]": Single-shot generation; if omitted, starts interactive mode.

--temperature [float]: Controls randomness (lower is more deterministic).

--top-k [int]: Limits sampling to the top K most likely tokens.

--top-p [float]: Nucleus sampling; limits tokens to a cumulative probability mass.

--max-new [int]: Maximum number of tokens to generate.
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

TOKENISER_JSON = Path("artefacts/tokeniser") / "qwen_style.json"


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
    print("\n=== CodingLM Interactive Mode (Alpaca Format) ===")
    print(REPL_HELP)

    # Stable defaults for small models to avoid "word salad"
    params = dict(max_new=256, temperature=0.1, top_k=1, top_p=0.95, rep_penalty=1.1)

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/quit"):
            break
        elif user_input.startswith("/temp"):
            params["temperature"] = float(user_input.split()[1])
            print(f"  temperature = {params['temperature']}")
        elif user_input.startswith("/topk"):
            params["top_k"] = int(user_input.split()[1])
            print(f"  top_k = {params['top_k']}")
        elif user_input.startswith("/topp"):
            params["top_p"] = float(user_input.split()[1])
            print(f"  top_p = {params['top_p']}")
        elif user_input.startswith("/rep"):
            params["rep_penalty"] = float(user_input.split()[1])
            print(f"  rep_penalty = {params['rep_penalty']}")
        elif user_input.startswith("/len"):
            params["max_new"] = int(user_input.split()[1])
            print(f"  max_new = {params['max_new']}")
        else:
            # 1. WRAP IN THE ALPACA TEMPLATE
            # Based on your data, the model expects "### Instruction:\n{input}\n\n### Response:\n"
            full_prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
            
            prompt_ids = encode_prompt(tokeniser, full_prompt, device)
            output_ids = generate(model, prompt_ids, **params)
            
            # 2. Decode only the new portion
            new_ids = output_ids[:, prompt_ids.size(1) :]
            completion = decode_ids(tokeniser, new_ids)
            
            # 3. Handle stopping logic
            # Your model was trained to end with the <|endoftext|> token (eot_id)
            eot_str = tokeniser.id_to_token(eot_id) or "<|endoftext|>"
            
            # Include ALPACA markers as stop sequences. 
            # This prevents the model from hallucinating a new question after answering.
            stop_sequences = [
                eot_str, 
                "### Instruction:", 
                "### Input:", 
                "### Response:", 
                "Instruction:"
            ]
            
            for stop_seq in stop_sequences:
                if stop_seq in completion:
                    completion = completion.split(stop_seq)[0]
            
            print(f"Model:\n{completion.strip()}\n")


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
