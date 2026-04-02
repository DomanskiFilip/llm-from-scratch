import json
import re
import struct
import unicodedata
from pathlib import Path
from typing import Generator

import regex
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    trainers,
)
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from tqdm import tqdm

from src.config import Config

# Paths 
DATA_DIR  = Path("data")
MODEL_DIR = Path("tokeniser")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ALPACA_JSONL        = DATA_DIR / "alpaca_cleaned.jsonl"
DOLLY_JSONL         = DATA_DIR / "dolly_15k.jsonl"
OPEN_INSTRUCT_JSONL = DATA_DIR / "open_instruct.jsonl"
TOKENISER_JSON      = MODEL_DIR / "qwen_style.json"
VOCAB_TXT           = MODEL_DIR / "vocab.txt"

ALL_DATASETS: list[tuple[Path, str]] = [
    (ALPACA_JSONL,        "alpaca"),
    (DOLLY_JSONL,         "dolly"),
    (OPEN_INSTRUCT_JSONL, "open_instruct")
]

# Regex (GPT-4 / Qwen cl100k_base pattern)
QWEN_REGEX_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

_PAT = regex.compile(QWEN_REGEX_PATTERN)


# Text cleaning 
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\x00", "")
    text = re.sub(r"\x1b\[[0-9;]*[mGKHF]", "", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


# Streaming helpers 
def iter_texts(
    jsonl_path: Path, max_lines: int | None = None
) -> Generator[str, None, None]:
    """Yield clean 'text' strings for BPE training (no mask needed here)."""
    n = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec  = json.loads(line)
            text = rec.get("text", "")
            if text:
                yield clean_text(text)
                n += 1
                if max_lines is not None and n >= max_lines:
                    return


def iter_records(jsonl_path: Path) -> Generator[dict, None, None]:
    """Yield full records (text + response_start_char) for encoding."""
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec  = json.loads(line)
            text = rec.get("text", "")
            if text:
                rec["text"] = clean_text(text)
                yield rec


def interleaved_texts(config: Config) -> Generator[str, None, None]:
    sample    = config.tokenizer_train_sample_lines
    available = [(p, n) for p, n in ALL_DATASETS if p.exists()]
    missing   = [n for p, n in ALL_DATASETS if not p.exists()]

    if missing:
        print(f"  [INFO] Skipping missing datasets: {missing}")
    if not available:
        raise FileNotFoundError("No dataset files found. Run download first.")

    iterators = [iter_texts(p, max_lines=sample) for p, _ in available]
    names     = [n for _, n in available]
    print(f"  Interleaving {len(available)} datasets: {names}")

    from itertools import zip_longest
    sentinel = object()
    for group in zip_longest(*iterators, fillvalue=sentinel):
        for text in group:
            if text is not sentinel:
                yield text


# Tokeniser construction
def build_tokeniser(config: Config) -> Tokenizer:
    tokeniser = Tokenizer(models.BPE(byte_fallback=True, unk_token=None))

    tokeniser.pre_tokenizer = Sequence([
        Split(pattern=QWEN_REGEX_PATTERN, behavior="isolated", invert=False),
        ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokeniser.decoder = decoders.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=config.tokenizer_vocab_size,
        min_frequency=2,
        special_tokens=config.tokenizer_special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"\nTraining BPE tokeniser (vocab_size={config.tokenizer_vocab_size}) …")
    print(f"  Sampling up to {config.tokenizer_train_sample_lines:,} lines per dataset.")

    tokeniser.train_from_iterator(interleaved_texts(config), trainer=trainer)
    print(f"  Vocabulary size after training: {tokeniser.get_vocab_size():,}")
    return tokeniser


# Encoding with loss mask
def encode_with_mask(
    tokeniser: Tokenizer,
    text: str,
    response_start_char: int,
    config: Config,
) -> tuple[list[int], list[int]]:
    """
    Encode `text` and return (token_ids, loss_mask).

    loss_mask[i] = 1  if token i is inside the response (train on it)
                 = 0  if token i is part of the prompt template (ignore)

    Strategy: encode the prompt prefix separately to get its token count,
    then encode the full text.  Everything up to prompt_token_count gets
    mask=0; the rest (response + EOT) gets mask=1.

    Edge case: if response_start_char is 0 or missing, the whole sequence
    is treated as trainable (safe fallback for plain-text data).
    """
    eot_id = tokeniser.token_to_id(config.tokenizer_eot_token)

    if response_start_char > 0:
        prompt_text    = text[:response_start_char]
        prompt_ids     = tokeniser.encode(prompt_text).ids
        prompt_len     = len(prompt_ids)
    else:
        prompt_len = 0

    all_ids  = tokeniser.encode(text).ids + [eot_id]
    mask     = [0] * min(prompt_len, len(all_ids)) + \
               [1] * max(0, len(all_ids) - prompt_len)

    # Safety: lengths must match
    assert len(all_ids) == len(mask), "Token/mask length mismatch — should never happen"
    return all_ids, mask


# Binary shard writers 
def write_token_shard(ids: list[int], path: Path) -> None:
    """Write uint16 token IDs."""
    with open(path, "wb") as f:
        for start in range(0, len(ids), 65536):
            chunk = ids[start : start + 65536]
            f.write(struct.pack(f"<{len(chunk)}H", *chunk))


def write_mask_shard(mask: list[int], path: Path) -> None:
    """Write uint8 loss mask (0 or 1 per token)."""
    with open(path, "wb") as f:
        f.write(bytes(mask))


# Per-dataset encoder 
def encode_dataset(
    tokeniser: Tokenizer,
    jsonl_path: Path,
    out_prefix: str,
    config: Config,
) -> None:
    """
    Encode every document in a .jsonl file into parallel token + mask shards.
    Skips gracefully if the file does not exist.
    """
    if not jsonl_path.exists():
        print(f"  [SKIP] {jsonl_path} not found — skipping '{out_prefix}'.")
        return

    shard_size = config.tokenizer_tokens_per_shard
    shard_idx  = 0
    tok_buf:  list[int] = []
    mask_buf: list[int] = []
    total = 0

    def _flush(final: bool = False) -> None:
        nonlocal shard_idx, tok_buf, mask_buf
        if not tok_buf:
            return
        tok_path  = DATA_DIR / f"{out_prefix}_shard_{shard_idx:04d}.bin"
        mask_path = DATA_DIR / f"{out_prefix}_shard_{shard_idx:04d}.mask.bin"
        write_token_shard(tok_buf, tok_path)
        write_mask_shard(mask_buf, mask_path)
        label = "final shard" if final else f"{shard_size / 1e6:.1f}M tokens"
        print(f"\n  Wrote {tok_path}  ({label})")
        tok_buf  = []
        mask_buf = []
        shard_idx += 1

    with tqdm(desc=f"Encoding {jsonl_path.name}", unit=" docs") as pbar:
        for rec in iter_records(jsonl_path):
            text = rec["text"]
            rsc  = rec.get("response_start_char", 0)
            ids, mask = encode_with_mask(tokeniser, text, rsc, config)

            tok_buf.extend(ids)
            mask_buf.extend(mask)
            total += len(ids)
            pbar.set_postfix(tokens=f"{total / 1e6:.1f}M", shards=shard_idx + 1)
            pbar.update(1)

            while len(tok_buf) >= shard_size:
                write_token_shard(tok_buf[:shard_size],  DATA_DIR / f"{out_prefix}_shard_{shard_idx:04d}.bin")
                write_mask_shard( mask_buf[:shard_size], DATA_DIR / f"{out_prefix}_shard_{shard_idx:04d}.mask.bin")
                print(f"\n  Wrote shard {shard_idx}  ({shard_size / 1e6:.1f}M tokens)")
                tok_buf  = tok_buf[shard_size:]
                mask_buf = mask_buf[shard_size:]
                shard_idx += 1

    # Write remaining tokens
    if tok_buf:
        tok_path  = DATA_DIR / f"{out_prefix}_shard_{shard_idx:04d}.bin"
        mask_path = DATA_DIR / f"{out_prefix}_shard_{shard_idx:04d}.mask.bin"
        write_token_shard(tok_buf,  tok_path)
        write_mask_shard(mask_buf, mask_path)
        print(f"\n  Wrote {tok_path}  ({len(tok_buf) / 1e6:.2f}M tokens — final shard)")

    print(f"  Total tokens ({out_prefix}): {total / 1e6:.2f}M")

    # Sanity check: count trainable (mask=1) tokens
    mask_ones = sum(
        sum(open(DATA_DIR / f"{out_prefix}_shard_{i:04d}.mask.bin", "rb").read())
        for i in range(shard_idx + 1)
        if (DATA_DIR / f"{out_prefix}_shard_{i:04d}.mask.bin").exists()
    )
    pct = 100.0 * mask_ones / max(total, 1)
    print(f"  Trainable tokens (mask=1): {mask_ones:,} / {total:,}  ({pct:.1f}%)")


# Vocab dump 
def save_vocab_txt(tokeniser: Tokenizer) -> None:
    vocab = tokeniser.get_vocab()
    with open(VOCAB_TXT, "w", encoding="utf-8") as f:
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{repr(token)}\n")
    print(f"  Vocab saved to {VOCAB_TXT}")


# Entry point
def main(config: Config) -> None:
    tokeniser = build_tokeniser(config)
    tokeniser.save(str(TOKENISER_JSON))
    print(f"\nTokeniser saved to {TOKENISER_JSON}")
    save_vocab_txt(tokeniser)

    print("\nEncoding datasets into binary shards + loss-mask shards …")
    for jsonl_path, prefix in ALL_DATASETS:
        encode_dataset(tokeniser, jsonl_path, prefix, config)

    print("\nDone!  Token shards and mask shards are in data/")
    print("Each .bin shard has a matching .mask.bin file.")
    print("Next step: run  embeddings then  train")


if __name__ == "__main__":
    config = Config()
    main(config)