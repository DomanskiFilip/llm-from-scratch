"""
WHAT THIS FILE DOES
-------------------
1. Reads the .jsonl files produced by 01_download_and_filter_datasets.py
2. Cleans and normalises the raw text (code + instruction data)
3. Trains a Byte-level BPE vocabulary on a sample of that text
4. Encodes the full dataset into integer token-ID sequences
5. Saves the vocabulary, merge table, and encoded binary shards to disk

REFERENCES & DESIGN CHOICES
----------------------------
The tokeniser here is deliberately modelled on Qwen's approach, described in:

  Bai et al. (2023). "Qwen Technical Report."
  arXiv:2309.16609  https://arxiv.org/abs/2309.16609
  (Section 2.1 — Tokenisation)

Key design decisions taken from that paper / the Qwen codebase:

  [QWEN-1] BYTE-LEVEL BPE
      Qwen operates on UTF-8 bytes, not Unicode code-points.  Every possible
      byte value 0x00–0xFF is a base token, so no text can ever produce an
      <unk> token.  This is critical for code, which is full of unusual
      symbols and byte sequences.
      Source: Bai et al. 2023 §2.1; also Yang et al. 2024 "Qwen2 Technical
      Report" arXiv:2407.10671 §2.1.

  [QWEN-2] REGEX PRE-TOKENISATION
      Before the BPE algorithm runs, text is split by a regular expression
      into "pre-tokens" (rough word-like chunks).  BPE merges are then only
      allowed *within* a pre-token — never across a boundary.  This prevents
      the tokeniser learning merges like "  the" (trailing space of one word
      fused with the next), which would waste vocabulary slots.
      Qwen uses the same cl100k_base regex as GPT-4 (tiktoken library).
      Source: Bai et al. 2023 §2.1; tiktoken openai_public.py cl100k_base.

  [QWEN-3] LARGE VOCABULARY (151,643 REGULAR TOKENS)
      Qwen's production vocabulary has 151,643 regular BPE tokens plus 208
      control/special tokens = 151,851 total.  A larger vocab means longer
      tokens on average → fewer tokens per document → shorter sequences →
      less memory at training time.  We use a smaller vocab here (32,768)
      because we are training from scratch on limited compute.
      Source: Bai et al. 2023 §2.1; QwenLM/Qwen tokenization_note.md.

  [QWEN-4] SPECIAL TOKENS WITH ChatML FORMATTING
      Qwen wraps every conversation turn with <|im_start|>role\n...<|im_end|>
      (the ChatML format, originally from OpenAI).  We adopt the same special
      tokens so that later fine-tuning on instruction data is straightforward.
      Source: Bai et al. 2023 §2.1

  [QWEN-5] NO BOS/EOS IN THE TRADITIONAL SENSE
      Qwen deliberately avoids a single bos/eos token.  Document boundaries
      are marked by <|endoftext|> (ID = VOCAB_SIZE − 1 in our scheme).
      Source: Bai et al. 2023 §2.1; QwenLM/Qwen tokenization_note.md.

Usage
-----
  python tokeniser.py            # train + encode
  python tokeniser.py --encode-only  # skip training, just encode
"""

import argparse
import json
import os
import re
import struct
import unicodedata
from pathlib import Path
from typing import Generator

import regex  # 'regex' library supports \p{L} Unicode categories
from tokenizers import (
    AddedToken,
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
)
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from tqdm import tqdm


# Paths

DATA_DIR = Path("data")
MODEL_DIR = Path("tokeniser")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ALPACA_JSONL    = DATA_DIR / "alpaca_cleaned.jsonl"
STACK_JSONL     = DATA_DIR / "stack_v2_filtered.jsonl"
TOKENISER_JSON  = MODEL_DIR / "qwen_style.json"
VOCAB_TXT       = MODEL_DIR / "vocab.txt"


# Hyperparameters


VOCAB_SIZE         = 32_768
TRAIN_SAMPLE_LINES = 200_000
TOKENS_PER_SHARD   = 10_000_000


# [QWEN-2] Pre-tokenisation regex — identical to GPT-4's cl100k_base
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


# [QWEN-4] Special tokens (ChatML set)

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
    "<|pad|>",
]

EOT_TOKEN = "<|endoftext|>"



# Text cleaning
def clean_text(text: str) -> str:
    """
    Lightweight text normalisation.

    Steps (in order):
      1. Unicode NFC normalisation
      2. Remove null bytes
      3. Strip ANSI escape codes
      4. Collapse runs of more than 3 blank lines into 3
      5. Strip leading/trailing whitespace
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\x00", "")
    text = re.sub(r"\x1b\[[0-9;]*[mGKHF]", "", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()



# Dataset streaming helpers


def iter_texts(
    jsonl_path: Path, max_lines: int | None = None
) -> Generator[str, None, None]:
    """Yield the 'text' field from every line of a .jsonl file."""
    n = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec.get("text", "")
            if text:
                yield clean_text(text)
                n += 1
                if max_lines is not None and n >= max_lines:
                    return


def interleaved_texts(sample: int = TRAIN_SAMPLE_LINES) -> Generator[str, None, None]:
    """
    Yield texts for BPE training from all available datasets.

    Stack is optional — if stack_v2_filtered.jsonl is absent, training
    proceeds on Alpaca alone, producing a conversational-only tokeniser.
    Both datasets are interleaved when both are present so the vocabulary
    covers natural language and code equally.
    """
    stack_available = STACK_JSONL.exists()

    if stack_available:
        alpaca_iter = iter_texts(ALPACA_JSONL, max_lines=sample)
        stack_iter  = iter_texts(STACK_JSONL,  max_lines=sample)
        for a, s in zip(alpaca_iter, stack_iter):
            yield a
            yield s
        for t in alpaca_iter:
            yield t
        for t in stack_iter:
            yield t
    else:
        print(
            "  [INFO] stack_v2_filtered.jsonl not found — "
            "training tokeniser on Alpaca only."
        )
        yield from iter_texts(ALPACA_JSONL, max_lines=sample)



# Tokeniser construction
def build_tokeniser() -> Tokenizer:
    """
    Construct and train a Byte-level BPE tokeniser in the Qwen style.
    """
    tokeniser = Tokenizer(models.BPE(byte_fallback=True, unk_token=None))

    tokeniser.pre_tokenizer = Sequence(
        [
            Split(
                pattern=QWEN_REGEX_PATTERN,
                behavior="isolated",
                invert=False,
            ),
            ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )

    tokeniser.decoder = decoders.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"\nTraining BPE tokeniser (vocab_size={VOCAB_SIZE}) …")
    print(f"  Sampling up to {TRAIN_SAMPLE_LINES:,} lines from each dataset.")

    tokeniser.train_from_iterator(
        interleaved_texts(sample=TRAIN_SAMPLE_LINES),
        trainer=trainer,
    )

    print(f"  Vocabulary size after training: {tokeniser.get_vocab_size():,}")
    return tokeniser



# Encoding helpers
def encode_with_eot(tokeniser: Tokenizer, text: str) -> list[int]:
    """Encode a single document and append the <|endoftext|> boundary token."""
    ids = tokeniser.encode(text).ids
    eot_id = tokeniser.token_to_id(EOT_TOKEN)
    ids.append(eot_id)
    return ids


def write_shard(ids: list[int], path: Path) -> None:
    """Write a flat list of uint16 token IDs to a binary file."""
    with open(path, "wb") as f:
        for chunk_start in range(0, len(ids), 65536):
            chunk = ids[chunk_start : chunk_start + 65536]
            f.write(struct.pack(f"<{len(chunk)}H", *chunk))


def encode_dataset(
    tokeniser: Tokenizer,
    jsonl_path: Path,
    out_prefix: str,
    shard_size: int = TOKENS_PER_SHARD,
) -> None:
    """
    Encode every document in a .jsonl file and write binary shards.
    Skips gracefully if the file does not exist.
    """
    if not jsonl_path.exists():
        print(f"  [SKIP] {jsonl_path} not found — skipping encoding for '{out_prefix}'.")
        return

    shard_idx = 0
    buf: list[int] = []
    total = 0
    shard_path = DATA_DIR / f"{out_prefix}_shard_{shard_idx:04d}.bin"

    with tqdm(desc=f"Encoding {jsonl_path.name}", unit=" docs") as pbar:
        for text in iter_texts(jsonl_path):
            ids = encode_with_eot(tokeniser, text)
            buf.extend(ids)
            total += len(ids)
            pbar.set_postfix(tokens=f"{total / 1e6:.1f}M", shards=shard_idx + 1)
            pbar.update(1)

            while len(buf) >= shard_size:
                write_shard(buf[:shard_size], shard_path)
                print(f"\n  Wrote {shard_path}  ({shard_size / 1e6:.1f}M tokens)")
                buf = buf[shard_size:]
                shard_idx += 1
                shard_path = DATA_DIR / f"{out_prefix}_shard_{shard_idx:04d}.bin"

    if buf:
        write_shard(buf, shard_path)
        print(f"\n  Wrote {shard_path}  ({len(buf) / 1e6:.2f}M tokens — final shard)")

    print(f"  Total tokens ({out_prefix}): {total / 1e6:.2f}M")


# Human-readable vocab dump
def save_vocab_txt(tokeniser: Tokenizer) -> None:
    """Write vocab.txt: one 'token_id  display_repr' line per token."""
    vocab = tokeniser.get_vocab()
    with open(VOCAB_TXT, "w", encoding="utf-8") as f:
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{repr(token)}\n")
    print(f"  Vocab saved to {VOCAB_TXT}")



# Entry point
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and/or run Qwen-style BPE tokeniser"
    )
    parser.add_argument(
        "--encode-only",
        action="store_true",
        help="Skip training; load existing tokeniser and encode datasets",
    )
    args = parser.parse_args()

    if args.encode_only:
        print(f"Loading tokeniser from {TOKENISER_JSON} …")
        tokeniser = Tokenizer.from_file(str(TOKENISER_JSON))
    else:
        tokeniser = build_tokeniser()
        tokeniser.save(str(TOKENISER_JSON))
        print(f"\nTokeniser saved to {TOKENISER_JSON}")
        save_vocab_txt(tokeniser)

    print("\nEncoding datasets into binary shards …")
    encode_dataset(tokeniser, ALPACA_JSONL, "alpaca")
    encode_dataset(tokeniser, STACK_JSONL,  "stack")   # skipped if missing

    print("\nDone!  Shards are in data/")
    print(
        "Next step: run  03_embeddings.py  to build GloVe-initialised weight matrices."
    )


if __name__ == "__main__":
    main()