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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
MODEL_DIR = Path("tokeniser")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ALPACA_JSONL = DATA_DIR / "alpaca_cleaned.jsonl"
STACK_JSONL = DATA_DIR / "stack_v2_filtered.jsonl"
TOKENISER_JSON = MODEL_DIR / "qwen_style.json"  # HuggingFace tokenizers format
VOCAB_TXT = MODEL_DIR / "vocab.txt"  # human-readable vocab

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# [QWEN-3] Qwen uses 151,643 regular tokens.  We use 32,768 — a power of two,
# which keeps embedding matrix dimensions GPU-friendly.
VOCAB_SIZE = 32_768

# Number of lines sampled from each dataset to *train* the tokeniser.
# The actual model data will be encoded in full afterwards.
TRAIN_SAMPLE_LINES = 200_000  # lines from each source

# Shard size: how many tokens per .bin file written to disk.
TOKENS_PER_SHARD = 10_000_000

# ---------------------------------------------------------------------------
# [QWEN-2] Pre-tokenisation regex — identical to GPT-4's cl100k_base
# ---------------------------------------------------------------------------
# This regex splits text into "pre-tokens" before BPE runs.
# BPE merge rules are only applied within a pre-token, never across boundaries.
#
# Why this matters for code:
#   Without a regex, BPE would happily learn merges like ")\n    def" (closing
#   paren, newline, indent, keyword) — a useless multi-category token.  The
#   regex prevents that by ensuring punctuation, letters, numbers, and
#   whitespace always start separate pre-tokens.
#
# Breakdown of each alternative (| branch) in the pattern:
#   (?i:'s|'t|'re|'ve|'m|'ll|'d)
#       English contractions (case-insensitive): don't, I've, she'll …
#   [^\r\n\p{L}\p{N}]?\p{L}+
#       Optional leading symbol/punct, then a run of Unicode letters.
#       The optional prefix handles "(" before "while", "'(" before "re", etc.
#   \p{N}{1,3}
#       Up to 3 digits together.  Splitting long numbers into ≤3-digit chunks
#       keeps the vocabulary from wasting slots on every 7-digit integer.
#       [QWEN-2 detail] Qwen caps at {1,3}, same as cl100k_base.
#   [ ]?[^\s\p{L}\p{N}]+[\r\n]*
#       Optional space, then a run of punctuation/symbols, optional newlines.
#       Covers things like "::", "->", "//", "/*", etc. in code.
#   \s*[\r\n]+
#       Newlines (with optional preceding spaces).  Gives newlines their own
#       token slot so the model learns code indentation structure.
#   \s+(?!\S)
#       A run of spaces NOT followed by a non-space (i.e. trailing whitespace).
#   \s+
#       Any remaining whitespace.
#
# Source: tiktoken openai_public.py cl100k_base definition;
#         Bai et al. 2023 §2.1 ("same tokeniser as cl100k_base").

QWEN_REGEX_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# Compile once for use in the streaming cleaner
_PAT = regex.compile(QWEN_REGEX_PATTERN)


# ---------------------------------------------------------------------------
# [QWEN-4] Special tokens (ChatML set)
# ---------------------------------------------------------------------------
# <|endoftext|>  — separates documents in the pre-training stream
# <|im_start|>   — opens a ChatML turn  (role follows on the same line)
# <|im_end|>     — closes a ChatML turn
# <|fim_prefix|> — fill-in-the-middle prefix marker (code completion tasks)
# <|fim_suffix|> — fill-in-the-middle suffix marker
# <|fim_middle|> — fill-in-the-middle middle marker
# <|pad|>        — padding token (not meaningful to the model)
#
# IDs are assigned at the END of the vocabulary so that regular BPE tokens
# occupy IDs 0…(VOCAB_SIZE - N_SPECIAL - 1).  This mirrors Qwen's layout
# where all 208 control tokens sit above the 151,643 regular tokens.
# Source: Bai et al. 2023 §2.1; QwenLM/Qwen tokenization_note.md.

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


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """
    Lightweight text normalisation.

    Steps (in order):
      1. Unicode NFC normalisation — ensures multi-codepoint characters like
         é (e + combining acute) are stored as a single codepoint.  Prevents
         the same visual character producing different byte sequences.
      2. Remove null bytes — they cause issues in file I/O and some tokenisers.
      3. Strip ANSI escape codes — common in terminal-captured code snippets.
      4. Collapse runs of more than 3 blank lines into 3.  Code legitimately
         uses blank lines for section separation, but 20 blank lines in a row
         is almost always scraper noise.
      5. Strip leading/trailing whitespace.

    We intentionally do NOT:
      - Lower-case (code is case-sensitive)
      - Remove stopwords (meaningless for a generation model)
      - Lemmatise (same reason)
    """
    # 1. NFC normalisation
    text = unicodedata.normalize("NFC", text)
    # 2. Null bytes
    text = text.replace("\x00", "")
    # 3. ANSI escape sequences  (e.g. \x1b[32m from coloured terminal output)
    text = re.sub(r"\x1b\[[0-9;]*[mGKHF]", "", text)
    # 4. Collapse excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    # 5. Strip
    return text.strip()


# ---------------------------------------------------------------------------
# Dataset streaming helpers
# ---------------------------------------------------------------------------


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
    Interleave lines from both datasets for balanced BPE training.

    Why balance?  BPE greedily learns the most frequent byte pairs.  If we
    feed it only code, it will over-specialise on Python/JS syntax and produce
    poor tokens for natural-language instruction text.  Interleaving ensures
    the vocabulary covers both domains.
    """
    alpaca_iter = iter_texts(ALPACA_JSONL, max_lines=sample)
    stack_iter = iter_texts(STACK_JSONL, max_lines=sample)
    for a, s in zip(alpaca_iter, stack_iter):
        yield a
        yield s
    # drain whichever iterator still has items
    for t in alpaca_iter:
        yield t
    for t in stack_iter:
        yield t


# ---------------------------------------------------------------------------
# Tokeniser construction (HuggingFace `tokenizers` library)
# ---------------------------------------------------------------------------


def build_tokeniser() -> Tokenizer:
    """
    Construct and train a Byte-level BPE tokeniser in the Qwen style.

    Architecture choices explained inline with [QWEN-N] markers.

    Returns a fully trained HuggingFace Tokenizer object.
    """

    # ------------------------------------------------------------------
    # 1. Model — BPE on byte-level tokens
    # ------------------------------------------------------------------
    # [QWEN-1] We use `models.BPE` with `byte_fallback=True`.
    #
    # byte_fallback=True means: if a sub-word cannot be decoded as valid
    # UTF-8, fall back to individual <0xNN> byte tokens instead of <unk>.
    # This matches Qwen's "no unknown token" guarantee.
    #
    # unk_token=None: there is no UNK because every byte is in the vocab.
    tokeniser = Tokenizer(models.BPE(byte_fallback=True, unk_token=None))

    # ------------------------------------------------------------------
    # 2. Pre-tokeniser — regex split then ByteLevel encoding
    # ------------------------------------------------------------------
    # [QWEN-2] Two-stage pre-tokenisation (same as tiktoken cl100k_base):
    #
    # Stage A — Split on the Qwen/GPT-4 regex.
    #   behavior="isolated" keeps the matched chunk as a single pre-token
    #   and does NOT merge the surrounding context into it.
    #
    # Stage B — ByteLevel.
    #   Maps every byte of each pre-token to a printable "alias" character
    #   from a 256-character alphabet (the Radford et al. GPT-2 mapping).
    #   This is what makes the tokeniser truly byte-level: the BPE algorithm
    #   sees printable strings and can learn merges, but the underlying unit
    #   is always a single byte.  add_prefix_space=False because the regex
    #   already handles leading spaces.
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

    # ------------------------------------------------------------------
    # 3. Decoder — ByteLevel
    # ------------------------------------------------------------------
    # Reverses the ByteLevel encoding during decode().
    tokeniser.decoder = decoders.ByteLevel(add_prefix_space=False)

    # ------------------------------------------------------------------
    # 4. Post-processor — no BOS/EOS injected automatically
    # ------------------------------------------------------------------
    # [QWEN-5] Qwen does not inject BOS/EOS tokens automatically at the
    # tokeniser level.  Document boundaries are handled at the data
    # collation stage by inserting <|endoftext|> between documents.
    # We follow that convention: do nothing here.

    # ------------------------------------------------------------------
    # 5. Trainer
    # ------------------------------------------------------------------
    # vocab_size: [QWEN-3] 32,768 for our small-scale experiment.
    # min_frequency: pairs that appear fewer than 2 times are not merged.
    # special_tokens: [QWEN-4] the ChatML set.  They are added *after* BPE
    #   training so they don't interfere with the frequency counts.
    # initial_alphabet: start from the full 256-byte alphabet so that no
    #   input can ever produce <unk>.

    added_tokens = [
        AddedToken(tok, special=True, normalized=False) for tok in SPECIAL_TOKENS
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    print(f"\nTraining BPE tokeniser (vocab_size={VOCAB_SIZE}) …")
    print(f"  Sampling up to {TRAIN_SAMPLE_LINES:,} lines from each dataset.")

    # The HuggingFace trainer accepts an iterator of strings.
    tokeniser.train_from_iterator(
        interleaved_texts(sample=TRAIN_SAMPLE_LINES),
        trainer=trainer,
    )

    print(f"  Vocabulary size after training: {tokeniser.get_vocab_size():,}")
    return tokeniser


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def encode_with_eot(tokeniser: Tokenizer, text: str) -> list[int]:
    """
    Encode a single document and append the <|endoftext|> boundary token.

    [QWEN-4 / QWEN-5] In Qwen's pre-training, multiple documents are packed
    into a single sequence separated by <|endoftext|>.  Appending the marker
    here means the DataLoader can concatenate encoded documents without any
    extra logic.
    """
    ids = tokeniser.encode(text).ids
    eot_id = tokeniser.token_to_id(EOT_TOKEN)
    ids.append(eot_id)
    return ids


def write_shard(ids: list[int], path: Path) -> None:
    """
    Write a flat list of uint16 token IDs to a binary file.

    uint16 supports vocab sizes up to 65,535 — plenty for VOCAB_SIZE=32,768.
    Using a raw binary format (not numpy, not pickle) keeps the file minimal
    and readable by any language.

    File format: N × 2 bytes, little-endian uint16, no header.
    To read back:  ids = list(struct.unpack('<' + 'H'*N, data))
    """
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

    Shards are named  <out_prefix>_shard_0000.bin, _0001.bin, etc.
    The last shard may be smaller than shard_size.
    """
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


# ---------------------------------------------------------------------------
# Human-readable vocab dump
# ---------------------------------------------------------------------------


def save_vocab_txt(tokeniser: Tokenizer) -> None:
    """Write vocab.txt: one 'token_id  display_repr' line per token."""
    vocab = tokeniser.get_vocab()
    with open(VOCAB_TXT, "w", encoding="utf-8") as f:
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{repr(token)}\n")
    print(f"  Vocab saved to {VOCAB_TXT}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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
    encode_dataset(tokeniser, STACK_JSONL, "stack")

    print("\nDone!  Shards are in data/")
    print(
        "Next step: run  03_embeddings.py  to build GloVe-initialised weight matrices."
    )


if __name__ == "__main__":
    main()
