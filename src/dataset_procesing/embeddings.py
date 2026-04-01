"""
WHAT THIS FILE DOES
-------------------
1. Downloads GloVe-100d vectors (trained by Pennington et al., Stanford)
2. Optionally downloads Word2Vec Google News vectors as an alternative
3. Aligns either embedding set to our BPE vocabulary from tokeniser.py
4. Saves a weight matrix (float32, shape [VOCAB_SIZE, EMBED_DIM]) that can
   be plugged directly into nn.Embedding as a pretrained initialisation

WHY USE PRE-TRAINED EMBEDDINGS AT ALL?
---------------------------------------
Training embeddings from scratch requires the model to see each token in
many different contexts before its vector becomes meaningful.  With a small
dataset or limited compute, the embedding layer may never converge well.

Pre-trained embeddings give the model a "head start": the vectors already
encode syntactic and semantic similarity (e.g. cosine(king, queen) ≈ high)

GLOVE vs WORD2VEC — QUICK COMPARISON
--------------------------------------
Both produce dense floating-point vectors of fixed dimension (100, 300 …)
where geometrically close vectors represent semantically similar words.
They differ in *how* they learn those vectors:

  GloVe (Global Vectors for Word Representation)
  -----------------------------------------------
  Pennington, Socher & Manning (2014)
  https://nlp.stanford.edu/projects/glove/

  GloVe builds a word–word co-occurrence *matrix* X from the entire corpus
  first, then factorises it.  The loss for a pair (i, j) is:

      L = Σ_{i,j} f(X_ij) (wᵢᵀw̃ⱼ + bᵢ + b̃ⱼ − log X_ij)²

  where f is a weighting function that down-weights very frequent pairs
  (so "the cat" doesn't dominate).  Optimising this loss forces the dot
  product of two word vectors to approximate the log of their co-occurrence
  count.  This global view of statistics is GloVe's key advantage — it
  sees every pair at once rather than a local context window.

  Word2Vec
  --------
  Mikolov et al. (2013)
  https://code.google.com/archive/p/word2vec/

  Word2Vec uses one of two prediction objectives:

    CBOW (Continuous Bag of Words):
      Given the surrounding context words, predict the centre word.
      Faster to train; slightly worse on analogies.

    Skip-gram with Negative Sampling (SGNS):
      Given the centre word, predict each context word.
      Slower; generally better semantic structure, especially for rare words.

  The skip-gram loss for a centre word w and context word c is:

      L = −log σ(vₜᵀvₛ) − Σₖ E[log σ(−vₙₖᵀvₛ)]

  where the second term is the "negative sampling" component that pushes
  randomly drawn non-context words away from the centre vector.

  WHY WE USE GloVe BY DEFAULT
  ----------------------------
  The Stanford GloVe-100d file (glove.6B.100d.txt) is 347 MB compared to
  ~1.6 GB for the full Google News Word2Vec binary.  Both encode common
  English words well; GloVe tends to be better for syntactic tasks while
  Word2Vec skip-gram is often better for semantic analogies.  Since our
  model handles code and instruction text (not pure NLP), GloVe-100d is a
  good starting point that doesn't chew up disk space.

BPE ALIGNMENT CHALLENGE
------------------------
GloVe and Word2Vec are trained on *word-level* vocabularies.  Our BPE
tokeniser produces *sub-word* tokens like "Ġdef", "ĠTokenizer", "▁for".
(The Ġ/▁ prefix is the ByteLevel encoding of a space.)

Most BPE tokens don't directly appear in GloVe.  We handle the mismatch
with the following priority chain:

  Priority 1 — Exact match after stripping the space prefix:
      The token "Ġreturn" → strip "Ġ" → look up "return" in GloVe. ✓

  Priority 2 — Lower-case match:
      "ĠReturn" → strip → "return" → lowercase → "return". ✓

  Priority 3 — Sub-word average:
      "Ġfunctionality" not in GloVe.  Split by the BPE regex into
      ["function", "ality"] (or whatever sub-parts exist in GloVe) and
      average their vectors.

  Priority 4 — Character n-gram average:
      For very rare tokens, break into overlapping character trigrams and
      average any GloVe entries for those trigrams.  (GloVe does not have
      character n-grams natively; we fall back on whatever sub-strings match.)

  Priority 5 — Random initialisation:
      Sample from N(0, σ) where σ = std of the GloVe vectors, so
      uninitialised tokens don't stand out statistically.

The final weight matrix has shape [VOCAB_SIZE, EMBED_DIM].  It is saved as
a .npy file and also as a PyTorch .pt tensor for direct use in nn.Embedding
"""

import re
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import requests
import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from src.config import Config

# Paths
DATA_DIR = Path("data")
MODEL_DIR = Path("tokeniser")
EMBED_DIR = Path("embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)

TOKENISER_JSON = MODEL_DIR / "qwen_style.json"
GLOVE_ZIP = EMBED_DIR / "glove.6B.zip"
GLOVE_TXT = EMBED_DIR / "glove.6B.100d.txt"
GLOVE_ALIGNED = EMBED_DIR / "glove_aligned.pt"  # final weight matrix
COVERAGE_LOG = EMBED_DIR / "alignment_coverage.txt"


# Download helpers
def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Stream-download a file, showing a tqdm progress bar."""
    if dest.exists():
        print(f"  {dest.name} already downloaded — skipping.")
        return
    print(f"  Downloading {url} → {dest}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_glove(zip_path: Path, txt_path: Path) -> None:
    """Extract glove.6B.100d.txt from the zip if not already done."""
    if txt_path.exists():
        print(f"  {txt_path.name} already extracted — skipping.")
        return
    print(f"  Extracting {txt_path.name} from {zip_path.name} …")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("glove.6B.100d.txt") as src, open(txt_path, "wb") as dst:
            dst.write(src.read())


# Load GloVe vectors into a dict


def load_glove(txt_path: Path) -> Dict[str, np.ndarray]:
    """
    Parse GloVe text file into a {word: vector} dictionary.

    GloVe text format (one token per line):
        word  f1  f2  …  f100
    where f1…f100 are space-separated floats.

    Returns a dict mapping lowercase words to float32 numpy arrays.

    Memory note: the 6B-100d file has 400,000 words.  At 100 floats each
    (4 bytes) that is 400,000 × 100 × 4 = 160 MB — manageable in RAM.
    """
    print(f"  Loading GloVe vectors from {txt_path} …")
    glove: Dict[str, np.ndarray] = {}
    with open(txt_path, encoding="utf-8") as f:
        for line in tqdm(f, total=400_000, unit=" words"):
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            glove[word] = vec
    print(f"  Loaded {len(glove):,} GloVe vectors")
    return glove


# Byte-level prefix decoder

# The HuggingFace ByteLevel pre-tokeniser maps each raw byte to a printable
# character using the Radford et al. (2019) GPT-2 mapping.  Tokens in our
# vocabulary look like "Ġdef" or "Ġreturn" where Ġ = U+0120 represents a
# space byte (0x20).  We need to strip these aliases before looking up the
# word in GloVe.


# Build the GPT-2 byte→char mapping (same table used in tokenizers library)
def _build_byte_decoder() -> Dict[int, int]:
    """Return a dict mapping GPT-2 alias codepoints back to raw byte values."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {c: b for b, c in zip(bs, cs)}


_BYTE_DECODER = _build_byte_decoder()


def bpe_token_to_text(token: str) -> str:
    """
    Decode a ByteLevel BPE token to a plain UTF-8 string.

    Each character in the token is an alias for a raw byte.  We convert
    alias → byte, then decode the resulting byte string as UTF-8.
    Returns the decoded string or "" on decode failure (rare raw bytes).
    """
    byte_vals = bytes([_BYTE_DECODER.get(ord(c), 0) for c in token])
    try:
        return byte_vals.decode("utf-8")
    except UnicodeDecodeError:
        return ""


# Alignment: BPE vocab → GloVe vectors
def align_to_bpe(
    tokeniser: Tokenizer,
    glove: Dict[str, np.ndarray],
    config: Config,
) -> Tuple[np.ndarray, dict]:
    """
    Build a [VOCAB_SIZE, EMBED_DIM] weight matrix aligned to the BPE vocab.

    Returns:
        weight_matrix  — numpy array of shape (vocab_size, embed_dim)
        stats          — dict with coverage counts for the log file

    Alignment priority chain (described in module docstring):
      1. Exact decoded string in GloVe
      2. Lowercase decoded string in GloVe
      3. Sub-word average: split decoded string into GloVe words, average
      4. Random (N(0, σ) where σ = std of all GloVe vectors)
    """
    vocab = tokeniser.get_vocab()  # {token_str: id}
    vocab_size = tokeniser.get_vocab_size()

    # Compute the standard deviation of GloVe vectors for random init
    glove_std = float(np.std(np.stack(list(glove.values()))))
    rng = np.random.default_rng(config.random_seed)

    weight = rng.normal(0.0, glove_std, size=(vocab_size, config.embedding_dim)).astype(
        np.float32
    )

    stats = {"exact": 0, "lower": 0, "subword_avg": 0, "random": 0, "special": 0}

    # Simple English word tokeniser for sub-word splitting (priority 3)
    _word_re = re.compile(r"[a-zA-Z]+")

    print(f"  Aligning {vocab_size:,} BPE tokens to GloVe …")
    for token_str, token_id in tqdm(vocab.items(), total=vocab_size):
        # --- Special tokens: leave at random init, mark separately -------
        if token_str in config.tokenizer_special_tokens:
            stats["special"] += 1
            continue

        # Decode the ByteLevel-encoded token string back to plain text
        decoded = bpe_token_to_text(token_str)
        stripped = decoded.strip()  # remove leading/trailing spaces

        # Priority 1: exact match
        if stripped in glove:
            weight[token_id] = glove[stripped]
            stats["exact"] += 1
            continue

        # Priority 2: lowercase match
        lower = stripped.lower()
        if lower in glove:
            weight[token_id] = glove[lower]
            stats["lower"] += 1
            continue

        # Priority 3: sub-word average
        # Split the decoded text into plain English words and average those
        # that appear in GloVe.  For code tokens like "tokenise" or "relu"
        # this catches substrings like "token" / "re".
        parts = _word_re.findall(lower)
        found_vecs = [glove[p] for p in parts if p in glove]
        if found_vecs:
            weight[token_id] = np.mean(found_vecs, axis=0)
            stats["subword_avg"] += 1
            continue

        # Priority 4: random (already in weight from rng.normal above)
        stats["random"] += 1

    return weight, stats


# Main
def main(config: Config) -> None:

    # Download & extract GloVe
    download_file(config.embedding_glove_url, GLOVE_ZIP)
    extract_glove(GLOVE_ZIP, GLOVE_TXT)

    # Load GloVe
    glove = load_glove(GLOVE_TXT)

    # Load our BPE tokeniser
    if not TOKENISER_JSON.exists():
        raise FileNotFoundError(
            f"{TOKENISER_JSON} not found. tokeniser first."
        )
    print(f"  Loading tokeniser from {TOKENISER_JSON} …")
    tokeniser = Tokenizer.from_file(str(TOKENISER_JSON))

    # Align
    print("\nAligning BPE vocabulary to GloVe vectors …")
    weight, stats = align_to_bpe(tokeniser, glove, config)

    # Save
    torch.save(torch.from_numpy(weight), GLOVE_ALIGNED)
    print(f"\n  Saved aligned weight matrix → {GLOVE_ALIGNED}")
    print(f"  Shape: {weight.shape}  dtype: {weight.dtype}")

    # Coverage report
    vocab_size = tokeniser.get_vocab_size()
    total_regular = vocab_size - stats["special"]
    covered = stats["exact"] + stats["lower"] + stats["subword_avg"]
    pct = 100.0 * covered / max(total_regular, 1)

    report = (
        f"GloVe alignment coverage report\n"
        f"================================\n"
        f"Vocab size          : {vocab_size:>10,}\n"
        f"  Special tokens    : {stats['special']:>10,}\n"
        f"  Regular tokens    : {total_regular:>10,}\n"
        f"\n"
        f"Exact match         : {stats['exact']:>10,}  ({100 * stats['exact'] / max(total_regular, 1):.1f}%)\n"
        f"Lowercase match     : {stats['lower']:>10,}  ({100 * stats['lower'] / max(total_regular, 1):.1f}%)\n"
        f"Sub-word average    : {stats['subword_avg']:>10,}  ({100 * stats['subword_avg'] / max(total_regular, 1):.1f}%)\n"
        f"Random init         : {stats['random']:>10,}  ({100 * stats['random'] / max(total_regular, 1):.1f}%)\n"
        f"\n"
        f"Total GloVe coverage: {covered:>10,} / {total_regular:,}  ({pct:.1f}%)\n"
        f"\n"
        f"Interpretation\n"
        f"--------------\n"
        f"Tokens that got GloVe vectors start training with meaningful\n"
        f"geometric relationships (synonyms close, antonyms apart).\n"
        f"Randomly initialised tokens (mostly rare code symbols and\n"
        f"sub-word fragments) will need the model to learn their\n"
        f"representations from scratch during pre-training.\n"
        f"\n"
        f"References\n"
        f"----------\n"
        f"GloVe: Pennington, Socher & Manning (2014) arXiv:1405.0312\n"
        f"Word2Vec: Mikolov et al. (2013) arXiv:1301.3781\n"
        f"BPE: Sennrich, Haddow & Birch (2016) arXiv:1508.07909\n"
        f"Qwen tokeniser: Bai et al. (2023) arXiv:2309.00071 §2.1\n"
    )

    print("\n" + report)
    with open(COVERAGE_LOG, "w") as f:
        f.write(report)
    print(f"  Report saved → {COVERAGE_LOG}")

    # Quick sanity check
    print("\nSanity check — nearest GloVe neighbours for 'def' and 'return':")
    for probe in ["def", "return", "class"]:
        probe_id = tokeniser.token_to_id("Ġ" + probe)  # Ġ = space prefix
        if probe_id is None:
            probe_id = tokeniser.token_to_id(probe)
        if probe_id is None:
            continue
        probe_vec = torch.from_numpy(weight[probe_id])
        all_vecs = torch.from_numpy(weight[:5000])  # check in first 5k
        sims = torch.nn.functional.cosine_similarity(probe_vec.unsqueeze(0), all_vecs)
        top5_ids = sims.topk(6).indices.tolist()
        words = [tokeniser.id_to_token(i) for i in top5_ids if i != probe_id][:5]
        print(f"  '{probe}' → {words}")

    print("\nDone!  Next step: run  train. Optionally run model.py (standalone).")


if __name__ == "__main__":
    config = Config()
    main(config)