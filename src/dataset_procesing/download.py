import json
import os
import logging
from pathlib import Path
from typing import Iterator

from datasets import load_dataset, Dataset
from tqdm import tqdm

# Configuration

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Permissive / open-source licences we want to keep
ALLOWED_LICENCES: set[str] = {
    # MIT family
    "mit",
    # Apache family
    "apache-2.0",
    "apache-1.1",
    # BSD family
    "bsd-2-clause",
    "bsd-3-clause",
    "bsd-4-clause",
    # Public domain / unlicensed
    "unlicense",
    "cc0-1.0",
    "public-domain",
    # Mozilla
    "mpl-2.0",
    # ISC
    "isc",
    # Boost
    "bsl-1.0",
    # Python
    "python-2.0",
    # zlib
    "zlib",
    # Artistic
    "artistic-2.0",
    # LGPL (weak copyleft, generally fine for model training data)
    "lgpl-2.0",
    "lgpl-2.1",
    "lgpl-3.0",
}

ALLOWED_LANGUAGES: set[str] = {
    "python",
    "javascript",
    "typescript",
    "rust",
    "java",
    "shell",
    "sql",
}

# Streaming chunk size
WRITE_BATCH = 5_000

# Cap on Stack v2 examples (set to None for no cap during full training runs)
STACK_MAX_EXAMPLES: int | None = 500_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Helpers

def normalise_licence(raw: str) -> str:
    """Lower-case and strip whitespace for consistent comparison."""
    return raw.strip().lower()


def is_permitted_licence(licence_field) -> bool:
    """
    The Stack v2 'license' column is a list[str] (can be empty).
    We keep a file if *any* of its declared licences is in ALLOWED_LICENCES.
    Files with no licence info are discarded
    """
    if not licence_field:
        return False
    if isinstance(licence_field, str):
        licence_field = [licence_field]
    normalised = {normalise_licence(lic) for lic in licence_field}
    return bool(normalised & ALLOWED_LICENCES)


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def stream_to_jsonl(
    stream: Iterator[dict],
    out_path: Path,
    filter_fn=None,
    transform_fn=None,
    max_examples: int | None = None,
    desc: str = "Streaming",
) -> int:
    """
    Generic streaming writer.

    Args:
        stream:       Iterable of raw dataset rows
        out_path:     Destination .jsonl file
        filter_fn:    Optional callable(row) -> bool. Rows returning False are skipped
        transform_fn: Optional callable(row) -> dict. Applied after filtering
        max_examples: Stop after this many *kept* examples (None = unlimited)
        desc:         tqdm label

    Returns:
        Number of examples written
    """
    out_path.unlink(missing_ok=True)   # start fresh
    buf: list[dict] = []
    kept = 0
    seen = 0

    with tqdm(desc=desc, unit=" ex") as pbar:
        for row in stream:
            seen += 1
            pbar.set_postfix(seen=seen, kept=kept)

            if filter_fn is not None and not filter_fn(row):
                continue

            record = transform_fn(row) if transform_fn else row
            buf.append(record)
            kept += 1
            pbar.update(1)

            if len(buf) >= WRITE_BATCH:
                write_jsonl(out_path, buf)
                buf.clear()

            if max_examples is not None and kept >= max_examples:
                log.info("Reached max_examples cap (%d). Stopping early.", max_examples)
                break

        if buf:
            write_jsonl(out_path, buf)

    log.info("Wrote %d examples to %s  (scanned %d total)", kept, out_path, seen)
    return kept


# Dataset 1 — Alpaca Cleaned (instruction tuning)
def download_alpaca() -> int:
    """
    unsloth/alpaca-cleaned schema:
      instruction (str), input (str), output (str)

    We reformat into a single 'text' field using the Alpaca prompt template,
    plus keep the raw fields for flexibility
    """
    log.info("=== Alpaca Cleaned ===")
    out_path = OUTPUT_DIR / "alpaca_cleaned.jsonl"

    ds = load_dataset(
        "unsloth/alpaca-cleaned",
        split="train",
        streaming=False,   # small enough to load fully (~52 k rows)
    )

    log.info("Loaded %d examples from alpaca-cleaned", len(ds))

    def transform(row: dict) -> dict:
        instruction = row["instruction"].strip()
        inp         = row.get("input", "").strip()
        output      = row.get("output", "").strip()

        if inp:
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{inp}\n\n"
                f"### Response:\n{output}"
            )
        else:
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output}"
            )

        return {
            "source":      "alpaca-cleaned",
            "text":        prompt,
            "instruction": instruction,
            "input":       inp,
            "output":      output,
        }

    out_path.unlink(missing_ok=True)
    records = [transform(row) for row in ds]
    write_jsonl(out_path, records)
    log.info("Wrote %d examples to %s", len(records), out_path)
    return len(records)


# Dataset 2 — The Stack v2 (code, licence-filtered)
def download_stack_v2() -> int:
    """
    bigcode/the-stack-v2-train-full-ids schema (relevant columns):
      content      (str)  — raw source code
      lang         (str)  — programming language
      license      (list) — list of SPDX licence strings
      max_stars_repo_name (str)
      max_stars_count     (int | None)

    We stream the dataset to avoid downloading ~4 TB up front.
    The 'train-full-ids' variant contains all rows with blob SHAs;
    actual file content may need a second lookup in some configs —
    check the dataset card if 'content' is missing in your stream.
    """
    log.info("=== The Stack v2 (streaming, licence-filtered) ===")
    out_path = OUTPUT_DIR / "stack_v2_filtered.jsonl"

    # Use streaming=True — the full dataset is enormous.
    ds = load_dataset(
        "bigcode/the-stack-v2-train-full-ids",
        split="train",
        streaming=True,
    )

    def filter_fn(row: dict) -> bool:
        # 1. Licence check
        if not is_permitted_licence(row.get("license")):
            return False
        # 2. Language check
        lang = (row.get("lang") or row.get("language") or "").lower()
        if lang not in ALLOWED_LANGUAGES:
            return False
        # 3. Skip empty content
        content = row.get("content") or ""
        if not content.strip():
            return False
        return True

    def transform_fn(row: dict) -> dict:
        lang    = (row.get("lang") or row.get("language") or "unknown").lower()
        licence = row.get("license") or []
        content = (row.get("content") or "").strip()
        repo    = row.get("max_stars_repo_name") or ""
        stars   = row.get("max_stars_count")

        return {
            "source":   "the-stack-v2",
            "text":     content,
            "language": lang,
            "license":  licence,
            "repo":     repo,
            "stars":    stars,
        }

    kept = stream_to_jsonl(
        stream=iter(ds),
        out_path=out_path,
        filter_fn=filter_fn,
        transform_fn=transform_fn,
        max_examples=STACK_MAX_EXAMPLES,
        desc="Stack v2 (filtered)",
    )
    return kept


# Summary / stats
def print_stats() -> None:
    log.info("=== Dataset Summary ===")
    for fname in ["alpaca_cleaned.jsonl", "stack_v2_filtered.jsonl"]:
        path = OUTPUT_DIR / fname
        if not path.exists():
            continue
        n = sum(1 for _ in open(path, encoding="utf-8"))
        size_mb = path.stat().st_size / 1e6
        log.info("  %-35s  %8d rows  %7.1f MB", fname, n, size_mb)

        # Language breakdown for Stack
        if "stack" in fname:
            from collections import Counter
            lang_counts: Counter = Counter()
            with open(path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    lang_counts[rec.get("language", "unknown")] += 1
            log.info("  Language breakdown:")
            for lang, cnt in lang_counts.most_common():
                log.info("    %-20s %d", lang, cnt)


# Entry point
def main():
    log.info("Starting dataset preparation...")
    log.info("Output directory: %s", OUTPUT_DIR.resolve())
    alpaca_count = download_alpaca()
    stack_count  = download_stack_v2()
    log.info("Done! Total examples: alpaca=%d  stack=%d", alpaca_count, stack_count)
    print_stats()
    log.info(
        "\nNext step: run  02_tokenise.py  to build the vocabulary and "
        "tokenise both datasets into binary shards for training."
    )

if __name__ == "__main__":
    main()