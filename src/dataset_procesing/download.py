import json
import logging
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from src.config import Config

# Configuration
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Maximum response length in characters to keep for "short response" datasets.
# Responses longer than this are truncated at the last sentence boundary so
# the model learns to stop rather than ramble.
MAX_RESPONSE_CHARS = 800


# Helpers

def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def truncate_to_sentences(text: str, max_chars: int) -> str:
    """
    Truncate text to at most max_chars, cutting at the last sentence boundary
    so the response feels complete rather than mid-sentence cut off.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Find the last sentence-ending punctuation
    for punct in (".", "!", "?", "\n"):
        idx = truncated.rfind(punct)
        if idx > max_chars // 2:   # don't cut too aggressively
            return truncated[: idx + 1].strip()
    return truncated.strip()


def alpaca_format(instruction: str, inp: str, output: str) -> str:
    """Format a single example into the Alpaca prompt template."""
    if inp:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


# Dataset 1 Alpaca Cleaned (instruction tuning, ~52k)

def download_alpaca() -> int:
    """
    unsloth/alpaca-cleaned
    License : CC BY-NC 4.0 (educational / research use)
    Schema  : instruction, input, output
    """
    log.info("=== Alpaca Cleaned ===")
    out_path = OUTPUT_DIR / "alpaca_cleaned.jsonl"

    ds = load_dataset("unsloth/alpaca-cleaned", split="train", streaming=False)
    log.info("Loaded %d examples from alpaca-cleaned", len(ds))

    def transform(row: dict) -> dict:
        instruction = row["instruction"].strip()
        inp         = row.get("input", "").strip()
        output      = row.get("output", "").strip()
        return {
            "source":      "alpaca-cleaned",
            "text":        alpaca_format(instruction, inp, output),
            "instruction": instruction,
            "input":       inp,
            "output":      output,
        }

    out_path.unlink(missing_ok=True)
    records = [transform(row) for row in ds]
    write_jsonl(out_path, records)
    log.info("Wrote %d examples to %s", len(records), out_path)
    return len(records)


# Dataset 2 Dolly 15k (human-written, concise QA, ~15k)

def download_dolly() -> int:
    """
    databricks/databricks-dolly-15k
    License : CC BY-SA 3.0 (open, educational use allowed)
    Schema  : instruction, context, response, category

    We keep all categories but apply MAX_RESPONSE_CHARS to teach the model
    to give focused answers.  Categories like 'open_qa' and 'classification'
    are naturally short; 'summarization' is truncated.
    """
    log.info("=== Databricks Dolly 15k ===")
    out_path = OUTPUT_DIR / "dolly_15k.jsonl"
    out_path.unlink(missing_ok=True)

    ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=False)
    log.info("Loaded %d examples", len(ds))

    buf = []
    for row in tqdm(ds, desc="dolly"):
        instruction = (row.get("instruction") or "").strip()
        context     = (row.get("context") or "").strip()
        response    = (row.get("response") or "").strip()

        if not instruction or not response:
            continue

        # Truncate long responses so the model learns brevity
        response = truncate_to_sentences(response, MAX_RESPONSE_CHARS)

        buf.append({
            "source":      "dolly-15k",
            "text":        alpaca_format(instruction, context, response),
            "instruction": instruction,
            "input":       context,
            "output":      response,
        })

    write_jsonl(out_path, buf)
    log.info("Wrote %d examples to %s", len(buf), out_path)
    return len(buf)


# ── Dataset 3 — Open Instruct V1 (diverse instructions, ~500k, capped) ───────

def download_open_instruct() -> int:
    """
    hakurei/open-instruct-v1
    License : Apache 2.0
    Schema  : instruction, response   (no input field)
    Size    : ~500k rows — we cap at 50k to keep things balanced

    This is an amalgamation of many open instruction datasets, cleaned into
    a single format.  Good source of short, direct Q&A pairs.
    """
    log.info("=== Open Instruct V1 (capped at 50k) ===")
    out_path = OUTPUT_DIR / "open_instruct.jsonl"
    out_path.unlink(missing_ok=True)

    CAP = 50_000

    ds = load_dataset(
        "hakurei/open-instruct-v1",
        split="train",
        streaming=True,   # stream — 500k rows, don't load all into RAM
    )

    buf = []
    kept = 0

    for row in tqdm(ds, desc="open_instruct", total=CAP):
        instruction = (row.get("instruction") or "").strip()
        response    = (row.get("output") or "").strip()

        if not instruction or not response:
            continue

        # Skip very long responses — we want short-answer behaviour
        response = truncate_to_sentences(response, MAX_RESPONSE_CHARS)

        buf.append({
            "source":      "open-instruct-v1",
            "text":        alpaca_format(instruction, "", response),
            "instruction": instruction,
            "input":       "",
            "output":      response,
        })
        kept += 1

        if len(buf) >= 5_000:
            write_jsonl(out_path, buf)
            buf.clear()

        if kept >= CAP:
            log.info("Reached cap (%d). Stopping.", CAP)
            break

    if buf:
        write_jsonl(out_path, buf)

    log.info("Wrote %d examples to %s", kept, out_path)
    return kept


# Dataset 4 Python Code Instructions (MIT, ~43k) 

def download_python_code_instructions() -> int:
    """
    1. iamtarun/python_code_instructions_18k_alpaca  (~18k, MIT)
    2. flytech/python-codes-25k                      (~25k, MIT)

    Both Alpaca-format with Python code in the output field.
    """
    log.info("=== Python code datasets (open, MIT-licensed) ===")
    out_path = OUTPUT_DIR / "python_code_instructions.jsonl"
    out_path.unlink(missing_ok=True)

    SOURCES = [
        ("iamtarun/python_code_instructions_18k_alpaca", "train"),
        ("flytech/python-codes-25k",                     "train"),
    ]

    UNSAFE_PATTERNS = (
        "-----begin rsa private key-----",
        "-----begin openssh private key-----",
        "-----begin private key-----",
        "aws_secret_access_key",
        "aws_access_key_id",
        "ghp_",
        "glpat-",
    )

    total_kept = 0

    for dataset_name, split in SOURCES:
        log.info("--- Loading %s ---", dataset_name)
        try:
            ds = load_dataset(dataset_name, split=split, streaming=False)
        except Exception as e:
            log.warning("Could not load %s: %s — skipping.", dataset_name, e)
            continue

        log.info("  Loaded %d examples", len(ds))
        buf = []

        for row in tqdm(ds, desc=dataset_name.split("/")[1]):
            instruction = (row.get("instruction") or "").strip()
            inp         = (row.get("input")       or "").strip()
            output      = (row.get("output") or row.get("text") or "").strip()

            if not output:
                continue
            if any(p in output.lower() for p in UNSAFE_PATTERNS):
                continue

            buf.append({
                "source":   dataset_name,
                "text":     alpaca_format(instruction, inp, output),
                "language": "python",
                "license":  ["mit"],
            })

        write_jsonl(out_path, buf)
        log.info("  Kept %d examples from %s", len(buf), dataset_name)
        total_kept += len(buf)

    log.info("Total code examples written: %d → %s", total_kept, out_path)
    return total_kept


# Summary / stats

def print_stats() -> None:
    log.info("=== Dataset Summary ===")
    files = [
        "alpaca_cleaned.jsonl",
        "dolly_15k.jsonl",
        "open_instruct.jsonl",
        "python_code_instructions.jsonl",
    ]
    total = 0
    for fname in files:
        path = OUTPUT_DIR / fname
        if not path.exists():
            log.info("  %-40s  (not found)", fname)
            continue
        n = sum(1 for _ in open(path, encoding="utf-8"))
        size_mb = path.stat().st_size / 1e6
        log.info("  %-40s  %8d rows  %6.1f MB", fname, n, size_mb)
        total += n
    log.info("  %-40s  %8d rows  total", "TOTAL", total)


# Entry point 

def main(config: Config) -> None:
    log.info("Starting dataset preparation...")
    log.info("Output directory: %s", OUTPUT_DIR.resolve())

    alpaca_count       = download_alpaca()
    dolly_count        = download_dolly()
    open_instruct_count = download_open_instruct()
   # python_code_count  = download_python_code_instructions()

    log.info("Done!")
    print_stats()
    log.info("Next step: tokenise")


if __name__ == "__main__":
    config = Config()
    main(config)