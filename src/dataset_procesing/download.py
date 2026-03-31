import json
import logging
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# Configuration
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# helper
def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# Dataset 1 — Alpaca Cleaned (instruction tuning)
def download_alpaca() -> int:
    """
    unsloth/alpaca-cleaned schema:
      instruction (str), input (str), output (str)
    """
    log.info("=== Alpaca Cleaned ===")
    out_path = OUTPUT_DIR / "alpaca_cleaned.jsonl"

    ds = load_dataset(
        "unsloth/alpaca-cleaned",
        split="train",
        streaming=False,
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

# Dataset 2 — Python Code Instructions (open, MIT-licensed)
def download_python_code_instructions() -> int:
    """
    Two open, MIT-licensed Python code datasets — no gating, no login required

      1. iamtarun/python_code_instructions_18k_alpaca  (~18k instruction+code pairs)
      2. flytech/python-codes-25k                      (~25k instruction+code pairs)

    Both are Alpaca-format: instruction / input / output fields where the
    output contains Python code. We format them the same way as Alpaca so
    the tokeniser sees consistent prompt structure across both datasets
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
        buf: list[dict] = []

        for row in tqdm(ds, desc=dataset_name.split("/")[1]):
            instruction = (row.get("instruction") or "").strip()
            inp         = (row.get("input") or "").strip()
            output      = (row.get("output") or row.get("text") or "").strip()

            if not output:
                continue

            # Credential safety filter on the code output
            if any(p in output.lower() for p in UNSAFE_PATTERNS):
                continue

            # Format identically to Alpaca so both datasets look the same
            if inp:
                text = (
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\n{inp}\n\n"
                    f"### Response:\n{output}"
                )
            else:
                text = (
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Response:\n{output}"
                )

            buf.append({
                "source":   dataset_name,
                "text":     text,
                "language": "python",
                "license":  ["mit"],
            })

        write_jsonl(out_path, buf)
        log.info("  Kept %d examples from %s", len(buf), dataset_name)
        total_kept += len(buf)

    log.info("Total code examples written: %d → %s", total_kept, out_path)
    return total_kept

#  Summary / stats 
def print_stats() -> None:
    log.info("=== Dataset Summary ===")
    for fname in ["alpaca_cleaned.jsonl", "python_code_instructions.jsonl"]:
        path = OUTPUT_DIR / fname
        if not path.exists():
            log.info("  %-35s  (not found)", fname)
            continue
        n = sum(1 for _ in open(path, encoding="utf-8"))
        size_mb = path.stat().st_size / 1e6
        log.info("  %-35s  %8d rows  %7.1f MB", fname, n, size_mb)

        if "python_code_instructions" in fname:
            from collections import Counter
            lang_counts: Counter = Counter()
            with open(path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    lang_counts[rec.get("language", "unknown")] += 1
            log.info("  Language breakdown:")
            for lang, cnt in lang_counts.most_common():
                log.info("    %-20s %d", lang, cnt)


#  Summary / stats 
def print_stats() -> None:
    log.info("=== Dataset Summary ===")
    for fname in ["alpaca_cleaned.jsonl", "stack_v2_filtered.jsonl"]:
        path = OUTPUT_DIR / fname
        if not path.exists():
            log.info("  %-35s  (not found)", fname)
            continue
        n = sum(1 for _ in open(path, encoding="utf-8"))
        size_mb = path.stat().st_size / 1e6
        log.info("  %-35s  %8d rows  %7.1f MB", fname, n, size_mb)

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
    python_code_count = download_python_code_instructions()

    log.info("Done! Total examples: python_code_instructions=%d", python_code_count)
    print_stats()
    log.info(
        "Next step: run  python main.py tokenise  to build the vocabulary "
        "and tokenise both datasets into binary shards for training."
    )



if __name__ == "__main__":
    main()