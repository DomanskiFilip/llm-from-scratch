import json
import logging
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from src.config import Config

OUTPUT_DIR = Path("artefacts/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MAX_RESPONSE_CHARS = 800


# Helpers 
def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def truncate_to_sentences(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    for punct in (".", "!", "?", "\n"):
        idx = truncated.rfind(punct)
        if idx > max_chars // 2:
            return truncated[: idx + 1].strip()
    return truncated.strip()


def alpaca_format(instruction: str, inp: str, output: str) -> tuple[str, int]:
    """
    Build the Alpaca-formatted string and return (text, response_start_char).

    response_start_char is the index in `text` where `output` begins —
    i.e. right after the '### Response:\\n' header.  tokeniser.py uses this
    to build a binary loss mask so gradients only flow through response tokens.
    """
    if inp:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    text = prompt + output
    return text, len(prompt)


#  Dataset 1 — Alpaca Cleaned (~52k)
def download_alpaca() -> int:
    log.info("=== Alpaca Cleaned ===")
    out_path = OUTPUT_DIR / "alpaca_cleaned.jsonl"

    ds = load_dataset("unsloth/alpaca-cleaned", split="train", streaming=False)
    log.info("Loaded %d examples from alpaca-cleaned", len(ds))

    out_path.unlink(missing_ok=True)
    records = []
    for row in ds:
        instruction = row["instruction"].strip()
        inp         = row.get("input", "").strip()
        output      = row.get("output", "").strip()
        text, rsc   = alpaca_format(instruction, inp, output)
        records.append({
            "source":               "alpaca-cleaned",
            "text":                 text,
            "instruction":          instruction,
            "input":                inp,
            "output":               output,
            "response_start_char":  rsc,
        })

    write_jsonl(out_path, records)
    log.info("Wrote %d examples to %s", len(records), out_path)
    return len(records)


# Dataset 2 — Dolly 15k (~15k) 
def download_dolly() -> int:
    log.info("=== Databricks Dolly 15k ===")
    out_path = OUTPUT_DIR / "dolly_15k.jsonl"
    out_path.unlink(missing_ok=True)

    ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=False)
    log.info("Loaded %d examples", len(ds))

    buf = []
    for row in tqdm(ds, desc="dolly"):
        instruction = (row.get("instruction") or "").strip()
        context     = (row.get("context")     or "").strip()
        response    = (row.get("response")    or "").strip()

        if not instruction or not response:
            continue

        response        = truncate_to_sentences(response, MAX_RESPONSE_CHARS)
        text, rsc       = alpaca_format(instruction, context, response)
        buf.append({
            "source":               "dolly-15k",
            "text":                 text,
            "instruction":          instruction,
            "input":                context,
            "output":               response,
            "response_start_char":  rsc,
        })

    write_jsonl(out_path, buf)
    log.info("Wrote %d examples to %s", len(buf), out_path)
    return len(buf)


# Dataset 3 — Open Instruct V1 (capped at 50k)
def download_open_instruct() -> int:
    log.info("=== Open Instruct V1 (capped at 50k) ===")
    out_path = OUTPUT_DIR / "open_instruct.jsonl"
    out_path.unlink(missing_ok=True)

    CAP = 50_000
    ds  = load_dataset("hakurei/open-instruct-v1", split="train", streaming=True)

    buf  = []
    kept = 0

    for row in tqdm(ds, desc="open_instruct", total=CAP):
        instruction = (row.get("instruction") or "").strip()
        response    = (row.get("output")      or "").strip()

        if not instruction or not response:
            continue

        response  = truncate_to_sentences(response, MAX_RESPONSE_CHARS)
        text, rsc = alpaca_format(instruction, "", response)
        buf.append({
            "source":               "open-instruct-v1",
            "text":                 text,
            "instruction":          instruction,
            "input":                "",
            "output":               response,
            "response_start_char":  rsc,
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


# Summary
def print_stats() -> None:
    log.info("=== Dataset Summary ===")
    files = [
        "alpaca_cleaned.jsonl",
        "dolly_15k.jsonl",
        "open_instruct.jsonl"
    ]
    total = 0
    for fname in files:
        path = OUTPUT_DIR / fname
        if not path.exists():
            log.info("  %-40s  (not found)", fname)
            continue
        n       = sum(1 for _ in open(path, encoding="utf-8"))
        size_mb = path.stat().st_size / 1e6
        log.info("  %-40s  %8d rows  %6.1f MB", fname, n, size_mb)
        total += n
    log.info("  %-40s  %8d rows  total", "TOTAL", total)


# synthetic dataset to train the model to say hello
def download_hello() -> int:
    log.info("=== Generating Synthetic 'Hello' Dataset ===")
    out_path = OUTPUT_DIR / "hello_synthetic.jsonl"
    out_path.unlink(missing_ok=True)
    
    num_examples = 5000
    buf = []
    
    # We create a mix of simple "Hello" instructions
    variations = [
        ("Say hello.", "Hello! How can I help you today?"),
        ("Greet me.", "Greetings! I am an AI assistant."),
        ("Hello", "Hello! It is nice to meet you."),
        ("Hi", "Hi there!"),
    ]

    for i in range(num_examples):
        # Cycle through variations
        instruction, output = variations[i % len(variations)]
        
        # Use existing alpaca_format to ensure consistency
        text, rsc = alpaca_format(instruction, "", output)
        
        buf.append({
            "source":               "synthetic-hello",
            "text":                 text,
            "instruction":          instruction,
            "input":                "",
            "output":               output,
            "response_start_char":  rsc,
        })
        
        if len(buf) >= 1000:
            write_jsonl(out_path, buf)
            buf.clear()
            
    if buf:
        write_jsonl(out_path, buf)
        
    log.info("Wrote %d synthetic examples to %s", num_examples, out_path)
    return num_examples

# Entry point 
def main(config: Config) -> None:
    log.info("Starting dataset preparation...")
    log.info("Output directory: %s", OUTPUT_DIR.resolve())

    download_alpaca()
    download_dolly()
    download_open_instruct()
    download_hello()

    log.info("Done!")
    print_stats()
    log.info("Next step: tokenise")


if __name__ == "__main__":
    config = Config()
    main(config)