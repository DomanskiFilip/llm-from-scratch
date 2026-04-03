"""
WHAT THIS FILE DOES
-------------------
1. Loads a trained checkpoint from train.py
2. Computes:
     - Per-epoch loss curves (train + validation)  →  PNG plot
     - Perplexity (the standard LM metric)
     - Token-level accuracy (top-1 and top-5)
     - Confusion matrix on the most frequent 50 tokens  →  PNG plot
     - Precision, recall, F1 for those tokens
3. Writes a human-readable Insights report covering:
     - Model strengths
     - Model limitations
     - Recommended improvements (BERT / transformer path)
     
METRICS EXPLAINED
-----------------

PERPLEXITY
    PPL = exp(L)  where L = average cross-entropy loss over the test set.
    Intuitively: if PPL = 100, the model is as uncertain as if it were
    choosing uniformly among 100 equally likely next tokens.
    Lower is better.  A char-level model on English might achieve PPL ≈ 2-3;
    a word-level model on Penn Treebank ≈ 60-80; our BPE code model at the
    start of training ≈ exp(log(32768)) ≈ 32,768 (random baseline)

TOKEN ACCURACY
    Top-1: fraction of positions where the most probable token equals the
    ground-truth next token.  This is a strict metric — the model must
    rank the correct token first.
    Top-5: fraction where the correct token is among the top 5 predictions.
    Useful for seeing whether the model is "almost right" even when top-1 fails

CONFUSION MATRIX
    For language models, a full V×V confusion matrix (32,768 × 32,768) is
    not practical.  We restrict to the K most frequent tokens in the
    evaluation set (K=50 by default) and show an K×K matrix where entry
    (i,j) counts how often token i was the ground truth and token j was
    the model's top prediction.  A good model has a diagonal matrix

PRECISION / RECALL / F1 (per-token)
    Treating each token as a binary classification problem (was token t
    predicted or not?):
    Precision_t = TP_t / (TP_t + FP_t)  — when we predict t, are we right?
    Recall_t    = TP_t / (TP_t + FN_t)  — of all ground-truth t's, how many
                                           did we catch?
    F1_t        = 2 × P × R / (P + R)   — harmonic mean
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import argparse
import json
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn.functional as F
from model import CodingLM, LMConfig
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm
from train import get_device

from src.config import Config

# Paths
DATA_DIR = Path("artefacts/data")
LOG_DIR = Path("artefacts/logs")
EVAL_DIR = Path("artefacts/evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

TOKENISER_JSON = Path("artefacts/tokeniser") / "qwen_style.json"

# How many of the most frequent tokens to include in the confusion matrix
TOP_K_TOKENS = 50


# Load checkpoint
def load_checkpoint(ckpt_path: Path, device: torch.device):
    """Load a checkpoint saved by train.py and return (model, metadata)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt["config"]
    model_cfg = LMConfig(
        **{k: v for k, v in cfg_dict.items() if k in LMConfig.__dataclass_fields__}
    )
    model = CodingLM(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  Trained for {ckpt['epoch']} epoch(s),  val_loss={ckpt['val_loss']:.4f}")
    return model, ckpt


# 1. Loss curves
def plot_loss_curves(log_csv_path: Path, out_path: Path) -> None:
    """
    Read the CSV log written by train.py and plot train vs val loss
    per epoch.  Loss curves are the primary tool for diagnosing:
      - Underfitting: both curves high and flat  →  model too small / LR too low
      - Overfitting: train continues falling but val plateaus or rises
      - Good fit:    both curves fall together and converge
    """
    import csv

    epochs, train_losses, val_losses = [], [], []
    with open(log_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # One row per epoch (we only kept epoch-level rows)
            # If there are also step rows, skip them
            try:
                e = int(row["epoch"])
                t = float(row["train_loss"])
                v = float(row["val_loss"])
                if e not in epochs:  # first occurrence per epoch
                    epochs.append(e)
                    train_losses.append(t)
                    val_losses.append(v)
            except (ValueError, KeyError):
                continue

    if not epochs:
        print(f"  No epoch data found in {log_csv_path} — skipping loss curves.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Training & Validation Curves", fontsize=14, fontweight="bold")

    # Left: loss
    ax = axes[0]
    ax.plot(
        epochs, train_losses, "o-", label="Train loss", color="#2196F3", linewidth=2
    )
    ax.plot(
        epochs, val_losses, "s--", label="Validation loss", color="#F44336", linewidth=2
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Loss vs Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: perplexity
    ax = axes[1]
    train_ppl = [math.exp(min(l, 20)) for l in train_losses]
    val_ppl = [math.exp(min(l, 20)) for l in val_losses]
    ax.plot(epochs, train_ppl, "o-", label="Train PPL", color="#2196F3", linewidth=2)
    ax.plot(
        epochs, val_ppl, "s--", label="Validation PPL", color="#F44336", linewidth=2
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity  (lower = better)")
    ax.set_title("Perplexity vs Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Loss curves saved → {out_path}")


# Collect predictions
@torch.no_grad()
@torch.no_grad()
def collect_predictions(
    model: CodingLM,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    vocab_size: int,
    max_batches: int = 200,
) -> dict:
    model.eval()
    all_targets, all_top1, all_logprobs = [], [], []
    hidden = None

    # Note: we unpack the batch as (x, y) because ShardDataset returns a tuple
    for batch_idx, (x, y) in enumerate(
        tqdm(loader, desc="Evaluating", total=max_batches)
    ):
        if max_batches and batch_idx >= max_batches:
            break
            
        x, y = x.to(device), y.to(device)
        logits, hidden = model(x, None)
        hidden = CodingLM.detach_hidden(hidden)

        # [B*T, V]
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = y.reshape(-1)

        # MASKING LOGIC: 
        # ShardDataset sets target_ids to -1 for instruction tokens.
        # We only collect metrics for the response tokens (targets != -1).
        mask = (targets_flat != -1)
        
        if not mask.any():
            continue

        masked_logits = logits_flat[mask]
        masked_targets = targets_flat[mask]

        # Top-1 prediction for accuracy
        top1 = masked_logits.argmax(dim=-1)

        # Log-probability for perplexity
        log_probs = F.log_softmax(masked_logits, dim=-1)
        gt_log_probs = log_probs[
            torch.arange(len(masked_targets), device=device), masked_targets
        ]

        all_targets.append(masked_targets.cpu())
        all_top1.append(top1.cpu())
        all_logprobs.append(gt_log_probs.cpu())

    return {
        "targets": torch.cat(all_targets).numpy(),
        "top1": torch.cat(all_top1).numpy(),
        "logprobs": torch.cat(all_logprobs).numpy(),
    }


# Compute scalar metrics
def compute_metrics(preds: dict) -> dict:
    """Compute perplexity, top-1 accuracy, top-5 accuracy (approx from top-1)."""
    targets = preds["targets"]
    top1 = preds["top1"]
    logprobs = preds["logprobs"]

    # Perplexity
    avg_nll = -float(np.mean(logprobs))
    perplexity = math.exp(min(avg_nll, 20))

    # Top-1 accuracy
    top1_acc = float(np.mean(targets == top1))

    return {
        "perplexity": perplexity,
        "avg_nll": avg_nll,
        "top1_accuracy": top1_acc,
    }


# Confusion matrix (top K tokens)
def plot_confusion_matrix(
    preds: dict,
    token_names: list,
    out_path: Path,
    top_k: int = TOP_K_TOKENS,
) -> None:
    """
    Plot a confusion matrix restricted to the `top_k` most frequent tokens.

    Rows = ground truth token, Columns = predicted token.
    Diagonal = correct predictions (good).
    Off-diagonal = confusions (the model thought it should be token j but
    the correct answer was token i).
    """
    targets = preds["targets"]
    top1 = preds["top1"]

    # Find the top_k most frequent token IDs in the ground truth
    unique, counts = np.unique(targets, return_counts=True)
    top_ids = unique[np.argsort(-counts)][:top_k]

    # Filter to positions where the ground truth is one of the top tokens
    mask = np.isin(targets, top_ids)
    t_masked = targets[mask]
    p_masked = top1[mask]

    # Build confusion matrix
    labels = sorted(top_ids.tolist())
    cm = confusion_matrix(t_masked, p_masked, labels=labels)

    # Normalise by row (ground truth frequency) so colours encode recall
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    # Token display names: truncate long tokens for readability
    label_names = []
    for tid in labels:
        name = token_names[tid] if tid < len(token_names) else str(tid)
        name = repr(name)[:12]  # truncate to fit
        label_names.append(name)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (row-normalised)")

    ax.set_xticks(range(top_k))
    ax.set_yticks(range(top_k))
    ax.set_xticklabels(label_names, rotation=90, fontsize=6)
    ax.set_yticklabels(label_names, fontsize=6)
    ax.set_xlabel("Predicted token")
    ax.set_ylabel("Ground-truth token")
    ax.set_title(f"Confusion Matrix — top {top_k} tokens (row-normalised recall)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved → {out_path}")


# Precision / Recall / F1 (top K tokens)
def compute_prf(preds: dict, top_k: int = TOP_K_TOKENS) -> str:
    """
    Compute precision, recall, and F1 for the top_k most frequent tokens.
    Returns a formatted string report.
    """
    targets = preds["targets"]
    top1 = preds["top1"]

    unique, counts = np.unique(targets, return_counts=True)
    top_ids = unique[np.argsort(-counts)][:top_k].tolist()
    mask = np.isin(targets, top_ids)

    report = classification_report(
        targets[mask],
        top1[mask],
        labels=top_ids,
        target_names=[str(t) for t in top_ids],
        digits=3,
        zero_division=0,
    )
    return report


INSIGHTS_TEMPLATE = """
CODING LLM — EVALUATION INSIGHTS REPORT
=========================================
Generated from: {ckpt_path}

QUANTITATIVE RESULTS
--------------------
  Perplexity (val)       : {perplexity:.2f}
  Average NLL loss       : {avg_nll:.4f}
  Top-1 token accuracy   : {top1_accuracy:.2%}

INTERPRETATION OF PERPLEXITY
-----------------------------
Perplexity = exp(average negative log-probability of the correct next token).
A value of {perplexity:.1f} means the model is as uncertain as if it were
choosing uniformly among ~{perplexity:.0f} equally likely tokens at each step.

"""


def write_insights(metrics: dict, ckpt_path: str, out_path: Path) -> None:
    report = INSIGHTS_TEMPLATE.format(ckpt_path=ckpt_path, **metrics)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Insights report saved → {out_path}")


# Entry point
def main(config: Config) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CodingLM checkpoint"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to .pt checkpoint"
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--batches",
        type=int,
        default=200,
        help="Max evaluation batches (None = full val set)",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    ckpt_path = Path(args.ckpt)

    # Load model
    model, ckpt = load_checkpoint(ckpt_path, device)
    cfg_dict = ckpt["train_cfg"]
    vocab_size = model.config.vocab_size

    # Loss curves find the matching log CSV
    run_name = ckpt.get("run_name", "default_run")
    log_csv = Path("artefacts/logs") / f"{run_name}.csv"
    if log_csv.exists():
        plot_loss_curves(log_csv, EVAL_DIR / "loss_curves.png")
    else:
        print(f"  Log CSV not found at {log_csv} — skipping loss curves.")

    # Build val dataloader
    from train import TrainConfig, build_dataloaders

    train_cfg = TrainConfig(
        seq_len=cfg_dict.get("seq_len", 256),
        batch_size=cfg_dict.get("batch_size", 32),
        device=args.device,
    )
    _, val_loader = build_dataloaders(train_cfg, device.type)

    # Collect predictions
    preds = collect_predictions(
        model, val_loader, device, vocab_size, max_batches=args.batches
    )

    # Scalar metrics
    metrics = compute_metrics(preds)
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")

    # Confusion matrix
    # Load vocab for token display names
    token_names = []
    if TOKENISER_JSON.exists():
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(TOKENISER_JSON))
        token_names = [tok.id_to_token(i) or "" for i in range(vocab_size)]
    plot_confusion_matrix(preds, token_names, EVAL_DIR / "confusion_matrix.png")

    # Precision / Recall / F1
    prf_report = compute_prf(preds)
    prf_path = EVAL_DIR / "precision_recall_f1.txt"
    with open(prf_path, "w") as f:
        f.write(prf_report)
    print(f"  PRF report saved → {prf_path}")

    # Full metrics JSON
    metrics_path = EVAL_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Insights report
    write_insights(metrics, str(ckpt_path), EVAL_DIR / "insights_report.txt")

    print(f"\nAll evaluation outputs saved to {EVAL_DIR}/")
    print("Next step: run  generate.py --ckpt", args.ckpt)


if __name__ == "__main__":
    config = Config()
    main(config)
