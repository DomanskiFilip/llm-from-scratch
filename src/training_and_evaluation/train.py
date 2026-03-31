"""
train.py — Training Loop, Hyperparameter Search & Early Stopping
=====================================================================

WHAT THIS FILE DOES
-------------------
1. Loads tokenised binary shards produced by tokeniser.py
2. Trains CodingLM (from model.py) with AdamW + cosine LR schedule
3. Implements early stopping on validation loss
4. Runs a grid search over learning rate, batch size, and dropout rate
5. Logs training/validation loss per epoch to CSV files for plotting
6. Saves the best checkpoint for use in evaluate.py and generate.py

REQUIREMENT COVERAGE (Req 1.2.2)
---------------------------------
  ✓ Batch size experimentation
  ✓ Learning rate experimentation
  ✓ Dropout rate experimentation
  ✓ Grid search for hyperparameter tuning
  ✓ Early stopping
  ✓ All attempts logged for comparison

TRAINING CONCEPTS EXPLAINED
----------------------------

TRUNCATED BACK-PROPAGATION THROUGH TIME (TBPTT)
    An LSTM's forward pass over a 512-token sequence produces a computation
    graph of depth 512.  Back-propagating through that full graph is:
      a) Slow — every node in the graph must be materialised in memory.
      b) Numerically fragile — gradients can vanish or explode over 512 steps.
    TBPTT solves this by splitting each sequence into chunks of length
    `bptt_len` (e.g. 64).  We process chunk by chunk, carrying the hidden
    state forward but *detaching* it so gradients only flow within a chunk.
    Chunk length is a hyperparameter: shorter = cheaper but less context.

AdamW OPTIMISER
    Loshchilov & Hutter (2019) "Decoupled Weight Decay Regularisation"
    arXiv:1711.05101.
    AdamW = Adam (adaptive per-parameter learning rates) + correct L2 weight
    decay.  Vanilla Adam applies weight decay incorrectly via gradient updates;
    AdamW applies it directly to the weights, which is more effective and
    allows larger weight decay values without destabilising training.

COSINE LEARNING RATE SCHEDULE
    After a short linear warmup (prevents large early gradient steps), the
    learning rate decays following a half-cosine curve from lr_max down to
    lr_min.  This allows the model to first explore the loss surface with
    large steps, then settle into a sharp minimum with small steps.

GRADIENT CLIPPING
    If the global gradient norm exceeds `clip_norm`, all gradients are scaled
    down proportionally.  This prevents the "exploding gradient" problem that
    can occur in LSTMs on long sequences.

CROSS-ENTROPY LOSS
    The standard loss for language models:
        L = −(1/T) Σ_t log P(xₜ₊₁ | x₁…xₜ)
    = average negative log-probability of the next token.
    Perfect memorisation would give L → 0; a random model gives L ≈ log(V).
    Perplexity = exp(L): lower is better; a value of 1 is perfect.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from model import CodingLM, LMConfig

import argparse
import csv
import json
import math
import struct
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from model import CodingLM, LMConfig   # model.py

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR  = Path("data")
CKPT_DIR  = Path("checkpoints")
LOG_DIR   = Path("logs")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

GLOVE_PT  = Path("embeddings") / "glove_aligned.pt"


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """
    All training hyperparameters.

    These are the values used for the *default* single run.
    The grid search overrides lr, batch_size, and dropout_rate.
    """
    # --- Data ---
    seq_len      : int   = 256      # tokens per training example (context window)
    bptt_len     : int   = 64       # TBPTT chunk length (how far back gradients flow)

    # --- Optimisation ---
    lr           : float = 3e-4     # peak learning rate for AdamW
    weight_decay : float = 0.1      # AdamW weight decay
    clip_norm    : float = 1.0      # gradient clipping norm
    warmup_steps : int   = 200      # linear LR warmup steps

    # --- Schedule ---
    epochs       : int   = 20
    batch_size   : int   = 32

    # --- Regularisation ---
    dropout_rate : float = 0.2      # applied to embed_drop, lstm_drop, out_drop

    # --- Early stopping ---
    patience     : int   = 3        # stop if val loss doesn't improve for this many epochs
    min_delta    : float = 1e-4     # minimum improvement to count as improvement

    # --- Hardware ---
    device       : str   = "auto"   # "auto" picks CUDA > MPS > CPU

    # --- Data split ---
    val_fraction : float = 0.05     # fraction of shards reserved for validation

    # --- Logging ---
    log_every    : int   = 100      # log loss every N batches


# ---------------------------------------------------------------------------
# Grid search space (Req 1.2.2)
# ---------------------------------------------------------------------------

GRID = {
    "lr"           : [1e-3, 3e-4, 1e-4],
    "batch_size"   : [32, 64],
    "dropout_rate" : [0.1, 0.3],
}

# For a full search, every combination is tried.
# 3 × 2 × 2 = 12 runs.  Each run trains for `grid_epochs` epochs only to
# keep the search affordable; the best config is then trained to convergence.
GRID_EPOCHS = 3   # short runs for the search
FULL_EPOCHS = 20  # full run with the best config


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


# ---------------------------------------------------------------------------
# Binary shard DataLoader
# ---------------------------------------------------------------------------
# The shards produced by tokeniser.py are flat arrays of uint16 token IDs.
# We read them into numpy arrays and slice out [seq_len+1]-length windows.
# The +1 is because for a context [x₀, x₁, …, x_{T-1}] the target is
# [x₁, x₂, …, x_T] — each target is the next token.

def load_shard(path: Path) -> np.ndarray:
    """Read a .bin shard into a uint16 numpy array."""
    data = path.read_bytes()
    n = len(data) // 2
    return np.frombuffer(data, dtype=np.uint16, count=n).astype(np.int32)


class ShardDataset(torch.utils.data.Dataset):
    """
    Dataset that spans multiple binary shard files.

    Each item is a pair (input_ids, target_ids) of length seq_len.
    input_ids[i] = token at position i
    target_ids[i] = token at position i+1  (the "next token" the model predicts)
    """

    def __init__(self, shard_paths: List[Path], seq_len: int) -> None:
        self.seq_len = seq_len
        # Load and concatenate all shards
        arrays = [load_shard(p) for p in shard_paths]
        self.data = np.concatenate(arrays)
        # Number of complete windows we can extract
        self.n = (len(self.data) - 1) // seq_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].copy()).long()
        y = torch.from_numpy(chunk[1:].copy()).long()
        return x, y


def build_dataloaders(cfg: TrainConfig, device_type: str):
    """
    Find all shards, split into train/val, return DataLoaders.
    Handles the case where only one shard is present.
    """
    shards = sorted(DATA_DIR.glob("*_shard_*.bin"))
    if not shards:
        raise FileNotFoundError(
            f"No shard files found in {DATA_DIR}. Run tokeniser.py first."
        )

    pin = device_type == "cuda"

    #  Multiple Shards
    if len(shards) > 1:
        n_val = max(1, int(len(shards) * cfg.val_fraction))
        val_shards   = shards[:n_val]
        train_shards = shards[n_val:]
        
        print(f"  Shards: {len(train_shards)} train, {len(val_shards)} val")
        train_ds = ShardDataset(train_shards, cfg.seq_len)
        val_ds   = ShardDataset(val_shards,   cfg.seq_len)

    # CASE B: Single Shard
    else:
        print(f"  Single shard detected. Splitting data within the shard.")
        full_ds = ShardDataset(shards, cfg.seq_len)
        
        # Split the single dataset into 95% train / 5% val (or based on val_fraction)
        val_size = max(1, int(len(full_ds) * cfg.val_fraction))
        train_size = len(full_ds) - val_size
        
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds, [train_size, val_size]
        )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=2, pin_memory=pin, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=2, pin_memory=pin, drop_last=True,
    )
    return train_loader, val_loader


# Learning rate schedule: linear warmup + cosine decay
def make_lr_lambda(warmup_steps: int, total_steps: int, lr_min_ratio: float = 0.1):
    """
    Returns a function step → lr_multiplier for LambdaLR.

    Phase 1 (steps 0…warmup_steps): linear ramp from 0 → 1.
    Phase 2 (steps warmup_steps…total_steps): cosine decay from 1 → lr_min_ratio.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cosine
    return lr_lambda


# Early stopping
class EarlyStopping:
    """
    Monitors validation loss and signals when training should stop.

    Implements the "patience" heuristic: if the validation loss does not
    improve by at least min_delta for `patience` consecutive epochs, the
    training loop should stop to prevent overfitting.

    Attributes:
        best_loss   Best validation loss seen so far.
        counter     Epochs without improvement.
        should_stop Flag set to True when patience is exhausted.
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-4) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """
        Call once per epoch.  Returns True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# Single training run
def run_training(
    cfg: TrainConfig,
    run_name: str,
    max_epochs: Optional[int] = None,
    use_glove: bool = True,
) -> dict:
    """
    Train the model with settings from `cfg`.

    Returns a dict with:
        best_val_loss   float
        train_losses    list of per-epoch average train losses
        val_losses      list of per-epoch val losses
        epochs_run      int
        ckpt_path       Path to saved best checkpoint
    """
    device = get_device(cfg.device)
    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"  device={device}  lr={cfg.lr}  batch={cfg.batch_size}  dropout={cfg.dropout_rate}")
    print(f"{'='*60}")

    epochs = max_epochs if max_epochs is not None else cfg.epochs

    # Build model
    model_cfg = LMConfig(
        vocab_size   = 32_768,
        embed_dim    = 100,
        hidden_dim   = 512,
        n_layers     = 2,
        embed_drop   = cfg.dropout_rate,
        lstm_drop    = cfg.dropout_rate,
        out_drop     = cfg.dropout_rate,
        tie_weights  = False,
        max_seq_len  = cfg.seq_len + 64,   # a little headroom
        pad_id       = 6,
    )

    pretrained = None
    if use_glove and GLOVE_PT.exists():
        pretrained = torch.load(GLOVE_PT, map_location="cpu")

    model = CodingLM(model_cfg, pretrained_embeddings=pretrained).to(device)
    print(f"  Model: {model.count_parameters():,} parameters")

    # Data
    train_loader, val_loader = build_dataloaders(cfg, device.type)

    # Optimiser
    # Separate parameter groups: embeddings + biases don't get weight decay.
    # This matches the convention in Loshchilov & Hutter (2019).
    decay_params     = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]

    optimiser = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)

    total_steps = epochs * len(train_loader)
    scheduler   = LambdaLR(optimiser, make_lr_lambda(cfg.warmup_steps, total_steps))

    # Loss function
    # ignore_index=-1 so we can mask padding tokens by setting their
    # target to -1 in the DataLoader (not done here but easy to add).
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # Logging
    log_path = LOG_DIR / f"{run_name}.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "train_loss", "val_loss", "lr", "perplexity"])

    early_stop  = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)
    ckpt_path   = CKPT_DIR / f"{run_name}_best.pt"
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    # Epoch loop
    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        step_count = 0
        hidden     = None
        t0         = time.time()

        for step, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)):
            x, y = x.to(device), y.to(device)
            B    = x.size(0)

            # TBPTT: process seq_len in bptt_len-sized chunks
            chunk_losses = []
            for t in range(0, x.size(1), cfg.bptt_len):
                xc = x[:, t : t + cfg.bptt_len]
                yc = y[:, t : t + cfg.bptt_len]
                if xc.size(1) == 0:
                    continue

                if hidden is not None:
                    hidden = CodingLM.detach_hidden(hidden)

                logits, hidden = model(xc, hidden)

                loss = criterion(
                    logits.reshape(-1, model_cfg.vocab_size),
                    yc.reshape(-1),
                )
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
                optimiser.step()
                scheduler.step()
                chunk_losses.append(loss.item())

            batch_loss = float(np.mean(chunk_losses)) if chunk_losses else 0.0
            epoch_loss += batch_loss
            step_count += 1

            if (step + 1) % cfg.log_every == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  e{epoch} step {step+1:5d}  loss={batch_loss:.4f}  lr={cur_lr:.2e}")

        avg_train_loss = epoch_loss / max(step_count, 1)
        train_losses.append(avg_train_loss)

        # --- Validate ---
        val_loss = evaluate(model, val_loader, criterion, device, model_cfg.vocab_size)
        val_losses.append(val_loss)
        perplexity = math.exp(min(val_loss, 20))   # cap to avoid overflow
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train={avg_train_loss:.4f}  val={val_loss:.4f}  "
            f"ppl={perplexity:.1f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"time={elapsed:.0f}s"
        )

        # Save log row
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, step_count, avg_train_loss, val_loss,
                scheduler.get_last_lr()[0], perplexity,
            ])

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optim_state": optimiser.state_dict(),
                "val_loss"   : val_loss,
                "config"     : asdict(model_cfg),
                "train_cfg"  : asdict(cfg),
                "run_name"   : run_name,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (val_loss={val_loss:.4f}) → {ckpt_path}")

        # Early stopping check
        if early_stop.step(val_loss):
            print(f"  Early stopping triggered after {epoch} epochs (patience={cfg.patience}).")
            break

    return {
        "run_name"      : run_name,
        "best_val_loss" : best_val_loss,
        "train_losses"  : train_losses,
        "val_losses"    : val_losses,
        "epochs_run"    : len(train_losses),
        "ckpt_path"     : str(ckpt_path),
        "config"        : asdict(cfg),
    }


# Validation loop
@torch.no_grad()
def evaluate(
    model: CodingLM,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    vocab_size: int,
) -> float:
    """Compute average cross-entropy loss over the validation set."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    hidden     = None

    for x, y in tqdm(loader, desc="Validating", leave=False):
        x, y = x.to(device), y.to(device)
        logits, hidden = model(x, None)
        hidden = CodingLM.detach_hidden(hidden)
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        total_loss += loss.item()
        n_batches  += 1

    model.train()
    return total_loss / max(n_batches, 1)


# Grid search (Req 1.2.2)
def run_grid_search(base_cfg: TrainConfig) -> dict:
    """
    Try every combination in GRID, train for GRID_EPOCHS, record val loss.

    Returns the best config as a TrainConfig.
    """
    keys   = list(GRID.keys())
    values = list(GRID.values())
    combos = list(product(*values))

    print(f"\nGrid search: {len(combos)} combinations × {GRID_EPOCHS} epochs each")
    results = []

    for combo in combos:
        params = dict(zip(keys, combo))
        cfg = TrainConfig(
            lr           = params["lr"],
            batch_size   = params["batch_size"],
            dropout_rate = params["dropout_rate"],
            epochs       = GRID_EPOCHS,
            seq_len      = base_cfg.seq_len,
            bptt_len     = base_cfg.bptt_len,
            device       = base_cfg.device,
        )
        name = f"grid_lr{params['lr']}_bs{params['batch_size']}_do{params['dropout_rate']}"
        result = run_training(cfg, run_name=name, max_epochs=GRID_EPOCHS)
        results.append(result)

    # Sort by best validation loss
    results.sort(key=lambda r: r["best_val_loss"])

    # Save grid search summary
    summary_path = LOG_DIR / "grid_search_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGrid search results saved → {summary_path}")

    print("\nTop 3 configurations:")
    for r in results[:3]:
        c = r["config"]
        print(
            f"  val_loss={r['best_val_loss']:.4f}  "
            f"lr={c['lr']}  batch={c['batch_size']}  dropout={c['dropout_rate']}"
        )

    # Return best config
    best = results[0]["config"]
    return TrainConfig(
        lr           = best["lr"],
        batch_size   = best["batch_size"],
        dropout_rate = best["dropout_rate"],
        epochs       = FULL_EPOCHS,
        seq_len      = base_cfg.seq_len,
        bptt_len     = base_cfg.bptt_len,
        device       = base_cfg.device,
    )


# Entry point
def main() -> None:
    parser = argparse.ArgumentParser(description="Train CodingLM")
    parser.add_argument("--grid-search", action="store_true",
                        help="Run grid search before final training")
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--seq-len",     type=int,   default=256)
    parser.add_argument("--device",      type=str,   default="auto")
    parser.add_argument("--no-glove",    action="store_true",
                        help="Train embeddings from scratch instead of loading GloVe")
    args = parser.parse_args()

    base_cfg = TrainConfig(
        lr           = args.lr,
        batch_size   = args.batch_size,
        dropout_rate = args.dropout,
        epochs       = args.epochs,
        seq_len      = args.seq_len,
        device       = args.device,
    )

    if args.grid_search:
        print("Running grid search to find best hyperparameters …")
        best_cfg = run_grid_search(base_cfg)
        print(f"\nBest config found. Running full training for {FULL_EPOCHS} epochs …")
        result = run_training(best_cfg, run_name="best_model", use_glove=not args.no_glove)
    else:
        result = run_training(base_cfg, run_name="default_run", use_glove=not args.no_glove)

    print(f"\nTraining complete.")
    print(f"  Best val loss : {result['best_val_loss']:.4f}")
    print(f"  Perplexity    : {math.exp(min(result['best_val_loss'], 20)):.1f}")
    print(f"  Checkpoint    : {result['ckpt_path']}")
    print(f"\nNext step: run  evaluate.py --ckpt {result['ckpt_path']}")


if __name__ == "__main__":
    main()