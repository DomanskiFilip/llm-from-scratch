"""
Manages the model training process, including hyperparameter grid searches and integration with GloVe embeddings.

Flags:
--grid-search: Runs a short trial of multiple hyperparameter combinations before full training.

--no-glove: Disables pre-trained GloVe initialization; trains embeddings from scratch.

--lr [float]: Overrides the default learning rate.

--batch-size [int]: Overrides the default training batch size.

--epochs [int]: Sets the number of training iterations.

--device [str]: Forces usage of cpu, cuda, or mps.
"""

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from model import CodingLM, LMConfig
from src.config import Config

# Paths 
DATA_DIR = Path("artefacts/data")
CKPT_DIR = Path("artefacts/checkpoints")
LOG_DIR  = Path("artefacts/logs")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

GLOVE_PT = Path("artefacts/embeddings") / "glove_aligned.pt"

# Training config 
_config_defaults = Config()


@dataclass
class TrainConfig:
    seq_len:      int   = _config_defaults.seq_len
    bptt_len:     int   = _config_defaults.bptt_len
    lr:           float = _config_defaults.lr
    weight_decay: float = _config_defaults.weight_decay
    clip_norm:    float = _config_defaults.clip_norm
    warmup_steps: int   = _config_defaults.warmup_steps
    epochs:       int   = _config_defaults.epochs
    batch_size:   int   = _config_defaults.batch_size
    device:       str   = _config_defaults.device
    val_fraction: float = _config_defaults.val_fraction
    log_every:    int   = _config_defaults.log_every
    dropout_rate: float = _config_defaults.dropout_rate
    patience:     int   = _config_defaults.patience
    min_delta:    float = _config_defaults.min_delta

# Dynamic Grid centered around Config values
GRID = {
    "lr":           [_config_defaults.lr * 2, _config_defaults.lr, _config_defaults.lr / 5],
    "batch_size":   [max(32, _config_defaults.batch_size // 2), _config_defaults.batch_size],
    "dropout_rate": [_config_defaults.dropout_rate, 0.1, 0.2],
}
GRID_EPOCHS = _config_defaults.grid_epochs
FULL_EPOCHS = _config_defaults.full_epochs


# Device 
def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


# Dataset
def load_shard(path: Path) -> np.ndarray:
    """Read a .bin shard into a uint16→int32 numpy array."""
    data = path.read_bytes()
    n    = len(data) // 2
    return np.frombuffer(data, dtype=np.uint16, count=n).astype(np.int32)


def load_mask_shard(path: Path) -> np.ndarray:
    """
    Read a .mask.bin shard into a uint8 numpy array.
    Returns an array of all-ones (train on everything) if the file is missing
    so the code still works with data that was tokenised without masking.
    """
    if not path.exists():
        # Fall back: treat all tokens as trainable
        token_path = Path(str(path).replace(".mask.bin", ".bin"))
        n = len(token_path.read_bytes()) // 2 if token_path.exists() else 0
        return np.ones(n, dtype=np.uint8)
    return np.frombuffer(path.read_bytes(), dtype=np.uint8).copy()


class ShardDataset(torch.utils.data.Dataset):
    """
    Dataset spanning multiple binary shard files.

    Each item is a triple (input_ids, target_ids, loss_mask):
      input_ids  [seq_len]  — token IDs fed to the model
      target_ids [seq_len]  — next-token labels (shifted by 1)
      loss_mask  [seq_len]  — 1 = compute loss, 0 = ignore (prompt tokens)

    target_ids positions where loss_mask=0 are set to -1 so that
    CrossEntropyLoss(ignore_index=-1) skips them.
    """

    def __init__(self, shard_paths: List[Path], seq_len: int) -> None:
        self.seq_len = seq_len

        tok_arrays  = [load_shard(p) for p in shard_paths]
        mask_arrays = [load_mask_shard(Path(str(p).replace(".bin", ".mask.bin")))
                       for p in shard_paths]

        self.tokens = np.concatenate(tok_arrays)
        self.masks  = np.concatenate(mask_arrays)

        # Trim to same length in case of any off-by-one
        min_len = min(len(self.tokens), len(self.masks))
        self.tokens = self.tokens[:min_len]
        self.masks  = self.masks[:min_len]

        self.n = (len(self.tokens) - 1) // seq_len

        # Report how much of the data is trainable
        pct = 100.0 * self.masks.sum() / max(len(self.masks), 1)
        print(f"  ShardDataset: {self.n:,} windows, "
              f"{pct:.1f}% tokens are response (mask=1)")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        chunk_tok  = self.tokens[start : start + self.seq_len + 1]
        chunk_mask = self.masks[ start : start + self.seq_len + 1]

        x    = torch.from_numpy(chunk_tok[:-1].copy()).long()
        y    = torch.from_numpy(chunk_tok[1:].copy()).long()
        mask = torch.from_numpy(chunk_mask[1:].copy()).long()  # mask aligns with targets

        # Wherever mask=0 set target to -1 so CrossEntropyLoss ignores it
        y = y.masked_fill(mask == 0, -1)
        return x, y


def build_dataloaders(cfg: TrainConfig, device_type: str):
    """Find all token shards, split into train/val, return DataLoaders."""
    shards = sorted(DATA_DIR.glob("*_shard_*.bin"))
    # Exclude mask shards from the list
    shards = [p for p in shards if ".mask." not in p.name]

    if not shards:
        raise FileNotFoundError(
            f"No shard files found in {DATA_DIR}. Run tokeniser.py first."
        )

    pin = device_type == "cuda"

    if len(shards) > 1:
        n_val      = max(1, int(len(shards) * cfg.val_fraction))
        val_shards = shards[:n_val]
        trn_shards = shards[n_val:]
        print(f"  Shards: {len(trn_shards)} train, {len(val_shards)} val")
        train_ds = ShardDataset(trn_shards, cfg.seq_len)
        val_ds   = ShardDataset(val_shards, cfg.seq_len)
    else:
        print("  Single shard detected — splitting within shard.")
        full_ds  = ShardDataset(shards, cfg.seq_len)
        val_size = max(1, int(len(full_ds) * cfg.val_fraction))
        trn_size = len(full_ds) - val_size
        train_ds, val_ds = torch.utils.data.random_split(full_ds, [trn_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=2, pin_memory=pin, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=2, pin_memory=pin, drop_last=True,
    )
    return train_loader, val_loader


# LR schedule 
def make_lr_lambda(warmup_steps: int, total_steps: int, lr_min_ratio: float = 0.1):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cosine
    return lr_lambda


# Early stopping 
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_loss   = float("inf")
        self.counter     = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# Checkpoint validation 
def validate_checkpoint_architecture(model: CodingLM, ckpt: dict, ckpt_path: Path) -> None:
    model_state = model.state_dict()
    ckpt_state  = ckpt["model_state"]
    checks = {
        "tok_embed.weight":   "Vocabulary / Embedding Size",
        "lstm.weight_ih_l0":  "Input / Hidden Size",
        "head.weight":        "Output Vocabulary Size",
    }
    mismatches = []
    for key, name in checks.items():
        if key in model_state and key in ckpt_state:
            ms = tuple(model_state[key].shape)
            cs = tuple(ckpt_state[key].shape)
            if ms != cs:
                mismatches.append(f"  - {name} ({key}): current={ms}  checkpoint={cs}")

    if mismatches:
        print("\n" + "!" * 60)
        print("ERROR: ARCHITECTURE MISMATCH")
        print("!" * 60)
        print(f"Checkpoint '{ckpt_path}' is incompatible with current config.")
        print("Differences:")
        print("\n".join(mismatches))
        print("\nFIX: delete the checkpoint or update your Config to match.")
        print("!" * 60 + "\n")
        sys.exit(1)


# Validation loop 
@torch.no_grad()
def evaluate(
    model:      CodingLM,
    loader:     torch.utils.data.DataLoader,
    criterion:  nn.Module,
    device:     torch.device,
    vocab_size: int,
) -> float:
    """
    Compute average cross-entropy loss over the validation set.
    Uses ignore_index=-1 so masked (prompt) tokens don't affect the metric.
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for x, y in tqdm(loader, desc="Validating", leave=False):
        x, y   = x.to(device), y.to(device)
        logits, _ = model(x, None)
        loss    = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        total_loss += loss.item()
        n_batches  += 1

    model.train()
    return total_loss / max(n_batches, 1)


# Single training run 
def run_training(
    cfg:        TrainConfig,
    run_name:   str,
    max_epochs: Optional[int] = None,
    use_glove:  bool = True,
) -> dict:
    device = get_device(cfg.device)
    print(f"\n{'=' * 60}")
    print(f"Run: {run_name}")
    print(f"  device={device}  lr={cfg.lr}  batch={cfg.batch_size}  dropout={cfg.dropout_rate}")
    print(f"{'=' * 60}")

    epochs = max_epochs if max_epochs is not None else cfg.epochs

    # Build model
    model_cfg = LMConfig(
        vocab_size  = _config_defaults.vocab_size,
        embed_dim   = _config_defaults.embed_dim,
        hidden_dim  = _config_defaults.hidden_dim,
        n_layers    = _config_defaults.n_layers,
        embed_drop  = cfg.dropout_rate,
        lstm_drop   = cfg.dropout_rate,
        out_drop    = cfg.dropout_rate,
        tie_weights = _config_defaults.tie_weights,
        max_seq_len = min(_config_defaults.max_seq_len, cfg.seq_len + 64),
        pad_id      = _config_defaults.pad_id,
    )

    pretrained = None
    if use_glove and GLOVE_PT.exists():
        pretrained = torch.load(GLOVE_PT, map_location="cpu")

    model = CodingLM(model_cfg, pretrained_embeddings=pretrained).to(device)
    print(f"  Model: {model.count_parameters():,} parameters")

    # Data
    train_loader, val_loader = build_dataloaders(cfg, device.type)

    # Optimiser
    decay_params    = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]

    optimiser = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    total_steps = epochs * len(train_loader)
    scheduler   = LambdaLR(optimiser, make_lr_lambda(cfg.warmup_steps, total_steps))

    # ignore_index=-1 means masked (prompt) token positions are skipped
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # Logging / checkpointing paths
    log_path  = LOG_DIR  / f"{run_name}.csv"
    ckpt_path = CKPT_DIR / f"{run_name}_best.pt"

    # Resume from checkpoint
    start_epoch   = 1
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    if ckpt_path.exists():
        print(f"  Resuming from {ckpt_path} …")
        ckpt = torch.load(ckpt_path, map_location=device)

        # Validate architecture before loading weights
        validate_checkpoint_architecture(model, ckpt, ckpt_path)

        model.load_state_dict(ckpt["model_state"])
        optimiser.load_state_dict(ckpt["optim_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_loss"]
        print(f"  Resumed at epoch {start_epoch-1}, best_val_loss={best_val_loss:.4f}")
    else:
        # Fresh run — write CSV header
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "step", "train_loss", "val_loss", "lr", "perplexity"]
            )

    early_stop = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)
    early_stop.best_loss = best_val_loss

    # Epoch loop 
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        hidden     = None
        t0         = time.time()

        for step, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        ):
            x, y = x.to(device), y.to(device)
            # y already has -1 at masked positions (set in ShardDataset.__getitem__)

            chunk_losses = []
            for t in range(0, x.size(1), cfg.bptt_len):
                xc = x[:, t : t + cfg.bptt_len]
                yc = y[:, t : t + cfg.bptt_len]
                if xc.size(1) == 0:
                    continue

                if hidden is not None:
                    hidden = CodingLM.detach_hidden(hidden)

                logits, hidden = model(xc, hidden)
                loss = criterion(logits.reshape(-1, model_cfg.vocab_size), yc.reshape(-1))

                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
                optimiser.step()
                scheduler.step()
                chunk_losses.append(loss.item())

            batch_loss  = float(np.mean(chunk_losses)) if chunk_losses else 0.0
            epoch_loss += batch_loss
            step_count += 1

            if (step + 1) % cfg.log_every == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  e{epoch} step {step + 1:5d}  loss={batch_loss:.4f}  lr={cur_lr:.2e}")

        avg_train_loss = epoch_loss / max(step_count, 1)
        train_losses.append(avg_train_loss)

        val_loss   = evaluate(model, val_loader, criterion, device, model_cfg.vocab_size)
        val_losses.append(val_loss)
        perplexity = math.exp(min(val_loss, 20))
        elapsed    = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train={avg_train_loss:.4f}  val={val_loss:.4f}  "
            f"ppl={perplexity:.1f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"time={elapsed:.0f}s"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, step_count, avg_train_loss, val_loss,
                scheduler.get_last_lr()[0], perplexity,
            ])

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimiser.state_dict(),
                    "val_loss":    val_loss,
                    "config":      asdict(model_cfg),
                    "train_cfg":   asdict(cfg),
                    "run_name":    run_name,
                },
                ckpt_path,
            )
            print(f"  ✓ Saved best checkpoint (val_loss={val_loss:.4f}) → {ckpt_path}")

        if early_stop.step(val_loss):
            print(f"  Early stopping after {epoch} epochs (patience={cfg.patience})")
            break

    return {
        "run_name":       run_name,
        "best_val_loss":  best_val_loss,
        "train_losses":   train_losses,
        "val_losses":     val_losses,
        "epochs_run":     len(train_losses),
        "ckpt_path":      str(ckpt_path),
        "config":         asdict(cfg),
    }


# Grid search 
def run_grid_search(base_cfg: TrainConfig) -> TrainConfig:
    keys   = list(GRID.keys())
    values = list(GRID.values())
    combos = list(product(*values))

    print(f"\nGrid search: {len(combos)} combinations × {GRID_EPOCHS} epochs each")
    results = []

    for combo in combos:
        params = dict(zip(keys, combo))
        cfg    = TrainConfig(
            lr=params["lr"],
            batch_size=params["batch_size"],
            dropout_rate=params["dropout_rate"],
            epochs=GRID_EPOCHS,
            seq_len=base_cfg.seq_len,
            bptt_len=base_cfg.bptt_len,
            device=base_cfg.device,
        )
        name   = f"grid_lr{params['lr']}_bs{params['batch_size']}_do{params['dropout_rate']}"
        result = run_training(cfg, run_name=name, max_epochs=GRID_EPOCHS)
        results.append(result)

    results.sort(key=lambda r: r["best_val_loss"])

    summary_path = LOG_DIR / "grid_search_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGrid search results saved → {summary_path}")

    print("\nTop 3 configurations:")
    for r in results[:3]:
        c = r["config"]
        print(f"  val_loss={r['best_val_loss']:.4f}  lr={c['lr']}  "
              f"batch={c['batch_size']}  dropout={c['dropout_rate']}")

    best = results[0]["config"]
    return TrainConfig(
        lr=best["lr"],
        batch_size=best["batch_size"],
        dropout_rate=best["dropout_rate"],
        epochs=FULL_EPOCHS,
        seq_len=base_cfg.seq_len,
        bptt_len=base_cfg.bptt_len,
        device=base_cfg.device,
    )


# Entry point 
def main(config: Config) -> None:
    parser = argparse.ArgumentParser(description="Train CodingLM")
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--dropout",    type=float, default=None)
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--seq-len",    type=int,   default=None)
    parser.add_argument("--device",     type=str,   default=None)
    parser.add_argument("--no-glove",   action="store_true",
                        help="Train embeddings from scratch instead of loading GloVe")
    args = parser.parse_args()

    base_cfg = TrainConfig(
        lr           = args.lr         if args.lr         is not None else config.lr,
        batch_size   = args.batch_size if args.batch_size is not None else config.batch_size,
        dropout_rate = args.dropout    if args.dropout    is not None else config.dropout_rate,
        epochs       = args.epochs     if args.epochs     is not None else config.epochs,
        seq_len      = args.seq_len    if args.seq_len    is not None else config.seq_len,
        device       = args.device     if args.device     is not None else config.device,
    )

    if args.grid_search:
        print("Running grid search …")
        best_cfg = run_grid_search(base_cfg)
        print(f"Best config found.  Running full training for {FULL_EPOCHS} epochs …")
        result = run_training(best_cfg, run_name="best_model", use_glove=not args.no_glove)
    else:
        result = run_training(base_cfg, run_name="default_run", use_glove=not args.no_glove)

    print("Training complete")
    print(f"  Best val loss : {result['best_val_loss']:.4f}")
    print(f"  Perplexity    : {math.exp(min(result['best_val_loss'], 20)):.1f}")
    print(f"  Checkpoint    : {result['ckpt_path']}")


if __name__ == "__main__":
    config = Config()
    main(config)