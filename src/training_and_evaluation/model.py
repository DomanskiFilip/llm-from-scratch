"""
model.py — LSTM Language Model Architecture

WHAT THIS FILE DOES
-------------------
Defines the PyTorch neural network used for the coding LLM.
The architecture is a multi-layer LSTM with:
  - An embedding layer (optionally initialised from GloVe via embeddings.py)
  - Stacked LSTM recurrent layers
  - Dropout regularisation between every major component
  - A linear projection head that maps hidden states → vocabulary logits

This file can also be run directly as a script to print a model summary
and verify shapes are correct before you start a full training run
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import project-wide default config as the single source of truth
from src.config import Config

# Create a module-level Config instance for default values
_global_cfg = Config()


@dataclass
class LMConfig:
    """
    All hyperparameters for the language model in one place.

    Default values are taken from src.config.Config so there is a single source
    of truth for hyperparameters across the codebase. Training scripts (train.py)
    may override some of these (for example dropout values) by constructing an
    LMConfig explicitly.
    """

    # Model capacity
    vocab_size: int = _global_cfg.vocab_size
    embed_dim: int = _global_cfg.embed_dim
    hidden_dim: int = _global_cfg.hidden_dim
    n_layers: int = _global_cfg.n_layers

    # Dropouts (these can be overridden by the training config)
    embed_drop: float = _global_cfg.dropout_rate
    lstm_drop: float = _global_cfg.dropout_rate
    out_drop: float = _global_cfg.dropout_rate

    # Weight tying and positional/padding ids
    tie_weights: bool = _global_cfg.tie_weights
    max_seq_len: int = _global_cfg.max_seq_len
    pad_id: int = _global_cfg.pad_id

    def __post_init__(self):
        # Basic sanity checks
        if self.embed_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("embed_dim and hidden_dim must be positive integers")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embedding table.

    Implements a learned embedding for absolute positions. The forward method
    clamps positions to the available table size so the model won't crash if
    asked to process slightly longer sequences than `max_seq_len`.
    """

    def __init__(self, max_seq_len: int, embed_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len]  (token IDs — only the shape matters here)
        Returns: [batch, seq_len, embed_dim]
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        positions = positions.clamp(max=self.embed.num_embeddings - 1)
        return self.embed(positions)  # broadcast over batch


class CodingLM(nn.Module):
    """
    Multi-layer LSTM Language Model for code and instruction text.

    Causal LM: at each position t predicts token at t+1 using only tokens 0..t.
    """

    def __init__(
        self, config: LMConfig, pretrained_embeddings: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_embed = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=config.pad_id,
        )
        if pretrained_embeddings is not None:
            assert pretrained_embeddings.shape == (
                config.vocab_size,
                config.embed_dim,
            ), (
                f"Embedding shape mismatch: expected ({config.vocab_size}, {config.embed_dim}), "
                f"got {tuple(pretrained_embeddings.shape)}"
            )
            self.tok_embed.weight.data.copy_(pretrained_embeddings)
            print("[CodingLM] Loaded pre-trained embeddings.")

        # Positional embedding
        self.pos_embed = PositionalEmbedding(config.max_seq_len, config.embed_dim)

        # Embedding dropout
        self.embed_drop = nn.Dropout(config.embed_drop)

        # Stacked LSTM
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.lstm_drop if config.n_layers > 1 else 0.0,
        )

        # Output dropout
        self.out_drop = nn.Dropout(config.out_drop)

        # Layer norm on hidden dimension
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

        # Linear projection head (no bias to save a little memory)
        self.head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying (only valid when embed_dim == hidden_dim)
        if config.tie_weights and config.embed_dim == config.hidden_dim:
            self.head.weight = self.tok_embed.weight
        elif config.tie_weights:
            # Warn but continue
            print(
                f"[CodingLM] Warning: tie_weights=True but embed_dim ({config.embed_dim}) "
                f"!= hidden_dim ({config.hidden_dim}). Skipping weight tying."
            )

        # Parameter initialisation
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialise model parameters.

        - Embeddings: normal with std = 1/sqrt(embed_dim)
        - LSTM weights: Xavier / orthogonal
        - LSTM biases: zero except forget gate bias set to 1
        - Head: normal(0, 0.02) if not weight-tied
        """
        nn.init.normal_(
            self.tok_embed.weight, mean=0.0, std=1.0 / math.sqrt(self.config.embed_dim)
        )
        if self.config.pad_id is not None:
            self.tok_embed.weight.data[self.config.pad_id].zero_()

        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.zero_()
                # Forget gate bias lives in the second quarter of the bias vector
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        # Initialise head if not tied
        if not (
            self.config.tie_weights and self.config.embed_dim == self.config.hidden_dim
        ):
            nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialise LSTM hidden and cell states to zeros.
        Returns (h_0, c_0) each of shape [n_layers, batch, hidden_dim]
        """
        h = torch.zeros(
            self.config.n_layers, batch_size, self.config.hidden_dim, device=device
        )
        c = torch.zeros(
            self.config.n_layers, batch_size, self.config.hidden_dim, device=device
        )
        return h, c

    @staticmethod
    def detach_hidden(
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detach hidden states from the computation graph (for TBPTT).
        """
        h, c = hidden
        return h.detach(), c.detach()

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len]
            hidden: optional (h, c) from previous chunk; if None initialised to zeros

        Returns:
            logits: [batch, seq_len, vocab_size]
            hidden: updated (h, c)
        """
        B, T = input_ids.shape
        device = input_ids.device

        if hidden is None:
            hidden = self.init_hidden(B, device)

        tok = self.tok_embed(input_ids)  # [B, T, E]
        pos = self.pos_embed(input_ids)  # [1, T, E] -> broadcast
        x = self.embed_drop(tok + pos)  # [B, T, E]

        x, hidden = self.lstm(x, hidden)  # [B, T, H]

        x = self.out_drop(x)
        x = self.layer_norm(x)

        logits = self.head(x)  # [B, T, V]
        return logits, hidden

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        """
        Autoregressive generation (top-k sampling + temperature)

        Args:
            prompt_ids: [1, prompt_len]
        Returns:
            [1, prompt_len + max_new_tokens]
        """
        self.eval()
        ids = prompt_ids.clone()
        hidden = None

        if ids.size(1) > 1:
            _, hidden = self.forward(ids[:, :-1], hidden)

        current = ids[:, -1:]  # [1,1]
        for _ in range(max_new_tokens):
            logits, hidden = self.forward(current, hidden)
            logits = logits[:, -1, :] / max(1e-8, temperature)

            if top_k > 0:
                top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_val = top_vals[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < min_val, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [1,1]
            ids = torch.cat([ids, next_id], dim=1)
            current = next_id

        return ids

    def count_parameters(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        cfg = self.config
        lines = [
            "CodingLM",
            f"  vocab_size  : {cfg.vocab_size:,}",
            f"  embed_dim   : {cfg.embed_dim}",
            f"  hidden_dim  : {cfg.hidden_dim}",
            f"  n_layers    : {cfg.n_layers}",
            f"  dropout     : embed={cfg.embed_drop}  lstm={cfg.lstm_drop}  out={cfg.out_drop}",
            f"  tie_weights : {cfg.tie_weights}",
            f"  max_seq_len : {cfg.max_seq_len}",
            f"  parameters  : {self.count_parameters():,}",
        ]
        return "\n".join(lines)


# Quick smoke-test when run as a script
if __name__ == "__main__":
    import time

    # Use defaults from the dataclass (which in turn used src.config.Config)
    cfg = LMConfig()

    model = CodingLM(cfg)
    print(model)
    print()

    # Dummy forward pass
    B, T = 4, min(128, cfg.max_seq_len)
    x = torch.randint(0, cfg.vocab_size, (B, T))
    t0 = time.time()
    logits, hidden = model(x)
    t1 = time.time()

    print(f"Input  shape : {x.shape}")
    print(f"Logits shape : {logits.shape}  (expected [{B}, {T}, {cfg.vocab_size}])")
    print(f"h shape      : {hidden[0].shape}")
    print(f"c shape      : {hidden[1].shape}")
    print(f"Forward pass : {(t1 - t0) * 1000:.1f} ms")

    # Dummy loss
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    print(
        f"Dummy CE loss: {loss.item():.4f}  (expected ≈ {math.log(cfg.vocab_size):.2f})"
    )

    # Generation test
    prompt = torch.randint(0, cfg.vocab_size, (1, min(10, cfg.max_seq_len)))
    gen = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=40)
    print(f"Generation   : {prompt.shape} → {gen.shape}")

    print("\nAll checks passed!")
