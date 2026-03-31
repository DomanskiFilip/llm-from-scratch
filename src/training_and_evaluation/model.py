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

WHY LSTM AND NOT A TRANSFORMER?
---------------------------------
Transformers (Vaswani et al. 2017, "Attention Is All You Need",
arXiv:1706.03762) dominate modern LLMs because self-attention scales better
and parallelises across the whole sequence at once.  However:

  1. Transformers require O(n²) memory in the sequence length.  For code,
     sequences can be very long (whole files); an LSTM handles arbitrary
     length with O(1) memory per step.
  2. LSTMs are conceptually simpler to explain (Req 1.2.1 asks for
     "detailed explanations") and their limitations lead naturally into
     the Req 1.3.2 discussion of "areas for improvement → transformers/BERT".
  3. The Karpathy-style approach (nanoGPT / minGPT) the project is modelled
     on actually *starts* with an RNN/LSTM before introducing transformers.

ARCHITECTURE OVERVIEW
---------------------
  Input token IDs  [batch, seq_len]
        │
        ▼
  ┌─────────────────────────────────────────┐
  │  Embedding layer  [batch, seq_len, d_e] │  token + positional
  └─────────────────────────────────────────┘
        │  dropout (embed_drop)
        ▼
  ┌─────────────────────────────────────────┐
  │  LSTM layer 1     [batch, seq_len, d_h] │
  └─────────────────────────────────────────┘
        │  dropout (lstm_drop — between layers only)
        ▼
  ┌─────────────────────────────────────────┐
  │  LSTM layer 2     [batch, seq_len, d_h] │
  └─────────────────────────────────────────┘
        │  … (repeat for n_layers)
        │  dropout (out_drop)
        ▼
  ┌─────────────────────────────────────────┐
  │  Layer Norm       [batch, seq_len, d_h] │
  └─────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────┐
  │  Linear head      [batch, seq_len, V]   │  V = vocab size
  └─────────────────────────────────────────┘
        │
        ▼
  Logits (raw scores before softmax)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Model configuration dataclass
@dataclass
class LMConfig:
    """
    All hyperparameters for the language model in one place.

    Keeping config separate from the model class makes hyperparameter
    sweeps (train.py) clean: you create a new LMConfig, pass it to
    CodingLM, and never touch the model code.

    Field descriptions:
      vocab_size    Number of tokens in the BPE vocabulary (from tokeniser.py).
      embed_dim     Dimensionality of the token embedding vectors.
                    Should match GloVe dim (100) if using pre-trained embeddings,
                    or be a free hyperparameter if training from scratch.
      hidden_dim    Width of the LSTM hidden state.  Wider = more capacity
                    but more memory and slower training.
      n_layers      Number of stacked LSTM layers.  More layers = deeper
                    feature hierarchy, but harder to train (vanishing gradients).
      embed_drop    Dropout probability applied to the embedding output.
      lstm_drop     Dropout probability applied between LSTM layers.
                    PyTorch's nn.LSTM applies this automatically when
                    n_layers > 1 and dropout > 0.
      out_drop      Dropout probability applied after the final LSTM layer
                    and before the linear projection head.
      tie_weights   If True, the linear projection head's weight matrix is
                    shared (transposed) with the embedding weight matrix.
                    This is the "weight tying" trick from Press & Wolf (2017)
                    arXiv:1608.05859.  It reduces parameters and often improves
                    perplexity because the model uses the same geometry to
                    both represent input tokens and score output tokens.
      max_seq_len   Maximum sequence length for positional embeddings.
      pad_id        Token ID used for padding (should match <|pad|> from vocab).
    """
    vocab_size  : int   = 32_768
    embed_dim   : int   = 100       # match GloVe-100d
    hidden_dim  : int   = 512
    n_layers    : int   = 2
    embed_drop  : float = 0.1
    lstm_drop   : float = 0.2
    out_drop    : float = 0.2
    tie_weights : bool  = True
    max_seq_len : int   = 512
    pad_id      : int   = 0         # overwritten by trainer once vocab is known

# Positional Embedding
class PositionalEmbedding(nn.Module):
    """
    Learnable positional embedding table.

    WHAT IT DOES
    ------------
    An LSTM processes tokens sequentially — it already has an implicit sense
    of "position" from the order in which it reads tokens.  So why add
    positional embeddings at all?

    Two reasons:
      1. Consistency with transformer-style models.  If you later swap the
         LSTM for a transformer (Req 1.3.2 improvement), the positional
         embedding layer is already there; you just change the recurrent
         part.
      2. The embedding layer concatenates token + position information
         *before* the first LSTM layer.  This gives the very first hidden
         state access to absolute position, which can help with tasks like
         "complete this function definition" where the model benefits from
         knowing it is near the start of a file vs. deep inside one.

    LEARNABLE vs SINUSOIDAL
    -----------------------
    The original Transformer (Vaswani et al. 2017) used fixed sinusoidal
    encodings.  BERT (Devlin et al. 2019, arXiv:1810.04805) switched to
    *learned* position embeddings.  We follow BERT: an nn.Embedding of shape
    [max_seq_len, embed_dim] whose values are updated by backprop.

    At inference time the model can handle sequences longer than max_seq_len
    by clamping the position index — position embeddings beyond the training
    range extrapolate poorly but won't crash.
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
        # Clamp to max_seq_len to allow inference on longer sequences
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        positions = positions.clamp(max=self.embed.num_embeddings - 1)
        return self.embed(positions)  # broadcast over batch

# Main model
class CodingLM(nn.Module):
    """
    Multi-layer LSTM Language Model for code and instruction text.

    This is a *causal* language model: at each position t it predicts the
    token at position t+1 using only tokens 0…t.  The LSTM enforces
    causality naturally because it processes tokens left-to-right.

    Layer-by-layer explanation
    --------------------------

    1. TOKEN EMBEDDING  (nn.Embedding)
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       Converts integer token IDs into dense floating-point vectors.
       Shape: [vocab_size, embed_dim]

       Why dense vectors instead of one-hot?
         A one-hot vector of size 32,768 is sparse and treats all tokens
         as equally dissimilar.  A dense embedding of size 100 learned by
         backprop (or pre-loaded from GloVe) encodes *similarity*: tokens
         that appear in similar contexts end up close in the embedding space.

       Weight tying (Press & Wolf 2017):
         When tie_weights=True, the output linear layer's weight matrix is
         set to the transpose of this embedding matrix.  The intuition is
         that the same geometric relationship used to *represent* a token
         as input should be used to *score* that token as output.  It also
         halves the parameter count of the largest matrix in the network.

    2. POSITIONAL EMBEDDING  (PositionalEmbedding)
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       See PositionalEmbedding docstring above.

    3. EMBEDDING DROPOUT  (nn.Dropout)
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       Applied to the sum of token + positional embeddings.

       Dropout (Srivastava et al. 2014):
         During training, randomly zeroes each element with probability p.
         Forces the network to not rely on any single embedding dimension,
         which acts as a regulariser and reduces overfitting.  During
         evaluation, dropout is disabled (nn.Module.eval() handles this).

       Why dropout *here* specifically?
         The embedding layer has by far the most parameters in the model
         (vocab_size × embed_dim = 32,768 × 100 = 3.3M).  Without
         regularisation, the model can memorise training examples by
         assigning very specific embeddings to rare tokens.

    4. LSTM  (nn.LSTM)
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       Long Short-Term Memory (Hochreiter & Schmidhuber 1997).

       An LSTM maintains two hidden vectors per layer:
         hₜ  — the "hidden state"  (short-term memory, also the output)
         cₜ  — the "cell state"    (long-term memory)

       At each time step t, three *gates* control information flow:

         Forget gate:   fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)
           What fraction of the previous cell state to keep.
           σ = sigmoid, output ∈ (0,1).

         Input gate:    iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)
           Which new information to write to the cell state.

         Cell candidate: g̃ₜ = tanh(Wg·[hₜ₋₁, xₜ] + bg)
           The "proposed" new memory content.

         Cell update:   cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ g̃ₜ
           Blend old memory (weighted by forget) with new content (weighted
           by input).  The ⊙ is element-wise multiplication.

         Output gate:   oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)
           How much of the cell state to expose as the hidden state output.

         Hidden update: hₜ = oₜ ⊙ tanh(cₜ)

       Why LSTM beats a plain RNN for code:
         A plain RNN suffers from vanishing gradients over long sequences —
         after ~20 tokens the gradient signal from early tokens becomes
         negligible and the model forgets them.  The LSTM cell state cₜ
         has an *additive* update (cₜ = fₜ⊙cₜ₋₁ + iₜ⊙g̃ₜ) so gradients
         can flow back through time without vanishing.  This is essential
         for code where a function name defined at line 1 must be remembered
         at line 50.

       Stacking layers:
         We use n_layers=2 by default.  Layer 1 reads the embeddings and
         learns local features (token n-grams, parenthesis pairs).  Layer 2
         reads layer 1's hidden states and learns higher-level features
         (function structure, control flow patterns).  Each additional layer
         adds capacity but also increases the risk of overfitting.

       PyTorch parameter:
         batch_first=True means inputs are [batch, seq_len, features]
         rather than [seq_len, batch, features].  This is more intuitive
         and consistent with the rest of the code.

    5. OUTPUT DROPOUT  (nn.Dropout)
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       Applied to the final LSTM layer's output before the projection head.
       Same motivation as embedding dropout.

    6. LAYER NORMALISATION  (nn.LayerNorm)
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       Ba et al. 2016 "Layer Normalization" arXiv:1607.06450.

       Normalises the last dimension of the input to zero mean and unit
       variance, then applies a learned scale (γ) and shift (β):

           LN(x) = γ · (x − μ) / (σ + ε) + β

       Why here?
         Without normalisation, the hidden state magnitudes can drift
         during training — large hidden states cause large logits which
         cause numerically unstable softmax probabilities.  LayerNorm
         stabilises training and often allows higher learning rates.

       LayerNorm vs BatchNorm:
         BatchNorm normalises across the batch dimension.  For variable-
         length sequences with padding, batch statistics are noisy.
         LayerNorm normalises across the feature dimension for *each*
         (batch, position) independently — no interference from padding.

    7. LINEAR PROJECTION HEAD  (nn.Linear)
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       Projects the hidden_dim-dimensional hidden state down to vocab_size
       logits.  One logit per vocabulary token.

       Shape: [batch, seq_len, vocab_size]

       The logits are *not* passed through softmax here.  PyTorch's
       nn.CrossEntropyLoss (used in the training loop) expects raw logits
       and applies log_softmax internally for numerical stability.
    """

    def __init__(self, config: LMConfig, pretrained_embeddings: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.config = config

        
        # 1. Token embedding
        self.tok_embed = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=config.pad_id,
        )
        if pretrained_embeddings is not None:
            # Load GloVe-aligned weights from embeddings.py
            assert pretrained_embeddings.shape == (config.vocab_size, config.embed_dim), (
                f"Embedding shape mismatch: expected ({config.vocab_size}, {config.embed_dim}), "
                f"got {tuple(pretrained_embeddings.shape)}"
            )
            self.tok_embed.weight.data.copy_(pretrained_embeddings)
            print("[CodingLM] Loaded pre-trained GloVe embeddings.")

        # 2. Positional embedding
        self.pos_embed = PositionalEmbedding(config.max_seq_len, config.embed_dim)

        
        # 3. Embedding dropout
        self.embed_drop = nn.Dropout(config.embed_drop)

        
        # 4. Stacked LSTM
        # When n_layers > 1 and dropout > 0, PyTorch automatically applies
        # dropout between each pair of LSTM layers (but NOT after the last
        # layer — that is handled by out_drop below).
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.lstm_drop if config.n_layers > 1 else 0.0,
        )

        
        # 5. Output dropout
        
        self.out_drop = nn.Dropout(config.out_drop)

        
        # 6. Layer normalisation
        
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

        
        # 7. Linear projection head
        
        # bias=False is conventional for output projection heads in LLMs —
        # the bias adds little capacity but wastes memory.
        self.head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        
        # Weight tying (Press & Wolf 2017)
        
        # This only works when embed_dim == hidden_dim.
        # If they differ, we skip tying and treat both as independent matrices.
        if config.tie_weights and config.embed_dim == config.hidden_dim:
            self.head.weight = self.tok_embed.weight
        elif config.tie_weights:
            print(
                f"[CodingLM] Warning: tie_weights=True but embed_dim ({config.embed_dim}) "
                f"≠ hidden_dim ({config.hidden_dim}). Weight tying skipped."
            )

        
        # Parameter initialisation
        
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialise model parameters.

        Embedding: N(0, 1/sqrt(embed_dim)) — same scale as random GloVe fallback.
        LSTM weights: Xavier uniform — balances forward/backward signal variance.
        LSTM biases: zero, except the forget gate bias is set to 1.0.
          Hochreiter & Schmidhuber (1997) and Jozefowicz et al. (2015)
          recommend initialising the forget gate bias to 1 so the LSTM starts
          by *remembering* most of its history.  This prevents the network
          from discarding everything at the beginning of training.
        Linear head: N(0, 0.02) — small init to keep logits near zero at start.
        """
        # Embedding
        nn.init.normal_(self.tok_embed.weight, mean=0.0, std=1.0 / math.sqrt(self.config.embed_dim))
        if self.config.pad_id is not None:
            self.tok_embed.weight.data[self.config.pad_id].zero_()

        # LSTM
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)   # orthogonal init for recurrent weights
            elif "bias" in name:
                param.data.zero_()
                # Forget gate bias is in the second quarter of the bias vector
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        # Head (only if not weight-tied — if tied, embedding init already ran)
        if not (self.config.tie_weights and self.config.embed_dim == self.config.hidden_dim):
            nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    
    # Hidden state management
    

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialise LSTM hidden and cell states to zeros.

        Returns (h_0, c_0) each of shape [n_layers, batch, hidden_dim].
        Called at the beginning of each training sequence.
        """
        h = torch.zeros(self.config.n_layers, batch_size, self.config.hidden_dim, device=device)
        c = torch.zeros(self.config.n_layers, batch_size, self.config.hidden_dim, device=device)
        return h, c

    @staticmethod
    def detach_hidden(
        hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detach hidden states from the computation graph.

        In "truncated BPTT" (truncated back-propagation through time), we
        chop long sequences into fixed-length chunks and pass the hidden
        state from the end of one chunk as the initial state of the next.
        Without detach(), the gradient would try to flow all the way back
        through every previous chunk, quickly running out of memory.
        Detaching breaks the graph at the chunk boundary while still letting
        the hidden state carry information forward.
        """
        h, c = hidden
        return h.detach(), c.detach()

    
    # Forward pass
    

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len]  integer token IDs
            hidden:    Optional (h, c) from a previous chunk (for TBPTT).
                       If None, initialised to zeros.

        Returns:
            logits:    [batch, seq_len, vocab_size]  raw (pre-softmax) scores
            hidden:    updated (h, c) for use in the next chunk
        """
        B, T = input_ids.shape
        device = input_ids.device

        if hidden is None:
            hidden = self.init_hidden(B, device)

        # 1. Token + positional embeddings  →  [B, T, embed_dim]
        tok = self.tok_embed(input_ids)
        pos = self.pos_embed(input_ids)
        x   = self.embed_drop(tok + pos)

        # 2. LSTM  →  [B, T, hidden_dim]
        x, hidden = self.lstm(x, hidden)

        # 3. Output dropout + LayerNorm  →  [B, T, hidden_dim]
        x = self.out_drop(x)
        x = self.layer_norm(x)

        # 4. Linear head  →  [B, T, vocab_size]
        logits = self.head(x)

        return logits, hidden

    
    # Inference helper
    

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        """
        Autoregressive token generation (greedy / top-k sampling).

        Args:
            prompt_ids:     [1, prompt_len]  integer tensor on the right device
            max_new_tokens: how many tokens to generate
            temperature:    > 1 → flatter distribution (more random)
                            < 1 → sharper distribution (more deterministic)
                            = 1 → unmodified logits
            top_k:          Only sample from the top-k most likely tokens.
                            Setting top_k=1 is equivalent to greedy decoding.
                            Nucleus (top-p) sampling is left as an exercise.

        Returns:
            [1, prompt_len + max_new_tokens]  — prompt + generated IDs
        """
        self.eval()
        ids    = prompt_ids.clone()
        hidden = None

        # Process the prompt through the model to warm up the hidden state
        if ids.size(1) > 1:
            _, hidden = self.forward(ids[:, :-1], hidden)

        current = ids[:, -1:]   # [1, 1]

        for _ in range(max_new_tokens):
            logits, hidden = self.forward(current, hidden)
            logits = logits[:, -1, :] / temperature   # [1, vocab_size]

            # Top-k filtering: zero out everything outside the top k
            if top_k > 0:
                top_vals, _ = torch.topk(logits, top_k)
                min_val = top_vals[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < min_val, float("-inf"))

            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)   # [1, 1]
            ids     = torch.cat([ids, next_id], dim=1)
            current = next_id

        return ids

    
    # Utility
    

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # noqa: D105
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

    cfg = LMConfig(
        vocab_size=32_768,
        embed_dim=100,
        hidden_dim=512,
        n_layers=2,
        embed_drop=0.1,
        lstm_drop=0.2,
        out_drop=0.2,
        tie_weights=False,   # embed_dim ≠ hidden_dim, so can't tie here
        max_seq_len=512,
        pad_id=6,
    )

    model = CodingLM(cfg)
    print(model)
    print()

    # Dummy forward pass
    B, T = 4, 128
    x = torch.randint(0, cfg.vocab_size, (B, T))
    t0 = time.time()
    logits, hidden = model(x)
    t1 = time.time()

    print(f"Input  shape : {x.shape}")
    print(f"Logits shape : {logits.shape}  (expected [{B}, {T}, {cfg.vocab_size}])")
    print(f"h shape      : {hidden[0].shape}")
    print(f"c shape      : {hidden[1].shape}")
    print(f"Forward pass : {(t1-t0)*1000:.1f} ms")

    # Dummy loss
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    print(f"Dummy CE loss: {loss.item():.4f}  (expected ≈ {math.log(cfg.vocab_size):.2f} = log(V))")

    # Generation test
    prompt = torch.randint(0, cfg.vocab_size, (1, 10))
    gen    = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=40)
    print(f"Generation   : {prompt.shape} → {gen.shape}")

    print("\nAll checks passed.")