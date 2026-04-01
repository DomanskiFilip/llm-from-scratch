# llm-from-scratch/src/config.py

from dataclasses import dataclass, field


@dataclass
class Config:
    # ── Model Parameters ──────────────────────────────────────────────────────
    vocab_size: int = 8192
    # embed_dim MUST equal hidden_dim when tie_weights=True.
    # It also MUST equal embedding_dim (GloVe dim) so the pretrained matrix
    # loads without shape errors.  We use 100 to match GloVe-100d.
    # If you want a larger model, switch to GloVe-300d and set all three to 300.
    embed_dim: int = 300
    hidden_dim: int = 300         # keep equal to embed_dim for weight tying
    n_layers: int = 3
    tie_weights: bool = True       # only valid when embed_dim == hidden_dim
    max_seq_len: int = 512
    pad_id: int = 0

    # ── Training Parameters ───────────────────────────────────────────────────
    seq_len: int = 300             # context window fed to the model each step
    bptt_len: int = 64             # TBPTT chunk length
    lr: float = 5e-4              # peak AdamW learning rate
    weight_decay: float = 0.1
    clip_norm: float = 1.0        # gradient clipping
    warmup_steps: int = 300       # linear LR warm-up steps
    epochs: int = 30
    batch_size: int = 64
    val_fraction: float = 0.05    # fraction of shards held out for validation
    log_every: int = 100          # print loss every N batches

    # ── Regularisation ────────────────────────────────────────────────────────
    dropout_rate: float = 0.15    # applied to embed_drop, lstm_drop, out_drop

    # ── Hardware ──────────────────────────────────────────────────────────────
    device: str = "auto"          # "auto" → CUDA > MPS > CPU

    # ── Early Stopping ────────────────────────────────────────────────────────
    patience: int = 5
    min_delta: float = 1e-4

    # ── Grid Search ───────────────────────────────────────────────────────────
    # Short runs used during the search; full run afterwards
    grid_epochs: int = 3
    full_epochs: int = 30

    # ── Tokeniser ────────────────────────────────────────────────────────────
    tokenizer_vocab_size: int = 8192
    tokenizer_train_sample_lines: int = 200000
    tokenizer_tokens_per_shard: int = 10000000
    tokenizer_special_tokens: list[str] = field(
        default_factory=lambda: [
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|fim_prefix|>",
            "<|fim_suffix|>",
            "<|fim_middle|>",
            "<|pad|>",
        ]
    )
    tokenizer_eot_token: str = "<|endoftext|>"

    # ── Embeddings ────────────────────────────────────────────────────────────
    # embedding_dim MUST match embed_dim above.
    # GloVe-100d → embedding_dim = 100  (default, ~347 MB zip)
    # GloVe-300d → embedding_dim = 300  (change embed_dim/hidden_dim too)
    embedding_dim: int = 300
    embedding_glove_url: str = "https://nlp.stanford.edu/data/glove.6B.zip"
    random_seed: int = 42

    # ── Download / Data Paths ─────────────────────────────────────────────────
    data_dir: str = "data"

    def __post_init__(self):
        # Enforce the constraints that are easy to get wrong
        assert self.embed_dim == self.hidden_dim or not self.tie_weights, (
            f"tie_weights=True requires embed_dim == hidden_dim, "
            f"got embed_dim={self.embed_dim}, hidden_dim={self.hidden_dim}"
        )
        assert self.embed_dim == self.embedding_dim, (
            f"embed_dim ({self.embed_dim}) must equal embedding_dim ({self.embedding_dim}) "
            f"so the GloVe weight matrix loads without a shape error."
        )
        assert self.vocab_size == self.tokenizer_vocab_size, (
            f"vocab_size ({self.vocab_size}) must equal tokenizer_vocab_size "
            f"({self.tokenizer_vocab_size}) — they must always be in sync."
        )