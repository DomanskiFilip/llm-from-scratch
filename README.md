# LSTM Language Model From Scratch

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

A conversational language model built entirely from scratch in PyTorch. No pretrained weights (with exception of optional glove). No transformer. 

## 1. Project Overview

 The goal was to build every component of the pipeline manually: from raw dataset collection through tokenisation, embedding alignment, training, and inference without relying on any pretrained model weights. Initially coding dataset and capability was planned but early results showed that at this scale coding confuses the model too much and leads to horrible results so it was abandoned.

The best model trainied with this project is "Pery", a multi-layer LSTM language model trained from scratch on ~122k instruction-following examples. an rtx 2080 was mainly used to acheve the "Pery" model, indicateing there is significant potential to use this project with more powerfull hardware and bigger datasets to produce actuall coherent llms.

The model demonstrates coherent short-form responses, basic persona consistency (it consistently introduces itself as "Pery"), and grammatically correct output within the constraints of a small from-scratch model.

The project allows for training of LSTM based LLMs with high customization through config.py file. 

**Key design choices at a glance:**

| Choice | Decision |
|--------|----------|
| Architecture | Multi-layer LSTM |
| Tokenisation | Byte-level BPE |
| Embeddings | GloVe-100d/GloVe-300d aligned to BPE vocab |
| Loss masking | Response tokens only |
| Weight tying | head.weight = embed.weight |

---

## 2. Architecture

### 2.1 Tokenisation — Byte-Level BPE

Tokenisation converts raw text into sequences of integer token IDs the model can process.

**Byte-Pair Encoding (BPE)** starts from individual bytes (all 256 possible values) and iteratively merges the most frequent adjacent pairs into new tokens. Because it operates at the byte level, every possible byte value `0x00`–`0xFF` is a valid base token — it is hard to produce an `<unk>` token regardless of the input language or characters used altho testing showed that with small vocab and hudge datasets it is possible to encounter `<unk>` token during training.

The vocabulary is trained from scratch on the project's own datasets, so common instruction-following phrases become single tokens while rare subwords are split into pieces.

**Special tokens** are given their own dedicated IDs and match the Alpaca prompt format exactly:

```
<|endoftext|>      — document boundary / generation stop
### Instruction:   — prompt header and stop sequence
### Input:         — optional context header and stop sequence
### Response:      — response header and stop sequence
\n\n               — double newline structural separator
\n                 — newline
<|pad|>            — padding
```

Making `### Instruction:` and `### Response:` single vocabulary tokens means the model has an unambiguous, one-token stop signal — limmiting risk of partially generating a stop sequence and continuing.

the <||> tokens were inspired by qwen and gpt tokenisers.

### 2.2 Embeddings - GloVe-100d/GloVe-300d Alignment

GloVe (Pennington et al. 2014) provides pretrained 100-dimensional(or 300-dimensional) word vectors where semantically similar words are geometrically close. Initialising from GloVe gives the embedding layer meaningful structure from epoch 1 rather than random noise — this accelerates early convergence without using a pretrained language model.

**The alignment challenge:** GloVe operates on plain words (`return`, `hello`). Our BPE tokens use the GPT-2 ByteLevel encoding where a leading space becomes `Ġ` — tokens look like `Ġreturn`, `ĠReturn`, `Ġinstruction`. A four-step priority chain resolves this:

```
1. Exact match      — strip Ġ prefix → look up in GloVe          Ġreturn → "return" 
2. Lowercase match  — ĠReturn → "return" 
3. Sub-word average — split into English words, average vectors   Ġinstruction → ["instruction"] 
4. Random init      — sample from N(0, σ_glove) so uninitialised tokens
                      don't stand out statistically
```

**Constraint enforced by the config:** `embed_dim == hidden_dim == embedding_dim`. All three must be 100 (or 300 if switching to GloVe-300d). A runtime assertion prevents silent shape mismatches. You can also customise this in config.py to ommit useing glove and train on any dimentionality but need to use ```--no-glove``` flag during training, this also allows to skip embeddings.py stage.

---

### 2.3 Model Layers

The full forward pass through `LM`:

```
Input IDs  [B, T]
    │
    ├── Token Embedding  [B, T, 100]   — GloVe-initialised lookup table
    │       (pad token zeroed after load)
    │
    ├── Positional Embedding  [1, T, 100]  — learnable table, 128 positions
    │       (positions clamped at max_seq_len to prevent crashes on long inputs)
    │
    ├── + (token + position)
    │
    ├── Embedding Dropout  p=0.05
    │       Randomly zeros dimensions → forces the model not to rely on any
    │       single dimension. Regularises without removing information entirely.
    │
    ├── LSTM  3 layers, hidden_dim
    │   │
    │   │   At each time step t, the LSTM computes:
    │   │
    │   │   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)   ← forget gate
    │   │   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   ← input gate
    │   │   g̃_t = tanh(W_g · [h_{t-1}, x_t] + b_g) ← cell candidate
    │   │   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g̃_t        ← cell update
    │   │   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)   ← output gate
    │   │   h_t = o_t ⊙ tanh(c_t)                  ← hidden output
    │   │
    │   │   The additive cell update (c_t = f_t⊙c_{t-1} + i_t⊙g̃_t) prevents
    │   │   vanishing gradients — when f_t ≈ 1 the gradient flows unchanged.
    │   │
    │   │   Layer 1 → local patterns (word associations, response phrasing)
    │   │   Layer 2 → higher-level structure (turn-taking, response length)
    │   │   Layer 3 → further refinement
    │   │   (more Layers can be added or Layers can be removed useing config.py)
    │   │
    │   │   Weight init:
    │   │     Input weights (weight_ih) — Xavier uniform
    │   │     Recurrent weights (weight_hh) — Orthogonal
    │   │     Forget gate bias — initialised to 1.0 (Jozefowicz et al. 2015)
    │   │       so the LSTM starts by retaining history rather than forgetting
    │
    ├── Output Dropout  p=0.05
    │
    ├── LayerNorm  (Ba et al. 2016)
    │       LN(x) = γ · (x − µ) / (σ + ε) + β
    │       Normalises each 100-dim hidden state independently per position.
    │       Used instead of BatchNorm because it works with variable-length
    │       sequences and is compatible with the loss masking scheme.
    │
    └── Linear Projection Head  Linear(100 → 4096, bias=False)
            Projects hidden state to raw logits over the vocabulary.
            No bias (per-token frequency is already in the embedding matrix).
            Logits are NOT softmax'd here — CrossEntropyLoss applies
            log-softmax internally for numerical stability.

Output Logits  [B, T, vocab_size]
```

### 2.4 Weight Tying

```python
self.head.weight = self.tok_embed.weight
```

The same 4096 × 100 matrix serves as both the input embedding lookup and the output scoring matrix. This works because `embed_dim == hidden_dim == 100`. It halves the parameter count of the largest matrix in the model and enforces a useful geometric constraint: the space used to represent tokens as inputs is the same space used to score them as outputs.

---

## 3. Training Pipeline

### 3.1 Response-Only Masked Loss

The single most impactful change in the project — it roughly halved validation loss compared to training on all tokens.

Every training example is stored as a pair of binary files:
- `*_shard_NNNN.bin` — uint16 token IDs
- `*_shard_NNNN.mask.bin` — uint8 loss mask (1 = response token, 0 = instruction token)

The mask is built at tokenisation time using `response_start_char` — the character index where the `### Response:` output begins in the full formatted string. Any token whose character offset falls at or after that index gets `mask=1`; all instruction/prompt tokens get `mask=0`.

In the training loop:

```python
# ShardDataset.__getitem__
y = y.masked_fill(mask == 0, -1)

# Training loop
criterion = nn.CrossEntropyLoss(ignore_index=-1)
```

The model only receives gradients for predicting response tokens. It never learns to predict `### Response:` given `### Instruction:` — it learns to produce answers.

---

### 3.2 Truncated Backpropagation Through Time

An LSTM's backward pass must traverse a computation graph of depth equal to the sequence length. For long sequences this is slow and numerically fragile (gradient explosion/vanishing).

TBPTT detaches the hidden state between batches, cutting the graph, while carrying the hidden state values forward so the model retains context:

```python
if hidden is not None:
    hidden = CodingLM.detach_hidden(hidden)   # detach between batches
logits, hidden = model(xc, hidden)             # carry values within sequence
```

In the final configuration `bptt_len = seq_len = 128`, so each sequence is processed as a single TBPTT chunk.

---

### 3.3 Optimiser & LR Schedule

**AdamW** (Loshchilov & Hutter 2019) applies weight decay directly to parameters rather than through the gradient update, correcting a bug in vanilla Adam's L2 regularisation. Weight decay is applied selectively:

```python
decay_params    = [p for n, p in model.named_parameters() if p.dim() >= 2]   # weight matrices
no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]    # biases, LayerNorm
```

**Cosine LR schedule with linear warmup:**

```
Steps 0 → warmup_steps : linear ramp  0 → peak_lr
Steps warmup → end     : cosine decay  peak_lr → 0.1 × peak_lr
```

Warmup prevents destructively large gradient steps at initialisation. Cosine decay allows broad exploration early, then fine convergence later. The scheduler steps after every batch.

**Gradient clipping:**

```python
nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

If the global gradient norm exceeds 1.0, all gradients are scaled down proportionally. LSTMs are particularly susceptible to exploding gradients on long sequences; this cap prevents training instability.

---
> All above mentioned (3.1 - 3.3) steps and processes were added dureing the project development and together with hyperparameter optimisation allowed to go from 1 hour lasting epochs to even couple of minutes or seconds lasting epochs with even better PPL(perplexity which was chosen as a main comparison metric) than innitially trained models . the final best "Pery" model was trainied in just 23 epochs in 2 hours acheaving 40ppl and best conversations compared to previously trained models that trained for even 8-9 hours.
---

### 3.4 Grid Search (this needs to be triggered with ```--grid-search``` flag dureing training process)

Before the full training run, a grid search tests every combination of candidate hyperparameters for 3 epochs each and selects the best:

| Hyperparameter | Values Searched |
|---------------|----------------|
| Learning rate | `lr×2`, `lr`, `lr÷5` |
| Batch size | `batch÷2`, `batch` |
| Dropout rate | `dropout`, `0.1`, `0.2` |

All results are saved to `artefacts/logs/grid_search_results.json`. The best configuration (lowest validation loss) is automatically used for the full training run.

---

### 3.5 Early Stopping & Checkpointing

Validation loss is computed after every epoch on a held-out 5% of shards. If it does not improve by at least `min_delta=1e-4` for `patience=20` consecutive epochs, training stops.

The best checkpoint (lowest validation loss ever seen) is saved separately from the most recent epoch — the best weights are never overwritten. On resumption, `validate_checkpoint_architecture()` checks three key weight shapes before loading to catch config mismatches that would otherwise cause silent incorrect behaviour.

This also allows you to stop training and restart it from where you finished at any moment it will restart from last saved best checkpoint allowing for flexibility and nice ux for the user of the project!

---

## 4. Datasets

All datasets are open-licensed. No gated or AI-generated content is used.

| Dataset | Rows | Licence | Purpose |
|---------|------|---------|---------|
| unsloth/alpaca-cleaned | ~52,000 | CC BY-NC 4.0 | General instruction following |
| databricks/databricks-dolly-15k | ~15,000 | CC BY-SA 3.0 | Human-written QA |
| hakurei/open-instruct-v1 | ~50,000 | Apache 2.0 | Diverse short instructions |
| Synthetic hello dataset | 5,000 | Custom | Greeting and persona behaviour |

Every example is normalised into the same Alpaca prompt template:

```
### Instruction:
{instruction}

### Response:
{output}
```

Dolly and Open Instruct responses are truncated to 800 characters at the nearest sentence boundary, teaching the model that responses end rather than rambling.

---

## 5. Evaluation

All metrics are computed exclusively over response tokens (`mask=1`). Instruction tokens are excluded using `ignore_index=-1`, consistent with training.

| Metric | Description |
|--------|-------------|
| **Perplexity** | `exp(average NLL loss)` over response tokens. A PPL of 40 means the model is as uncertain as choosing uniformly among 40 tokens. |
| **Top-1 accuracy** | Fraction of response positions where the highest-probability token matches ground truth. |
| **Confusion matrix** | 50×50 row-normalised recall matrix restricted to the 50 most frequent tokens. Strong diagonal = good recall. |
| **Precision / Recall / F1** | Per-token binary classification metrics over the top 50 tokens. |

**Final evaluation results for "Pery" model:**

| Metric | Value |
|--------|-------|
| Validation perplexity | < 40 |
| Micro-avg precision | 0.585 |
| Micro-avg recall | 0.383 |
| Micro-avg F1 | 0.463 |
| Weighted-avg F1 | 0.447 |

---

Training converged over 23 epochs before early stopping (patience=20). Both train and validation loss fell together throughout with no divergence — no overfitting observed.

```
Epoch  1:  train=7.44  val=6.95  ppl=1041
Epoch  5:  train=5.09  val=4.91  ppl=135
Epoch 10:  train=4.24  val=4.18  ppl=65
Epoch 15:  train=3.95  val=3.94  ppl=51
Epoch 23:  train=3.73  val=3.74  ppl=42
```
<img width="1784" height="667" alt="loss_curves" src="https://github.com/user-attachments/assets/305fd967-7a01-4821-9b1c-089afbdfdb40" />
<img width="1984" height="1781" alt="confusion_matrix" src="https://github.com/user-attachments/assets/48708a6e-25ed-46cb-be6a-38c80b28a82c" />

              precision    recall  f1-score   support

           4      0.835     0.367     0.510     87269
           5      0.665     0.731     0.696     77904
          20      0.523     0.632     0.573     67801
         227      0.500     0.342     0.406     60393
         305      0.584     0.431     0.496     55427
         283      0.496     0.246     0.329     53113
          89      0.393     0.167     0.235     52844
         296      0.363     0.287     0.321     48598
           0      0.454     0.816     0.584     46179
         273      0.441     0.442     0.441     45146
         263      0.689     0.588     0.634     40706
         308      0.540     0.436     0.483     37081
         280      0.544     0.229     0.322     33921
         278      0.612     0.501     0.551     33842
         266      0.722     0.227     0.345     31133
         334      0.526     0.461     0.492     31016
          75      0.611     0.314     0.415     31013
          19      0.671     0.300     0.415     30956
         269      0.516     0.329     0.402     30491
          73      0.496     0.211     0.296     29670
         275      0.671     0.381     0.486     29616
          95      0.730     0.262     0.385     29260
         268      0.597     0.436     0.504     28831
          72      0.479     0.217     0.299     28657
         336      0.581     0.386     0.463     28596
         281      0.779     0.356     0.489     27532
         415      0.491     0.202     0.286     26601
         291      0.671     0.574     0.618     26515
         267      0.801     0.324     0.462     26364
          90      0.674     0.468     0.553     25782
          74      0.713     0.234     0.352     25593
         285      0.786     0.382     0.514     25389
         326      0.555     0.241     0.337     24059
         274      0.857     0.320     0.466     23447
         324      0.629     0.461     0.532     22786
         353      0.528     0.071     0.125     22681
         297      0.796     0.525     0.633     22356
          83      0.577     0.166     0.258     22221
          76      0.417     0.134     0.202     21910
         379      0.607     0.352     0.446     21660
          82      0.718     0.283     0.406     21613
           8      0.655     0.524     0.583     21551
         461      0.378     0.224     0.281     20782
         369      0.709     0.588     0.643     20652
         441      0.705     0.231     0.348     19677
          57      0.627     0.217     0.322     19634
          88      0.647     0.254     0.365     19624
         272      0.789     0.440     0.565     19594
         306      0.585     0.415     0.486     19571
          86      0.564     0.206     0.302     19483

   micro avg      0.585     0.383     0.463   1636540
   
   macro avg      0.610     0.359     0.433   1636540
   
weighted avg      0.601     0.383     0.447   1636540

---

## 6. Project Structure

```
.
├── src/
│   ├── config.py                    — Single source of truth for all hyperparameters
│   ├── main.py                      — Central CLI entry point
│   ├── dataset_processing/
│   │   ├── download.py              — Fetch, clean, normalise datasets to Alpaca format
│   │   ├── tokeniser.py             — Train BPE vocab, encode to binary token + mask shards
│   │   └── embeddings.py            — Download GloVe, align to BPE vocabulary
│   └── training_and_evaluation/
│       ├── model.py                 — LM architecture definition
│       ├── train.py                 — Grid search + full LSTM training with masked loss + TBPTT
│       ├── evaluate.py              — Perplexity, accuracy, confusion matrix, precision/recall/F1
│       └── generate.py              — Interactive REPL inference
│
└── artefacts/                       — Generated at runtime (not committed)
    ├── data/                        — JSONL datasets + binary token/mask shards
    ├── tokeniser/                   — qwen_style.json + vocab.txt
    ├── embeddings/                  — glove_aligned.pt
    ├── checkpoints/                 — *_best.pt model checkpoints
    ├── logs/                        — Training CSVs + grid search results
    └── evaluation/                  — Loss curves, confusion matrix, PRF report
```

---

## 8. Installation

**Requirements:** Python 3.10+, PyTorch 2.0+

```bash
# Clone the repository
git clone <repo-url>
cd llm-from-scratch

# Install dependencies
pip install torch torchvision torchaudio
pip install tokenizers datasets transformers
pip install numpy scikit-learn matplotlib tqdm requests regex
```

> **GPU:** The model will automatically detect and use CUDA or MPS if available !!! You have to have correct torch version !!!. CPU training is supported but slow — expect ~5–10 minutes per epoch on CPU vs ~30 seconds on a modern GPU.

---

## 9. How to Use

All commands are run through the central `main.py` entry point.

### Step 1 — Download and preprocess datasets

```bash
python -m src.main download
```

Downloads all four datasets, normalises them to Alpaca format, and writes JSONL files to `artefacts/data/`.

---

### Step 2 — Tokenise

```bash
python -m src.main tokenise
```

Trains the BPE tokeniser on the downloaded data and encodes every document into paired binary shards:
- `*_shard_NNNN.bin` — uint16 token IDs
- `*_shard_NNNN.mask.bin` — uint8 loss mask

---

### Step 3 — Align GloVe embeddings (omit if you dont want to use glove)

```bash
python -m src.main embeddings
```

Downloads GloVe-100d/300d (~860 MB), aligns it to the BPE vocabulary using the four-priority chain, and saves `artefacts/embeddings/glove_aligned.pt`. A coverage report is printed showing what fraction of tokens were aligned vs randomly initialised.

---

### Step 4 — Train

**Standard training:**
```bash
python -m src.main train
```

**With hyperparameter grid search first (recommended):**
```bash
python -m src.main train --grid-search
```

**Override specific parameters:**
```bash
python -m src.main train --lr 3e-4 --batch-size 64 --epochs 40
```

**Train without GloVe (embeddings from scratch):**
```bash
python -m src.main train --no-glove
```

**Force a specific device:**
```bash
python -m src.main train --device cuda
python -m src.main train --device cpu
```

Training automatically resumes from the best checkpoint if one exists. Progress is logged to `artefacts/logs/<run_name>.csv`.

---

### Step 5 — Evaluate

```bash
python -m src.main evaluate --ckpt artefacts/checkpoints/default_run_best.pt
```

**Limit evaluation to N batches (faster):**
```bash
python -m src.main evaluate --ckpt artefacts/checkpoints/default_run_best.pt --batches 50
```

Outputs saved to `artefacts/evaluation/`:
- `loss_curves.png` — train/val loss and perplexity over epochs
- `confusion_matrix.png` — top-50 token recall matrix
- `precision_recall_f1.txt` — per-token classification report
- `metrics.json` — perplexity and accuracy as JSON
- `insights_report.txt` — human-readable summary

---

### Step 6 — Generate (Interactive)

```bash
python -m src.main generate --ckpt artefacts/checkpoints/default_run_best.pt
```

This starts an interactive REPL. Type any instruction and the model responds
Dont expect coherent responces or commertial level responces it all depends on configuration and dataset used and thus is also limmited by your hardware.

**REPL commands:**
```
/temp 0.5     — set sampling temperature (default 0.1)
/topk 10      — set top-k (default 1)
/topp 0.9     — set nucleus sampling threshold (default 0.95)
/rep  1.2     — set repetition penalty (default 1.1)
/len  300     — set max new tokens (default 256)
/quit         — exit
```

**Single-shot generation:**
```bash
python -m src.main generate \
  --ckpt artefacts/checkpoints/default_run_best.pt \
  --prompt "Explain what a neural network is." \
  --temperature 0.2 \
  --max-new 150
```

> **Note on generation settings:** The default temperature of 0.1 and top_k of 1 (near-greedy) are intentional. At this model scale the probability distributions are relatively flat — higher temperatures cause incoherent output. If you retrain on more data or a larger architecture, higher temperatures will produce more diverse and natural responses. it is also good to set repetition penalty higher if the model repeats one word or token too much.

> **To switch to GloVe-300d:** set `embed_dim`, `hidden_dim`, and `embedding_dim` all to 300 in `config.py`. The assertion in `__post_init__` will catch any inconsistency at startup.

## References

- Hochreiter & Schmidhuber (1997). Long Short-Term Memory. *Neural Computation* 9(8).
- Pennington, Socher & Manning (2014). GloVe: Global Vectors for Word Representation
- Jozefowicz et al. (2015). An Empirical Exploration of Recurrent Network Architectures. *ICML 2015*.
- Ba et al. (2016). Layer Normalization
- Press & Wolf (2017). Using the Output Embedding to Improve Language Models.
- Loshchilov & Hutter (2019). Decoupled Weight Decay Regularisation (AdamW).
- Vaswani et al. (2017). Attention Is All You Need.
- Devlin et al. (2019). BERT.
- Taori et al. (2023). Alpaca: A Strong, Replicable Instruction-Following Model. Stanford CRFM.
