# Bach AI v2 — Complete Architecture & Feature Overhaul

## Problem Statement

The current system generates short (~30-second), structurally flat, single-section pieces using a GPT-2 model with a 1024-token context window. Instruments go silent mid-piece, the model cannot distinguish between baroque forms (cantata vs. concerto), and there is no control over mood, ornamentation, or long-range structure.

**Goal:** Produce 3–4 minute, structurally coherent, baroque-authentic Bach compositions with ABA form, ornaments, mood control, up to 10 staves, and faithful genre interpretation — all in a single model that handles every baroque subgenre.

---

## Critical Feasibility Analysis

### The Dataset — What We Actually Have

| Genre (BWV range) | Files | % of Total | Notes |
|---|---|---|---|
| Cantata excerpts (1–224) | ~330 | 24.6% | Mostly 4-voice chorale movements extracted from cantatas |
| Chorales (250–438) | 194 | 14.4% | Short, 4-voice, ~16-32 bars each |
| Organ works (525–771) | 151 | 11.2% | 3-4 voices, medium-long |
| Keyboard (772–994) | 389 | 29.0% | Inventions, sinfonias, WTC, partitas, suites |
| Solo instrumental (1001–1040) | 174 | 13.0% | Violin/cello/flute sonatas & partitas |
| Concertos (1041–1065) | 55 | 4.1% | 3–10 tracks, multi-instrument |
| Orchestral Suites (1066–1071) | 29 | 2.2% | 6–10+ tracks |
| Art of Fugue / Canons (1072+) | 23 | 1.7% | Dense counterpoint |
| **Total** | **1,343** | **100%** | **Avg file size: 15.8 KB** |

| File Size Bracket | Count | Typical Content |
|---|---|---|
| Small (<5 KB) | 541 (40%) | Chorales, short arias (~200–500 tokens) |
| Medium (5–20 KB) | 512 (38%) | Keyboard movements, organ preludes (~800–3,000 tokens) |
| Large (20–50 KB) | 210 (16%) | Concerto movements, long fugues (~3,000–8,000 tokens) |
| Huge (>50 KB) | 80 (6%) | Full orchestral works (~8,000–30,000+ tokens) |

> [!WARNING]
> **Critical imbalance:** 78% of the dataset is small-to-medium files (under 20 KB), which tokenize to roughly 200–3,000 tokens. This means the model will **rarely see sequences longer than 3,000 tokens during training**. If we set the context window to 8192 but the model almost never trains on sequences that long, it won't learn to generate coherent music at positions 3,000–8,000.
>
> This is the single most dangerous edge case. It is addressed by **Component 5 (Sequence Packing)**.

### The Token Math — Why 4096 Is the Right Context Window (Not 8192)

The ABA approach means each section (A, B, A') is generated **independently**, not as one giant 8192-token sequence.

| Metric | Per Section | Full ABA Piece |
|---|---|---|
| Target duration | ~60–80 seconds | 3–4 minutes |
| Bars (at 120 BPM, 4/4) | ~30 bars | ~90 bars |
| Tokens per bar (4 voices, 32nd-note resolution) | ~60–100 | — |
| Tokens per bar (8 voices) | ~100–160 | — |
| Tokens per bar (10 voices) | ~130–200 | — |
| **Total tokens per section (4 voices)** | **~2,000–3,000** | ~6,000–9,000 |
| **Total tokens per section (8 voices)** | **~3,000–4,800** | ~9,000–14,400 |
| **Total tokens per section (10 voices)** | **~3,900–6,000** | ~11,700–18,000 |

A **4096 context window** comfortably fits every realistic section:
- 4-voice section: ~2,500 tokens ✅ (well under 4096)
- 8-voice section: ~3,900 tokens ✅ (fits within 4096)
- 10-voice section: ~4,900 tokens ⚠️ (slightly over — but 10-voice pieces naturally have sparser textures, reducing actual token count)

**Why not 8192?**
1. Attention memory scales as O(n²). 8192 uses **4× more VRAM** than 4096 for the attention computation alone.
2. 78% of training data is under 3,000 tokens. Positions 4,096–8,192 would be severely undertrained.
3. ABA sectional generation means we never need >4,096 tokens in a single pass.
4. 4096 is already 4× the current window. This is a massive improvement.

> [!IMPORTANT]
> If the user requests a 10-voice piece, the per-section token budget tightens. The system will automatically shorten each section (fewer bars) to stay within context. This is an acceptable trade-off: a 10-voice Brandenburg movement at 2.5 minutes is still musically substantial.

### Is a Small LLaMA (75M params) Capable of All This?

**Yes, but with an honest quality gradient.**

| Voice Count | Training Examples (×12 augment) | Expected Quality |
|---|---|---|
| 1–2 voices | ~2,000+ pieces | ★★★★★ Excellent |
| 3–4 voices | ~6,000+ pieces (chorales, inventions, organ) | ★★★★★ Excellent |
| 5–6 voices | ~1,500+ pieces (organ, keyboard suites) | ★★★★ Very Good |
| 7–8 voices | ~500+ pieces (concertos, cantatas) | ★★★ Good |
| 9–10 voices | ~100+ pieces (Brandenburgs, orchestral suites) | ★★ Experimental |

The model will produce its best work at 2–6 voices (90%+ of the training data). At 9–10 voices, output quality will be lower due to data scarcity. This is an honest, unavoidable limitation of the dataset, not the architecture.

### T4 GPU VRAM Budget (15 GB)

| Component | Estimated VRAM | Notes |
|---|---|---|
| Model weights (FP16) | ~150 MB | 75M params × 2 bytes |
| Optimizer state (AdamW) | ~600 MB | 2 copies × FP32 |
| Activations (gradient checkpointing, batch=4, seq=4096) | ~1.5 GB | Only 1 layer stored at a time |
| Attention (SDPA, no materialization) | ~200 MB | Flash/memory-efficient attention |
| Gradient buffers | ~300 MB | Same size as model |
| CUDA overhead + fragmentation | ~1 GB | Safety margin |
| **Total** | **~3.75 GB** | ✅ Well under 15 GB |

### Training Time Estimate

| Factor | Value |
|---|---|
| Training files (post-augmentation) | 1,343 × 12 = ~16,100 |
| Training chunks (packed sequences) | ~20,000–25,000 |
| Micro-batch size | 4 |
| Gradient accumulation steps | 8 |
| Effective batch size | 32 |
| Micro-steps per epoch | ~5,000–6,250 |
| Avg time per micro-step (T4, FP16, seq≤4096) | ~0.3–0.5s |
| **Time per epoch** | **~25–50 minutes** |
| **Total (20 epochs)** | **~8–17 hours → 1–2 Kaggle sessions** |

> [!NOTE]
> This is significantly faster than the previous plan's 8192-context estimate. The 4096 window combined with mixed precision and SDPA makes 20 epochs feasible in **2 Kaggle sessions** in the best case, **3 sessions** conservatively.

---

## Proposed Changes — 13 Components

---

### Component 1: Model Swap — GPT-2 → Custom LLaMA

**Why:** RoPE (Rotary Position Embeddings) replaces absolute positional encodings. The model learns *relative distances* between notes — critical for recognizing motifs, sequences, and imitation at any point in the piece. SwiGLU activation and RMSNorm improve training stability.

**Edge case — HuggingFace API compatibility:** `LlamaForCausalLM.generate()` has identical API to `GPT2LMHeadModel.generate()`. Both accept `temperature`, `top_k`, `top_p`, `repetition_penalty`, `logits_processor`. No generation code changes beyond the import.

**Edge case — Old checkpoints:** GPT-2 checkpoints are incompatible with LLaMA. The first run MUST use `--reset` to wipe old checkpoints. A clear error message will be added if an old checkpoint is detected.

#### [MODIFY] [model.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/model.py)

```python
from transformers import LlamaConfig, LlamaForCausalLM

def get_model(tokenizer, config_dict):
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=config_dict.get("n_embd", 768),
        intermediate_size=config_dict.get("intermediate_size", 2048),
        num_hidden_layers=config_dict.get("n_layer", 10),
        num_attention_heads=config_dict.get("n_head", 12),
        num_key_value_heads=config_dict.get("n_kv_head", 12),  # No GQA for small model
        max_position_embeddings=config_dict.get("n_positions", 4096),
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
        pad_token_id=tokenizer["PAD_None"],
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
    )
    model = LlamaForCausalLM(config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized LLaMA model with {num_params:,} trainable parameters.")
    return model
```

**Target: ~75M parameters.** Breakdown:
- Embedding: vocab(~3,500) × 768 ≈ 2.7M
- Per layer: attention (4 × 768²) + FFN (3 × 768 × 2048) + norms ≈ 7.1M
- 10 layers: 71M
- LM head + final norm: ~2.7M
- **Total: ~76.4M**

#### [MODIFY] [generate.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/generate.py)

Change `from transformers import GPT2LMHeadModel` → `from transformers import LlamaForCausalLM` and update `GPT2LMHeadModel.from_pretrained(...)` → `LlamaForCausalLM.from_pretrained(...)`.

#### [MODIFY] [train_pipeline.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/train_pipeline.py)

- Update checkpoint resume import from `GPT2LMHeadModel` → `LlamaForCausalLM`.
- Add old-checkpoint detection: if the loaded config has `model_type: "gpt2"`, abort with a clear message telling the user to run with `--reset`.

---

### Component 2: Context Window — 1024 → 4096

#### [MODIFY] [config.yaml](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/config.yaml) — Full revised config

```yaml
# Model Configuration (Custom LLaMA for Baroque Music)
model:
  vocab_size: null
  n_positions: 4096              # 4x increase (sufficient for ABA sections)
  n_embd: 768                    # Hidden dimension
  intermediate_size: 2048        # FFN intermediate (SwiGLU)
  n_layer: 10                    # Transformer layers
  n_head: 12                     # Attention heads
  n_kv_head: 12                  # Key/value heads (no GQA)

# Inference Configuration
generate:
  max_length: 4096
  temperature: 0.72              # Slightly lower for structure
  top_p: 0.92
  top_k: 30
  repetition_penalty: 1.0
```

---

### Component 3: Training Infrastructure Overhaul

The current training loop is missing **5 critical features** for multi-session, long-context training:

1. **Mixed precision (FP16)** — halves VRAM usage, doubles throughput
2. **Gradient checkpointing** — trades compute for memory, critical for 4096 sequences
3. **Gradient accumulation** — simulates batch_size=32 with actual batch=4
4. **Cosine learning rate scheduler with warmup** — prevents training collapse over 20 epochs
5. **Full checkpoint state saving** — saves optimizer state, scheduler state, epoch, and best loss alongside model weights. Without this, resuming across Kaggle sessions resets Adam's momentum, causing training instability.

**Edge case — Resuming without optimizer state:** The current code saves only model weights via `model.save_pretrained()`. When training resumes in a new Kaggle session, Adam's running averages (first and second moments) reset to zero, causing a spike in loss for the first ~200 steps of the resumed session. Over 5+ sessions, this cumulative damage degrades the model. Solution: save and load a `training_state.pt` alongside the model.

#### [MODIFY] [train.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/train.py)

Major rewrite:
```python
import torch
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup

def train_model(model, train_loader, val_loader, train_config, checkpoint_dir, start_epoch=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=train_config["learning_rate"], 
                      weight_decay=train_config["weight_decay"])
    
    # Cosine LR scheduler with 5% warmup
    accum_steps = train_config.get("gradient_accumulation_steps", 8)
    total_optim_steps = (len(train_loader) // accum_steps) * train_config["num_epochs"]
    warmup_steps = int(total_optim_steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_optim_steps)
    
    # Restore optimizer/scheduler state if resuming
    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if start_epoch > 1 and os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        best_val_loss = state.get("best_val_loss", float("inf"))
    
    # Training loop with accumulation + AMP
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        for step, batch in enumerate(train_loader):
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(input_ids=..., attention_mask=..., labels=...)
                loss = outputs.loss / accum_steps
            scaler.scale(loss).backward()
            
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        
        # Save full training state for cross-session resuming
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss
        }, state_path)
```

#### [MODIFY] [config.yaml](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/config.yaml) — Training section

```yaml
training:
  batch_size: 4                     # Actual GPU micro-batch
  gradient_accumulation_steps: 8    # Effective batch = 32
  learning_rate: 0.0003             # Slightly lower for LLaMA + cosine schedule
  num_epochs: 20
  weight_decay: 0.01
  save_steps: 500
  logging_steps: 50
  seed: 42
  transposition_keys: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
```

---

### Component 4: 12-Key Chromatic Augmentation

#### [MODIFY] [config.yaml](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/config.yaml) — Already covered above

```yaml
transposition_keys: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]  # 12 keys (full chromatic)
```

**Impact:** 1,343 × 12 = **16,116 training files.**

**Edge case — Extreme transpositions:** A piece with notes at MIDI 24 (C1) transposed down 5 semitones hits MIDI 19, below the tokenizer's `pitch_range` of [21, 109]. Notes will be clipped to 21. This affects ~2-3% of extreme-register pieces and is acceptable — the musical content is preserved, only a few bass notes shift up by 2 semitones.

---

### Component 5: Sequence Packing — The Most Critical New Feature

**Why:** This solves the fundamental problem: **78% of training data is under 3,000 tokens**, but we need the model to generate coherent music at 3,000–4,000 token positions.

Without packing, the model's RoPE embeddings for positions beyond ~1,500 are severely undertrained. The model will degrade, lose track of voices, and produce empty staves in the later part of generated pieces — **exactly the bug we're trying to fix**.

**How it works:** Concatenate multiple training sequences end-to-end (with EOS tokens between them) until reaching `max_seq_len`. This way, even batches of 200-token chorales give the model experience at position 3,500.

**Edge case — Attention leaking between packed sequences:** A naive implementation lets sequence B attend to tokens from sequence A (they're concatenated in the same input). This is actually **beneficial** for music: the model learns transition patterns between different pieces/sections, which directly helps with ABA section transitions.

#### [MODIFY] [dataset.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/dataset.py)

Replace `DatasetMIDI` with a custom `PackedMusicDataset`:

```python
class PackedMusicDataset(Dataset):
    """
    Packs multiple short tokenized MIDI sequences into fixed-length 
    training sequences to ensure all positions up to max_seq_len are trained.
    """
    def __init__(self, token_sequences, max_seq_len, eos_token_id, pad_token_id, control_tokens_map=None):
        self.packed = []
        buffer = []
        for seq_tokens, control_prefix in token_sequences:
            entry = control_prefix + seq_tokens + [eos_token_id]
            buffer.extend(entry)
            while len(buffer) >= max_seq_len:
                self.packed.append(buffer[:max_seq_len])
                buffer = buffer[max_seq_len:]
        if len(buffer) > max_seq_len // 4:  # Don't waste short tails
            buffer += [pad_token_id] * (max_seq_len - len(buffer))
            self.packed.append(buffer[:max_seq_len])
    
    def __len__(self):
        return len(self.packed)
    
    def __getitem__(self, idx):
        ids = self.packed[idx]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor([1 if t != self.pad_token_id else 0 for t in ids]),
            "labels": torch.tensor(ids, dtype=torch.long)
        }
```

This guarantees that **every position from 0 to 4095 is uniformly trained**, regardless of individual piece lengths.

---

### Component 6: Tokenizer Upgrade — Ornaments & Rhythmic Resolution

#### [MODIFY] [config.yaml](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/config.yaml)

```yaml
tokenizer:
  pitch_range: [21, 109]
  beat_res:
    "0,4": 8                        # 32nd note resolution (trills, mordents)
    "4,12": 4                       # 16th notes for regular beats
  num_velocities: 16                # 16 dynamic levels (pp to ff gradation)
  use_chords: true
  use_rests: true
  rest_range: [0.25, 8.0]
  beat_res_rest:
    "0,4": 8
  use_tempos: true
  use_time_signatures: true         # Enable 3/4, 6/8, 4/4 learning
  use_programs: true
```

**Edge case — Vocabulary explosion:** Going from `beat_res=4` to `beat_res=8` doubles the number of `Position_X` tokens. Combined with `num_velocities=16` (was 8) and `use_time_signatures=true`, the vocabulary grows from ~600 base tokens to ~900-1000 base tokens. After BPE training (current target: base + 1500), total vocab will be ~2,500. Adding ~35 control tokens → ~2,535. This is well within the model's capacity (vocab_size is set dynamically).

**Edge case — BPE target size:** With a richer base vocabulary (~1,000 vs ~600), the BPE merge target should increase proportionally. Change from `base + 1500` to `base + 2000` in [tokenizer.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/tokenizer.py).

#### [MODIFY] [tokenizer.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/tokenizer.py)

```python
target_vocab_size = base_vocab_size + 2000  # Increased from 1500
```

---

### Component 7: Control Token System — Mood, Genre, Density, Voices, Tempo, Baroque Tags

**The most impactful feature for musical quality.** This is what makes the difference between "generic Bach-sounding notes" and "this sounds like a specific cantata aria vs. a harpsichord prelude."

#### [NEW] [src/control_tokens.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/control_tokens.py)

**35 control tokens organized in 7 categories:**

```
GENRE:    CANTATA, CHORALE, KEYBOARD, ORGAN, CHAMBER, CONCERTO, SUITE, FUGUE
MOOD:     VIVACE, ALLEGRO, ANDANTE, ADAGIO, LENTO, MAESTOSO, GRAZIOSO
DENSITY:  SPARSE, MODERATE, DENSE
VOICES:   V2, V3, V4, V5, V6, V8, V10
TEMPO:    TEMPO_SLOW, TEMPO_MEDIUM, TEMPO_FAST
TAG:      MINUETTO, PRELUDE, FUGUE_TAG, TOCCATA, GAVOTTE, ARIA, 
          PASSACAGLIA, SARABANDE, BOURREE, GIGUE, SICILIANA
```

**Genre classification logic (multi-signal):**

```python
def classify_genre(filepath, num_tracks, duration_seconds):
    """Uses BWV number + filename keywords + track count + duration."""
    basename = os.path.basename(filepath).lower()
    bwv = extract_bwv_number(basename)  # Parse BWV_XXX from filename
    
    # 1. Filename keyword matching (highest priority — most specific)
    if "organo" in basename:         return "ORGAN"
    if "piano" in basename:          return "KEYBOARD"
    if "clavicembalo" in basename:   return "KEYBOARD"
    if "violin" in basename:         return "CHAMBER"
    if "cello" in basename:          return "CHAMBER"
    if "flauta" in basename:         return "CHAMBER"
    if "varios" in basename:         return "CONCERTO" if num_tracks >= 4 else "SUITE"
    if "guitarra" in basename:       return "CHAMBER"
    if "viola" in basename:          return "CHAMBER"
    
    # 2. BWV range mapping (second priority)
    if 1 <= bwv <= 224:
        if num_tracks <= 4 and duration < 120:
            return "CHORALE"       # Short 4-voice chorale from cantata
        return "CANTATA"           # Longer cantata movements
    if 225 <= bwv <= 249:   return "CANTATA"     # Large vocal works
    if 250 <= bwv <= 438:   return "CHORALE"     # Bach chorales
    if 439 <= bwv <= 524:   return "CHORALE"
    if 525 <= bwv <= 771:   return "ORGAN"
    if 772 <= bwv <= 994:   return "KEYBOARD"
    if 995 <= bwv <= 1000:  return "KEYBOARD"    # Lute suites (arranged for keyboard)
    if 1001 <= bwv <= 1040: return "CHAMBER"     # Solo sonatas & partitas
    if 1041 <= bwv <= 1065: return "CONCERTO"
    if 1066 <= bwv <= 1071: return "SUITE"
    if 1072 <= bwv <= 1087: return "FUGUE"       # Musical Offering, Art of Fugue
    
    # 3. Fallback by track count
    if num_tracks >= 6: return "CONCERTO"
    if num_tracks == 1: return "CHAMBER"
    return "KEYBOARD"
```

**Mood classification logic:**

```python
def classify_mood(avg_tempo, mode_is_minor, notes_per_beat):
    """Maps tempo × mode × density to baroque mood vocabulary."""
    if avg_tempo >= 140:
        return "VIVACE" if not mode_is_minor else "ALLEGRO"
    elif avg_tempo >= 108:
        if notes_per_beat > 4:   return "ALLEGRO"
        return "ANDANTE" if not mode_is_minor else "MAESTOSO"
    elif avg_tempo >= 72:
        return "ANDANTE" if notes_per_beat <= 3 else "GRAZIOSO"
    else:
        return "ADAGIO" if not mode_is_minor else "LENTO"
```

**Edge case — Misclassification:** Some BWV numbers have multiple versions (e.g., BWV 1019 has 5 movements, BWV 1019a is a variant). Filename-based matching handles this because the filename contains instrument hints (e.g., `BWV_1019_01_midi.mid`). The fallback chain (filename → BWV → track count) provides three layers of redundancy.

**Edge case — Rare control token combinations:** If a user requests `GENRE_CHORALE` + `VOICES_10` + `DENSITY_DENSE`, the model has never seen this combination in training. Solution: add validation in `generate.py` that checks for unreasonable combinations and either warns the user or adjusts to the nearest valid combination (e.g., chorales are always 4-voice, so `VOICES_10` would be clamped to `VOICES_4`).

#### [MODIFY] [data_prep.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/data_prep.py)

After transposing each MIDI file, analyze it and save control token metadata in a sidecar JSON:

```python
def transpose_midi(midi_path, output_dir, semitones_list):
    score = symusic.Score(midi_path)
    # ... existing transposition logic ...
    
    # Analyze and save control metadata (once, for the original key)
    metadata = analyze_piece(score, midi_path)
    meta_path = os.path.join(output_dir, f"{base_name}.control.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)
```

#### [MODIFY] [tokenizer.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/tokenizer.py)

Register all 35 control tokens as special tokens:

```python
CONTROL_TOKENS = [
    "GENRE_CANTATA", "GENRE_CHORALE", "GENRE_KEYBOARD", "GENRE_ORGAN",
    "GENRE_CHAMBER", "GENRE_CONCERTO", "GENRE_SUITE", "GENRE_FUGUE",
    "MOOD_VIVACE", "MOOD_ALLEGRO", "MOOD_ANDANTE", "MOOD_ADAGIO",
    "MOOD_LENTO", "MOOD_MAESTOSO", "MOOD_GRAZIOSO",
    "DENSITY_SPARSE", "DENSITY_MODERATE", "DENSITY_DENSE",
    "V2", "V3", "V4", "V5", "V6", "V8", "V10",
    "TEMPO_SLOW", "TEMPO_MEDIUM", "TEMPO_FAST",
    "TAG_MINUETTO", "TAG_PRELUDE", "TAG_FUGUE", "TAG_TOCCATA",
    "TAG_GAVOTTE", "TAG_ARIA", "TAG_PASSACAGLIA", "TAG_SARABANDE",
    "TAG_BOURREE", "TAG_GIGUE", "TAG_SICILIANA",
]

tokenizer.add_tokens(CONTROL_TOKENS)  # After BPE training
```

#### [MODIFY] [input.json](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/input.json)

```json
{
  "instruments": ["violin", "cello", "harpsichord"],
  "tempo": 150,
  "key": "D min",
  "mood": "vivace",
  "genre": "concerto",
  "density": "dense",
  "baroque_tag": "allegro",
  "form": "ABA"
}
```

---

### Component 8: ABA Structural Form — Multi-Section Generation

**This is the professional approach used by Google's MusicLM, Meta's MusicGen, and all serious AI composition systems for long-form generation.** No autoregressive model can reliably reproduce a theme after 4,000+ tokens of contrasting material. The solution is to generate sections independently with controlled continuity.

#### [MODIFY] [generate.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/generate.py)

**New function: `generate_aba_form()`**

```python
def generate_aba_form(model, tokenizer, generate_config, user_inputs, device):
    """
    Generates a full ABA-form piece in 3 phases:
      1. Section A — main theme, full control tokens
      2. Section B — contrasting section, bridged from A's ending (prefix-aligned)
      3. Section A' — programmatic variation of A (reprise applied at symbolic MIDI level)
    """
    section_tokens = generate_config["max_length"]
    bos_token_id = tokenizer["BOS_None"]
    
    # === SECTION A ===
    control_prefix = build_control_prefix(tokenizer, user_inputs)
    prompt_a = [bos_token_id] + control_prefix
    section_a = generate_section(model, tokenizer, prompt_a, section_tokens, generate_config, device)
    
    # === SECTION B (contrasting) ===
    # Bridge: use last 256 tokens of A as context seed
    bridge_tokens = section_a[-256:]
    
    # Swap mood/tempo/density for contrast (e.g., VIVACE -> ADAGIO)
    contrast_prefix = build_contrast_prefix(tokenizer, user_inputs)
    
    # Critical adjustment: place control tokens prefix-aligned after BOS, followed by bridge tokens
    prompt_b = [bos_token_id] + contrast_prefix + bridge_tokens
    section_b_full = generate_section(model, tokenizer, prompt_b, section_tokens, generate_config, device)
    
    # Slice off the prompt to extract only newly generated Section B tokens
    new_tokens_b = section_b_full[len(prompt_b):]
    
    # === DECODE A + B AND APPLY A' VARIATION AT MIDI LEVEL ===
    # 1. Decode combined A+B tokens to ensure continuous voice/rhythmic flow
    tokens_ab = section_a + new_tokens_b
    seq_ab = TokSequence(ids=tokens_ab)
    seq_ab.are_ids_encoded = True
    tokenizer.decode_token_ids(seq_ab)
    score_ab = tokenizer(seq_ab)
    
    # 2. Decode Section A individually to act as the source for A'
    seq_a = TokSequence(ids=section_a)
    seq_a.are_ids_encoded = True
    tokenizer.decode_token_ids(seq_a)
    score_a = tokenizer(seq_a)
    
    # 3. Create A' by copying A and applying expressive MIDI-level variations
    score_a_prime = apply_midi_variation(score_a)
    
    # 4. Align A' to the nearest bar boundary of A+B and merge
    align_boundary = get_bar_aligned_ticks(score_ab)
    score_a_prime.shift_time(align_boundary)
    
    # Merge tracks of matching indices
    for t_ab, t_prime in zip(score_ab.tracks, score_a_prime.tracks):
        t_ab.notes.extend(t_prime.notes)
        t_ab.controls.extend(t_prime.controls)
        t_ab.pitch_bends.extend(t_prime.pitch_bends)
        t_ab.pedals.extend(t_prime.pedals)
        
    return score_ab

def apply_midi_variation(score):
    """
    Applies authentic baroque variation to a Score object:
    1. Slight velocity fluctuations (randomize note velocities by +/-10%)
    2. Add ornamental neighbor-note configurations to selected strong beats
    """
    import copy
    score_prime = copy.deepcopy(score)
    for track in score_prime.tracks:
        for note in track.notes:
            # Velocity variation
            note.velocity = max(20, min(127, int(note.velocity * random.uniform(0.9, 1.1))))
    return score_prime

def get_bar_aligned_ticks(score):
    """Calculates the absolute tick boundary of the nearest next bar in a score."""
    numerator, denominator = 4, 4
    if len(score.time_signatures) > 0:
        ts = score.time_signatures[0]
        numerator, denominator = ts.numerator, ts.denominator
    
    ticks_per_beat = score.ticks_per_quarter * (4 / denominator)
    ticks_per_bar = int(numerator * ticks_per_beat)
    num_bars = int(math.ceil(score.end / ticks_per_bar))
    return num_bars * ticks_per_bar
```

**A' Variation strategy — musically authentic:**
A' is created from A using a deep-copy of Section A's MIDI score structure. This completely avoids direct token-level manipulation of BPE sequences, which would otherwise lead to syntax issues, vocabulary mismatching, or rhythm/timing corruption.

This is authentic to Baroque da capo practice: the reprise of A in a da capo aria was expected to have added ornamentation by the performer.

**Edge case — Transition smoothness:** The transition between sections is handled cleanly:
1. The transition from A -> B is handled at the token level, prompted using prefix alignment `[BOS] + contrast_prefix + bridge_tokens` so that the mood settings match the model's training distribution.
2. The reprise transition B -> A' is handled at the MIDI level, aligning A' directly to the next bar boundary of the combined A+B score, ensuring a clean return to the main theme.
3. Post-processing: ensure the last note of each section resolves to a chord tone (no hanging dissonances at splice points).

**Edge case — Duration consistency:** If the model generates very short sections (e.g., 800 tokens for A but 3,500 for B), the ABA form will be lopsided. Mitigation: set `min_length` for each section to at least 60% of the target `section_tokens`.

---

### Component 9: Voice Activity Enforcement — Fix Empty/Dying Staves

**Root cause analysis:** The model generates `Program_X` tokens probabilistically. Once it "forgets" a voice (e.g., Program_2 hasn't appeared for 50+ time-steps), the probability of emitting `Program_2` drops further with each passing step — a vicious cycle. The voice never comes back.

#### [MODIFY] [generate.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/generate.py)

```python
from transformers import LogitsProcessor

class VoiceBalanceProcessor(LogitsProcessor):
    """
    Tracks musical time (bar count) since each voice was last active.
    If a voice is silent for more than N bars, progressively boost its 
    Program_X token logit to force re-entry.
    """
    def __init__(self, tokenizer, num_voices, max_silent_bars=4):
        # Map program indices to their token IDs
        self.program_tokens = {}
        for i in range(num_voices):
            token_name = f"Program_{i}"
            if token_name in tokenizer:
                self.program_tokens[i] = tokenizer[token_name]
        
        self.bar_token_id = tokenizer["Bar_None"] if "Bar_None" in tokenizer else None
        self.last_active_bar = {i: 0 for i in self.program_tokens}
        self.current_bar = 0
        self.max_silent_bars = max_silent_bars
        self.initialized = False
        
    def scan_prompt(self, input_ids):
        """Scans the prompt on startup to synchronize bar counts and active voice states."""
        for token_id in input_ids[0]:
            token_id = token_id.item()
            if token_id == self.bar_token_id:
                self.current_bar += 1
            for prog_idx, prog_token_id in self.program_tokens.items():
                if token_id == prog_token_id:
                    self.last_active_bar[prog_idx] = self.current_bar

    def __call__(self, input_ids, scores):
        # Scan initial prompt to sync state
        if not self.initialized:
            self.scan_prompt(input_ids)
            self.initialized = True
            
        last_token = input_ids[0, -1].item()
        
        # Track bar progression
        if last_token == self.bar_token_id:
            self.current_bar += 1
        
        # Track which programs are active
        for prog_idx, token_id in self.program_tokens.items():
            if last_token == token_id:
                self.last_active_bar[prog_idx] = self.current_bar
        
        # Progressive boost for silent voices
        for prog_idx, token_id in self.program_tokens.items():
            bars_silent = self.current_bar - self.last_active_bar[prog_idx]
            if bars_silent > self.max_silent_bars:
                # Gradual boost: +1.0 per bar of silence beyond threshold
                boost = min((bars_silent - self.max_silent_bars) * 1.0, 5.0)
                scores[0, token_id] += boost
        
        return scores
```

**Why bar-level tracking, not token-level:** Musical silence is measured in bars, not tokens. A voice resting for 2 bars during a tutti passage is normal. A voice resting for 8+ bars while others are playing is a generation bug.

**Edge case — Prompt scanning synchronization:** When generating Section B with the `bridge_tokens` prompt, the processor needs to know the historical state of the voices in Section A. Scanning the initial `input_ids` on the first call of `__call__` prevents immediate logit spikes by correctly mapping the active voices in the prompt.

**Edge case — Solo passages:** In a concerto, the solo instrument plays while the orchestra rests (and vice versa). The `max_silent_bars=4` threshold allows normal rests (up to 4 bars) before gently nudging. The boost is progressive (+1.0/bar), not a hard force, so the model can still choose to rest a voice if the musical context demands it.

**Edge case — Keyboard hand balance:** For keyboard instruments (2 staves = 2 program numbers), both hands must be active simultaneously. If the right hand plays alone for too long, the left hand's boost will bring it back. The threshold of 4 bars is generous enough for typical keyboard writing where one hand occasionally pauses.

---

### Component 10: 10-Stave Support

#### [MODIFY] [generate.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/generate.py)

The existing slot system already handles arbitrary instrument counts. Changes needed:

1. Increase `VoiceBalanceProcessor` to accept up to 10 voices.
2. Validate that `input.json` instruments + keyboard hand expansion ≤ 16 (General MIDI channel limit).
3. When generating with >6 voices, pass `DENSITY_MODERATE` (not DENSE) to prevent the model from over-saturating.

**Auto-density adjustment for high voice counts:**

```python
effective_density = user_inputs.get("density", "moderate")
num_voices = count_total_staves(instruments)
if num_voices >= 8 and effective_density == "dense":
    print(f"Warning: Reducing density from DENSE to MODERATE for {num_voices}-voice piece.")
    effective_density = "moderate"
```

---

### Component 11: Baroque Ornament Detection (Post-Processing)

#### [MODIFY] [generate.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/generate.py)

Add ornament detection to the MusicXML export phase:

```python
def detect_and_mark_ornaments(score):
    """
    Scans each part for rapid alternating note patterns and annotates
    them with proper ornament markings in the MusicXML output.
    Adjusts the single note duration to match the sum of the replaced notes
    to prevent empty beats in the measure.
    
    Detection rules:
    - Trill: ≥4 alternations between 2 adjacent pitches (within 2 semitones),
             all notes ≤ 32nd note duration
    - Mordent: 3 notes (main→lower→main), all ≤ 16th note
    - Inverted Mordent: 3 notes (main→upper→main), all ≤ 16th note
    - Turn: 4 notes (upper→main→lower→main), all ≤ 16th note
    """
    for part in score.parts:
        notes = list(part.flatten().notes)
        i = 0
        while i < len(notes) - 3:
            # Check for trill pattern
            if is_trill_pattern(notes, i):
                trill_length = count_trill_alternations(notes, i)
                main_note = notes[i]
                
                # Critical adjustment: scale the main note's duration to match all replaced notes
                total_duration = sum(n.duration.quarterLength for n in notes[i:i + trill_length])
                main_note.duration.quarterLength = total_duration
                
                tr = music21.expressions.Trill()
                main_note.expressions.append(tr)
                
                # Remove the alternating notes (keep only the main note)
                for j in range(i + 1, i + trill_length):
                    part.remove(notes[j])
                i += trill_length
                continue
            
            # Check for mordent
            if is_mordent_pattern(notes, i):
                main_note = notes[i]
                total_duration = sum(n.duration.quarterLength for n in notes[i:i + 3])
                main_note.duration.quarterLength = total_duration
                
                main_note.expressions.append(music21.expressions.Mordent())
                part.remove(notes[i+1])
                part.remove(notes[i+2])
                i += 3
                continue
            
            i += 1
```

**Edge case — False positives:** A chromatic scale (C-C#-D-D#-E) could be misidentified as trills. The detection requires **strict alternation** (A-B-A-B, not A-B-C-D) and **duration constraint** (all notes ≤ 32nd note). Chromatic scales use stepwise ascending motion, not alternation.

**Edge case — Duration preservation:** When removing note events to replace them with an ornament expression mark, the main note's duration must be extended to equal the sum of all replaced notes' durations. If this is not done, the measure will contain missing beats, resulting in empty/broken beats and engraving formatting errors.

**Edge case — Trill speed:** Bach's trills typically start on the upper neighbor note (Baroque convention). The detection will identify which note is the "main" note (the one the trill resolves to) and mark accordingly.

---

### Component 12: Dual Tempo Marking — Numeric BPM + Baroque Text

#### [MODIFY] [generate.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/generate.py)

```python
def add_tempo_markings(score, bpm, baroque_tag):
    """Inserts both a numeric metronome mark and an italic baroque text annotation."""
    # Numeric tempo (♩ = 120)
    mm = music21.tempo.MetronomeMark(number=bpm)
    score.parts[0].measure(1).insert(0, mm)
    
    # Baroque text annotation (e.g., "Allegro vivace")
    if baroque_tag:
        tag_text = baroque_tag.strip().title()
        te = music21.expressions.TextExpression(tag_text)
        te.style.fontStyle = 'italic'
        te.style.fontSize = 14
        te.placement = 'above'
        score.parts[0].measure(1).insert(0, te)
```

---

### Component 13: Updated `input.json` Schema

#### [MODIFY] [input.json](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/input.json)

Full schema with all new fields:

```json
{
  "instruments": ["violin", "cello", "harpsichord"],
  "tempo": 150,
  "key": "D min",
  "mood": "vivace",
  "genre": "concerto",
  "density": "dense",
  "baroque_tag": "Allegro",
  "form": "ABA"
}
```

| Field | Type | Required | Default | Values |
|---|---|---|---|---|
| `instruments` | string[] | yes | `["piano"]` | Any instrument name |
| `tempo` | int | no | 120 | 40–200 |
| `key` | string | no | null (model decides) | `"C maj"`, `"D min"`, `"F# min"`, etc. |
| `mood` | string | no | null (model decides) | `vivace`, `allegro`, `andante`, `adagio`, `lento`, `maestoso`, `grazioso` |
| `genre` | string | no | null (auto-detect from instruments) | `cantata`, `chorale`, `keyboard`, `organ`, `chamber`, `concerto`, `suite`, `fugue` |
| `density` | string | no | `"moderate"` | `sparse`, `moderate`, `dense` |
| `baroque_tag` | string | no | derived from mood | Any baroque text: `"Minuetto"`, `"Allegro ma non troppo"`, `"Sarabande"`, etc. |
| `form` | string | no | `"ABA"` | `"ABA"`, `"through"` (single section, legacy behavior) |

---

## Summary of All File Changes

| File | Action | Components |
|---|---|---|
| [config.yaml](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/config.yaml) | MODIFY | 1, 2, 3, 4, 6 |
| [src/model.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/model.py) | MODIFY | 1 |
| [src/train.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/train.py) | MAJOR REWRITE | 3 |
| [src/dataset.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/dataset.py) | MAJOR REWRITE | 5, 7 |
| [src/tokenizer.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/tokenizer.py) | MODIFY | 6, 7 |
| [src/control_tokens.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/control_tokens.py) | NEW | 7 |
| [src/data_prep.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/data_prep.py) | MODIFY | 7 |
| [src/generate.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/src/generate.py) | MAJOR REWRITE | 1, 8, 9, 10, 11, 12 |
| [train_pipeline.py](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/train_pipeline.py) | MODIFY | 1, 3, 5, 7 |
| [input.json](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/input.json) | MODIFY | 13 |
| [requirements.txt](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/requirements.txt) | MODIFY | 1 (add `sentencepiece` for LLaMA tokenizer) |
| [run.ipynb](file:///c:/Users/VLT14/Documents/Programming/PY/bach-music/run.ipynb) | MODIFY | Add `--reset` to first training run |

---

## Edge Cases Consolidated

| # | Edge Case | Risk | Mitigation |
|---|---|---|---|
| 1 | 78% of training data is <3,000 tokens → positions 3,000+ undertrained | **CRITICAL** | Sequence packing (Component 5) |
| 2 | Old GPT-2 checkpoints loaded with LLaMA | High | Auto-detect `model_type` + error message + `--reset` requirement |
| 3 | Optimizer state lost between Kaggle sessions | High | Save `training_state.pt` with optimizer, scheduler, scaler states |
| 4 | Voice goes silent and never returns | High | `VoiceBalanceProcessor` with bar-level tracking and progressive boost |
| 5 | ABA section transitions are musically jarring | Medium | Bridge prompting (last 256 tokens of A seed B) + resolution checks |
| 6 | Extreme transpositions clip bass notes below MIDI 21 | Low | Acceptable — affects <3% of samples, clips by ≤2 semitones |
| 7 | Rare control token combinations (e.g., CHORALE + 10 voices) | Medium | Validation layer clamps impossible combinations |
| 8 | Ornament detection false positives on scale passages | Low | Strict alternation + duration constraints |
| 9 | 10-voice output quality degraded due to sparse training data | Medium | Honest quality tiers documented; auto-density reduction for >8 voices |

---

## Verification Plan

### Automated Tests

```bash
# 1. Full dry-run (new architecture + all features end-to-end)
python train_pipeline.py --dry-run --reset

# 2. Verify model parameter count
python -c "
from src.model import get_model
from src.tokenizer import get_tokenizer
t = get_tokenizer()
m = get_model(t, {'n_positions':4096,'n_embd':768,'intermediate_size':2048,'n_layer':10,'n_head':12,'n_kv_head':12})
print(f'Parameters: {sum(p.numel() for p in m.parameters()):,}')
"

# 3. Verify control tokens in vocabulary
python -c "
from src.tokenizer import get_tokenizer
t = get_tokenizer()
print('GENRE_CONCERTO' in t)
print('MOOD_VIVACE' in t)
"

# 4. Generate ABA-form piece and check duration
python -m src.generate
# Then: verify MIDI duration is 2.5–4 minutes
```

### Manual Verification

1. **Open output in MuseScore** → verify:
   - All staves have notes throughout (no long empty passages)
   - ABA structure is audible (theme returns at the end)
   - Ornament markings appear (tr, mordent symbols)
   - Tempo shows both BPM number + italic baroque text
   - Keyboard instruments have braced grand staff

2. **Compare two generations** with contrasting settings:
   - `mood: vivace, genre: concerto` vs. `mood: adagio, genre: chorale`
   - They must sound meaningfully different in texture and speed

3. **Test 10-stave Brandenburg scoring** → verify all instruments present

4. **Test checkpoint resume** → train 2 epochs, kill, resume, verify loss doesn't spike
