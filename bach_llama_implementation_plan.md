# Bach AI v3 — Comprehensive Fix Plan

> Supersedes v2. This document is the single source of truth for the overhaul.
> Status: **APPROVED — implementation in progress.**

---

## 1. Problem Statement (v3)

### What v2 delivered
A custom 75M-parameter LLaMA trained from scratch on 1,343 Bach pieces × 12 transpositions (16,116 files), 4096 context, REMI tokenizer with BPE, control tokens (genre/mood/density/voices/tempo/tag), ABA generation, voice-balance processor, ornament detection.

### What v3 must fix
An epoch-8 generation was analyzed and is musically broken:

| Symptom | Evidence |
|---------|----------|
| Mutant 12-pitch "chords" | Violin part: 318 chords, e.g. `[F#.F#.A.F#.A.F#.A.A.F#.F#.A.A]` |
| Repetition loops | D6 repeated 7× consecutively, A4 repeated 4× |
| Wrong key | Asked D minor in `input.json`, got D major (corr 0.86) |
| Voice imbalance | Violin 483 events, oboe 649, harpsichord 37 / 13 |
| Time-sig corruption | 4/4 ×4 then 12/8 appearing mid-piece after ABA merge |
| Overfitting | Train 1.24 / Val 2.23 by epoch 4 (gap still growing at epoch 8) |
| Lost best_model | Only `epoch_8/` exists; `best_model/` and `training_state.pt` missing |

### Root causes (ranked by impact)

| # | Root cause | File:line | Evidence |
|---|-----------|-----------|----------|
| 1 | `PackedMusicDataset` concatenates pieces with only BOS/EOS; attention leaks across pieces → model learns to mash unrelated music | `dataset.py:67-81` | 12-pitch stacked "chords" are the symptom |
| 2 | Zero dropout on 75M model | `model.py:14-28`, config.json `attention_dropout: 0.0` | Train/val gap by epoch 4 |
| 3 | 12× transposition = near-duplicate data → memorization | `config.yaml:46` | Repetition loops |
| 4 | `repetition_penalty: 1.0` (disabled), `temperature: 0.72` (too low) | `config.yaml:52-53` | Note repetition runs |
| 5 | No key/mode control token; post-transposition cannot change major↔minor | `control_tokens.py:5-16` | D-minor request → D major |
| 6 | ABA merge appends A' by raw tick shift; uses only first TS for alignment | `generate.py:464-472,560-583` | 4/4→12/8 corruption |
| 7 | Track→instrument mapping is positional, not by register | `generate.py:847-883` | Harpsichord starved |
| 8 | No chord-stacking guard; `use_chords:true` happily groups 12 NoteOns | `config.yaml:16` | 318 chords in violin |
| 9 | `track.program = idx` in data_prep destroys real register identity — track order in MIDI is arbitrary | `data_prep.py:58` | Model can't learn register→voice mapping |
| 10 | `training_state.pt` never saved; `best_model` overwritten every epoch because `best_val_loss` resets to inf on resume | `train.py:45-57` | Only epoch_8 exists |

---

## 2. Approved Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Retrain from scratch? | **YES** | All Phase 1 changes require fresh weights |
| Key/mode control tokens? | **YES** — add `MODE_MAJOR`, `MODE_MINOR` | Only way to make minor-key generation work |
| Packing fix? | **Option A2** — single-piece samples + dynamic per-batch padding + length-grouped sampler | 100% kills cross-piece bleed; 5-8× faster than fixed 4096 padding |
| DDP for both T4s? | **YES** — `torchrun` launch via `train_ddp.py` | Halves Kaggle training time |
| Immediate generation fixes? | **YES** — apply to existing epoch_8 first | Quick win while retrain runs |
| Fix `track.program = idx`? | **YES** — sort tracks by average pitch (highest→lowest) before assigning program idx | Gives consistent register→voice mapping without fragile real-program remapping |

---

## 3. Deep Edge-Case Analysis

### 3.1 Long pieces (>4096 tokens, ~6% of dataset)
**Risk:** truncating loses concerto/suite data; naive splitting creates orphan fragments.
**Decision:** Split into 4096-token chunks with **256-token overlap**. Each chunk gets `BOS + control_prefix + tokens + EOS`. Drop any chunk where real tokens <25% of 4096 (avoids near-empty padded chunks from a 4097-token piece).
**Edge case:** overlap creates near-duplicate boundaries. Acceptable — the model benefits from seeing long contexts and the duplication is small.

### 3.2 Dynamic padding + gradient accumulation
**Risk:** micro-batches of different lengths make loss normalization wrong.
**Decision:** Normalize loss by the **number of non-pad target tokens** in each micro-batch (not by batch size). This is the mathematically correct per-token cross-entropy. Implemented via a custom `DataCollator` that returns `labels` with `-100` on PAD and a `loss_scale` count.
**Edge case:** with `DistributedSampler`, lengths may be skewed across ranks. Use `shuffle=False` after length-sorting so both ranks see similar length distributions.

### 3.3 Mode detection reliability
**Risk:** many Bach MIDI files have no key signature; filename heuristic is unreliable.
**Decision:** Priority chain: (1) symusic `key_signatures[0].tonality` if present, (2) filename heuristic ("minor", " min", "_m_"), (3) default `MAJOR`. Log how many fall to each tier. Do NOT run music21 Krumhansl analysis on 16k files (too slow).
**Edge case:** ~30% of files may default to MAJOR incorrectly. The model still learns mode patterns from the 70% correctly labeled — acceptable noise.

### 3.4 Track sorting by register (`track.program = idx` fix)
**Risk:** real MIDI program numbers are inconsistent (violin as piano, cello as ensemble, etc.).
**Decision:** Sort tracks by **average pitch (descending)** before assigning `program = idx`. This guarantees:
- `Program_0` = highest voice (soprano / violin / right hand)
- `Program_1` = next (alto / viola)
- `Program_2` = next (tenor / cello / left hand)
- `Program_3` = lowest (bass)

At generation, the track→instrument mapper also sorts by average pitch, so the highest generated track correctly maps to the user's highest instrument (violin), etc.
**Edge case:** a track with 0 notes has avg pitch 0 → sorts last. Correct (empty track = lowest priority).
**Edge case:** drum channel (program 128 / channel 9). Skip drum tracks entirely (Bach has no drums).

### 3.5 DDP on Kaggle
**Risk:** Kaggle notebooks are single-process; `torchrun` needs subprocess launch.
**Decision:** Create `train_ddp.py` wrapper. Launch via `!python -m torch.distributed.run --nproc_per_node=2 train_ddp.py` in a notebook cell. If `LOCAL_RANK` env var is absent, fall back to single-GPU mode automatically.
**Edge case:** DDP + gradient accumulation. Use `model.no_sync()` context for accum sub-steps to avoid redundant all-reduces. Only the final sub-step syncs.
**Edge case:** only `rank==0` saves checkpoints, prints progress, runs validation aggregation.
**Edge case:** `DistributedSampler` must call `.set_epoch(epoch)` each epoch or all epochs see the same order.

### 3.6 `no_repeat_ngram_size` vs. legitimate baroque repetition
**Risk:** baroque music legitimately repeats short motifs (sequences, imitations). Too-small n-gram block kills musicality.
**Decision:** `no_repeat_ngram_size: 8` (≈1-2 beats of tokens). Configurable. At 8 tokens, a 1-beat motif can repeat but a full 2-beat exact copy is blocked.
**Edge case:** with `do_sample=True`, HF's implementation scans the generated sequence. Verified compatible.

### 3.7 ChordGuardProcessor threshold
**Risk:** a legit 4-voice chord has 4 Pitch tokens at one Position. Threshold of 4 would block valid SATB.
**Decision:** Threshold = **6**. Allows 5-voice chords (rare but valid in Bach). At 6+ Pitch tokens at one Position, boost the next Position/Bar/Rest token by +4.0.
**Edge case:** need to track Position token IDs and Pitch token IDs (tokenizer-dependent). Build maps in `__init__` by scanning `tokenizer.vocab`.
**Edge case:** the processor must reset its note counter when a new Position or Bar token is emitted.

### 3.8 ABA time-signature corruption
**Risk:** `get_bar_aligned_ticks` uses `score.time_signatures[0]` but the model may emit multiple TS tokens.
**Decision:** After decoding each section, **strip all TimeSig tokens** from the symusic Score. Apply a single user-specified (or default 4/4) TimeSig at tick 0. This guarantees both A and A' use the same bar grid before merge.
**Edge case:** user wants a mid-piece TS change. Out of scope for v3 — enforce single TS. Add `time_signature` field to `input.json` (default "4/4").

### 3.9 Dropout in LLaMA + AMP
**Risk:** HF LLaMA applies `attention_dropout` but `hidden_dropout` support varies by version.
**Decision:** Set `attention_dropout=0.1`. For hidden dropout, add a custom `nn.Dropout(0.1)` after the embedding layer and after each residual block (monkey-patch or subclass). Verify with a forward pass that dropout is active in `model.train()` mode.
**Edge case:** dropout + FP16 AMP can cause numerical issues with some ops. PyTorch handles this — dropout runs in the autocast dtype. Verified safe.

### 3.10 Weight decay on biases/norms
**Risk:** applying weight decay to biases and LayerNorm parameters destabilizes training.
**Decision:** Use parameter groups in AdamW: group 0 = weights (decay 0.05), group 1 = biases + norm weights (decay 0.0). Standard practice.

### 3.11 Resume without optimizer state
**Risk:** if `training_state.pt` is missing but `epoch_N` exists, optimizer momentum is lost → loss spike for ~200 steps.
**Decision:** Warn loudly. Optionally re-run 5% warmup on resume. Do not silently continue — the user must know.

### 3.12 `best_model` safety
**Risk:** if `best_val_loss` resets to `inf` on resume, every epoch overwrites `best_model`.
**Decision:** Always restore `best_val_loss` from `training_state.pt`. If `training_state.pt` is missing, scan existing `epoch_N/config.json` + a sidecar `epoch_N/val_loss.json` to reconstruct the best epoch. Save `val_loss.json` next to every epoch checkpoint.

### 3.13 5 transpositions — key coverage
**Decision:** `[-2, 0, 2, -3, 5]` → a C-major piece becomes C, Bb, D, A, F. Covers 5 distinct keys spanning the circle of fifths. 1343 × 5 = 6,715 files.
**Edge case:** `+5` transposition may push high notes above 109. Tokenizer clips. Affects <2% of pieces. Acceptable.

### 3.14 Early stopping
**Risk:** even with dropout, 12 epochs may overfit on 6,715 files.
**Decision:** `num_epochs: 12` with **early stopping patience = 4**. If val loss doesn't improve for 4 consecutive epochs, stop. Always keep `best_model`.

### 3.15 Velocity token handling in chord guard
**Risk:** REMI encodes a note as `Pitch → Velocity → Duration`. The chord guard must count Pitch tokens, not Velocity/Duration.
**Decision:** Build a set of Pitch token IDs from the tokenizer vocab (tokens matching `Pitch_*`). Only count these toward the per-Position note limit.

---

## 4. Implementation Plan

### Phase 0 — Immediate Generation Fixes (no retrain, test on epoch_8)

| File | Change | Edge case handled |
|------|--------|-------------------|
| `config.yaml:52-54` | `temperature 0.72→0.95`, `top_p 0.92→0.9`, `top_k 30→40`, `repetition_penalty 1.0→1.15` | 3.6 |
| `generate.py:449-461` (both ABA + legacy paths) | add `no_repeat_ngram_size=8` to `model.generate()` | 3.6 |
| `generate.py` (new class) | `ChordGuardProcessor`: count Pitch tokens per Position; at ≥6, boost Position/Bar/Rest logits +4.0 | 3.7, 3.15 |
| `generate.py:464-472` | `get_bar_aligned_ticks`: strip all TimeSig from both A and A' before merge; apply single TS at tick 0 | 3.8 |
| `generate.py:847-883` | track→instrument mapping by average pitch (highest→first requested instrument) instead of positional | 3.4 |
| `generate.py` (post-decode) | `filter_degenerate_chords`: any chord with >4 notes or duplicate pitch-classes → keep 4 highest-pitch notes | safety net |
| `generate.py:186-246` | `VoiceBalanceProcessor`: `max_silent_bars 4→2`, cap boost at 3.0 (was 5.0) | gentler nudge |
| `generate.py:973` | default `--model_path`: try `best_model`, fall back to latest `epoch_N` with warning | 3.12 |

### Phase 1 — Data & Model Retrain (requires `--reset`)

| File | Change | Edge case |
|------|--------|-----------|
| `model.py:14-28` | add `attention_dropout=0.1`; add custom `nn.Dropout(0.1)` after embedding + residual blocks | 3.9 |
| `control_tokens.py:5-16` | add `MODE_MAJOR`, `MODE_MINOR` (vocab → 43 control tokens) | — |
| `control_tokens.py:114-205` | `analyze_piece` emits `mode` field; `get_control_prefix` appends `MODE_*` | 3.3 |
| `data_prep.py:55-60` | **sort tracks by avg pitch desc**, skip drum tracks, then `track.program = idx` | 3.4 |
| `data_prep.py:64-70` | write `mode` into `.control.json` sidecar | 3.3 |
| `dataset.py` (rewrite) | `SinglePieceDataset`: one piece per sample; split long pieces into 4096-chunks with 256-overlap; drop <25%-real chunks | 3.1 |
| `dataset.py` (new) | `DynamicDataCollator`: pad to batch-max; labels=-100 on PAD; return non-pad token count | 3.2 |
| `dataset.py` (new) | `LengthGroupedSampler`: sort by length, form uniform-length batches, distribute across DDP ranks | 3.2 |
| `dataset.py:43-64` | read `mode` from sidecar, emit `MODE_*` token in prefix | 3.3 |
| `config.yaml:46` | `transposition_keys: [-2, 0, 2, -3, 5]` (5 keys) | 3.13 |
| `config.yaml:42` | `weight_decay: 0.01→0.05` | 3.10 |
| `config.yaml:41` | `num_epochs: 20→12`, add `early_stopping_patience: 4` | 3.14 |
| `config.yaml` (new) | `time_signature: "4/4"` default | 3.8 |
| `tokenizer.py` | BPE target unchanged (base + 2000); verify new control tokens register | — |

### Phase 2 — DDP & Speed

| File | Change | Edge case |
|------|--------|-----------|
| `train_ddp.py` (new) | thin wrapper: `torch.distributed.init`, set device to `local_rank`, call `train_model` | 3.5 |
| `train.py:14-16` | init process group if `LOCAL_RANK` env set; single-GPU fallback if not | 3.5 |
| `train.py:19-21` | **remove `gradient_checkpointing_enable()`** (75M fits in 15GB T4 without it; ~25% faster) | — |
| `train.py` | wrap model in `DistributedDataParallel`; use `model.no_sync()` for accum sub-steps | 3.5 |
| `train.py` | use `DistributedSampler` on train/val; call `.set_epoch(epoch)` | 3.5 |
| `train.py:94-98` | loss normalization by non-pad token count (from collator) | 3.2 |
| `train.py:135-157` | save checkpoints only on `rank==0`; save `val_loss.json` sidecar | 3.12 |
| `train.py:45-57` | save `training_state.pt` every epoch; restore `best_val_loss` on resume | 3.11 |
| `train.py` (new) | early stopping: track `epochs_no_improve`; break if > patience | 3.14 |
| `train.py` (optimizer) | parameter groups: weights decay 0.05, biases/norms decay 0.0 | 3.10 |
| `config.yaml:38-39` | `batch_size: 4→8`, `gradient_accumulation_steps: 8→4` (effective batch 32 unchanged) | — |
| `run.ipynb` | launch cell: `!python -m torch.distributed.run --nproc_per_node=2 train_ddp.py --reset` | 3.5 |

### Phase 3 — Checkpoint & Pipeline Hygiene

| File | Change | Edge case |
|------|--------|-----------|
| `train.py:45-57` | verify `training_state.pt` saves optimizer+scheduler+scaler+epoch+best_val_loss | 3.11 |
| `train.py` (resume) | if `training_state.pt` missing but `epoch_N` exists: warn, reconstruct `best_val_loss` from sidecars, re-run 5% warmup | 3.11, 3.12 |
| `train_pipeline.py:79-96` | on resume, prefer `best_model` if no `training_state.pt` | 3.12 |
| `generate.py:973` | default path: `best_model` → lowest-val-loss `epoch_N` → latest `epoch_N` (warn) | 3.12 |

---

## 5. File Change Summary

| File | Action | Phase |
|------|--------|-------|
| `config.yaml` | MODIFY | 0, 1, 2 |
| `src/model.py` | MODIFY | 1 |
| `src/control_tokens.py` | MODIFY | 1 |
| `src/data_prep.py` | MODIFY | 1 |
| `src/dataset.py` | MAJOR REWRITE | 1 |
| `src/tokenizer.py` | VERIFY | 1 |
| `src/train.py` | MAJOR REWRITE | 2 |
| `train_ddp.py` | NEW | 2 |
| `src/generate.py` | MAJOR REWRITE | 0, 3 |
| `train_pipeline.py` | MODIFY | 3 |
| `run.ipynb` | MODIFY | 2 |
| `input.json` | ADD `time_signature` field | 0 |

---

## 6. Execution Order

1. **Phase 0** (generation fixes) — apply first, regenerate on existing `epoch_8`, let user hear the difference. ~30 min of work.
2. **Phase 1 + 2 + 3** (data + model + DDP + hygiene) — implement together, then `--reset` retrain on Kaggle. Estimated retrain time: **~2-3 hours** on 2×T4 with DDP (5× augmentation, dynamic padding, no grad checkpointing, 12 epochs with early stop).
3. **Verify** — run dry-run, then full train, then generate and listen.

---

## 7. Verification Plan

### 7.1 Automated
```bash
# Dry run (validates all code paths end-to-end)
python train_pipeline.py --dry-run --reset

# DDP dry run
python -m torch.distributed.run --nproc_per_node=2 train_ddp.py --dry-run --reset

# Verify dropout active
python -c "from src.model import get_model; m=get_model(...); print(sum(p.requires_grad for p in m.parameters())); import torch; m.train(); ..."

# Verify mode tokens in vocab
python -c "from src.control_tokens import CONTROL_TOKENS; print('MODE_MAJOR' in CONTROL_TOKENS, 'MODE_MINOR' in CONTROL_TOKENS)"

# Verify track sorting
python -c "from src.data_prep import transpose_midi; ..."  # check programs are pitch-sorted

# Generate and check duration / chord count
python -m src.generate
python analyze_output.py  # existing analysis script
```

### 7.2 Manual
1. Open `generated_bach.mid` in MuseScore:
   - No mutant chords (>4 notes)
   - No repetition loops
   - Key matches `input.json` mode (minor if requested)
   - All instruments have notes (no starved staves)
   - Single time signature throughout
2. Compare two generations: `mood: vivace, genre: concerto` vs `mood: adagio, genre: chorale` — must differ in texture/speed.
3. Check train/val gap: val loss should stay within ~0.3 of train loss through epoch 8.
4. Check checkpoint resume: train 2 epochs, kill, resume, verify no loss spike.

---

## 8. Expected Outcomes

| Metric | v2 (epoch 8) | v3 (target) |
|--------|--------------|-------------|
| Train/val gap | 1.0 (1.24 vs 2.23) | <0.3 |
| Max chord size | 12 pitches | ≤4 |
| Repetition runs (≥3) | 5+ per piece | 0-1 |
| Key mode accuracy | 0% (D maj instead of D min) | >85% |
| Voice balance (min/max events) | 37 / 649 | >150 / <500 |
| Time signature stability | corrupt (4/4→12/8) | single TS |
| Training time (Kaggle) | 30h, single GPU | 2-3h, dual T4 DDP |
