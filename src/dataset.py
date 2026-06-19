import os
import random
import re
import json
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
from src.control_tokens import get_control_prefix, CONTROL_TOKENS


class SinglePieceDataset(Dataset):
    """
    One piece per sample. Long pieces are split into overlapping chunks.
    Each chunk is: [BOS] + control_prefix + tokens + [EOS], padded to max_seq_len.
    """
    def __init__(self, files_paths, tokenizer, max_seq_len, bos_token_id, eos_token_id, pad_token_id,
                 overlap=256, min_real_ratio=0.25, rank=0, world_size=1):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.vocab_size = len(tokenizer)
        self.overlap = min(overlap, max_seq_len // 2) # Cap overlap to prevent infinite loops when max_seq_len is small (e.g. in dry-run)
        self.min_real_ratio = min_real_ratio
        self.samples = []
        self.tokenizer = tokenizer

        import hashlib
        import pickle

        cache_file = None
        if len(files_paths) > 0:
            cache_dir = os.path.dirname(files_paths[0])
            if os.path.exists(cache_dir):
                hasher = hashlib.md5()
                # Include tokenizer vocab size and all dataset parameters
                hasher.update(
                    f"{max_seq_len}_{self.vocab_size}_{bos_token_id}_{eos_token_id}_{pad_token_id}_{self.overlap}_{self.min_real_ratio}".encode()
                )
                # Include full file paths and modification times for cache invalidation
                for p in sorted(files_paths, key=str):
                    hasher.update(str(p).encode())
                    try:
                        hasher.update(str(os.path.getmtime(p)).encode())
                    except OSError:
                        pass
                cache_file = os.path.join(cache_dir, f"dataset_cache_{hasher.hexdigest()}.pkl")

        loaded = False
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    self.samples = pickle.load(f)
                loaded = True
                if rank == 0:
                    print(f"Loaded dataset cache from {cache_file} ({len(self.samples)} samples)")
            except Exception as e:
                if rank == 0:
                    print(f"Warning: Failed to load dataset cache {cache_file}: {e}")

        if not loaded:
            if rank == 0:
                print(f"No valid cache found. Tokenizing {len(files_paths)} files...")
                # Limit thread count to prevent context switching overhead
                num_workers = min(16, (os.cpu_count() or 4) * 2)
                
                def worker(fp):
                    try:
                        return self._process_file(Path(fp))
                    except Exception as e:
                        print(f"  Error: Failed to process MIDI file {fp}: {e}")
                        return []
                
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    results = list(executor.map(worker, files_paths))
                
                for chunks in results:
                    self.samples.extend(chunks)
                
                if cache_file:
                    try:
                        with open(cache_file, "wb") as f:
                            pickle.dump(self.samples, f)
                        print(f"Saved dataset cache to {cache_file}")
                    except Exception as e:
                        print(f"Warning: Failed to save dataset cache {cache_file}: {e}")
            
            # Synchronize all ranks: rank 0 must finish saving before other ranks load.
            if dist.is_initialized() and world_size > 1:
                dist.barrier()
            
            # All ranks (including rank 0) load the freshly saved cache to ensure identical samples.
            if cache_file and os.path.exists(cache_file):
                try:
                    with open(cache_file, "rb") as f:
                        self.samples = pickle.load(f)
                    loaded = True
                    if rank != 0:
                        print(f"Rank {rank} loaded dataset cache ({len(self.samples)} samples)")
                except Exception as e:
                    print(f"Rank {rank} warning: Failed to load dataset cache after sync: {e}")
        
        if not loaded:
            # Fallback: tokenize locally if cache could not be created or loaded.
            if rank == 0:
                print(f"Tokenizing {len(files_paths)} files as fallback...")
            for filepath in files_paths:
                try:
                    self.samples.extend(self._process_file(Path(filepath)))
                except Exception as e:
                    print(f"Rank {rank} error: Failed to process MIDI file {filepath}: {e}")

    def _process_file(self, filepath):
        # 1. Tokenize
        tokens_seq = self.tokenizer(filepath)
        if isinstance(tokens_seq, list):
            if len(tokens_seq) > 0:
                ids = tokens_seq[0].ids
            else:
                return []
        else:
            ids = tokens_seq.ids

        # 2. Load control prefix
        basename = os.path.basename(filepath)
        original_base = re.sub(r"_transposed_[-+]?\d+$", "", os.path.splitext(basename)[0])
        control_json_path = os.path.join(os.path.dirname(filepath), f"{original_base}.control.json")

        control_prefix_ids = []
        vocab_offset = len(self.tokenizer)
        if os.path.exists(control_json_path):
            try:
                with open(control_json_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                prefix_tokens = get_control_prefix(metadata)
                for t in prefix_tokens:
                    if t in CONTROL_TOKENS:
                        control_prefix_ids.append(vocab_offset + CONTROL_TOKENS.index(t))
            except Exception as e:
                print(f"  Warning: Failed to load control JSON for {filepath}: {e}")

        if not control_prefix_ids:
            fallback_tokens = ["GENRE_KEYBOARD", "MOOD_ANDANTE", "DENSITY_MODERATE", "V2", "TEMPO_MEDIUM", "MODE_MAJOR"]
            for t in fallback_tokens:
                if t in CONTROL_TOKENS:
                    control_prefix_ids.append(vocab_offset + CONTROL_TOKENS.index(t))

        # 3. Assemble full sequence: BOS + control + ids + EOS
        prefix = []
        if self.bos_token_id is not None:
            prefix.append(self.bos_token_id)
        prefix.extend(control_prefix_ids)
        if self.eos_token_id is not None:
            full_ids = prefix + list(ids) + [self.eos_token_id]
        else:
            full_ids = prefix + list(ids)

        # 4. Chunk with overlap
        chunk_start = 0
        chunks = []
        while chunk_start < len(full_ids):
            chunk_end = min(chunk_start + self.max_seq_len, len(full_ids))
            chunk = full_ids[chunk_start:chunk_end]
            # Count real (non-pad) tokens in the chunk
            real_count = sum(1 for t in chunk if t != self.pad_token_id)
            min_real_tokens = max(1, int(self.max_seq_len * self.min_real_ratio))
            if real_count >= min_real_tokens or (chunk_start == 0 and real_count > 0):
                chunks.append(chunk)
            chunk_start = chunk_end - self.overlap if chunk_end < len(full_ids) else chunk_end
        return chunks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        labels = [
            -100 if (t == self.pad_token_id or t == self.bos_token_id or t >= self.vocab_size) else t
            for t in ids
        ]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor([1 if t != self.pad_token_id else 0 for t in ids], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


class DynamicDataCollator:
    """
    Pads a batch to the maximum length in the batch. Labels are -100 on PAD.
    Returns the number of non-pad target tokens for loss normalization.
    """
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        loss_scale = 0
        for f in features:
            ids = f["input_ids"].tolist()
            attn = f["attention_mask"].tolist()
            labels = f["labels"].tolist()
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
                attn = attn + [0] * pad_len
                labels = labels + [-100] * pad_len
            batch_input_ids.append(ids)
            batch_attention_mask.append(attn)
            batch_labels.append(labels)
            loss_scale += sum(1 for lab in labels if lab != -100)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "loss_scale": torch.tensor(loss_scale, dtype=torch.long)
        }


class LengthGroupedSampler(Sampler):
    """
    Sorts samples by length, creates batches of similar lengths, and distributes across ranks.
    """
    def __init__(self, dataset, batch_size, rank=0, world_size=1, shuffle=False, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        # Precompute lengths
        lengths = [len(s) for s in dataset.samples]
        self.indices = list(range(len(lengths)))
        self.indices.sort(key=lambda i: lengths[i])

    def __iter__(self):
        indices = list(self.indices)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # Shuffle within length-ordered blocks for stability
            block_size = max(self.batch_size * 4, 1)
            blocks = [indices[i:i + block_size] for i in range(0, len(indices), block_size)]
            random.Random(self.seed + self.epoch).shuffle(blocks)
            indices = [i for b in blocks for i in b]

        total_batches = math.ceil(len(indices) / self.batch_size)
        per_rank = (total_batches + self.world_size - 1) // self.world_size

        yielded = 0
        for batch_idx in range(total_batches):
            if batch_idx % self.world_size != self.rank:
                continue
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(indices))
            yield indices[start:end]
            yielded += 1

        # Pad with repeated last batch if this rank has fewer batches than the max
        last_start = (total_batches - 1) * self.batch_size if total_batches > 0 else 0
        last_batch = indices[last_start:last_start + self.batch_size]
        while yielded < per_rank:
            yield last_batch
            yielded += 1

    def __len__(self):
        total_batches = math.ceil(len(self.indices) / self.batch_size)
        per_rank = (total_batches + self.world_size - 1) // self.world_size
        return per_rank

    def set_epoch(self, epoch):
        self.epoch = epoch


def prepare_dataset_loaders(files_paths, tokenizer, max_seq_len, batch_size, val_split=0.1,
                            rank=0, world_size=1):
    """
    Splits files list into train and validation sets, tokenizes them as single-piece samples,
    and returns PyTorch DataLoader objects with dynamic padding and length grouping.
    """
    # Filter files_paths to only include MIDI files (excluding control JSON files)
    midi_files = [Path(f) for f in files_paths if os.path.splitext(str(f))[1].lower() in ['.mid', '.midi']]

    # Group files by their original base name to prevent data leakage
    grouped_files = {}
    for f in midi_files:
        original_base = re.sub(r"_transposed_[-+]?\d+$", "", f.stem)
        if original_base not in grouped_files:
            grouped_files[original_base] = []
        grouped_files[original_base].append(f)

    # Shuffle and split original pieces
    unique_pieces = list(grouped_files.keys())
    random.seed(42)
    sorted_pieces = sorted(unique_pieces)
    random.shuffle(sorted_pieces)

    val_size = int(len(sorted_pieces) * val_split)
    if val_size == 0 and len(sorted_pieces) > 1 and val_split > 0:
        val_size = 1

    val_pieces = sorted_pieces[:val_size]
    train_pieces = sorted_pieces[val_size:]

    val_paths = []
    for p in val_pieces:
        val_paths.extend(grouped_files[p])

    train_paths = []
    for p in train_pieces:
        train_paths.extend(grouped_files[p])

    print(f"Dataset split: {len(train_paths)} training files, {len(val_paths)} validation files.")

    pad_token_id = tokenizer["PAD_None"] if "PAD_None" in tokenizer else tokenizer.pad_token_id
    bos_token_id = tokenizer["BOS_None"] if "BOS_None" in tokenizer else None
    eos_token_id = tokenizer["EOS_None"] if "EOS_None" in tokenizer else None

    train_dataset = SinglePieceDataset(
        files_paths=train_paths,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        rank=rank,
        world_size=world_size
    )

    val_dataset = SinglePieceDataset(
        files_paths=val_paths,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        rank=rank,
        world_size=world_size
    ) if len(val_paths) > 0 else None

    print(f"Single-piece dataset: {len(train_dataset)} training chunks, {len(val_dataset) if val_dataset else 0} validation chunks.")

    collator = DynamicDataCollator(pad_token_id=pad_token_id)

    train_sampler = LengthGroupedSampler(
        train_dataset, batch_size=batch_size, rank=rank, world_size=world_size, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collator
    )

    val_loader = None
    if val_dataset:
        val_sampler = LengthGroupedSampler(
            val_dataset, batch_size=batch_size, rank=rank, world_size=world_size, shuffle=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=collator
        )

    return train_loader, val_loader
