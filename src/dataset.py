import os
import random
import re
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from src.control_tokens import get_control_prefix

class PackedMusicDataset(Dataset):
    """
    Packs multiple short tokenized MIDI sequences into fixed-length 
    training sequences to ensure all positions up to max_seq_len are trained.
    """
    def __init__(self, files_paths, tokenizer, max_seq_len, bos_token_id, eos_token_id, pad_token_id):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        self.packed = []
        buffer = []
        
        # Load and tokenize all MIDI files in memory
        for filepath in files_paths:
            try:
                # 1. Load and tokenize the MIDI file
                tokens_seq = tokenizer(Path(filepath))
                if isinstance(tokens_seq, list):
                    if len(tokens_seq) > 0:
                        ids = tokens_seq[0].ids
                    else:
                        continue
                else:
                    ids = tokens_seq.ids
                
                # 2. Get control prefix tokens
                basename = os.path.basename(filepath)
                original_base = re.sub(r"_transposed_[-+]?\d+$", "", os.path.splitext(basename)[0])
                control_json_path = os.path.join(os.path.dirname(filepath), f"{original_base}.control.json")
                
                control_prefix_ids = []
                vocab_offset = len(tokenizer)
                if os.path.exists(control_json_path):
                    try:
                        with open(control_json_path, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                        from src.control_tokens import CONTROL_TOKENS
                        prefix_tokens = get_control_prefix(metadata)
                        for t in prefix_tokens:
                            if t in CONTROL_TOKENS:
                                control_prefix_ids.append(vocab_offset + CONTROL_TOKENS.index(t))
                    except Exception:
                        pass
                
                # Fallback if no controls
                if not control_prefix_ids:
                    fallback_tokens = ["GENRE_KEYBOARD", "MOOD_ANDANTE", "DENSITY_MODERATE", "V2", "TEMPO_MEDIUM"]
                    from src.control_tokens import CONTROL_TOKENS
                    for t in fallback_tokens:
                        if t in CONTROL_TOKENS:
                            control_prefix_ids.append(vocab_offset + CONTROL_TOKENS.index(t))
                
                # Assemble sequence: [BOS] + control_prefix + ids + [EOS]
                entry = []
                if self.bos_token_id is not None:
                    entry.append(self.bos_token_id)
                entry.extend(control_prefix_ids)
                entry.extend(ids)
                if self.eos_token_id is not None:
                    entry.append(self.eos_token_id)
                
                # Add to buffer
                buffer.extend(entry)
                
                # Pack buffer
                while len(buffer) >= self.max_seq_len:
                    self.packed.append(buffer[:self.max_seq_len])
                    buffer = buffer[self.max_seq_len:]
            except Exception:
                pass
                
        # Don't waste short tails
        if len(buffer) > self.max_seq_len // 4:
            padded_buffer = buffer + [self.pad_token_id] * (self.max_seq_len - len(buffer))
            self.packed.append(padded_buffer)
            
    def __len__(self):
        return len(self.packed)
        
    def __getitem__(self, idx):
        ids = self.packed[idx]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor([1 if t != self.pad_token_id else 0 for t in ids], dtype=torch.long),
            "labels": torch.tensor(ids, dtype=torch.long)
        }

def prepare_dataset_loaders(files_paths, tokenizer, max_seq_len, batch_size, val_split=0.1):
    """
    Splits files list into train and validation sets, tokenizes and packs them in memory,
    and returns PyTorch DataLoader objects.
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
    
    train_dataset = PackedMusicDataset(
        files_paths=train_paths,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id
    )
    
    val_dataset = PackedMusicDataset(
        files_paths=val_paths,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id
    ) if len(val_paths) > 0 else None
    
    print(f"Packed dataset: {len(train_dataset)} training chunks, {len(val_dataset) if val_dataset else 0} validation chunks.")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    ) if val_dataset else None
    
    return train_loader, val_loader

