import os
import random
from pathlib import Path
from torch.utils.data import DataLoader
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training

def prepare_dataset_loaders(files_paths, tokenizer, max_seq_len, batch_size, val_split=0.1):
    """
    Splits MIDI files into chunks of max_seq_len, separates them into train and validation sets,
    and returns PyTorch DataLoader objects.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chunks_dir = os.path.join(base_dir, "processed", "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    
    # 1. Clean existing chunks to prevent stale files
    for f in Path(chunks_dir).glob("*.mid"):
        try:
            f.unlink()
        except OSError:
            pass
            
    print(f"Splitting {len(files_paths)} files into chunks of max length {max_seq_len}...")
    # 2. Split MIDI files into chunks
    path_objects = [Path(p) for p in files_paths]
    split_files_for_training(
        files_paths=path_objects,
        tokenizer=tokenizer,
        save_dir=Path(chunks_dir),
        max_seq_len=max_seq_len
    )
    
    # 3. Retrieve all split chunks
    chunk_paths = list(Path(chunks_dir).glob("*.mid"))
    print(f"Generated {len(chunk_paths)} music chunks for training.")
    
    if len(chunk_paths) == 0:
        raise ValueError("No music chunks generated. Ensure your training MIDI files are valid and contain notes.")
        
    # 4. Shuffle and split into Train / Validation sets
    random.seed(42)
    random.shuffle(chunk_paths)
    
    val_size = int(len(chunk_paths) * val_split)
    # Ensure there is at least 1 file in validation if split > 0
    if val_size == 0 and len(chunk_paths) > 1 and val_split > 0:
        val_size = 1
        
    val_paths = chunk_paths[:val_size]
    train_paths = chunk_paths[val_size:]
    
    print(f"Dataset split: {len(train_paths)} training chunks, {len(val_paths)} validation chunks.")
    
    # 5. Extract special token IDs safely
    pad_token_id = tokenizer["PAD_None"] if "PAD_None" in tokenizer else tokenizer.pad_token_id
    bos_token_id = tokenizer["BOS_None"] if "BOS_None" in tokenizer else None
    eos_token_id = tokenizer["EOS_None"] if "EOS_None" in tokenizer else None
    
    # 6. Create DatasetMIDI objects
    train_dataset = DatasetMIDI(
        files_paths=train_paths,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id
    )
    
    val_dataset = DatasetMIDI(
        files_paths=val_paths,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id
    ) if len(val_paths) > 0 else None
    
    # 7. Create DataCollator for batch assembly
    collator = DataCollator(
        pad_token_id=pad_token_id,
        copy_inputs_as_labels=True
    )
    
    # 8. Create PyTorch Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collator
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collator
    ) if val_dataset else None
    
    return train_loader, val_loader
