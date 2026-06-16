import os
import sys
import argparse
from src.config import (
    DATASET_DIR, PROCESSED_DIR, CHECKPOINT_DIR, OUTPUT_DIR,
    MODEL_CONFIG, TRAIN_CONFIG, GENERATE_CONFIG
)
from src.data_prep import prepare_dataset, get_midi_files
from src.tokenizer import get_tokenizer
from src.dataset import prepare_dataset_loaders
from src.model import get_model
from src.train import train_model
from src.generate import generate_music

def main():
    parser = argparse.ArgumentParser(description="J.S. Bach Polyphonic Sheet Music Generator Pipeline")
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Run a quick end-to-end dry-run with a tiny model and dataset to verify code correctness."
    )
    args = parser.parse_args()
    
    # Define model and training configs (which can be overridden in dry-run)
    model_config = MODEL_CONFIG.copy()
    train_config = TRAIN_CONFIG.copy()
    generate_config = GENERATE_CONFIG.copy()
    checkpoint_dir = CHECKPOINT_DIR
    
    print("==========================================================")
    print("   J.S. Bach Polyphonic Sheet Music AI Training Pipeline  ")
    print("==========================================================\n")
    
    if args.dry_run:
        print("!!! DRY-RUN MODE ENABLED !!!")
        print("Overriding configurations for a rapid end-to-end test...\n")
        checkpoint_dir = os.path.join(CHECKPOINT_DIR, "dry_run")
        # Override configs for speed
        model_config.update({
            "n_positions": 64,
            "n_embd": 64,
            "n_layer": 2,
            "n_head": 2,
        })
        train_config.update({
            "batch_size": 2,
            "num_epochs": 1,
        })
        generate_config.update({
            "max_length": 64,
        })
        
    # Phase 1: Data Preparation & Augmentation
    print("--- Phase 1: Preparing and Augmenting Dataset ---")
    prepare_dataset(DATASET_DIR, PROCESSED_DIR)
    processed_files = get_midi_files(PROCESSED_DIR)
    
    if len(processed_files) == 0:
        print("Error: No augmented MIDI files found in processed/ directory.")
        sys.exit(1)
        
    if args.dry_run:
        # Keep only a tiny fraction of files for dry-run
        processed_files = processed_files[:6]
        print(f"Dry-run: truncated dataset to {len(processed_files)} files.")
        
    # Phase 2: Tokenizer Loading & BPE Training
    print("\n--- Phase 2: Loading & Training Tokenizer ---")
    # Train BPE vocabulary on raw unaugmented files.
    # Transposing to 12 keys doesn't change rhythm or structural patterns, so training on 
    # augmented files is 12x redundant and slows tokenizer training down unnecessarily.
    tokenizer_files = get_midi_files(DATASET_DIR) if not args.dry_run else processed_files
    tokenizer = get_tokenizer(tokenizer_files, force_rebuild=args.dry_run)
    
    # Phase 3: DataLoader Preparation
    print("\n--- Phase 3: Preparing PyTorch DataLoaders ---")
    train_loader, val_loader = prepare_dataset_loaders(
        files_paths=processed_files,
        tokenizer=tokenizer,
        max_seq_len=model_config["n_positions"],
        batch_size=train_config["batch_size"],
        val_split=0.2 if not args.dry_run else 0.0 # No val split for tiny dry-run
    )
    
    # Phase 4: Model Instantiation
    print("\n--- Phase 4: Initializing Transformer Model ---")
    model = get_model(tokenizer, model_config)
    
    # Phase 5: Model Training
    print("\n--- Phase 5: Training Model on CPU ---")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_config=train_config,
        checkpoint_dir=checkpoint_dir
    )
    
    # Phase 6: Output Generation
    print("\n--- Phase 6: Generating Sample Sheet Music ---")
    best_model_path = os.path.join(checkpoint_dir, "best_model")
    out_mid = os.path.join(OUTPUT_DIR, "dry_run_bach.mid" if args.dry_run else "generated_bach.mid")
    out_xml = os.path.join(OUTPUT_DIR, "dry_run_bach.xml" if args.dry_run else "generated_bach.xml")
    
    generate_music(
        model_path=best_model_path,
        tokenizer=tokenizer,
        generate_config=generate_config,
        output_midi_path=out_mid,
        output_xml_path=out_xml
    )
    
    print("\n==========================================================")
    print("             Pipeline Executed Successfully!             ")
    print("==========================================================")

if __name__ == "__main__":
    main()
