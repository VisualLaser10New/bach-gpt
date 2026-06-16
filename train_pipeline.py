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
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Only prepare dataset, build tokenizer, and initialize/save model, but do not run training."
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
        
    # Detect checkpoint epoch early to know if we are starting a fresh run
    start_epoch = 1
    latest_epoch_path = None
    
    if os.path.exists(checkpoint_dir) and not args.dry_run:
        import re
        epoch_dirs = []
        for name in os.listdir(checkpoint_dir):
            match = re.match(r"^epoch_(\d+)$", name)
            if match:
                epoch_num = int(match.group(1))
                epoch_dirs.append((epoch_num, os.path.join(checkpoint_dir, name)))
        
        if epoch_dirs:
            # Sort by epoch number to get the latest
            epoch_dirs.sort(key=lambda x: x[0])
            latest_epoch, latest_epoch_path = epoch_dirs[-1]
            start_epoch = latest_epoch + 1
            
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
    # Train BPE vocabulary on raw unaugmented files with program mapping applied.
    # To do this, we select processed files that do not contain "_transposed_" in their filenames.
    tokenizer_files = [f for f in processed_files if "_transposed_" not in os.path.basename(f)]
    if args.dry_run:
        tokenizer_files = tokenizer_files[:6]
        
    # Force rebuild the tokenizer if we are starting a fresh run (epoch 1) to ensure the mapping matches
    force_rebuild = (start_epoch == 1) or args.dry_run
    if force_rebuild:
        print("Starting a fresh run. Rebuilding tokenizer from scratch on mapped files...")
    tokenizer = get_tokenizer(tokenizer_files, force_rebuild=force_rebuild)
    
    # Phase 3: DataLoader Preparation
    print("\n--- Phase 3: Preparing PyTorch DataLoaders ---")
    train_loader, val_loader = prepare_dataset_loaders(
        files_paths=processed_files,
        tokenizer=tokenizer,
        max_seq_len=model_config["n_positions"],
        batch_size=train_config["batch_size"],
        val_split=0.2 if not args.dry_run else 0.0 # No val split for tiny dry-run
    )
    
    # Phase 4: Model Instantiation / Resume Check
    print("\n--- Phase 4: Initializing/Resuming Transformer Model ---")
    if latest_epoch_path:
        print(f"Found existing checkpoint. Resuming training from {latest_epoch_path} (Starting at Epoch {start_epoch})...")
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(latest_epoch_path)
    else:
        print("No existing checkpoints found. Initializing new model weights...")
        model = get_model(tokenizer, model_config)
        
    # Save the initialized model configuration/weights so inference script can load it
    best_model_path = os.path.join(checkpoint_dir, "best_model")
    if start_epoch == 1:
        os.makedirs(best_model_path, exist_ok=True)
        model.save_pretrained(best_model_path)
        print(f"Saved baseline model configuration to {best_model_path}")
    
    # Phase 5: Model Training
    if args.no_train:
        print("\n--- Phase 5: Model Training (SKIPPED via --no-train flag) ---")
    else:
        print("\n--- Phase 5: Training Model ---")
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_config=train_config,
            checkpoint_dir=checkpoint_dir,
            start_epoch=start_epoch
        )
    
    # Phase 6: Output Generation
    print("\n--- Phase 6: Generating Sample Sheet Music ---")
    out_mid = os.path.join(OUTPUT_DIR, "dry_run_bach.mid" if args.dry_run else "generated_bach.mid")
    out_xml = os.path.join(OUTPUT_DIR, "dry_run_bach.xml" if args.dry_run else "generated_bach.xml")
    
    if os.path.exists(best_model_path):
        generate_music(
            model_path=best_model_path,
            tokenizer=tokenizer,
            generate_config=generate_config,
            output_midi_path=out_mid,
            output_xml_path=out_xml
        )
    else:
        print("Skipping generation: no model found at best_model.")
    
    print("\n==========================================================")
    print("             Pipeline Executed Successfully!             ")
    print("==========================================================")

if __name__ == "__main__":
    main()
