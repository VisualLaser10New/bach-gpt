import os
import sys
import argparse
import torch.distributed as dist
from src.config import (
    DATASET_DIR, PROCESSED_DIR, CHECKPOINT_DIR, OUTPUT_DIR,
    MODEL_CONFIG, TRAIN_CONFIG, GENERATE_CONFIG, TOKENIZER_PATH
)
from src.data_prep import prepare_dataset, get_midi_files
from src.tokenizer import get_tokenizer
from src.dataset import prepare_dataset_loaders
from src.model import get_model, BachLlamaForCausalLM
from src.train import train_model
from src.generate import generate_music

def sanitize_hf_repo(repo_str):
    if not repo_str:
        return repo_str
    # Remove protocol and domain if present
    clean = repo_str.strip()
    if "huggingface.co/" in clean:
        clean = clean.split("huggingface.co/")[-1]
    # Strip leading/trailing slashes
    clean = clean.strip("/")
    return clean

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
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all existing checkpoints and tokenizer to start a clean run from epoch 1."
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=os.environ.get("HF_REPO", None),
        help="Hugging Face repository name (e.g., 'username/bach-llama') for checkpoint syncing."
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN", None),
        help="Hugging Face access token with WRITE permissions."
    )
    args = parser.parse_args()
    
    if args.hf_repo:
        args.hf_repo = sanitize_hf_repo(args.hf_repo)
    
    # Detect DDP context
    is_ddp = "LOCAL_RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.dry_run:
        os.environ["DRY_RUN"] = "1"
    
    # Define model and training configs (which can be overridden in dry-run)
    model_config = MODEL_CONFIG.copy()
    train_config = TRAIN_CONFIG.copy()
    generate_config = GENERATE_CONFIG.copy()
    checkpoint_dir = CHECKPOINT_DIR
    
    if args.reset:
        print("Reset flag active. Cleaning up existing checkpoints and tokenizer...")
        import shutil
        # Clean checkpoint directory
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Clean tokenizer
        tokenizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer.json")
        if os.path.exists(tokenizer_path):
            try:
                os.remove(tokenizer_path)
                print("Deleted existing tokenizer.json")
            except OSError as e:
                print(f"Warning: Could not delete tokenizer.json: {e}")
                
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
        
    # Hugging Face Checkpoint Syncing (Download phase)
    if args.hf_repo and not args.reset and not args.dry_run:
        print(f"Hugging Face sync active. Checking repository '{args.hf_repo}' for existing checkpoints...")
        try:
            from huggingface_hub import snapshot_download, HfApi
            api = HfApi(token=args.hf_token)
            if api.repo_exists(repo_id=args.hf_repo, repo_type="model"):
                print(f"  Downloading existing checkpoints from Hugging Face Hub...")
                os.makedirs(checkpoint_dir, exist_ok=True)
                snapshot_download(
                    repo_id=args.hf_repo,
                    local_dir=checkpoint_dir,
                    token=args.hf_token,
                    ignore_patterns=["*.git*", "README.md", ".gitattributes"]
                )
                print(f"  Checkpoints downloaded successfully.")
                
                # Copy tokenizer.json if present
                local_tokenizer_in_checkpoints = os.path.join(checkpoint_dir, "tokenizer.json")
                if os.path.exists(local_tokenizer_in_checkpoints):
                    import shutil
                    shutil.copy(local_tokenizer_in_checkpoints, TOKENIZER_PATH)
                    print(f"  Copied tokenizer.json from checkpoints to workspace root.")
            else:
                print(f"  Repository '{args.hf_repo}' does not exist yet. Will create it on the first save.")
        except Exception as e:
            print(f"  Warning: Could not sync from Hugging Face Hub: {e}")

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
            # If a training_state.pt exists, resume from the latest epoch (train.py will restore state).
            # Otherwise, prefer best_model so the latest best checkpoint is used rather than the last epoch.
            state_path = os.path.join(checkpoint_dir, "training_state.pt")
            if os.path.exists(state_path):
                epoch_dirs.sort(key=lambda x: x[0])
                latest_epoch, latest_epoch_path = epoch_dirs[-1]
                start_epoch = latest_epoch + 1
            else:
                best_model_path = os.path.join(checkpoint_dir, "best_model")
                if os.path.exists(best_model_path):
                    print(f"Warning: training_state.pt not found. Resuming from best_model instead of latest epoch.")
                    latest_epoch_path = best_model_path
                    # Determine start epoch from best_model/config.json if possible, else latest epoch
                    start_epoch = 1
                    for num, path in epoch_dirs:
                        if os.path.samefile(path, best_model_path):
                            continue
                        start_epoch = max(start_epoch, num + 1)
                else:
                    epoch_dirs.sort(key=lambda x: x[0])
                    latest_epoch, latest_epoch_path = epoch_dirs[-1]
                    start_epoch = latest_epoch + 1
                    
    # Phase 1: Data Preparation & Augmentation
    print("--- Phase 1: Preparing and Augmenting Dataset ---")
    semitones_list = train_config.get("transposition_keys", [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    prepare_dataset(DATASET_DIR, PROCESSED_DIR, semitones_list=semitones_list)
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
    
    # Copy tokenizer.json to /kaggle/working/ for easy download if running on Kaggle
    if os.path.exists("/kaggle/working") and os.path.isdir("/kaggle/working"):
        try:
            import shutil
            shutil.copy(TOKENIZER_PATH, os.path.join("/kaggle/working", "tokenizer.json"))
            print("Copied tokenizer.json to /kaggle/working/ for easy download.")
        except Exception as e:
            print(f"Warning: Could not copy tokenizer.json to /kaggle/working/: {e}")
    
    # Phase 3: DataLoader Preparation
    print("\n--- Phase 3: Preparing PyTorch DataLoaders ---")
    train_loader, val_loader = prepare_dataset_loaders(
        files_paths=processed_files,
        tokenizer=tokenizer,
        max_seq_len=model_config["n_positions"],
        batch_size=train_config["batch_size"],
        val_split=0.2 if not args.dry_run else 0.0, # No val split for tiny dry-run
        rank=rank,
        world_size=world_size
    )
    
    # Phase 4: Model Instantiation / Resume Check
    print("\n--- Phase 4: Initializing/Resuming Transformer Model ---")
    if latest_epoch_path:
        print(f"Found existing checkpoint. Resuming training from {latest_epoch_path} (Starting at Epoch {start_epoch})...")
        # Add check to verify architecture mismatch
        config_file_path = os.path.join(latest_epoch_path, "config.json")
        if os.path.exists(config_file_path):
            try:
                with open(config_file_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                if cfg_data.get("model_type") == "gpt2":
                    print("\n" + "="*80)
                    print("CRITICAL ERROR: Detected existing checkpoint of type 'gpt2'.")
                    print("The project architecture has been updated to LLaMA.")
                    print("Please run with --reset to wipe old checkpoints and start training from scratch.")
                    print("="*80 + "\n")
                    sys.exit(1)
            except Exception:
                pass
        model = BachLlamaForCausalLM.from_pretrained(latest_epoch_path)
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
            start_epoch=start_epoch,
            hf_repo=args.hf_repo,
            hf_token=args.hf_token
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
