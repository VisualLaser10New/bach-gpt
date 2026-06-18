import os
import torch
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

def train_model(model, train_loader, val_loader, train_config, checkpoint_dir, start_epoch=1, hf_repo=None, hf_token=None):
    """
    Standard PyTorch training loop with CUDA GPU support, gradient accumulation,
    mixed precision training (AMP), gradient checkpointing, and cosine scheduler.
    Saves and restores full training state for multi-session checkpointing.
    Uploads checkpoints dynamically to Hugging Face Hub if hf_repo is active.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training. Using device: {device.type.upper()}")
    model.to(device)
    
    # Enable gradient checkpointing to save memory
    if device.type == "cuda":
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
    
    # Initialize Optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=train_config["learning_rate"], 
        weight_decay=train_config["weight_decay"]
    )
    
    # Learning rate scheduler details
    accum_steps = train_config.get("gradient_accumulation_steps", 8)
    batches_per_epoch = len(train_loader)
    steps_per_epoch = (batches_per_epoch + accum_steps - 1) // accum_steps
    total_optim_steps = steps_per_epoch * train_config["num_epochs"]
    warmup_steps = int(total_optim_steps * 0.05) # 5% warmup
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_optim_steps)
    
    # AMP Gradient Scaler
    scaler = GradScaler(enabled=(device.type == "cuda"))
    
    best_val_loss = float("inf")
    num_epochs = train_config["num_epochs"]
    
    # Check if there is an existing training state to load
    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if start_epoch > 1 and os.path.exists(state_path):
        print(f"Loading training state from {state_path}...")
        try:
            state = torch.load(state_path, map_location=device)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            best_val_loss = state.get("best_val_loss", float("inf"))
            print("Successfully restored optimizer, scheduler, and scaler states.")
        except Exception as e:
            print(f"Warning: Failed to restore training state: {e}. Starting fresh.")
            
    print(f"Train batches: {batches_per_epoch}, Val batches: {len(val_loader) if val_loader else 0}")
    print(f"Effective batch size: {train_config['batch_size'] * accum_steps} (Micro-batch: {train_config['batch_size']}, Accumulation: {accum_steps})")
    
    for epoch in range(start_epoch, num_epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        optimizer.zero_grad()
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for step, batch in enumerate(train_bar):
            # Unpack batch onto active device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with mixed precision
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Correct loss scaling for partial steps at the end of the epoch
                is_last_step = (step + 1) == len(train_loader)
                current_accum_steps = accum_steps if not is_last_step else (len(train_loader) % accum_steps or accum_steps)
                loss = outputs.loss / current_accum_steps
                
            # Backward pass
            scaler.scale(loss).backward()
            
            # Optimization step
            if (step + 1) % accum_steps == 0 or is_last_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
            total_train_loss += loss.item() * current_accum_steps
            train_steps += 1
            train_bar.set_postfix({"Loss": f"{(loss.item() * current_accum_steps):.4f}"})
            
        avg_train_loss = total_train_loss / train_steps
        
        # --- VALIDATION PHASE ---
        avg_val_loss = None
        if val_loader:
            model.eval()
            total_val_loss = 0
            val_steps = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                    total_val_loss += outputs.loss.item()
                    val_steps += 1
                    
            avg_val_loss = total_val_loss / val_steps
            print(f"Epoch {epoch} summary - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch} summary - Train Loss: {avg_train_loss:.4f}")
            
        # --- SAVE CHECKPOINTS ---
        compare_loss = avg_val_loss if avg_val_loss is not None else avg_train_loss
        if compare_loss < best_val_loss:
            best_val_loss = compare_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model")
            os.makedirs(best_model_path, exist_ok=True)
            
            # Save Hugging Face model weights and configuration
            model.save_pretrained(best_model_path)
            print(f"--> Saved new best model to {best_model_path} (Loss: {best_val_loss:.4f})")
            
        # Save regular epoch checkpoint
        epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
        os.makedirs(epoch_path, exist_ok=True)
        model.save_pretrained(epoch_path)
        
        # Save training state (optimizer, scheduler, scaler) for checkpoint resuming
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss
        }, state_path)
        print(f"Saved training state to {state_path}")
        
        # --- HUGGING FACE SYNC ---
        if hf_repo:
            print(f"Syncing checkpoints with Hugging Face Hub ({hf_repo})...")
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=hf_token)
                api.create_repo(repo_id=hf_repo, private=True, exist_ok=True)
                
                # Upload current epoch checkpoint
                api.upload_folder(
                    folder_path=epoch_path,
                    path_in_repo=f"epoch_{epoch}",
                    repo_id=hf_repo
                )
                
                # Upload best model if updated
                if compare_loss < best_val_loss:
                    api.upload_folder(
                        folder_path=best_model_path,
                        path_in_repo="best_model",
                        repo_id=hf_repo
                    )
                
                # Upload training state & tokenizer
                api.upload_file(
                    path_or_fileobj=state_path,
                    path_in_repo="training_state.pt",
                    repo_id=hf_repo
                )
                from src.config import TOKENIZER_PATH
                if os.path.exists(TOKENIZER_PATH):
                    api.upload_file(
                        path_or_fileobj=TOKENIZER_PATH,
                        path_in_repo="tokenizer.json",
                        repo_id=hf_repo
                    )
                print(f"--> Successfully synchronized Epoch {epoch} checkpoints with Hugging Face Hub.")
            except Exception as e:
                print(f"  Warning: Hugging Face sync failed: {e}")
        
    print("Training finished successfully!")

