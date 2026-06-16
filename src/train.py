import os
import torch
from torch.optim import AdamW
from tqdm import tqdm

def train_model(model, train_loader, val_loader, train_config, checkpoint_dir):
    """
    Standard PyTorch training loop with CUDA GPU support.
    Automatically detects and runs on CUDA GPU if available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training. Using device: {device.type.upper()}")
    model.to(device)
    
    # Initialize Optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=train_config["learning_rate"], 
        weight_decay=train_config["weight_decay"]
    )
    
    best_val_loss = float("inf")
    num_epochs = train_config["num_epochs"]
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader) if val_loader else 0}")
    
    for epoch in range(1, num_epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for batch in train_bar:
            optimizer.zero_grad()
            
            # Unpack batch onto active device (CPU or GPU)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_steps += 1
            train_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
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
        
    print("Training finished successfully!")
