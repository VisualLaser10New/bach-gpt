import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import contextlib
import json


def _setup_distributed():
    """Initializes distributed process group if LOCAL_RANK is set."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if local_rank >= 0 and world_size > 1:
        if not dist.is_initialized():
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        else:
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0 if torch.cuda.is_available() else -1


def _cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def _add_weight_decay(model, weight_decay):
    """Splits parameters into decay / no-decay groups (bias + norm)."""
    decay = set()
    no_decay = set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            if not p.requires_grad:
                continue
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
                decay.add(fpn)
            elif pn.endswith("weight") and "norm" in mn.lower():
                no_decay.add(fpn)

    # Validate all parameters are covered
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    if len(inter_params) > 0:
        print(f"Warning: parameters in both decay/no_decay sets: {inter_params}")
    if len(param_dict.keys() - union_params) > 0:
        print(f"Warning: parameters not in decay/no_decay sets: {param_dict.keys() - union_params}")

    decay_params = [param_dict[pn] for pn in sorted(list(decay))]
    no_decay_params = [param_dict[pn] for pn in sorted(list(no_decay))]
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def _reconstruct_best_val_loss(checkpoint_dir):
    """If training_state.pt is missing, reconstruct best_val_loss from val_loss.json sidecars."""
    import re
    best_val = float("inf")
    best_epoch = None
    for name in os.listdir(checkpoint_dir):
        match = re.match(r"^epoch_(\d+)$", name)
        if match:
            val_loss_path = os.path.join(checkpoint_dir, name, "val_loss.json")
            if os.path.exists(val_loss_path):
                try:
                    with open(val_loss_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    val_loss = data.get("val_loss", float("inf"))
                    if val_loss < best_val:
                        best_val = val_loss
                        best_epoch = int(match.group(1))
                except Exception:
                    pass
    return best_val, best_epoch


def train_model(model, train_loader, val_loader, train_config, checkpoint_dir, start_epoch=1, hf_repo=None, hf_token=None):
    """
    Standard PyTorch training loop with CUDA GPU support, gradient accumulation,
    mixed precision training (AMP), cosine scheduler, DDP support, and early stopping.
    Saves and restores full training state for multi-session checkpointing.
    """
    ddp_enabled, rank, world_size, local_rank = _setup_distributed()
    if local_rank >= 0:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rank {rank}: Starting training. Using device: {device.type.upper()}")
    model.to(device)

    # 75M model fits in 15GB T4 without gradient checkpointing; removing it gives ~25% speedup
    # (gradient checkpointing is intentionally disabled in v3)

    # Parameter groups with weight decay on weights only
    weight_decay = train_config.get("weight_decay", 0.05)
    optimizer = AdamW(
        _add_weight_decay(model, weight_decay),
        lr=train_config["learning_rate"],
        betas=(0.9, 0.95),
    )

    # Learning rate scheduler details
    accum_steps = train_config.get("gradient_accumulation_steps", 4)
    batches_per_epoch = len(train_loader)
    steps_per_epoch = (batches_per_epoch + accum_steps - 1) // accum_steps
    total_optim_steps = steps_per_epoch * train_config["num_epochs"]
    warmup_steps = int(total_optim_steps * 0.05)  # 5% warmup

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_optim_steps)

    # AMP Gradient Scaler
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    num_epochs = train_config["num_epochs"]
    early_stopping_patience = train_config.get("early_stopping_patience", 4)
    epochs_no_improve = 0

    # Check if there is an existing training state to load
    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if start_epoch > 1 and os.path.exists(state_path):
        if rank == 0:
            print(f"Loading training state from {state_path}...")
        try:
            # Map to the correct device
            state = torch.load(state_path, map_location=device)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            best_val_loss = state.get("best_val_loss", float("inf"))
            epochs_no_improve = state.get("epochs_no_improve", 0)
            if rank == 0:
                print("Successfully restored optimizer, scheduler, and scaler states.")
        except Exception as e:
            if rank == 0:
                print(f"Warning: Failed to restore training state: {e}. Starting fresh.")
    elif start_epoch > 1 and rank == 0:
        print(f"Warning: Resuming from epoch {start_epoch} but {state_path} is missing. Optimizer momentum lost.")
        best_val_loss, best_epoch = _reconstruct_best_val_loss(checkpoint_dir)
        print(f"  Reconstructed best validation loss {best_val_loss:.4f} from epoch sidecars (best_epoch={best_epoch}).")
        print("  Re-running 5% warmup on resume to reduce loss spike...")
        # Reset scheduler to a short warmup: 5% of remaining steps
        remaining_steps = (num_epochs - start_epoch + 1) * steps_per_epoch
        warmup_steps = max(1, int(remaining_steps * 0.05))
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, remaining_steps)

    # Broadcast best_val_loss to all ranks if DDP is enabled
    # (Removed MIN reduction since training and validation losses are now globally aggregated)

    if rank == 0:
        print(f"Train batches: {batches_per_epoch}, Val batches: {len(val_loader) if val_loader else 0}")
        print(f"Effective batch size: {train_config['batch_size'] * accum_steps * world_size} (Micro-batch: {train_config['batch_size']}, Accumulation: {accum_steps}, World size: {world_size})")

    # Wrap model in DDP after optimizer creation
    if ddp_enabled:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    for epoch in range(start_epoch, num_epochs + 1):
        # Tell samplers which epoch this is
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        if val_loader and hasattr(val_loader.batch_sampler, "set_epoch"):
            val_loader.batch_sampler.set_epoch(epoch)

        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0.0
        total_train_tokens = 0.0
        train_steps = 0

        optimizer.zero_grad()

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", disable=rank != 0)
        for step, batch in enumerate(train_bar):
            # Unpack batch onto active device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            loss_scale = batch["loss_scale"].to(device).float()

            is_last_step = (step + 1) == len(train_loader)
            should_sync = (step + 1) % accum_steps == 0 or is_last_step

            # Forward pass with mixed precision
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                if loss_scale.item() == 0:
                    if rank == 0:
                        print(f"  [Warning] Batch at step {step} has 0 valid labels. Using dummy zero loss to maintain sync.")
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=None
                    )
                    loss = (outputs.logits * 0.0).sum()
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / accum_steps

            # Backward with optional no_sync for DDP gradient accumulation
            sync_context = model.no_sync() if (ddp_enabled and not should_sync) else contextlib.nullcontext()
            with sync_context:
                scaler.scale(loss).backward()

            if loss_scale.item() > 0:
                total_train_loss += outputs.loss.item() * loss_scale.item()
                total_train_tokens += loss_scale.item()
            train_steps += 1

            # Optimization step
            if should_sync:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                
                optimizer.zero_grad()
                
                if scale_after >= scale_before:
                    scheduler.step()
                else:
                    if rank == 0:
                        print(f"  [Overflow] Skipped optimizer step (scale: {scale_before:.0f} -> {scale_after:.0f})")

            if rank == 0:
                train_bar.set_postfix({
                    "Loss": f"{total_train_loss / max(total_train_tokens, 1):.4f}",
                    "Scale": f"{scaler.get_scale():.0f}"
                })

        if ddp_enabled:
            metrics = torch.tensor([total_train_loss, total_train_tokens], device=device, dtype=torch.float64)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_train_loss = metrics[0].item()
            total_train_tokens = metrics[1].item()

        avg_train_loss = total_train_loss / max(total_train_tokens, 1)

        # --- VALIDATION PHASE ---
        avg_val_loss = None
        if val_loader:
            model.eval()
            total_val_loss = 0.0
            total_val_tokens = 0
            val_steps = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]", disable=rank != 0):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    loss_scale = batch["loss_scale"].to(device).float()

                    if loss_scale.item() == 0:
                        continue
                    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                    total_val_loss += outputs.loss.item() * loss_scale.item()
                    total_val_tokens += loss_scale.item()
                    val_steps += 1

            if ddp_enabled:
                metrics = torch.tensor([total_val_loss, total_val_tokens], device=device, dtype=torch.float64)
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                total_val_loss = metrics[0].item()
                total_val_tokens = metrics[1].item()

            avg_val_loss = total_val_loss / max(total_val_tokens, 1)
            if rank == 0:
                print(f"Epoch {epoch} summary - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        else:
            if rank == 0:
                print(f"Epoch {epoch} summary - Train Loss: {avg_train_loss:.4f}")

        # --- SAVE CHECKPOINTS ---
        compare_loss = avg_val_loss if avg_val_loss is not None else avg_train_loss
        is_best = compare_loss < best_val_loss
        if is_best:
            best_val_loss = compare_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if rank == 0:
            unwrapped_model = model.module if ddp_enabled else model
            # Validate weights are finite before saving
            has_bad_weights = False
            for name, param in unwrapped_model.named_parameters():
                if not torch.isfinite(param).all():
                    print(f"  [ERROR] Parameter '{name}' contains nan/inf. Aborting checkpoint save.")
                    has_bad_weights = True
                    break
            if has_bad_weights:
                raise RuntimeError(
                    f"Cannot save checkpoint: model weights contain nan/inf. "
                    f"This indicates training diverged at epoch {epoch}. "
                    f"Consider reducing learning rate or gradient clip norm."
                )

            if is_best:
                best_model_path = os.path.join(checkpoint_dir, "best_model")
                os.makedirs(best_model_path, exist_ok=True)
                unwrapped_model.save_pretrained(best_model_path)
                print(f"--> Saved new best model to {best_model_path} (Loss: {best_val_loss:.4f})")

            # Save regular epoch checkpoint
            epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
            os.makedirs(epoch_path, exist_ok=True)
            unwrapped_model.save_pretrained(epoch_path)

            # Save validation loss sidecar
            if avg_val_loss is not None:
                with open(os.path.join(epoch_path, "val_loss.json"), "w", encoding="utf-8") as f:
                    json.dump({"val_loss": avg_val_loss, "epoch": epoch}, f, indent=2)

            # Save training state (optimizer, scheduler, scaler) for checkpoint resuming
            torch.save({
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "epochs_no_improve": epochs_no_improve,
            }, state_path)
            print(f"Saved training state to {state_path}")

            # Clean up older epoch checkpoints to manage disk space
            # Keep only the current epoch checkpoint and the best_model
            import re
            for name in os.listdir(checkpoint_dir):
                match = re.match(r"^epoch_(\d+)$", name)
                if match:
                    old_epoch = int(match.group(1))
                    if old_epoch < epoch:
                        old_epoch_path = os.path.join(checkpoint_dir, name)
                        try:
                            import shutil
                            shutil.rmtree(old_epoch_path)
                            print(f"Deleted old epoch checkpoint to save space: {old_epoch_path}")
                        except Exception as e:
                            print(f"Warning: Could not delete old checkpoint {old_epoch_path}: {e}")

        # --- HUGGING FACE SYNC ---
        if hf_repo and rank == 0 and not os.environ.get("DRY_RUN"):
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
                if is_best:
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
                    import shutil
                    shutil.copy(TOKENIZER_PATH, os.path.join(epoch_path, "tokenizer.json"))
                    api.upload_file(
                        path_or_fileobj=TOKENIZER_PATH,
                        path_in_repo="tokenizer.json",
                        repo_id=hf_repo
                    )
                print(f"--> Successfully synchronized Epoch {epoch} checkpoints with Hugging Face Hub.")
            except Exception as e:
                print(f"  Warning: Hugging Face sync failed: {e}")

        # Synchronize all ranks before next epoch / early stopping decision.
        # Rank 0 may spend significant time saving checkpoints and uploading to HuggingFace;
        # without this barrier, ranks 1+ would enter the next epoch's training loop and
        # hit DDP all-reduce, causing an NCCL timeout while waiting for rank 0.
        if ddp_enabled:
            dist.barrier()

        # --- EARLY STOPPING ---
        if epochs_no_improve >= early_stopping_patience:
            if rank == 0:
                print(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
            break

    if rank == 0:
        print("Training finished successfully!")

    _cleanup_distributed()
