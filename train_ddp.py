import os
import sys
import argparse

# Lightweight DDP entry point: delegates all logic to train_pipeline.py after setting env vars.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP wrapper for Bach training pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick dry-run.")
    parser.add_argument("--reset", action="store_true", help="Start training from scratch.")
    parser.add_argument("--hf-repo", type=str, default=os.environ.get("HF_REPO", None))
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", None))
    args = parser.parse_args()

    # If launched without torchrun, fall back to single-process mode.
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        print("Warning: LOCAL_RANK not set. Running DDP wrapper in single-process fallback mode.")

    from train_pipeline import main
    sys.argv = ["train_pipeline.py"]
    if args.dry_run:
        sys.argv.append("--dry-run")
    if args.reset:
        sys.argv.append("--reset")
    if args.hf_repo:
        sys.argv.extend(["--hf-repo", args.hf_repo])
    if args.hf_token:
        sys.argv.extend(["--hf-token", args.hf_token])
    main()
