import os
from pathlib import Path
from miditok import REMI, TokenizerConfig
from src.config import TOKENIZER_PARAMS, TOKENIZER_PATH

def get_tokenizer(files_paths=None, force_rebuild=False):
    """
    Load or create the REMI tokenizer.
    If files_paths is provided and tokenizer doesn't exist, BPE vocabulary is trained.
    """
    if os.path.exists(TOKENIZER_PATH) and not force_rebuild:
        print(f"Loading existing tokenizer from {TOKENIZER_PATH}")
        # Load the tokenizer with its trained BPE vocabulary using new params argument
        tokenizer = REMI(params=TOKENIZER_PATH)
        return tokenizer
    
    print("Initializing a new REMI tokenizer configuration...")
    config = TokenizerConfig(**TOKENIZER_PARAMS)
    tokenizer = REMI(config)
    
    if files_paths:
        print(f"Training Byte Pair Encoding (BPE) tokenizer on {len(files_paths)} files...")
        base_vocab_size = len(tokenizer.vocab)
        target_vocab_size = base_vocab_size + 1500
        print(f"Base vocabulary size: {base_vocab_size}, Target vocabulary size: {target_vocab_size}")
        
        # Convert path strings to pathlib.Path objects for compatibility
        path_objects = [Path(p) for p in files_paths]
        
        # Using modern miditok v3.0 methods (train instead of learn_bpe, save instead of save_params)
        tokenizer.train(vocab_size=target_vocab_size, files_paths=path_objects)
        tokenizer.save(TOKENIZER_PATH)
        print(f"Saved trained BPE tokenizer to {TOKENIZER_PATH}")
    else:
        # Save baseline configuration if no training files provided
        tokenizer.save(TOKENIZER_PATH)
        print(f"Saved baseline tokenizer config to {TOKENIZER_PATH}")
        
    return tokenizer

if __name__ == "__main__":
    # Test script
    tokenizer = get_tokenizer()
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
