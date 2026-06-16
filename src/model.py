from transformers import GPT2Config, GPT2LMHeadModel

def get_model(tokenizer, config_dict):
    """
    Instantiates an untrained GPT-2 language model with parameters specified in config_dict
    and sized dynamically to match the tokenizer's vocabulary.
    """
    # Dynamically inject vocabulary size and special token IDs from tokenizer
    config_dict = config_dict.copy()
    config_dict["vocab_size"] = len(tokenizer)
    
    config_dict["bos_token_id"] = tokenizer["BOS_None"] if "BOS_None" in tokenizer else None
    config_dict["eos_token_id"] = tokenizer["EOS_None"] if "EOS_None" in tokenizer else None
    config_dict["pad_token_id"] = tokenizer["PAD_None"] if "PAD_None" in tokenizer else tokenizer.pad_token_id
    
    # Convert configuration dictionary to Hugging Face GPT2Config object
    config = GPT2Config(**config_dict)
    
    # Initialize model from config (new weights, ready to train)
    model = GPT2LMHeadModel(config)
    
    # Print parameter count for verification
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized GPT-2 model with {num_params:,} trainable parameters.")
    
    return model
