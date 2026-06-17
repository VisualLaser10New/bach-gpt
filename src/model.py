from transformers import LlamaConfig, LlamaForCausalLM
from src.control_tokens import CONTROL_TOKENS

def get_model(tokenizer, config_dict):
    """
    Instantiates an untrained LLaMA language model with parameters specified in config_dict
    and sized dynamically to match the tokenizer's vocabulary + custom control tokens.
    """
    n_head = config_dict.get("n_head", 12)
    n_kv_head = config_dict.get("n_kv_head", n_head)
    if n_kv_head > n_head or n_head % n_kv_head != 0:
        n_kv_head = n_head

    config = LlamaConfig(
        vocab_size=len(tokenizer) + len(CONTROL_TOKENS),
        hidden_size=config_dict.get("n_embd", 768),
        intermediate_size=config_dict.get("intermediate_size", 2048),
        num_hidden_layers=config_dict.get("n_layer", 10),
        num_attention_heads=n_head,
        num_key_value_heads=n_kv_head,
        max_position_embeddings=config_dict.get("n_positions", 4096),
        bos_token_id=tokenizer["BOS_None"] if "BOS_None" in tokenizer else None,
        eos_token_id=tokenizer["EOS_None"] if "EOS_None" in tokenizer else None,
        pad_token_id=tokenizer["PAD_None"] if "PAD_None" in tokenizer else tokenizer.pad_token_id,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
    )
    
    # Initialize model from config (new weights, ready to train)
    model = LlamaForCausalLM(config)
    
    # Print parameter count for verification
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized LLaMA model with {num_params:,} trainable parameters.")
    
    return model

