from transformers import LlamaConfig, LlamaForCausalLM
import torch.nn as nn
from src.control_tokens import CONTROL_TOKENS


class BachLlamaForCausalLM(LlamaForCausalLM):
    """
    LLaMA with extra dropout regularization for the Bach music domain.
    Applies dropout after the embedding layer and after each transformer layer's residual output.
    """
    def __init__(self, config):
        super().__init__(config)
        self.embed_dropout = nn.Dropout(0.1)
        self.residual_dropout = nn.Dropout(0.1)

        # Wrap embedding forward to apply dropout on token embeddings
        orig_embed_forward = self.model.embed_tokens.forward
        def embed_forward(input_ids, *args, orig_forward=orig_embed_forward, **kwargs):
            emb = orig_forward(input_ids, *args, **kwargs)
            return self.embed_dropout(emb)
        self.model.embed_tokens.forward = embed_forward

        # Wrap each decoder layer to apply residual dropout on the hidden state output
        for layer in self.model.layers:
            orig_layer_forward = layer.forward
            def layer_forward(hidden_states, *args, orig_forward=orig_layer_forward, **kwargs):
                out = orig_forward(hidden_states, *args, **kwargs)
                if isinstance(out, tuple):
                    hidden_states = self.residual_dropout(out[0])
                    return (hidden_states,) + out[1:]
                return self.residual_dropout(out)
            layer.forward = layer_forward


def get_model(tokenizer, config_dict):
    """
    Instantiates an untrained Bach LLaMA language model with parameters specified in config_dict
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
        attention_dropout=0.1,
    )

    # Initialize model from config (new weights, ready to train)
    model = BachLlamaForCausalLM(config)

    # Print parameter count for verification
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized Bach LLaMA model with {num_params:,} trainable parameters.")

    return model

