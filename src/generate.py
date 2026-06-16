import os
import torch
from transformers import GPT2LMHeadModel
import music21

def generate_music(model_path, tokenizer, generate_config, output_midi_path, output_xml_path):
    """
    Loads a trained model, generates a sequence of tokens on the active device (GPU if available),
    decodes them to MIDI, and exports MIDI and MusicXML files.
    """
    print(f"Loading trained model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    
    # Auto-detect CUDA GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using inference device: {device.type.upper()}")
    model.to(device)
    
    # Get special token IDs
    bos_token_id = tokenizer["BOS_None"] if "BOS_None" in tokenizer else None
    eos_token_id = tokenizer["EOS_None"] if "EOS_None" in tokenizer else None
    pad_token_id = tokenizer["PAD_None"] if "PAD_None" in tokenizer else tokenizer.pad_token_id
    
    # Initialize input with BOS token
    if bos_token_id is not None:
        input_ids = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
    else:
        input_ids = torch.tensor([[0]], dtype=torch.long, device=device)
        
    print("Generating music tokens from model...")
    with torch.no_grad():
        generation_output = model.generate(
            input_ids,
            max_length=generate_config["max_length"],
            do_sample=True,
            temperature=generate_config["temperature"],
            top_p=generate_config["top_p"],
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        
    # Extract token list and move back to CPU for decoding
    generated_tokens = generation_output[0].cpu().tolist()
    print(f"Generated sequence of {len(generated_tokens)} tokens.")
    
    # Decode tokens back into a MIDI/Score object
    print("Decoding tokens back to MIDI...")
    try:
        decoded_midi = tokenizer(generated_tokens)
        
        # Save MIDI file
        os.makedirs(os.path.dirname(output_midi_path), exist_ok=True)
        if hasattr(decoded_midi, "dump_midi"):
            decoded_midi.dump_midi(output_midi_path)
        elif hasattr(decoded_midi, "write"):
            decoded_midi.write(output_midi_path)
        else:
            decoded_midi.dump(output_midi_path)
            
        print(f"Saved generated MIDI to {output_midi_path}")
    except Exception as e:
        print(f"Failed to decode tokens or write MIDI: {e}")
        return
        
    # Export to MusicXML
    print("Converting MIDI output to MusicXML sheet music format...")
    try:
        score = music21.converter.parse(output_midi_path)
        score.write('musicxml', fp=output_xml_path)
        print(f"Saved sheet music to {output_xml_path}")
    except Exception as e:
        print(f"Failed to convert MIDI to MusicXML: {e}")
        print("Note: This is common if the generated MIDI contains empty tracks. The MIDI file is still fully functional.")

if __name__ == "__main__":
    from src.config import CHECKPOINT_DIR, GENERATE_CONFIG, OUTPUT_DIR
    from src.tokenizer import get_tokenizer
    
    tokenizer = get_tokenizer()
    model_path = os.path.join(CHECKPOINT_DIR, "best_model")
    
    if os.path.exists(model_path):
        out_mid = os.path.join(OUTPUT_DIR, "generated_bach.mid")
        out_xml = os.path.join(OUTPUT_DIR, "generated_bach.xml")
        generate_music(model_path, tokenizer, GENERATE_CONFIG, out_mid, out_xml)
    else:
        print(f"Model path {model_path} does not exist. Please train the model first.")
