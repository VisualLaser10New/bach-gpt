import os
import torch
from transformers import GPT2LMHeadModel
from miditok import TokSequence
import music21

# General MIDI Instrument mappings to Program Numbers
INSTRUMENT_TO_PROGRAM = {
    "piano": 0,
    "acoustic grand piano": 0,
    "bright acoustic piano": 1,
    "electric grand piano": 2,
    "honky-tonk piano": 3,
    "electric piano": 4,
    "harpsichord": 6,
    "harpsicord": 6, # Common spelling error support
    "clavier": 6,
    "cembalo": 6,
    "clavinet": 7,
    "celesta": 8,
    "glockenspiel": 9,
    "music box": 10,
    "vibraphone": 11,
    "marimba": 12,
    "xylophone": 13,
    "tubular bells": 14,
    "dulcimer": 15,
    "drawbar organ": 16,
    "percussive organ": 17,
    "rock organ": 18,
    "church organ": 19,
    "organ": 19,
    "reed organ": 20,
    "accordion": 21,
    "harmonica": 22,
    "tango accordion": 23,
    "acoustic guitar (nylon)": 24,
    "acoustic guitar (steel)": 25,
    "electric guitar": 26,
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "contrabass": 43,
    "double bass": 43,
    "tremolo strings": 44,
    "pizzicato strings": 45,
    "orchestral harp": 46,
    "harp": 46,
    "timpani": 47,
    "string ensemble": 48,
    "synth strings": 50,
    "choir aahs": 52,
    "choir": 52,
    "chorus": 52,
    "voice oohs": 53,
    "voice": 53,
    "trumpet": 56,
    "trombone": 57,
    "tuba": 58,
    "french horn": 60,
    "horn": 60,
    "soprano sax": 64,
    "alto sax": 65,
    "tenor sax": 66,
    "baritone sax": 67,
    "oboe": 68,
    "english horn": 69,
    "bassoon": 70,
    "clarinet": 71,
    "piccolo": 72,
    "flute": 73,
    "recorder": 74,
    "pan flute": 75,
    "blown bottle": 76,
    "shakuhachi": 77,
    "whistle": 78,
    "ocarina": 79,
    "synth lead": 80,
    "synth pad": 88,
    "banjo": 105,
    "shamisen": 106,
    "koto": 107,
    "kalimba": 108,
    "bagpipe": 109,
    "fiddle": 110,
    "shana": 111,
}

def get_program_number(name):
    """Maps a string instrument name to a General MIDI program number."""
    name_clean = name.strip().lower()
    
    # Try exact match first
    if name_clean in INSTRUMENT_TO_PROGRAM:
        return INSTRUMENT_TO_PROGRAM[name_clean]
        
    # Try substring match
    for key, val in INSTRUMENT_TO_PROGRAM.items():
        if key in name_clean or name_clean in key:
            return val
            
    return None

def load_instrument_file(file_path):
    """
    Reads a list of instruments from a file, maps them to MIDI program numbers,
    and returns a list of valid program numbers (max 5).
    """
    if not os.path.exists(file_path):
        return []
        
    print(f"Reading instrument request file from {file_path}...")
    valid_programs = []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            
        for line in lines:
            if len(valid_programs) >= 5:
                print(f"Warning: Maximum limit of 5 instruments reached. Skipping '{line}'.")
                continue
                
            prog = get_program_number(line)
            if prog is not None:
                valid_programs.append(prog)
                print(f"  Mapped '{line}' -> MIDI Program {prog}")
            else:
                print(f"  Warning: Unrecognized instrument '{line}'. Skipping.")
    except Exception as e:
        print(f"Failed to read instrument file: {e}")
        
    return valid_programs

def generate_music(model_path, tokenizer, generate_config, output_midi_path, output_xml_path, program_ids=None):
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
    prompt_tokens = []
    if bos_token_id is not None:
        prompt_tokens.append(bos_token_id)
    else:
        prompt_tokens.append(0)
                
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        
    print("Generating music tokens from model...")
    # Force a longer minimum length to maximize output without exceeding 512 limit
    min_length = max(256, generate_config["max_length"] - 64)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids,
            max_length=generate_config["max_length"],
            min_length=min_length,
            do_sample=True,
            temperature=generate_config["temperature"],
            top_p=generate_config["top_p"],
            top_k=generate_config.get("top_k", 30),
            repetition_penalty=generate_config.get("repetition_penalty", 1.15),
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        
    # Extract token list and move back to CPU for decoding
    generated_tokens = generation_output[0].cpu().tolist()
    print(f"Generated raw sequence of {len(generated_tokens)} tokens.")
    
    # Decompose BPE tokens into base tokens before checking for bar boundaries
    seq = TokSequence(ids=generated_tokens)
    seq.are_ids_encoded = True
    tokenizer.decode_token_ids(seq)
    
    # Truncate at the last completed Bar token to prevent incomplete measures and hanging notes
    bar_token_id = tokenizer["Bar_None"] if "Bar_None" in tokenizer else None
    if bar_token_id is not None and bar_token_id in seq.ids:
        # Find index of last Bar token in base tokens
        last_bar_idx = len(seq.ids) - 1 - seq.ids[::-1].index(bar_token_id)
        # Keep tokens up to and including the bar boundary
        seq.ids = seq.ids[:last_bar_idx + 1]
        print(f"Cleaned end of sequence. Truncated at last bar boundary. New length: {len(seq.ids)} base tokens.")
    
    # Decode tokens back into a MIDI/Score object
    print("Decoding tokens back to MIDI...")
    try:
        decoded_midi = tokenizer(seq)
        
        # Enforce a single stable tempo (clear chaotic tempo changes from under-trained model)
        if hasattr(decoded_midi, "tempos"):
            initial_qpm = 120.0
            if len(decoded_midi.tempos) > 0:
                initial_qpm = decoded_midi.tempos[0].qpm
            
            decoded_midi.tempos.clear()
            import symusic
            decoded_midi.tempos.append(symusic.Tempo(0, initial_qpm))
            print(f"Cleaned tempo changes. Enforced a stable constant tempo of {initial_qpm:.2f} BPM.")
            
        # Re-map tracks to requested instruments in order (avoiding out-of-distribution logits issues)
        if program_ids and hasattr(decoded_midi, "tracks"):
            print(f"Applying custom instrument re-mapping for tracks: {program_ids}")
            for i, track in enumerate(decoded_midi.tracks):
                if i < len(program_ids):
                    old_prog = track.program
                    new_prog = program_ids[i]
                    track.program = new_prog
                    
                    # Set a descriptive track name based on program number
                    names = [k for k, v in INSTRUMENT_TO_PROGRAM.items() if v == new_prog]
                    track.name = names[0].title() if names else f"Voice {i+1}"
                    print(f"  Track {i}: Remapped Program {old_prog} -> Program {new_prog} ({track.name})")
        
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
    from src.config import CHECKPOINT_DIR, GENERATE_CONFIG, OUTPUT_DIR, BASE_DIR
    from src.tokenizer import get_tokenizer
    
    tokenizer = get_tokenizer()
    model_path = os.path.join(CHECKPOINT_DIR, "best_model")
    
    if os.path.exists(model_path):
        out_mid = os.path.join(OUTPUT_DIR, "generated_bach.mid")
        out_xml = os.path.join(OUTPUT_DIR, "generated_bach.xml")
        
        # Check for instruments request file in the project base directory
        inst_file = os.path.join(BASE_DIR, "instruments.txt")
        program_ids = load_instrument_file(inst_file)
        
        generate_music(model_path, tokenizer, GENERATE_CONFIG, out_mid, out_xml, program_ids=program_ids)
    else:
        print(f"Model path {model_path} does not exist. Please train the model first.")
