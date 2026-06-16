import os
import torch
from transformers import GPT2LMHeadModel, LogitsProcessor, LogitsProcessorList
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

class RestrictInstrumentsLogitsProcessor(LogitsProcessor):
    """
    Modifies logits during generation to prevent the model from composing 
    for any instrument (program) not selected by the user, including 
    BPE-merged tokens containing those programs.
    """
    def __init__(self, blocked_token_ids):
        self.blocked_token_ids = set(blocked_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Penalize blocked program tokens by setting their logit to -infinity
        for token_id in self.blocked_token_ids:
            scores[:, token_id] = -float("inf")
        return scores

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
        
    # Setup logits processor to block forbidden instruments during generation
    logits_processor = LogitsProcessorList()
    
    if program_ids:
        print(f"Enforcing instrument constraints via logits processor for programs: {program_ids}")
        allowed_program_tokens = []
        for prog in program_ids:
            token_name = f"Program_{prog}"
            if token_name in tokenizer:
                allowed_program_tokens.append(tokenizer[token_name])
            else:
                print(f"Warning: Token '{token_name}' not in tokenizer vocabulary. Skipping.")
                
        # Find all program tokens in vocabulary
        all_program_tokens = [v for k, v in tokenizer.vocab.items() if k.startswith("Program_")]
        blocked_programs = set(all_program_tokens) - set(allowed_program_tokens)
        
        # Pre-analyze vocabulary to block BPE merged tokens containing blocked programs
        blocked_token_ids = set()
        
        for token_id in range(len(tokenizer)):
            # Create sequence and decode BPE to check its base component tokens
            seq = TokSequence(ids=[token_id])
            seq.are_ids_encoded = True
            tokenizer.decode_token_ids(seq)
            
            # If any component base ID is in blocked_programs, block the entire token ID
            if any(base_id in blocked_programs for base_id in seq.ids):
                blocked_token_ids.add(token_id)
                
        print(f"Blocked {len(blocked_token_ids)} tokens (including BPE merges) containing forbidden programs.")
        
        # Append the restriction logits processor
        logits_processor.append(
            RestrictInstrumentsLogitsProcessor(blocked_token_ids)
        )
                
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
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            logits_processor=logits_processor,
        )
        
    # Extract token list and move back to CPU for decoding
    generated_tokens = generation_output[0].cpu().tolist()
    print(f"Generated raw sequence of {len(generated_tokens)} tokens.")
    
    # Truncate at the last completed Bar token to prevent incomplete measures and hanging notes
    bar_token_id = tokenizer["Bar_None"] if "Bar_None" in tokenizer else None
    if bar_token_id is not None and bar_token_id in generated_tokens:
        # Find index of last Bar token
        last_bar_idx = len(generated_tokens) - 1 - generated_tokens[::-1].index(bar_token_id)
        # Keep tokens up to and including the bar boundary
        generated_tokens = generated_tokens[:last_bar_idx + 1]
        print(f"Cleaned end of sequence. Truncated at last bar boundary. New length: {len(generated_tokens)} tokens.")
    
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
