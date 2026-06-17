import os
import json
import re
import torch
import music21
import symusic
import copy
from transformers import GPT2LMHeadModel
from miditok import TokSequence

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

# Keyboard instruments that require Right Hand (Treble) and Left Hand (Bass) staves
KEYBOARD_INSTRUMENTS = {
    "piano", "acoustic grand piano", "bright acoustic piano", "electric grand piano", 
    "honky-tonk piano", "electric piano", "harpsichord", "harpsicord", "clavier", 
    "cembalo", "clavinet", "celesta", "church organ", "organ", "reed organ"
}

# Physical pitch ranges for instruments (min_note, max_note)
INSTRUMENT_RANGES = {
    "piano": (21, 108),
    "acoustic grand piano": (21, 108),
    "bright acoustic piano": (21, 108),
    "electric grand piano": (21, 108),
    "honky-tonk piano": (21, 108),
    "electric piano": (21, 108),
    "harpsichord": (21, 89),
    "harpsicord": (21, 89),
    "clavier": (21, 89),
    "cembalo": (21, 89),
    "clavinet": (21, 89),
    "celesta": (21, 89),
    "glockenspiel": (56, 89),
    "music box": (60, 84),
    "vibraphone": (45, 89),
    "marimba": (45, 89),
    "xylophone": (56, 89),
    "tubular bells": (54, 78),
    "dulcimer": (40, 80),
    "drawbar organ": (36, 96),
    "percussive organ": (36, 96),
    "rock organ": (36, 96),
    "church organ": (36, 96),
    "organ": (36, 96),
    "reed organ": (36, 96),
    "accordion": (45, 89),
    "harmonica": (45, 89),
    "tango accordion": (45, 89),
    "acoustic guitar (nylon)": (40, 84),
    "acoustic guitar (steel)": (40, 84),
    "electric guitar": (40, 84),
    "violin": (55, 100),
    "viola": (48, 88),
    "cello": (36, 76),
    "contrabass": (28, 55),
    "double bass": (28, 55),
    "tremolo strings": (40, 96),
    "pizzicato strings": (40, 96),
    "orchestral harp": (24, 100),
    "harp": (24, 100),
    "timpani": (36, 57),
    "string ensemble": (36, 96),
    "synth strings": (36, 96),
    "choir aahs": (36, 84),
    "choir": (36, 84),
    "chorus": (36, 84),
    "voice oohs": (36, 84),
    "voice": (36, 84),
    "trumpet": (55, 88),
    "trombone": (34, 72),
    "tuba": (18, 55),
    "french horn": (29, 77),
    "horn": (29, 77),
    "soprano sax": (50, 86),
    "alto sax": (43, 79),
    "tenor sax": (38, 74),
    "baritone sax": (31, 67),
    "oboe": (58, 91),
    "english horn": (50, 81),
    "bassoon": (26, 67),
    "clarinet": (50, 94),
    "piccolo": (74, 108),
    "flute": (60, 96),
    "recorder": (60, 96),
    "pan flute": (60, 96),
    "blown bottle": (60, 96),
    "shakuhachi": (50, 80),
    "whistle": (60, 96),
    "ocarina": (60, 96),
    "synth lead": (36, 96),
    "synth pad": (36, 96),
    "banjo": (48, 84),
    "shamisen": (48, 84),
    "koto": (48, 84),
    "kalimba": (48, 84),
    "bagpipe": (48, 84),
    "fiddle": (55, 100),
    "shana": (48, 84),
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

def python_array_split(lst, n):
    """Splits a list into n approximately equal parts, preserving order."""
    k, m = divmod(len(lst), n)
    return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def fit_track_to_range(track, min_pitch, max_pitch):
    """Shifts the entire track by octaves to fit within the instrument range."""
    if len(track.notes) == 0:
        return
    pitches = [n.pitch for n in track.notes]
    avg_pitch = sum(pitches) / len(pitches)
    
    target_center = (min_pitch + max_pitch) / 2
    
    # Find the octave shift closest to the center of instrument range
    best_shift = 0
    min_dist = float('inf')
    for shift in [-36, -24, -12, 0, 12, 24, 36]:
        dist = abs((avg_pitch + shift) - target_center)
        if dist < min_dist:
            min_dist = dist
            best_shift = shift
            
    if best_shift != 0:
        print(f"  Track '{track.name}': Shifting pitches by {best_shift} semitones (octaves) to fit range [{min_pitch}, {max_pitch}]")
        for n in track.notes:
            n.pitch = max(min_pitch, min(max_pitch, n.pitch + best_shift))
    else:
        # Fallback clipping
        for n in track.notes:
            n.pitch = max(min_pitch, min(max_pitch, n.pitch))

def parse_key_string(key_str):
    """Parses a user input key string (e.g. 'G maj', 'F min') into a music21.key.Key object."""
    clean = key_str.strip().lower()
    
    # Extract tonic note
    match = re.match(r"^([a-g][#\-b]*)", clean)
    if not match:
        raise ValueError(f"Could not parse tonic note from key string: {key_str}")
        
    tonic = match.group(1)
    
    # Determine mode
    is_minor = False
    if "min" in clean or "minor" in clean or clean.endswith("m"):
        is_minor = True
        
    # Standardize flat accidental notation
    if len(tonic) > 1:
        tonic = tonic[0] + tonic[1:].replace('b', '-')
        
    if is_minor:
        return music21.key.Key(tonic.lower())
    else:
        return music21.key.Key(tonic.upper())

def transpose_to_target_key(midi_path, xml_path, target_key_str):
    """Detects the key of the generated score and transposes it to the target key."""
    if not target_key_str:
        return
        
    print(f"Detecting key of the generated music...")
    try:
        score = music21.converter.parse(midi_path)
        detected_key = score.analyze('key')
        print(f"  Detected key: {detected_key.name} (confidence: {detected_key.correlationCoefficient:.2f})")
        
        target_key = parse_key_string(target_key_str)
        print(f"  Target key: {target_key.name}")
        
        interval = music21.interval.Interval(detected_key.tonic, target_key.tonic)
        semitones = interval.semitones
        
        if semitones != 0:
            print(f"  Transposing score by {semitones} semitones ({detected_key.tonic.name} -> {target_key.tonic.name})...")
            transposed_score = score.transpose(interval)
            transposed_score.write('midi', fp=midi_path)
            transposed_score.write('musicxml', fp=xml_path)
            print(f"  Saved transposed MIDI and MusicXML.")
        else:
            print("  Score is already in the target key. No transposition needed.")
    except Exception as e:
        print(f"  Warning: Key transposition failed: {e}")

def load_input_json(file_path):
    """Loads input configuration (instruments, tempo, key) from a JSON file."""
    default_inputs = {
        "instruments": [],
        "tempo": None,
        "key": None
    }
    
    if not os.path.exists(file_path):
        return default_inputs
        
    print(f"Reading input configuration from {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        inputs = {
            "instruments": data.get("instruments", []),
            "tempo": data.get("tempo", None),
            "key": data.get("key", None)
        }
        return inputs
    except Exception as e:
        print(f"Failed to read input JSON file: {e}")
        return default_inputs

def generate_music(model_path, tokenizer, generate_config, output_midi_path, output_xml_path, user_inputs=None):
    """
    Loads a trained model, generates a sequence of tokens, decodes them to MIDI,
    and applies custom layouts (keyboard hands), tempo, pitch constraints, and transposition.
    """
    print(f"Loading trained model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using inference device: {device.type.upper()}")
    model.to(device)
    
    bos_token_id = tokenizer["BOS_None"] if "BOS_None" in tokenizer else None
    eos_token_id = tokenizer["EOS_None"] if "EOS_None" in tokenizer else None
    pad_token_id = tokenizer["PAD_None"] if "PAD_None" in tokenizer else tokenizer.pad_token_id
    
    prompt_tokens = [bos_token_id] if bos_token_id is not None else [0]
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        
    print("Generating music tokens from model...")
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
            repetition_penalty=generate_config.get("repetition_penalty", 1.0),
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        
    generated_tokens = generation_output[0].cpu().tolist()
    print(f"Generated raw sequence of {len(generated_tokens)} tokens.")
    
    seq = TokSequence(ids=generated_tokens)
    seq.are_ids_encoded = True
    tokenizer.decode_token_ids(seq)
    
    bar_token_id = tokenizer["Bar_None"] if "Bar_None" in tokenizer else None
    if bar_token_id is not None and bar_token_id in seq.ids:
        last_bar_idx = len(seq.ids) - 1 - seq.ids[::-1].index(bar_token_id)
        seq.ids = seq.ids[:last_bar_idx + 1]
        print(f"Cleaned end of sequence. Truncated at last bar boundary. New length: {len(seq.ids)} base tokens.")
    
    print("Decoding tokens back to MIDI...")
    try:
        decoded_midi = tokenizer(seq)
        
        # Enforce inputs
        instruments = user_inputs.get("instruments", []) if user_inputs else []
        tempo = user_inputs.get("tempo", None) if user_inputs else None
        
        # Enforce constant tempo
        if hasattr(decoded_midi, "tempos"):
            initial_qpm = 120.0
            if tempo:
                initial_qpm = float(tempo)
            elif len(decoded_midi.tempos) > 0:
                initial_qpm = decoded_midi.tempos[0].qpm
                
            decoded_midi.tempos.clear()
            decoded_midi.tempos.append(symusic.Tempo(0, initial_qpm))
            print(f"Enforced stable constant tempo of {initial_qpm:.2f} BPM.")
            
        # Parse slots for instruments, accounting for 2 hands on keyboards
        if not instruments:
            instruments = ["piano"]
            
        slots = []
        for inst_name in instruments:
            prog = get_program_number(inst_name)
            if prog is None:
                print(f"Warning: Unrecognized instrument '{inst_name}'. Defaulting to Piano.")
                prog = 0
                inst_name = "piano"
                
            is_keyboard = inst_name.strip().lower() in KEYBOARD_INSTRUMENTS
            if is_keyboard:
                slots.append({"name": inst_name, "program": prog, "hand": "right"})
                slots.append({"name": inst_name, "program": prog, "hand": "left"})
            else:
                slots.append({"name": inst_name, "program": prog, "hand": "solo"})
                
        num_tracks = len(decoded_midi.tracks)
        num_slots = len(slots)
        print(f"Applying custom instrument re-mapping. Distributing {num_tracks} tracks into {num_slots} slots...")
        
        new_tracks = []
        
        if num_slots >= num_tracks:
            # Map each track to a slot. Cycle slots if there are fewer tracks than slots.
            for idx in range(num_tracks):
                slot = slots[idx]
                track = decoded_midi.tracks[idx]
                track.program = slot["program"]
                track.name = f"{slot['name'].title()} ({slot['hand'].upper()})" if slot["hand"] != "solo" else slot["name"].title()
                
                min_p, max_p = INSTRUMENT_RANGES.get(slot["name"], (21, 108))
                fit_track_to_range(track, min_p, max_p)
                new_tracks.append(track)
        else:
            # Map more tracks than slots: group and merge tracks into slots
            indices_split = python_array_split(range(num_tracks), num_slots)
            for slot_idx, slot in enumerate(slots):
                group_indices = indices_split[slot_idx]
                print(f"  Slot '{slot['name']} ({slot['hand']})' merged from tracks: {group_indices}")
                
                merged_track = symusic.Track(
                    program=slot["program"],
                    name=f"{slot['name'].title()} ({slot['hand'].upper()})" if slot["hand"] != "solo" else slot["name"].title()
                )
                for t_idx in group_indices:
                    orig_track = decoded_midi.tracks[t_idx]
                    for note in orig_track.notes:
                        merged_track.notes.append(note)
                        
                min_p, max_p = INSTRUMENT_RANGES.get(slot["name"], (21, 108))
                fit_track_to_range(merged_track, min_p, max_p)
                new_tracks.append(merged_track)
                
        decoded_midi.tracks = new_tracks
        
        # Save MIDI file
        os.makedirs(os.path.dirname(output_midi_path), exist_ok=True)
        if hasattr(decoded_midi, "dump_midi"):
            decoded_midi.dump_midi(output_midi_path)
        else:
            decoded_midi.write(output_midi_path)
            
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
        
    # Transpose to target key signature
    if user_inputs and user_inputs.get("key"):
        transpose_to_target_key(output_midi_path, output_xml_path, user_inputs["key"])

if __name__ == "__main__":
    from src.config import CHECKPOINT_DIR, GENERATE_CONFIG, OUTPUT_DIR, BASE_DIR
    from src.tokenizer import get_tokenizer
    
    tokenizer = get_tokenizer()
    model_path = os.path.join(CHECKPOINT_DIR, "best_model")
    
    if os.path.exists(model_path):
        out_mid = os.path.join(OUTPUT_DIR, "generated_bach.mid")
        out_xml = os.path.join(OUTPUT_DIR, "generated_bach.xml")
        
        # Load inputs from JSON configuration file
        input_file = os.path.join(BASE_DIR, "input.json")
        user_inputs = load_input_json(input_file)
        
        generate_music(model_path, tokenizer, GENERATE_CONFIG, out_mid, out_xml, user_inputs=user_inputs)
    else:
        print(f"Model path {model_path} does not exist. Please train the model first.")
