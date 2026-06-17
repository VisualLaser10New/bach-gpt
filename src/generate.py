import os
import json
import re
import copy
import random
import torch
import music21
import symusic
import math
from transformers import LlamaForCausalLM
from transformers import LogitsProcessor
from miditok import TokSequence
from src.control_tokens import CONTROL_TOKENS

# General MIDI Instrument mappings to Program Numbers
INSTRUMENT_TO_PROGRAM = {
    "piano": 0,
    "acoustic grand piano": 0,
    "bright acoustic piano": 1,
    "electric grand piano": 2,
    "honky-tonk piano": 3,
    "electric piano": 4,
    "harpsichord": 6,
    "harpsicord": 6,
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

class VoiceBalanceProcessor(LogitsProcessor):
    """
    Tracks musical time (bar count) since each voice was last active.
    If a voice is silent for more than N bars, progressively boost its 
    Program_X token logit to force re-entry.
    """
    def __init__(self, tokenizer, num_voices, max_silent_bars=4):
        self.program_tokens = {}
        for i in range(num_voices):
            token_name = f"Program_{i}"
            if token_name in tokenizer.vocab:
                self.program_tokens[i] = tokenizer[token_name]
        
        self.bar_token_id = tokenizer["Bar_None"] if "Bar_None" in tokenizer else None
        self.last_active_bar = {i: 0 for i in self.program_tokens}
        self.current_bar = 0
        self.max_silent_bars = max_silent_bars
        self.initialized = False
        self.tokens_since_last_bar = 0
        
    def scan_prompt(self, input_ids):
        """Scans the prompt on startup to synchronize bar counts and active voice states."""
        for token_id in input_ids[0]:
            token_id = token_id.item()
            if token_id == self.bar_token_id:
                self.current_bar += 1
            for prog_idx, prog_token_id in self.program_tokens.items():
                if token_id == prog_token_id:
                    self.last_active_bar[prog_idx] = self.current_bar

    def __call__(self, input_ids, scores):
        # Scan initial prompt to sync state
        if not self.initialized:
            self.scan_prompt(input_ids)
            self.initialized = True
            
        last_token = input_ids[0, -1].item()
        
        # Track bar progression
        self.tokens_since_last_bar += 1
        if last_token == self.bar_token_id:
            self.current_bar += 1
            self.tokens_since_last_bar = 0
        elif self.tokens_since_last_bar >= 100:  # Fallback: assume 1 bar ≈ 100 tokens if Bar_None is missed
            self.current_bar += 1
            self.tokens_since_last_bar = 0
        
        # Track which programs are active
        for prog_idx, token_id in self.program_tokens.items():
            if last_token == token_id:
                self.last_active_bar[prog_idx] = self.current_bar
        
        # Progressive boost for silent voices
        for prog_idx, token_id in self.program_tokens.items():
            bars_silent = self.current_bar - self.last_active_bar[prog_idx]
            if bars_silent > self.max_silent_bars:
                # Gradual boost: +1.0 per bar of silence beyond threshold
                boost = min((bars_silent - self.max_silent_bars) * 1.0, 5.0)
                scores[0, token_id] += boost
        
        return scores

def get_program_number(name):
    """Maps a string instrument name to a General MIDI program number."""
    name_clean = name.strip().lower()
    if name_clean in INSTRUMENT_TO_PROGRAM:
        return INSTRUMENT_TO_PROGRAM[name_clean]
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
        for n in track.notes:
            n.pitch = max(min_pitch, min(max_pitch, n.pitch))

def parse_key_string(key_str):
    """Parses a user input key string into a music21.key.Key object."""
    clean = key_str.strip().lower()
    match = re.match(r"^([a-g][#\-b]*)", clean)
    if not match:
        raise ValueError(f"Could not parse tonic note from key string: {key_str}")
    tonic = match.group(1)
    is_minor = "min" in clean or "minor" in clean or clean.endswith("m")
    if len(tonic) > 1:
        tonic = tonic[0] + tonic[1:].replace('b', '-')
    return music21.key.Key(tonic.lower() if is_minor else tonic.upper())

def group_keyboard_staves(score):
    """Groups consecutive parts sharing the same name with a curly brace."""
    parts = list(score.parts)
    groups = []
    i = 0
    while i < len(parts):
        part_name = parts[i].partName
        group = [parts[i]]
        j = i + 1
        while j < len(parts) and parts[j].partName == part_name:
            group.append(parts[j])
            j += 1
        if len(group) > 1:
            groups.append((part_name, group))
            i = j
        else:
            i += 1
    for name, group_parts in groups:
        print(f"  Grouping {len(group_parts)} staves under '{name}' with a brace...")
        sg = music21.layout.StaffGroup(group_parts, name=name, symbol="brace")
        score.insert(0, sg)

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
            group_keyboard_staves(score)
            transposed_score = score.transpose(interval)
            transposed_score.write('musicxml', fp=xml_path)
            sym_score = symusic.Score(midi_path)
            sym_score.shift_pitch(semitones)
            sym_score.dump_midi(midi_path)
            print(f"  Saved transposed MIDI and MusicXML.")
        else:
            print("  Score is already in the target key. No transposition needed.")
    except Exception as e:
        print(f"  Warning: Key transposition failed: {e}")

def load_input_json(file_path):
    """Loads input configuration from a JSON file."""
    default_inputs = {
        "instruments": ["piano"],
        "tempo": 120,
        "key": None,
        "mood": "andante",
        "genre": "keyboard",
        "density": "moderate",
        "baroque_tag": None,
        "form": "ABA"
    }
    if not os.path.exists(file_path):
        return default_inputs
    print(f"Reading input configuration from {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in default_inputs.items():
            if k not in data:
                data[k] = v
        return data
    except Exception as e:
        print(f"Failed to read input JSON file: {e}")
        return default_inputs

def count_total_staves(instruments):
    staves = 0
    for inst in instruments:
        if inst.strip().lower() in KEYBOARD_INSTRUMENTS:
            staves += 2
        else:
            staves += 1
    return staves

def build_control_prefix_tokens(user_inputs):
    prefix = []
    genre = user_inputs.get("genre", "keyboard").strip().upper()
    prefix.append(f"GENRE_{genre}")
    mood = user_inputs.get("mood", "andante").strip().upper()
    prefix.append(f"MOOD_{mood}")
    density = user_inputs.get("density", "moderate").strip().upper()
    prefix.append(f"DENSITY_{density}")
    
    instruments = user_inputs.get("instruments", ["piano"])
    num_staves = count_total_staves(instruments)
    voices_map = {1: "V2", 2: "V2", 3: "V3", 4: "V4", 5: "V5", 6: "V6", 7: "V6", 8: "V8", 9: "V8", 10: "V10"}
    voices_token = voices_map.get(num_staves, "V4")
    if num_staves > 10:
        voices_token = "V10"
    prefix.append(voices_token)
    
    tempo = user_inputs.get("tempo")
    if tempo:
        try:
            bpm = int(tempo)
            if bpm < 80:
                tempo_token = "TEMPO_SLOW"
            elif bpm > 125:
                tempo_token = "TEMPO_FAST"
            else:
                tempo_token = "TEMPO_MEDIUM"
        except ValueError:
            tempo_token = "TEMPO_MEDIUM"
    else:
        tempo_token = "TEMPO_MEDIUM"
    prefix.append(tempo_token)
    
    baroque_tag = user_inputs.get("baroque_tag")
    if baroque_tag:
        tag_clean = baroque_tag.strip().upper()
        valid_tags = ["MINUETTO", "PRELUDE", "FUGUE", "TOCCATA", "GAVOTTE", "ARIA", "PASSACAGLIA", "SARABANDE", "BOURREE", "GIGUE", "SICILIANA"]
        if tag_clean in valid_tags:
            prefix.append(f"TAG_{tag_clean}")
    return prefix

def build_contrast_prefix_tokens(user_inputs):
    inputs_b = user_inputs.copy()
    current_mood = user_inputs.get("mood", "andante").strip().lower()
    if current_mood in ["vivace", "allegro", "grazioso"]:
        inputs_b["mood"] = "adagio"
        inputs_b["density"] = "sparse"
        tempo = user_inputs.get("tempo")
        if tempo:
            try:
                inputs_b["tempo"] = int(int(tempo) * 0.7)
            except ValueError:
                inputs_b["tempo"] = 70
    else:
        inputs_b["mood"] = "allegro"
        inputs_b["density"] = "dense"
        tempo = user_inputs.get("tempo")
        if tempo:
            try:
                inputs_b["tempo"] = int(int(tempo) * 1.3)
            except ValueError:
                inputs_b["tempo"] = 120
    if "baroque_tag" in inputs_b:
        del inputs_b["baroque_tag"]
    return build_control_prefix_tokens(inputs_b)

def generate_section(model, tokenizer, prompt_ids, max_length, generate_config, user_inputs, device):
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    num_voices = count_total_staves(user_inputs.get("instruments", ["piano"]))
    
    voice_processor = VoiceBalanceProcessor(
        tokenizer=tokenizer,
        num_voices=num_voices,
        max_silent_bars=generate_config.get("max_silent_bars", 4)
    )
    
    from transformers import LogitsProcessorList
    processors = LogitsProcessorList([voice_processor])
    
    prompt_len = len(prompt_ids)
    max_new_tokens = max(16, max_length - prompt_len)
    
    min_length = max(64, int(max_length * 0.6))
    min_new_tokens = max(8, min_length - prompt_len) if min_length > prompt_len else 8
    
    eos_token_id = tokenizer["EOS_None"] if "EOS_None" in tokenizer else None
    pad_token_id = tokenizer["PAD_None"] if "PAD_None" in tokenizer else tokenizer.pad_token_id
    
    with torch.no_grad():
        generation_output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=True,
            temperature=generate_config.get("temperature", 0.72),
            top_p=generate_config.get("top_p", 0.92),
            top_k=generate_config.get("top_k", 30),
            repetition_penalty=generate_config.get("repetition_penalty", 1.0),
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            logits_processor=processors
        )
    return generation_output[0].cpu().tolist()

def get_bar_aligned_ticks(score):
    numerator, denominator = 4, 4
    if len(score.time_signatures) > 0:
        ts = score.time_signatures[0]
        numerator, denominator = ts.numerator, ts.denominator
    ticks_per_beat = score.ticks_per_quarter * (4 / denominator)
    ticks_per_bar = int(numerator * ticks_per_beat)
    num_bars = int(math.ceil(score.end() / ticks_per_bar))
    return num_bars * ticks_per_bar

def apply_midi_variation(score):
    score_prime = copy.deepcopy(score)
    for track in score_prime.tracks:
        for note in track.notes:
            note.velocity = max(20, min(127, int(note.velocity * random.uniform(0.9, 1.1))))
    return score_prime

def generate_aba_form(model, tokenizer, generate_config, user_inputs, device):
    """Generates an ABA structure using prefix-aligned generation and MIDI-level merging."""
    # Sanity check for load-bearing coupling with virtual control tokens
    assert model.config.vocab_size == len(tokenizer) + len(CONTROL_TOKENS), \
        f"Model vocab_size ({model.config.vocab_size}) does not match tokenizer + control tokens size ({len(tokenizer) + len(CONTROL_TOKENS)}). This is a critical coupling error."
        
    # Split token budget per section
    section_tokens = generate_config.get("max_length", 4096) // 2
    bos_token_id = tokenizer["BOS_None"] if "BOS_None" in tokenizer else 0
    
    # === SECTION A ===
    print("Generating Section A (Main Theme)...")
    control_prefix = build_control_prefix_tokens(user_inputs)
    vocab_offset = len(tokenizer)
    prompt_a_ids = []
    for t in control_prefix:
        if t in CONTROL_TOKENS:
            prompt_a_ids.append(vocab_offset + CONTROL_TOKENS.index(t))
    prompt_a = [bos_token_id] + prompt_a_ids
    section_a = generate_section(model, tokenizer, prompt_a, section_tokens, generate_config, user_inputs, device)
    
    # === SECTION B ===
    print("Generating Section B (Contrasting Section)...")
    bridge_tokens = section_a[-256:]
    contrast_prefix = build_contrast_prefix_tokens(user_inputs)
    prompt_b_ids = []
    for t in contrast_prefix:
        if t in CONTROL_TOKENS:
            prompt_b_ids.append(vocab_offset + CONTROL_TOKENS.index(t))
    prompt_b = [bos_token_id] + prompt_b_ids + bridge_tokens
    section_b_full = generate_section(model, tokenizer, prompt_b, section_tokens, generate_config, user_inputs, device)
    new_tokens_b = section_b_full[len(prompt_b):]
    
    # === DECODE A + B AND VARIATE A' ===
    print("Decoding Section A + B...")
    # Slice off prompt_a (BOS + control prefix) to avoid BPE KeyError on virtual tokens
    notes_a = [t for t in section_a[len(prompt_a):] if t < len(tokenizer)]
    new_tokens_b_clean = [t for t in new_tokens_b if t < len(tokenizer)]
    tokens_ab = notes_a + new_tokens_b_clean
    seq_ab = TokSequence(ids=tokens_ab)
    seq_ab.are_ids_encoded = True
    tokenizer.decode_token_ids(seq_ab)
    score_ab = tokenizer(seq_ab)
    
    print("Decoding Section A for Reprise...")
    seq_a = TokSequence(ids=notes_a)
    seq_a.are_ids_encoded = True
    tokenizer.decode_token_ids(seq_a)
    score_a = tokenizer(seq_a)
    
    print("Applying MIDI variations to create Section A'...")
    score_a_prime = apply_midi_variation(score_a)
    
    # Align and Merge
    align_boundary = get_bar_aligned_ticks(score_ab)
    
    # Re-tempo the reprise section to match Section A's initial tempo at the boundary
    if len(score_a.tempos) > 0:
        tempo_val = score_a.tempos[0].qpm
        score_ab.tempos.append(symusic.Tempo(align_boundary, tempo_val))
        print(f"Set reprise Section A' tempo to {tempo_val:.2f} BPM at tick {align_boundary}.")
    
    print(f"Merging Section A' into the final score at tick {align_boundary}...")
    # Group score_ab tracks by program number to align instruments correctly and avoid mismatches
    ab_tracks_by_program = {t.program: t for t in score_ab.tracks}
    
    for t_prime in score_a_prime.tracks:
        # Shift the track events directly to avoid duplicate global events
        t_prime.shift_time(align_boundary)
        
        prog = t_prime.program
        if prog in ab_tracks_by_program:
            t_ab = ab_tracks_by_program[prog]
            t_ab.notes.extend(t_prime.notes)
            t_ab.controls.extend(t_prime.controls)
            t_ab.pitch_bends.extend(t_prime.pitch_bends)
            t_ab.pedals.extend(t_prime.pedals)
        else:
            # If the track doesn't exist in score_ab (e.g. generation dropped it), append it
            score_ab.tracks.append(t_prime)
            
    return score_ab

def is_trill_pattern(notes, i):
    if i + 3 >= len(notes):
        return False
    n0, n1, n2, n3 = notes[i], notes[i+1], notes[i+2], notes[i+3]
    durations = [n.duration.quarterLength for n in [n0, n1, n2, n3]]
    if any(d > 0.15 for d in durations):
        return False
    p0, p1, p2, p3 = n0.pitch.ps, n1.pitch.ps, n2.pitch.ps, n3.pitch.ps
    if abs(p0 - p1) <= 2.0 and p0 == p2 and p1 == p3 and p0 != p1:
        return True
    return False

def count_trill_alternations(notes, i):
    p0, p1 = notes[i].pitch.ps, notes[i+1].pitch.ps
    count = 2
    j = i + 2
    while j < len(notes):
        n = notes[j]
        if n.duration.quarterLength > 0.15:
            break
        expected_pitch = p0 if count % 2 == 0 else p1
        if n.pitch.ps != expected_pitch:
            break
        count += 1
        j += 1
    return count

def is_mordent_pattern(notes, i):
    if i + 2 >= len(notes):
        return False
    n0, n1, n2 = notes[i], notes[i+1], notes[i+2]
    durations = [n.duration.quarterLength for n in [n0, n1, n2]]
    if any(d > 0.3 for d in durations):
        return False
    p0, p1, p2 = n0.pitch.ps, n1.pitch.ps, n2.pitch.ps
    if p0 == p2 and abs(p0 - p1) <= 2.0 and p0 != p1:
        return True
    return False

def detect_and_mark_ornaments(score):
    """
    Scans each part for rapid alternating note patterns and annotates
    them with proper ornament markings in the MusicXML output.
    Adjusts remaining note's duration to match the sum of the replaced notes
    to prevent empty beats in the measure.
    """
    for part in score.parts:
        notes = [n for n in part.flatten().notes if isinstance(n, music21.note.Note)]
        i = 0
        while i < len(notes) - 3:
            # Check for trill pattern
            if is_trill_pattern(notes, i):
                trill_length = count_trill_alternations(notes, i)
                main_note = notes[i]
                total_duration = sum(n.duration.quarterLength for n in notes[i:i + trill_length])
                main_note.duration.quarterLength = total_duration
                
                tr = music21.expressions.Trill()
                main_note.expressions.append(tr)
                for j in range(i + 1, i + trill_length):
                    n_to_remove = notes[j]
                    if n_to_remove.activeSite:
                        n_to_remove.activeSite.remove(n_to_remove)
                i += trill_length
                continue
            
            # Check for mordent
            if is_mordent_pattern(notes, i):
                main_note = notes[i]
                total_duration = sum(n.duration.quarterLength for n in notes[i:i + 3])
                main_note.duration.quarterLength = total_duration
                
                p0, p1 = notes[i].pitch.ps, notes[i+1].pitch.ps
                if p1 < p0:
                    main_note.expressions.append(music21.expressions.Mordent())
                else:
                    main_note.expressions.append(music21.expressions.Turn())
                    
                for j in [i+1, i+2]:
                    n_to_remove = notes[j]
                    if n_to_remove.activeSite:
                        n_to_remove.activeSite.remove(n_to_remove)
                i += 3
                continue
            i += 1

def add_tempo_markings(score, bpm, baroque_tag):
    """Inserts both a numeric metronome mark and an italic baroque text annotation."""
    if len(score.parts) == 0:
        return
    try:
        mm = music21.tempo.MetronomeMark(number=bpm)
        score.parts[0].measure(1).insert(0, mm)
        if baroque_tag:
            tag_text = baroque_tag.strip().title()
            te = music21.expressions.TextExpression(tag_text)
            te.style.fontStyle = 'italic'
            te.style.fontSize = 14
            te.placement = 'above'
            score.parts[0].measure(1).insert(0, te)
    except Exception as e:
        print(f"Warning: Failed to add tempo markings: {e}")

def validate_inputs(user_inputs):
    """Validates user inputs to prevent illegal/rare control token combinations."""
    instruments = user_inputs.get("instruments", ["piano"])
    genre = user_inputs.get("genre", "keyboard").lower()
    density = user_inputs.get("density", "moderate").lower()
    
    # Chorales are strictly 4 voices
    if genre == "chorale":
        user_inputs["instruments"] = ["soprano", "alto", "tenor", "bass"]
        print("Heuristic validation: Clamping chorale instruments to standard 4-voice SATB.")
        
    # Density cap for high voice count to prevent clipping
    num_staves = count_total_staves(instruments)
    if num_staves >= 8 and density == "dense":
        print(f"Heuristic validation: Reducing density from DENSE to MODERATE for {num_staves}-voice piece.")
        user_inputs["density"] = "moderate"

def generate_music(model_path, tokenizer, generate_config, output_midi_path, output_xml_path, user_inputs=None):
    """
    Loads LLaMA model, generates tokens (or runs ABA multi-section generation),
    and applies custom instrument layouts, tempo, pitch, and post-processing ornaments.
    """
    print(f"Loading trained model from {model_path}...")
    model = LlamaForCausalLM.from_pretrained(model_path)
    model.eval()
    
    # Sanity check for load-bearing coupling with virtual control tokens
    assert model.config.vocab_size == len(tokenizer) + len(CONTROL_TOKENS), \
        f"Model vocab_size ({model.config.vocab_size}) does not match tokenizer + control tokens size ({len(tokenizer) + len(CONTROL_TOKENS)}). This is a critical coupling error."
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using inference device: {device.type.upper()}")
    model.to(device)
    
    # 0. Validate inputs
    if not user_inputs:
        user_inputs = {
            "instruments": ["piano"],
            "tempo": 120,
            "key": None,
            "mood": "andante",
            "genre": "keyboard",
            "density": "moderate",
            "baroque_tag": None,
            "form": "ABA"
        }
    validate_inputs(user_inputs)
    
    form = user_inputs.get("form", "ABA").upper()
    tempo = user_inputs.get("tempo", 120)
    baroque_tag = user_inputs.get("baroque_tag")
    instruments = user_inputs.get("instruments", ["piano"])
    
    # 1. Run ABA generation or standard generation
    if form == "ABA":
        decoded_midi = generate_aba_form(model, tokenizer, generate_config, user_inputs, device)
    else:
        # Single section legacy generation
        print("Generating single-section (legacy) piece...")
        bos_token_id = tokenizer["BOS_None"] if "BOS_None" in tokenizer else 0
        control_prefix = build_control_prefix_tokens(user_inputs)
        vocab_offset = len(tokenizer)
        prompt_tokens_ids = []
        for t in control_prefix:
            if t in CONTROL_TOKENS:
                prompt_tokens_ids.append(vocab_offset + CONTROL_TOKENS.index(t))
        prompt_tokens = [bos_token_id] + prompt_tokens_ids
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        
        num_voices = count_total_staves(instruments)
        voice_processor = VoiceBalanceProcessor(
            tokenizer=tokenizer,
            num_voices=num_voices,
            max_silent_bars=generate_config.get("max_silent_bars", 4)
        )
        from transformers import LogitsProcessorList
        processors = LogitsProcessorList([voice_processor])
        
        prompt_len = len(prompt_tokens)
        max_new_tokens = max(16, generate_config["max_length"] - prompt_len)
        min_new_tokens = max(8, max(256, generate_config["max_length"] - 64) - prompt_len)
        
        with torch.no_grad():
            generation_output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=True,
                temperature=generate_config["temperature"],
                top_p=generate_config["top_p"],
                top_k=generate_config.get("top_k", 30),
                repetition_penalty=generate_config.get("repetition_penalty", 1.0),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer["EOS_None"] if "EOS_None" in tokenizer else None,
                logits_processor=processors
            )
        generated_tokens = generation_output[0].cpu().tolist()
        print(f"Generated raw sequence of {len(generated_tokens)} tokens.")
        
        # Slice off prompt_tokens (BOS + control prefix) to avoid BPE KeyError on virtual tokens
        notes = [t for t in generated_tokens[len(prompt_tokens):] if t < len(tokenizer)]
        seq = TokSequence(ids=notes)
        seq.are_ids_encoded = True
        tokenizer.decode_token_ids(seq)
        decoded_midi = tokenizer(seq)
        
    # Enforce constant tempo in symusic Score
    if hasattr(decoded_midi, "tempos"):
        initial_qpm = 120.0
        if tempo:
            try:
                initial_qpm = float(tempo)
            except ValueError:
                pass
        elif len(decoded_midi.tempos) > 0:
            initial_qpm = decoded_midi.tempos[0].qpm
            
        decoded_midi.tempos.clear()
        decoded_midi.tempos.append(symusic.Tempo(0, initial_qpm))
        print(f"Enforced stable constant tempo of {initial_qpm:.2f} BPM.")
        
    # Distribute staves/tracks
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
            
    keyboard_counts = {}
    for slot in slots:
        if slot["hand"] != "solo":
            keyboard_counts[slot["name"]] = keyboard_counts.get(slot["name"], 0) + 1
            
    keyboard_indices = {}
    for slot in slots:
        if slot["hand"] != "solo":
            total_keyboard_slots = keyboard_counts[slot["name"]]
            if total_keyboard_slots > 2:
                if slot["hand"] == "right":
                    keyboard_indices[slot["name"]] = keyboard_indices.get(slot["name"], 0) + 1
                slot["track_name"] = f"{slot['name'].title()} {keyboard_indices[slot['name']]}"
            else:
                slot["track_name"] = slot["name"].title()
        else:
            slot["track_name"] = slot["name"].title()
            
    num_tracks = len(decoded_midi.tracks)
    num_slots = len(slots)
    print(f"Applying custom instrument re-mapping. Distributing {num_tracks} tracks into {num_slots} slots...")
    
    new_tracks = []
    if num_slots >= num_tracks:
        for idx in range(num_tracks):
            slot = slots[idx]
            track = decoded_midi.tracks[idx]
            track.program = slot["program"]
            track.name = slot["track_name"]
            min_p, max_p = INSTRUMENT_RANGES.get(slot["name"], (21, 108))
            fit_track_to_range(track, min_p, max_p)
            new_tracks.append(track)
    else:
        indices_split = python_array_split(range(num_tracks), num_slots)
        for slot_idx, slot in enumerate(slots):
            group_indices = indices_split[slot_idx]
            print(f"  Slot '{slot['name']} ({slot['hand']})' merged from tracks: {group_indices}")
            merged_track = symusic.Track(program=slot["program"], name=slot["track_name"])
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
    
    # Export to MusicXML
    print("Converting MIDI output to MusicXML sheet music format...")
    try:
        score = music21.converter.parse(output_midi_path)
        group_keyboard_staves(score)
        
        # Ornament Detection
        print("Running baroque ornament detection...")
        detect_and_mark_ornaments(score)
        
        # Add tempo markings (Component 12)
        bpm = tempo if tempo else 120
        try:
            bpm = int(bpm)
        except ValueError:
            bpm = 120
        tag = baroque_tag if baroque_tag else user_inputs.get("mood", "")
        add_tempo_markings(score, bpm, tag)
        
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
        input_file = os.path.join(BASE_DIR, "input.json")
        user_inputs = load_input_json(input_file)
        generate_music(model_path, tokenizer, GENERATE_CONFIG, out_mid, out_xml, user_inputs=user_inputs)
    else:
        print(f"Model path {model_path} does not exist. Please train the model first.")
