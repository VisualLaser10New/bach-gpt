import os
import re
import symusic

CONTROL_TOKENS = [
    "GENRE_CANTATA", "GENRE_CHORALE", "GENRE_KEYBOARD", "GENRE_ORGAN",
    "GENRE_CHAMBER", "GENRE_CONCERTO", "GENRE_SUITE", "GENRE_FUGUE",
    "MOOD_VIVACE", "MOOD_ALLEGRO", "MOOD_ANDANTE", "MOOD_ADAGIO",
    "MOOD_LENTO", "MOOD_MAESTOSO", "MOOD_GRAZIOSO",
    "DENSITY_SPARSE", "DENSITY_MODERATE", "DENSITY_DENSE",
    "V2", "V3", "V4", "V5", "V6", "V8", "V10",
    "TEMPO_SLOW", "TEMPO_MEDIUM", "TEMPO_FAST",
    "TAG_MINUETTO", "TAG_PRELUDE", "TAG_FUGUE", "TAG_TOCCATA",
    "TAG_GAVOTTE", "TAG_ARIA", "TAG_PASSACAGLIA", "TAG_SARABANDE",
    "TAG_BOURREE", "TAG_GIGUE", "TAG_SICILIANA",
]

def extract_bwv_number(filename):
    """Parses BWV numbers from filenames like BWV_1041_01.mid or bwv1043.mid."""
    match = re.search(r"bwv[_\s-]*(\d+)", filename.lower())
    if match:
        return int(match.group(1))
    return None

def classify_genre(filepath, num_tracks, duration, bwv):
    """Classifies the baroque genre using filename keywords, BWV range, and tracks/duration."""
    basename = os.path.basename(filepath).lower()
    
    # 1. Filename keyword matching (highest specificity)
    if "organo" in basename or "organ" in basename:
        return "ORGAN"
    if "piano" in basename or "clavicembalo" in basename or "clavier" in basename or "harpsichord" in basename:
        return "KEYBOARD"
    if "violin" in basename or "cello" in basename or "flute" in basename or "flauta" in basename or "guitarra" in basename or "viola" in basename:
        return "CHAMBER"
    if "concerto" in basename:
        return "CONCERTO"
    if "suite" in basename or "partita" in basename:
        return "SUITE"
    if "chorale" in basename or "choral" in basename:
        return "CHORALE"
    if "fugue" in basename or "fuga" in basename:
        return "FUGUE"
        
    # 2. BWV range mapping (Bach Work Catalog classification)
    if bwv is not None:
        if 1 <= bwv <= 224:
            if num_tracks <= 4 and duration < 120:
                return "CHORALE"
            return "CANTATA"
        if 225 <= bwv <= 249:
            return "CANTATA"
        if 250 <= bwv <= 438:
            return "CHORALE"
        if 439 <= bwv <= 524:
            return "CHORALE"
        if 525 <= bwv <= 771:
            return "ORGAN"
        if 772 <= bwv <= 994:
            return "KEYBOARD"
        if 995 <= bwv <= 1000:
            return "KEYBOARD"  # Lute suites
        if 1001 <= bwv <= 1040:
            return "CHAMBER"   # Solo sonatas & partitas
        if 1041 <= bwv <= 1065:
            return "CONCERTO"
        if 1066 <= bwv <= 1071:
            return "SUITE"
        if 1072 <= bwv <= 1087:
            return "FUGUE"     # Art of Fugue, Musical Offering
            
    # 3. Fallback heuristics based on track count
    if num_tracks >= 6:
        return "CONCERTO"
    if num_tracks == 1:
        return "CHAMBER"
    return "KEYBOARD"

def classify_mood(avg_tempo, mode_is_minor, notes_per_beat):
    """Maps musical parameters to Baroque mood terms."""
    if avg_tempo >= 135:
        return "VIVACE" if not mode_is_minor else "ALLEGRO"
    elif avg_tempo >= 105:
        if notes_per_beat > 5.0:
            return "ALLEGRO"
        return "ANDANTE" if not mode_is_minor else "MAESTOSO"
    elif avg_tempo >= 72:
        return "ANDANTE" if notes_per_beat <= 3.5 else "GRAZIOSO"
    else:
        return "ADAGIO" if not mode_is_minor else "LENTO"

def classify_baroque_tag(filepath):
    """Detects movement/form tags like Minuetto, Gigue, Sarabande, etc."""
    basename = os.path.basename(filepath).lower()
    tags = {
        "minuet": "MINUETTO", "menuet": "MINUETTO",
        "prelude": "PRELUDE", "praeludium": "PRELUDE",
        "fugue": "FUGUE", "fuga": "FUGUE",
        "toccata": "TOCCATA",
        "gavotte": "TAG_GAVOTTE", # Wait, let's keep the naming aligned with vocabulary
        "aria": "ARIA",
        "passacaglia": "PASSACAGLIA",
        "sarabande": "SARABANDE",
        "bourree": "BOURREE", "bourrée": "BOURREE",
        "gigue": "GIGUE", "gig": "GIGUE",
        "siciliana": "SICILIANA", "siciliano": "SICILIANA"
    }
    for key, val in tags.items():
        if key in basename:
            # Strip standard prefix tags to match control token vocabulary exactly
            return val.replace("TAG_", "")
    return None

def analyze_piece(score, filepath):
    """
    Analyzes a symusic Score and filepath to produce control metadata.
    """
    # Track voice count (number of non-empty tracks)
    num_tracks = len([t for t in score.tracks if len(t.notes) > 0])
    
    # Calculate duration
    duration = 0.0
    if len(score.tempos) > 0:
        # Standard estimation: score.end is in ticks. tpq is ticks per quarter note.
        # total_quarters = score.end / score.ticks_per_quarter
        # duration = total_quarters * (60.0 / score.tempos[0].qpm)
        # symusic Score end is usually accurate. Let's do a simple estimation.
        avg_qpm = score.tempos[0].qpm if score.tempos[0].qpm > 0 else 120.0
        total_quarters = score.end() / (score.ticks_per_quarter or 384)
        duration = total_quarters * (60.0 / avg_qpm)
    else:
        duration = (score.end() / (score.ticks_per_quarter or 384)) * 0.5 # Default 120BPM
        
    bwv = extract_bwv_number(filepath)
    genre = classify_genre(filepath, num_tracks, duration, bwv)
    
    # Estimate average tempo
    tempos = [t.qpm for t in score.tempos if t.qpm > 0]
    avg_tempo = sum(tempos) / len(tempos) if len(tempos) > 0 else 120.0
    
    # Estimate major/minor mode
    # Standard heuristic: key_signatures in symusic
    mode_is_minor = False
    if len(score.key_signatures) > 0:
        # Check if mode is minor
        ks = score.key_signatures[0]
        # In symusic, key signature mode can be major/minor (represented as minor=True/False)
        mode_is_minor = getattr(ks, "minor", False)
    else:
        # Fallback: minor if 'm' or 'min' in filename or key representation
        basename = os.path.basename(filepath).lower()
        if "minor" in basename or " min" in basename or "_m_" in basename:
            mode_is_minor = True
            
    # Calculate note density: total notes / total duration
    total_notes = sum(len(t.notes) for t in score.tracks)
    # Notes per beat (quarter note)
    total_quarters = score.end() / (score.ticks_per_quarter or 384)
    notes_per_beat = (total_notes / total_quarters) if total_quarters > 0 else 0.0
    
    density = "MODERATE"
    if notes_per_beat > 8.0:
        density = "DENSE"
    elif notes_per_beat < 2.5:
        density = "SPARSE"
        
    mood = classify_mood(avg_tempo, mode_is_minor, notes_per_beat)
    
    # Voices count mapping to control tokens (V2, V3, etc.)
    voices_map = {1: "V2", 2: "V2", 3: "V3", 4: "V4", 5: "V5", 6: "V6", 7: "V6", 8: "V8", 9: "V8", 10: "V10"}
    voices_token = voices_map.get(num_tracks, "V4")
    if num_tracks > 10:
        voices_token = "V10"
        
    # Tempo category
    if avg_tempo < 80:
        tempo_cat = "SLOW"
    elif avg_tempo > 125:
        tempo_cat = "FAST"
    else:
        tempo_cat = "MEDIUM"
        
    tag = classify_baroque_tag(filepath)
    
    return {
        "genre": genre,
        "mood": mood,
        "density": density,
        "voices": voices_token,
        "tempo": tempo_cat,
        "tag": tag,
        "avg_tempo": avg_tempo
    }

def get_control_prefix(metadata):
    """Translates metadata dictionary to a list of control token strings."""
    prefix = [
        f"GENRE_{metadata['genre']}",
        f"MOOD_{metadata['mood']}",
        f"DENSITY_{metadata['density']}",
        metadata['voices'],
        f"TEMPO_{metadata['tempo']}"
    ]
    if metadata.get("tag"):
        prefix.append(f"TAG_{metadata['tag']}")
    return prefix
