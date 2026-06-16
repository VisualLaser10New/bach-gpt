import os
import glob
import shutil
import music21
import symusic
import copy
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def get_midi_files(directory):
    """Recursively list all MIDI files in a directory, deduplicating paths to handle case-insensitivity."""
    files = []
    for ext in ['*.mid', '*.midi', '*.MID', '*.MIDI']:
        matched = glob.glob(os.path.join(directory, '**', ext), recursive=True)
        files.extend([os.path.abspath(f) for f in matched])
    return sorted(list(set(files)))

def download_builtin_bach_corpus(target_dir):
    """
    Extracts Bach chorales and keyboard works from music21's built-in corpus
    and saves them as MIDI files in target_dir.
    """
    print("No files found in dataset/. Extracting default Bach corpus from music21...")
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all Bach works in the music21 corpus
    bach_paths = music21.corpus.getComposer('bach')
    print(f"Found {len(bach_paths)} Bach pieces in music21 corpus.")
    
    success_count = 0
    for path in tqdm(bach_paths[:150], desc="Extracting Bach files to MIDI"):
        try:
            score = music21.corpus.parse(path)
            base_name = os.path.splitext(os.path.basename(str(path)))[0]
            midi_path = os.path.join(target_dir, f"{base_name}.mid")
            score.write('midi', fp=midi_path)
            success_count += 1
        except Exception:
            continue
            
    print(f"Successfully extracted {success_count} Bach files to {target_dir}")

def transpose_midi(midi_path, output_dir):
    """
    Transpose a MIDI file into all 12 keys (from -6 to +5 semitones)
    and save them to output_dir using symusic (extremely fast C++ parser).
    """
    try:
        score = symusic.Score(midi_path)
        base_name = os.path.splitext(os.path.basename(midi_path))[0]
        
        # Transpose from -6 to +5 semitones (12 keys total)
        for semitones in range(-6, 6):
            suffix = f"_transposed_{semitones}" if semitones != 0 else ""
            out_name = f"{base_name}{suffix}.mid"
            out_path = os.path.join(output_dir, out_name)
            
            if semitones == 0:
                score.dump_midi(out_path)
            else:
                # Create a copy and shift pitch
                transposed_score = copy.deepcopy(score)
                transposed_score.shift_pitch(semitones)
                transposed_score.dump_midi(out_path)
    except Exception as e:
        print(f"Failed to transpose {midi_path} using symusic: {e}")

# Module-level helper for ProcessPoolExecutor pickling
def _transpose_worker(args):
    midi_path, output_dir = args
    transpose_midi(midi_path, output_dir)

def prepare_dataset(raw_dir, processed_dir):
    """
    Main preprocessing pipeline:
    1. Checks if raw_dir is empty. If so, downloads/extracts default Bach files.
    2. Transposes all files in raw_dir into processed_dir in parallel (12x augmentation).
    """
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Check if there are MIDI files in raw_dir
    midi_files = get_midi_files(raw_dir)
    
    if len(midi_files) == 0:
        default_dir = os.path.join(raw_dir, "default_bach")
        download_builtin_bach_corpus(default_dir)
        midi_files = get_midi_files(default_dir)
        
    print(f"Starting data preparation. Processing {len(midi_files)} files with 12x augmentation...")
    
    # Clear processed_dir first to avoid duplicates
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Run transposition in parallel using ProcessPoolExecutor for CPU speedup
    num_workers = min(os.cpu_count() or 4, 12)
    print(f"Spawning {num_workers} parallel processes for data transposition...")
    
    tasks = [(f, processed_dir) for f in midi_files]
    
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Consume the iterator to trigger processing and show progress
            list(tqdm(
                executor.map(_transpose_worker, tasks),
                total=len(tasks),
                desc="Augmenting dataset (transposing to 12 keys)"
            ))
    except Exception as e:
        print(f"Parallel augmentation failed: {e}. Falling back to sequential execution...")
        for f in tqdm(midi_files, desc="Augmenting dataset (sequential fallback)"):
            transpose_midi(f, processed_dir)
        
    augmented_files = get_midi_files(processed_dir)
    print(f"Data preparation complete. Total files in processed directory: {len(augmented_files)}")

if __name__ == "__main__":
    from src.config import DATASET_DIR, PROCESSED_DIR
    prepare_dataset(DATASET_DIR, PROCESSED_DIR)
