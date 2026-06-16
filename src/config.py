import os
import yaml

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML_PATH = os.path.join(BASE_DIR, "config.yaml")

# Load YAML file
try:
    with open(YAML_PATH, "r") as f:
        config_dict = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Error: Configuration file not found at {YAML_PATH}. Please ensure it exists.")
    raise

# Resolve paths
paths_config = config_dict["paths"]
DATASET_DIR = os.path.join(BASE_DIR, paths_config["dataset_dir"])
PROCESSED_DIR = os.path.join(BASE_DIR, paths_config["processed_dir"])
CHECKPOINT_DIR = os.path.join(BASE_DIR, paths_config["checkpoint_dir"])
TOKENIZER_PATH = os.path.join(BASE_DIR, paths_config["tokenizer_path"])
OUTPUT_DIR = os.path.join(BASE_DIR, paths_config["output_dir"])

# Ensure directories exist
for directory in [DATASET_DIR, PROCESSED_DIR, CHECKPOINT_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Parse tokenizer parameters (converting lists and string keys back to Python tuples)
t_config = config_dict["tokenizer"]
pitch_range = tuple(t_config["pitch_range"])
rest_range = tuple(t_config["rest_range"])

# Convert beat_res string keys like "0,4" to tuples (0, 4)
beat_res = {}
for k, v in t_config["beat_res"].items():
    try:
        tuple_key = tuple(map(int, k.split(",")))
        beat_res[tuple_key] = v
    except ValueError:
        # Fallback if key is already a tuple/numeric in structured format
        beat_res[k] = v

beat_res_rest = {}
for k, v in t_config["beat_res_rest"].items():
    try:
        tuple_key = tuple(map(int, k.split(",")))
        beat_res_rest[tuple_key] = v
    except ValueError:
        beat_res_rest[k] = v

TOKENIZER_PARAMS = {
    "pitch_range": pitch_range,
    "beat_res": beat_res,
    "num_velocities": t_config["num_velocities"],
    "use_chords": t_config["use_chords"],
    "use_rests": t_config["use_rests"],
    "rest_range": rest_range,
    "beat_res_rest": beat_res_rest,
    "use_tempos": t_config["use_tempos"],
    "use_time_signatures": t_config["use_time_signatures"],
    "use_programs": t_config["use_programs"],
}

# Export Model, Training, and Generation configurations directly
MODEL_CONFIG = config_dict["model"]
TRAIN_CONFIG = config_dict["training"]
GENERATE_CONFIG = config_dict["generate"]
