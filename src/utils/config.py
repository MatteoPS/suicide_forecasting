import yaml
from pathlib import Path
from functools import lru_cache

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

@lru_cache(maxsize=None) #forces Python to memorize the dictionary the first time it loads. Subsequent calls take 0 milliseconds because it never reads the disk again.
def load_config(config_name: str = "data.yaml") -> dict:
    """Loads and caches a YAML configuration file."""
    config_path = PROJECT_ROOT / "config" / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file missing: {config_path}")
    
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def get_data_path(file_key: str, data_folder: str = "raw") -> Path:
    """Convenience function to get the absolute path to a data file."""
    config = load_config("data.yaml")
    
    try:
        relative_path = config["paths"][data_folder][file_key]
        return PROJECT_ROOT / relative_path
    except KeyError:
        raise KeyError(f"Could not find paths -> {data_folder} -> {file_key} in data.yaml")