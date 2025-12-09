import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../../config/settings.yaml')

def load_config():
    """
    Loads the YAML configuration file.
    """
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
        
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        
    return config
