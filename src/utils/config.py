import yaml
import os

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def save_config(config, save_path):
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
