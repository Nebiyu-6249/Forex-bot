import os, yaml, json
from pathlib import Path
from dotenv import load_dotenv

def load_env(): load_dotenv()

def env(key, default=None, required=False, cast=None):
    v = os.getenv(key, default)
    if required and (v is None or v == ''):
        raise RuntimeError(f'Missing env: {key}')
    if cast and v is not None:
        v = cast(v)
    return v

def load_config(path:str)->dict:
    import yaml
    with open(path,'r') as fh:
        return yaml.safe_load(fh)

def ensure_dirs(out_dir:str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir)/'logs').mkdir(exist_ok=True)
    (Path(out_dir)/'state').mkdir(exist_ok=True)

def jdump(obj,path):
    with open(path,'w') as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)

def jload(path, default=None):
    try:
        with open(path,'r') as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default
