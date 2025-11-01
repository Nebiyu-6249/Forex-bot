import logging, sys
from pathlib import Path

def setup_logging(out_dir:str):
    logger = logging.getLogger('mt5fx'); logger.setLevel(logging.INFO); logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(ch)
    fh = logging.FileHandler(Path(out_dir)/'logs'/'mt5fx.log'); fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    return logger
