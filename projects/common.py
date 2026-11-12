from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
SEED = 42

def output(name):
    p = ROOT / 'generated' / name
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_metrics(path, metrics):
    def encode(x):
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(metrics, indent=2, default=encode, allow_nan=False) + '\n', encoding='utf-8')

def finish(name, metrics):
    save_metrics(output(name) / 'metrics.json', metrics)
    plt.close('all')
    return metrics
