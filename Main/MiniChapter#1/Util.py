import pandas as pd

def load_files():
    mb = pd.read_csv('mercedes-benz 300k.txt', header=None, names=['article'], encoding='utf-8')
    not_mb = pd.read_csv('not mercedes-benz 300k.txt', header=None, names=['article'], encoding='utf-8')
    return mb, not_mb
