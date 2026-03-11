#initial Imports and Analysis

import pandas as pd

#Loading
"""Named first two df as md and non_mb case its Mercedes Bents, looks dope"""

mb = pd.read_csv('mercedes-benz 300k.txt', header=None, names=['article'], encoding='utf-8')
not_mb = pd.read_csv('not mercedes-benz 300k.txt', header=None, names=['article'], encoding='utf-8')

print("cleaning module")

#Data Cleaning/Filtration
"""Kinda useless here, but i liked PfDS lectures so why not"""

mb['article'] = mb['article'].str.strip()
not_mb['article'] = not_mb['article'].str.strip()

mb = mb[mb['article'].notna() & (mb['article'] != '')]
not_mb = not_mb[not_mb['article'].notna() & (not_mb['article'] != '')]


#Папа чуть перекинул на лишний десяток в Мерседес, лол
"""Taking close look on what we have inside (as if I cant open .txt myself)"""

print(f"Mercedes: {len(mb)} articles")
print(f"Не-Mercedes: {len(not_mb)} articles")

print(mb.head(10))
print(not_mb.head(10))

#'header=None, names=['article']' is working, cool.
print(mb.info())
print(not_mb.info())



"""Pattern research, even tho patter is technically known"""
print("Length of MB ", mb['article'].str.len().value_counts().sort_index())
print("Length of non-MB", not_mb['article'].str.len().value_counts().sort_index())

