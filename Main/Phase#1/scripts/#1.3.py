#CREATING A TRAINING SET
import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from utils.loaders import load_files

mb, not_mb = load_files()

mb["label"] = 1
not_mb["label"] = 0

# 150k balanced sample (75k + 75k)
mixed = pd.concat([
    mb.sample(n=75000, random_state=42),
    not_mb.sample(n=75000, random_state=42),
], ignore_index=True).sample(frac=1).reset_index(drop=True)

print(mixed.head(10))
print("\nFirst 10 labels:", mixed["label"].head(10).tolist())

#mixed.to_csv("mixed_train_300k.csv", index=False)

# 300k balanced sample (150k + 150k)
giga_mixed = pd.concat([
    mb.sample(n=150000, random_state=42),
    not_mb.sample(n=150000, random_state=42),
], ignore_index=True).sample(frac=1).reset_index(drop=True)

print(giga_mixed.head(10))
print("\nFirst 10 labels:", giga_mixed["label"].head(10).tolist())

#giga_mixed.to_csv("giga_mixed_train_600k.csv", index=False)
