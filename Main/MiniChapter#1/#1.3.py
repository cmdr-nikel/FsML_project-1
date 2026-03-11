#CREATING A TRAINING SET(3rd file)
import pandas as pd
import random as rnd #pandas works with random_states as well
from Util import load_files
mb, not_mb = load_files()


mb['label'] = 1
not_mb['label'] = 0

mb_sample = mb.sample(frac=0.5) #150k lines
#print(mb_sample.head())
"""
                article  label
143373      A2048172215      1
141857  A20473064018P26      1
...
"""

not_mb_sample = not_mb.sample(frac=0.5) #another 150k
#print(not_mb_sample.head())
"""
            article  label
135394      SDHD364      0
190158  11127550855      0
...
"""
#indexes are cool, but useless here (if not even harmful)

#mixed = pd.concat([mb_sample, not_mb_sample]) #there were not mixing at first
#mixed = mixed.sample(frac=1, random_state=42).reset_index(drop=True)
#print(mixed.head(20))
#.concat is great, but it's kinda ugly even by mine standards

mixed = pd.concat([
    mb.sample(n=75000, random_state=42), #at least not 67, im a man of culture
    not_mb.sample(n=75000, random_state=42)
], ignore_index=True).sample(frac=1).reset_index(drop=True) #bingo ##pround, even tho Ai 'friend' helped me a bit with that

print(mixed.head(10))
print("\nFirst 10 labels:", mixed["label"].head(10).tolist())

"""
            article  label
0   A21269700517M87      1
1         1298009SX      0
...

First 10: [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
"""

#mixed.to_csv('mixed_train_300k#.csv', index=False)
