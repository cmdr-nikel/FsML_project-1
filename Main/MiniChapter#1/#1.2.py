#Rule Based Solution

import re
from Util import load_files
mb, not_mb = load_files()

"""
I decided to use regex library(it`s a big thing, apparently) 
for pattern description of MB articles(there are 4 of them)
"""

"""
A    164    720126    28    K56
[ABNC] \d{3}  \d{6}  \d{2}  [A-Z]\d{2}  (more-or-less)
"""

#dictionary instead of list. That was pain in ass indeed
patterns = {
    'org_pattern': r'^[ABNC]\d{10}$',                       # 11 sym - base pattern
    'es1_pattern': r'^[ABNC]\d{10}\d{2}$',                  # 13 sym - base + es1
    'es2_pattern': r'^[ABNC]\d{10}[A-Z0-9]{4}$',            # 15 sym + es2(suffix)
    'color_code':  r'^[ABNC]\d{10}\d{2}[A-Z0-9]{4}$'        # 17 sym + es1 + es2(colour)
}

def is_mb(article):
    for pattern in patterns.values():
        if re.match(pattern, article):
            return True
    return False

mb['regex_match'] = mb['article'].apply(is_mb)
not_mb['regex_match'] = not_mb['article'].apply(is_mb)

print(f"MB:     {mb['regex_match'].mean():%}")
print(f"Non-MB: {not_mb['regex_match'].mean():%}")
###At first got 'Non-MB: 7.666666666666667e-05' not good, then after i used :%
###It's 0.007667%, that means there ~20 articles that still passed """

false_positives = not_mb[not_mb['regex_match'] == True]
print(false_positives['article'].values)

"""
I was able to extract those specific articules:

['C2011391000' 'A7252703707' 'A307370112101' 'B3111101000' 'C2010140101'
 'C2011083000' 'A1087298247' 'B2110320300' 'A274100301020' 'A2741009009'
 'B3110310101' 'A2741009155' 'C2010700100' 'C2010690500' 'B2011410700'
 'C230100100055' 'C2010701400' 'C2010530101' 'A2741003020' 'C2011353000'
 'B3111041600' 'A0280160557' 'A2741011009']

Some of them are indeed MB (Or at least side-manufacturer ones)
[MIGHT BE IMPORTANT FOR LATER]
"""

