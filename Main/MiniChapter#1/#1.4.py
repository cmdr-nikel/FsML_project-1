from Util import load_files
mb, not_mb = load_files()

from Util import load_mixed_fixed
train_csv = load_mixed_fixed()

#That how patterns were at #1.2, might be useful here
"""
patterns = {
    'org_pattern': r'^[ABNC]\d{10}$',                       # 11 sym - base pattern
    'es1_pattern': r'^[ABNC]\d{10}\d{2}$',                  # 13 sym - es1
    'es2_pattern': r'^[ABNC]\d{10}[A-Z0-9]{4}$',            # 15 sym - es2(suffix)
    'color_code':  r'^[ABNC]\d{10}\d{2}[A-Z0-9]{4}$'        # 17 sym - es1 + es2(colour)
}
"""

#trying simplest feature
train_csv['length'] = train_csv['article'].str.len() #checking for length
result = train_csv.groupby('label')['length'].agg(['mean', 'std', 'min', 'max'])
print(result)

"""
            mean       std  min  max
label                               
0       8.298773  2.362622    1   22 - those are not_mb
1      13.212640  1.987926   11   17 - and here are mb results 
"""
#diffrence between mb and non-mb is whole 5 digits(almost)/looking dope
#i`m bout to start testing patterns in here



