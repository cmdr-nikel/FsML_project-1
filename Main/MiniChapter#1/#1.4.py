from sklearn.linear_model import LogisticRegression
    # ^-for practice purposes, simple binary choice that is enough,
    # but in real case i need to switch for smth more complicated (Y/N/Mb)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from Util import load_files
mb, not_mb = load_files()

from Util import load_mixed_fixed
train_csv = load_mixed_fixed()


#That how patterns were at #1.2, might be useful here
"""
patterns = {
    'org_pattern': r'^[ABNC]\d{10}$',                       # 11 sym - base pattern
    'es1_pattern': r'^[ABNC]\d{10}\d{2}$',                  # 13 sym - es1 (version)
    'es2_pattern': r'^[ABNC]\d{10}[A-Z0-9]{4}$',            # 15 sym - es2 ()
    'color_code':  r'^[ABNC]\d{10}\d{2}[A-Z0-9]{4}$'        # 17 sym - es1 + es2
    #two more from AI research on mercedes patterns 
    'color_suffix': r'^[ABNC]\d{10}\d{2}[A-Z]{1,4}$',       # es1 + colour (1-4 letters)
    'short_color': r'^[ABNC]\d{10}[A-Z]{1,4}$'             # base + short colour
    #including UNITED pattern
    'master_pattern' = r'^[ABNC0]\d{10}([A-Z0-9]{0,2}([A-Z0-9]{0,4})?)?$'

}
"""

#Putting WHOLE patterns as model might be harmful, i better split them on little ones
#Or no
#IDK
#Might try few test, if i made slearn work, but MB better keep both

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
#diffrence between mb and non-mb is whole 5 digits(almost)/looking dope, mb even leaving it as a feature
#i`m bout to start testing patterns in here

train_csv = load_mixed_fixed()
train_csv['length'] = train_csv['article'].str.len()

X = train_csv[['length']] #featue here
y = train_csv['label'] #labels/correct answers

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#LEaving 20% for testing, rest for learning

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print({accuracy_score(pred, y_test)*100})
print(classification_report(y_test, pred))



"""
So, 'by the end of a day', what features am i leaving?
1.base pattern + es1 + es2 + colour
2.microfeatures, like:
    2.1.beginning with a letter 
    2.2.three num digits first
    2.3.last two digits are num
    Like this:
A1647201278
^ A     [ABNC]  ← prefix MB
  164   \d{3}  ← model
    720 \d{3}  ← group  
       12 \d{2} ← version
          78 \d{2} ← number of party $
3.length?

And at last it is somewhat like
df['pattern_type'] = np.select([
    df['article'].str.match('org_pattern'), 
    df['article'].str.match('es1_pattern'),
    ...
"""