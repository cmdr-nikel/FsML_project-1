import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from utils.loaders import load_mixed_fixed

train_csv = load_mixed_fixed()

# Length is surprisingly discriminative: MB mean=13.2, non-MB mean=8.3
train_csv["length"] = train_csv["article"].str.len()
result = train_csv.groupby("label")["length"].agg(["mean", "std", "min", "max"])
print(result)

X = train_csv[["length"]]
y = train_csv["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print({accuracy_score(pred, y_test) * 100})
print(classification_report(y_test, pred))

"""
Feature ideas for the next step:
1. base pattern + es1 + es2 + colour flags
2. microfeatures:
   - starts with [ABNC]
   - first 3 are digits
   - last 2 are digits
   A1647201278:
     A     [ABNC]  ← MB prefix
     164   \d{3}  ← model
     720   \d{3}  ← group
      12   \d{2}  ← version
       78  \d{2}  ← revision
3. length
"""
