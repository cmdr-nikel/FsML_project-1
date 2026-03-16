from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

from Util import load_files
mb, not_mb = load_files()

from Util import load_mixed_fixed
train_csv = load_mixed_fixed()

from Util import load_giga_mixed_fixed
giga_train_csv = load_giga_mixed_fixed()

from Util import build_feature_matrix, load_mixed_fixed
train_df = load_mixed_fixed()
X_all = build_feature_matrix(train_df)
y_all = train_df["label"]



train_df = load_giga_mixed_fixed()
X_all = build_feature_matrix(train_df)
y_all = train_df["label"]

# 60% train / 20% val / 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.4, random_state=42, stratify=y_all)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# training section
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# hard predictions + evaluation
pred_val = model.predict(X_val)
print(f"Val accuracy: {accuracy_score(y_val, pred_val):.3f}")
print(classification_report(y_val, pred_val))

# soft probabilities + three decision zones
threshold_high = 0.90
threshold_low  = 0.50

probs_val = model.predict_proba(X_val)[:, 1]  # P(Mercedes) for each article

print(f"\nMercedes      (>= 0.90): {(probs_val >= threshold_high).sum()}")
print(f"Manual review (0.50-0.90): {((probs_val > threshold_low) & (probs_val < threshold_high)).sum()}")
print(f"Not Mercedes  (<= 0.50): {(probs_val <= threshold_low).sum()}")











#Servise Section
"""
mask = (probs_val > threshold_low) & (probs_val < threshold_high)

# indexes
uncertain_idx = X_val[mask].index

# mistery articles
print(train_df.loc[uncertain_idx, ["article", "label"]])
print(f"\nProbability: {probs_val[mask]}")
"""

"""
            article  label
110944  X0129150022      0
85797   Z9999990198      0

Probability: [0.51048099 0.74068329]
"""

"""
non_mb = train_df[train_df["label"] == 0]
X_non_mb = build_feature_matrix(non_mb)
print(X_non_mb["matches_core_pattern"].sum())
print(X_non_mb["prefix_is_mb_set"].sum())
"""

"""
#weight of a features(why 1.00 on training)
feature_names = X_train.columns.tolist()
coefs = model.coef_[0]

importance = pd.DataFrame({
    "feature": feature_names,
    "weight": coefs
}).sort_values("weight", ascending=False)

print(importance)
"""

"""                    feature    weight
10         prefix_is_mb_set  7.222313
9                has_prefix  5.422108
4   first_char_is_mb_prefix  4.294320
3      first_char_is_letter  1.221151
11                 core_len  0.305138
8      matches_core_pattern  0.030514
15             revision_int  0.012052
14              version_int  0.009530
7             suffix_digits  0.007641
13                group_int  0.005994
2                num_digits  0.000965
12                model_int -0.000427
6            suffix_letters -0.469382
1               num_letters -0.470561
0               article_len -0.471955
5            prefix_letters -3.868158
16    possible_truncated_mb -5.391595

#########

СИЛЬНЫЕ СИГНАЛЫ "ЭТО MB":
prefix_is_mb_set   +7.22  ← главный сигнал: буква A/B/N/C
has_prefix         +5.42  ← есть вообще буква перед цифрами
first_char_is_mb_prefix +4.29  ← дублирует первое (похожая фича)

СИЛЬНЫЕ СИГНАЛЫ "НЕ MB":
possible_truncated_mb  -5.39  ← без буквы + 10 цифр = скорее НЕ MB
prefix_letters         -3.87  ← много букв в начале = НЕ MB ("FILTEROIL...")
suffix_letters         -0.47  ← буквы в конце = немного против MB
article_len            -0.47  ← длинная строка = немного против MB (???)
"""

