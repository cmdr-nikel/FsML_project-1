import pandas as pd
import os

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
labeled_path = os.path.join(_BASE, "Data", "processed", "1M_parts_numbers_labeled.csv")

df = pd.read_csv(labeled_path, dtype=str)

print(df['label'].value_counts())
print(df[df['label'] == 'manual_check'].sample(20).to_string())

mc = df[df['label'] == 'manual_check'][['mb_prob','bmw_prob','vag_prob']].astype(float)
print(mc.describe())

print(os.getcwd())
print(os.listdir())

import os, json
import pandas as pd
import plotly.express as px

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

candidates = [
    os.path.join(_BASE, "Data", "processed", "1M_parts_numbers_labeled.csv"),
    os.path.join(_BASE, "1M_parts_numbers_labeled.csv"),
    os.path.join(_BASE, "Data", "processed", "labeled_report.csv"),
    os.path.join(_BASE, "labeled_report.csv"),
]

labeled_path = next((p for p in candidates if os.path.exists(p)), None)
if labeled_path is None:
    raise FileNotFoundError("Could not find labeled CSV in expected locations")

outdir = os.path.join(_BASE, "output")
os.makedirs(outdir, exist_ok=True)

df = pd.read_csv(labeled_path, dtype=str)
for c in ["mb_prob", "bmw_prob", "vag_prob"]:
    df[c] = df[c].astype(float)

df["max_prob"] = df[["mb_prob", "bmw_prob", "vag_prob"]].max(axis=1)

bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 1.000001]
labels = ["0-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", "0.95-0.97", "0.97-0.98", "0.98-0.99", "0.99-1.0"]
df["prob_bin"] = pd.cut(df["max_prob"], bins=bins, labels=labels, include_lowest=True, right=False)

hist = df["prob_bin"].value_counts().reindex(labels).reset_index()
hist.columns = ["prob_bin", "count"]
hist["pct"] = (hist["count"] / len(df) * 100).round(4)
hist.to_csv(os.path.join(outdir, "max_prob_histogram.csv"), index=False)

fig = px.bar(hist, x="prob_bin", y="count", text="count")
fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
fig.update_layout(
    title="Max probability distribution (1M rows)<br><span style='font-size: 18px; font-weight: normal;'>Source: labeled CSV | confidence concentration</span>"
)
fig.update_xaxes(title_text="Prob bin")
fig.update_yaxes(title_text="Count")
fig.write_image(os.path.join(outdir, "max_prob_histogram.png"))
with open(os.path.join(outdir, "max_prob_histogram.png.meta.json"), "w") as f:
    json.dump(
        {
            "caption": "Max probability distribution (1M rows)",
            "description": "Bar chart of the maximum class probability across all labeled rows.",
        },
        f,
    )

mc = df[df["label"] == "manual_check"].copy()
top100 = mc["article"].astype(str).value_counts().head(100).reset_index()
top100.columns = ["article", "count"]
top100.to_csv(os.path.join(outdir, "manual_check_top100.csv"), index=False)

fig2 = px.bar(top100.sort_values("count", ascending=True), x="count", y="article", orientation="h")
fig2.update_layout(
    title="Top manual_check articles (top 100)<br><span style='font-size: 18px; font-weight: normal;'>Source: labeled CSV | repeated ambiguous patterns</span>"
)
fig2.update_xaxes(title_text="Count")
fig2.update_yaxes(title_text="Article")
fig2.write_image(os.path.join(outdir, "manual_check_top100.png"))
with open(os.path.join(outdir, "manual_check_top100.png.meta.json"), "w") as f:
    json.dump(
        {
            "caption": "Top manual_check articles (top 100)",
            "description": "Horizontal bar chart showing the most frequent articles sent to manual review.",
        },
        f,
    )

summary = pd.DataFrame(
    [
        ["rows_total", len(df)],
        ["manual_check_rows", len(mc)],
        ["manual_check_pct", round(len(mc) / len(df) * 100, 4)],
        ["rows_ge_0.98", int((df["max_prob"] >= 0.98).sum())],
        ["rows_ge_0.99", int((df["max_prob"] >= 0.99).sum())],
    ],
    columns=["metric", "value"],
)
summary.to_csv(os.path.join(outdir, "confidence_summary.csv"), index=False)

print(f"Loaded: {labeled_path}")
print(hist.to_string(index=False))
print(top100.head(10).to_string(index=False))


#FOR 0.95 THRESHOLD
"""Total: 999999 | manual_check: 131,576
label
vag             583202
bmw             263071
manual_check    131576
mercedes         22150
Name: count, dtype: int64
               article         label mb_prob bmw_prob vag_prob                        comment
507177           11334  manual_check     0.0     0.64     0.36  low confidence, review needed
311568          802571  manual_check     0.0     0.92     0.08  low confidence, review needed
923798          571447  manual_check     0.0     0.92     0.08  low confidence, review needed
164249           61210  manual_check     0.0     0.64     0.36  low confidence, review needed
42455             4303  manual_check     0.0     0.38     0.62  low confidence, review needed
2088    P999G12PLUS005  manual_check     0.0     0.21     0.79  low confidence, review needed
168302           77224  manual_check     0.0     0.64     0.36  low confidence, review needed
337624          428500  manual_check     0.0     0.92     0.08  low confidence, review needed
568135          802666  manual_check     0.0     0.92     0.08  low confidence, review needed
490510            9249  manual_check     0.0     0.38     0.62  low confidence, review needed
202377          125101  manual_check     0.0     0.92     0.08  low confidence, review needed
390440          135999  manual_check     0.0     0.92     0.08  low confidence, review needed
202944           15627  manual_check     0.0     0.64     0.36  low confidence, review needed
568329           63991  manual_check     0.0     0.64     0.36  low confidence, review needed
258621          6PK906  manual_check     0.0     0.58     0.42  low confidence, review needed
632201         500237T  manual_check     0.0     0.87     0.13  low confidence, review needed
710995          1628JF  manual_check     0.0     0.58     0.42  low confidence, review needed
742749           72290  manual_check     0.0     0.64     0.36  low confidence, review needed
487267          246026  manual_check     0.0     0.92     0.08  low confidence, review needed
361582         6PK2120  manual_check     0.0     0.86     0.14  low confidence, review needed
             mb_prob       bmw_prob       vag_prob
count  131576.000000  131576.000000  131576.000000
mean        0.000005       0.749339       0.250653
std         0.000343       0.185964       0.185952
min         0.000000       0.000000       0.060000
25%         0.000000       0.640000       0.080000
50%         0.000000       0.860000       0.140000
75%         0.000000       0.920000       0.360000
max         0.060000       0.940000       0.950000
/Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Scripts/models_scr
['predictor.py', 'service_corner.py', '__init__.py', 'unsuprv.py', 'classic.py', 'inference.py']
Loaded: /Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Data/processed/1M_parts_numbers_labeled.csv
 prob_bin  count     pct
    0-0.5      0  0.0000
  0.5-0.6  12812  1.2812
  0.6-0.7  47943  4.7943
  0.7-0.8    224  0.0224
  0.8-0.9  15004  1.5004
 0.9-0.95  55587  5.5587
0.95-0.97     24  0.0024
0.97-0.98      8  0.0008
0.98-0.99  14760  1.4760
 0.99-1.0 853637 85.3638
article  count
6PK1210     99
6PK1115     92
 4PK845     86
5PK1750     71
 4PK815     70
 4PK850     69
6PK1700     68
6PK1555     66
 6PK995     61
6PK1220     61"""

#FOR 0.9
"""Total: 999999 | manual_check: 75985 (офигеть разница)

label
vag 583291
bmw 318573
manual_check 75985
mercedes 22150
Name: count, dtype: int64
article label mb_prob bmw_prob vag_prob comment
921466 14672 manual_check 0.0 0.64 0.36 low confidence, review needed
880761 31BJ627 manual_check 0.0 0.86 0.14 low confidence, review needed
40248 6208R manual_check 0.0 0.4 0.6 low confidence, review needed
649740 87589 manual_check 0.0 0.64 0.36 low confidence, review needed
989047 4501 manual_check 0.0 0.38 0.62 low confidence, review needed
203870 4PK668 manual_check 0.0 0.58 0.42 low confidence, review needed
293806 A21R223410015 manual_check 0.0 0.18 0.82 low confidence, review needed
741786 77743 manual_check 0.0 0.64 0.36 low confidence, review needed
923556 05P661 manual_check 0.0 0.59 0.41 low confidence, review needed
11252 6PK1736 manual_check 0.0 0.86 0.14 low confidence, review needed
313664 42230 manual_check 0.0 0.64 0.36 low confidence, review needed
228624 0810 manual_check 0.0 0.38 0.62 low confidence, review needed
664335 04347 manual_check 0.0 0.64 0.36 low confidence, review needed
84357 70622 manual_check 0.0 0.64 0.36 low confidence, review needed
962817 20336 manual_check 0.0 0.64 0.36 low confidence, review needed
796859 94785 manual_check 0.0 0.64 0.36 low confidence, review needed
961891 02931 manual_check 0.0 0.64 0.36 low confidence, review needed
948082 40043 manual_check 0.0 0.64 0.36 low confidence, review needed
694468 7010E9 manual_check 0.0 0.59 0.41 low confidence, review needed
876617 51937 manual_check 0.0 0.64 0.36 low confidence, review needed
mb_prob bmw_prob vag_prob
count 75985.000000 75985.000000 75985.000000
mean 0.000008 0.625476 0.374513
std 0.000386 0.150764 0.150752
min 0.000000 0.100000 0.110000
25% 0.000000 0.580000 0.360000
50% 0.000000 0.640000 0.360000
75% 0.000000 0.640000 0.420000
max 0.020000 0.890000 0.900000
/Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Scripts/models_scr
['predictor.py', 'service_corner.py', '__init__.py', 'unsuprv.py', 'classic.py', 'inference.py']
Loaded: /Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Data/processed/1M_parts_numbers_labeled.csv
prob_bin count pct
0-0.5 0 0.0000
0.5-0.6 12812 1.2812
0.6-0.7 47943 4.7943
0.7-0.8 224 0.0224
0.8-0.9 15004 1.5004
0.9-0.95 55587 5.5587
0.95-0.97 24 0.0024
0.97-0.98 8 0.0008
0.98-0.99 14760 1.4760
0.99-1.0 853637 85.3638
article count
6PK1210 99
6PK1115 92
4PK845 86
5PK1750 71
4PK815 70
4PK850 69
6PK1700 68
6PK1555 66
6PK995 61
6PK1220 61

Process finished with exit code 0"""

#BERDIKT - AHUENA
#0.85
"""Total: 999999 | manual_check: 62142
label
vag             583299
bmw             332408
manual_check     62142
mercedes         22150
Name: count, dtype: int64
       article         label mb_prob bmw_prob vag_prob                        comment
464179    4851  manual_check     0.0     0.38     0.62  low confidence, review needed
117440  6441S6  manual_check     0.0     0.59     0.41  low confidence, review needed
780398   15611  manual_check     0.0     0.64     0.36  low confidence, review needed
181923  6466NG  manual_check     0.0     0.58     0.42  low confidence, review needed
276529  15F470  manual_check     0.0     0.59     0.41  low confidence, review needed
400546   01654  manual_check     0.0     0.64     0.36  low confidence, review needed
30958    50200  manual_check     0.0     0.64     0.36  low confidence, review needed
221060  7401JR  manual_check     0.0     0.58     0.42  low confidence, review needed
769526  0816G3  manual_check     0.0     0.59     0.41  low confidence, review needed
33681    43733  manual_check     0.0     0.64     0.36  low confidence, review needed
147691   87872  manual_check     0.0     0.64     0.36  low confidence, review needed
880184    4445  manual_check     0.0     0.38     0.62  low confidence, review needed
229915    8889  manual_check     0.0     0.38     0.62  low confidence, review needed
87053    20822  manual_check     0.0     0.64     0.36  low confidence, review needed
465445  17381N  manual_check     0.0     0.59     0.41  low confidence, review needed
165256   98107  manual_check     0.0     0.64     0.36  low confidence, review needed
245204   90765  manual_check     0.0     0.64     0.36  low confidence, review needed
151212   19031  manual_check     0.0     0.64     0.36  low confidence, review needed
170633   60497  manual_check     0.0     0.64     0.36  low confidence, review needed
562746   20668  manual_check     0.0     0.64     0.36  low confidence, review needed
            mb_prob      bmw_prob      vag_prob
count  62142.000000  62142.000000  62142.000000
mean       0.000009      0.572669      0.427317
std        0.000426      0.111399      0.111385
min        0.000000      0.160000      0.150000
25%        0.000000      0.580000      0.360000
50%        0.000000      0.640000      0.360000
75%        0.000000      0.640000      0.420000
max        0.020000      0.850000      0.840000
/Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Scripts/models_scr
['predictor.py', 'service_corner.py', '__init__.py', 'unsuprv.py', 'classic.py', 'inference.py']
Loaded: /Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Data/processed/1M_parts_numbers_labeled.csv
 prob_bin  count     pct
    0-0.5      0  0.0000
  0.5-0.6  12812  1.2812
  0.6-0.7  47943  4.7943
  0.7-0.8    224  0.0224
  0.8-0.9  15004  1.5004
 0.9-0.95  55587  5.5587
0.95-0.97     24  0.0024
0.97-0.98      8  0.0008
0.98-0.99  14760  1.4760
 0.99-1.0 853637 85.3638
article  count
 4PK845     86
 4PK815     70
 4PK850     69
 6PK995     61
 5PK970     59
   1001     53
   2262     43
 4PK890     43
   9005     41
 75D23L     41
"""

"""
| Порог | manual_check | % от 1M | BMW bias в manual_check | Verdict                                              |
| ----- | ------------ | ------- | ----------------------- | -----------------------------------------------------|
| 0.95  | 131,576      | 13.2%   | 0.75                    | Too conservative labeled_report.csv                  |
| 0.90  | 75,985       | 7.6%    | 0.63                    | Good labeled_report.csv                              |
| 0.85  | 62,142       | 6.2%    | 0.57                    | Better labeled_report.csv                            |
| 0.80  | 60,983       | 6.1%    | 0.58                    | automatisation stopped here, so 0.85 might be better | (IDK)
0.7 -- 6.07%
"""