

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

"""
label
unknown_article    693974
vag                242346
bmw                 38573
mercedes            22150
manual_check         2956
Name: count, dtype: int64
                     article         label mb_prob bmw_prob vag_prob                        comment
600071       236031384311200  manual_check     0.0     0.27     0.73  low confidence, review needed
245314       316300374301110  manual_check     0.0     0.25     0.75  low confidence, review needed
148102       211201001368008  manual_check     0.0     0.25     0.75  low confidence, review needed
884081  42000374100160230002  manual_check     0.0     0.49     0.51  low confidence, review needed
652198       236020350207000  manual_check     0.0     0.27     0.73  low confidence, review needed
325232     30015048746000020  manual_check     0.0     0.43     0.57  low confidence, review needed
667839       210800101200508  manual_check     0.0     0.23     0.77  low confidence, review needed
382422       111801041082008  manual_check     0.0     0.23     0.77  low confidence, review needed
916715     15095060110951001  manual_check     0.0     0.41     0.59  low confidence, review needed
637511       600200382900000  manual_check     0.0     0.26     0.74  low confidence, review needed
27128        121210000401999  manual_check     0.0     0.26     0.74  low confidence, review needed
759433       040524100040001  manual_check     0.0     0.25     0.75  low confidence, review needed
421042       316200290604400  manual_check     0.0     0.26     0.74  low confidence, review needed
656227       000010011983738  manual_check     0.0     0.26     0.74  low confidence, review needed
129093       111114600501999  manual_check     0.0     0.26     0.74  low confidence, review needed
878546       316300840326111  manual_check     0.0     0.25     0.75  low confidence, review needed
878530       316300110919200  manual_check     0.0     0.25     0.75  low confidence, review needed
814810       316300340805000  manual_check     0.0     0.25     0.75  low confidence, review needed
442336       210802901056008  manual_check     0.0     0.24     0.76  low confidence, review needed
533007       000010060443219  manual_check     0.0     0.26     0.74  low confidence, review needed
       mb_prob     bmw_prob     vag_prob
count   2956.0  2956.000000  2956.000000
mean       0.0     0.264516     0.735477
std        0.0     0.039639     0.039638
min        0.0     0.230000     0.500000
25%        0.0     0.250000     0.740000
50%        0.0     0.260000     0.740000
75%        0.0     0.260000     0.750000
max        0.0     0.500000     0.770000
/Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Scripts/models_scr
['predictor.py', 'service_corner.py', '__init__.py', 'unsuprv.py', 'classic.py', 'inference.py']
Loaded: /Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Data/processed/1M_parts_numbers_labeled.csv
 prob_bin  count     pct
    0-0.5      0  0.0000
  0.5-0.6     87  0.0087
  0.6-0.7     76  0.0076
  0.7-0.8   2793  0.2793
  0.8-0.9  11201  1.1201
 0.9-0.95   5203  0.5203
0.95-0.97   4872  0.4872
0.97-0.98     28  0.0028
0.98-0.99    134  0.0134
 0.99-1.0 281631 28.1631
        article  count
212302201012060      4
402613170107003      4
330202120100840      4
040600101200600      3
316006110908000      3
220695220301010      3
642263502136010      3
000010061050118      2
000010061041118      2
040520370700810      2
Loaded: /Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Data/processed/1M_parts_numbers_labeled.csv
 prob_bin  count     pct
    0-0.5      0  0.0000
  0.5-0.6     87  0.0087
  0.6-0.7     76  0.0076
  0.7-0.8   2793  0.2793
  0.8-0.9  11201  1.1201
 0.9-0.95   5203  0.5203
0.95-0.97   4872  0.4872
0.97-0.98     28  0.0028
0.98-0.99    134  0.0134
 0.99-1.0 281631 28.1631
        article  count
212302201012060      4
402613170107003      4
330202120100840      4
040600101200600      3
316006110908000      3
220695220301010      3
642263502136010      3
000010061050118      2
000010061041118      2
040520370700810      2

Process finished with exit code 0
"""

#####DO FIXA BMW
"""
Full dataset size: (900000, 2)
brand
bmw         300000
mercedes    300000
vag         300000
Name: count, dtype: int64
Filtering unknown articles by pattern...
True unknowns after filter: 217,194
Full dataset size after adding unknowns: (1117194, 2)
brand
bmw                300000
mercedes           300000
vag                300000
unknown_article    217194
Name: count, dtype: int64
Train: 670316 | Val: 223439 | Test: 223439
brand
vag                180000
bmw                180000
mercedes           180000
unknown_article    130316
Name: count, dtype: int64

Extracting features...
Done. Shape: (670316, 51)
No NaNs. Ready for scaling.
Fixed shapes: (670316, 51) (223439, 51) (223439, 51)

--- C tuning ---

Training with C=0.01 (pre-validated)...
Done.

--- Val set results ---
Accuracy: 0.9936537488979095
Macro F1: 0.992995914882518

Classification report:
                 precision    recall  f1-score   support

            bmw       1.00      0.98      0.99     60000
       mercedes       1.00      1.00      1.00     60000
unknown_article       0.97      1.00      0.98     43439
            vag       1.00      1.00      1.00     60000

       accuracy                           0.99    223439
      macro avg       0.99      0.99      0.99    223439
   weighted avg       0.99      0.99      0.99    223439


Confusion matrix (MB / BMW / VAG):
[[60000     0     0     0]
 [    0 58935     0  1065]
 [    0     3 59839   158]
 [    0   192     0 43247]]

--- TEST set results (final) ---
Accuracy: 0.9933628417599434
Macro F1: 0.9926743512954425

Classification report:
                 precision    recall  f1-score   support

            bmw       1.00      0.98      0.99     60000
       mercedes       1.00      1.00      1.00     60000
unknown_article       0.97      1.00      0.98     43439
            vag       1.00      1.00      1.00     60000

       accuracy                           0.99    223439
      macro avg       0.99      0.99      0.99    223439
   weighted avg       0.99      0.99      0.99    223439


Confusion matrix (MB / BMW / VAG / UNK):
[[60000     0     0     0]
 [    0 58875     0  1125]
 [    0     0 59833   167]
 [    0   191     0 43248]]

Model saved → /Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Models/linearcvs/linearsvc_atom_1117k_4cls.pkl
Saved 51 features to /Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Scripts/models_scr/../features_scr/feature_order.npy

Process finished with exit code 0
"""

#MORE STATISTICS
"""
Classification report:
precision recall f1-score support

bmw 1.00 0.98 0.99 60000
mercedes 1.00 1.00 1.00 60000
unknown_article 0.97 1.00 0.98 43439
vag 1.00 1.00 1.00 60000

accuracy 0.99 223439
macro avg 0.99 0.99 0.99 223439
weighted avg 0.99 0.99 0.99 223439


Confusion matrix (MB / BMW / VAG / UNK):
[[60000 0 0 0]
[ 0 58875 0 1125]
[ 0 0 59833 167]
[ 0 204 0 43235]]

Model saved → /Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Models/linearcvs/linearsvc_atom_1117k_4cls.pkl
Saved 50 features to /Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Scripts/models_scr/../features_scr/feature_order.npy

=== LOST BMW ARTICLES ===
Total lost: 1125

First 2 digits distribution:
article
51 207
61 152
85 151
65 102
80 74
36 64
32 52
66 50
83 32
11 26
63 26
62 25
52 19
34 17
64 16
Name: count, dtype: int64

Length distribution:
article
11 1125
Name: count, dtype: int64

Sample (20):
['34515A36BF1', '64125A4BA57', '54105A0C758', '24405A131F7', '80145A0A480', '85455A09F08', '65125A38AB8', '85455A2E1B6', '62105A2A3E0', '51215A0C771', '36115A276D4', '65825A12F52', '67625A3F285', '83305A40808', '85455A102A7', '65775A1FC42', '61315A25474', '66125A40695', '65125A318D8', '85455A17E06']

Process finished with exit code 0
"""

#NOW

"""
Full dataset size: (900000, 2)
brand
bmw         300000
mercedes    300000
vag         300000
Name: count, dtype: int64
Filtering unknown articles by pattern...
True unknowns after filter: 217,063
Full dataset size after adding unknowns: (1117063, 2)
brand
bmw                300000
vag                300000
mercedes           300000
unknown_article    217063
Name: count, dtype: int64
Train: 670237 | Val: 223413 | Test: 223413
brand
bmw                180000
mercedes           180000
vag                180000
unknown_article    130237
Name: count, dtype: int64

Extracting features...
Done. Shape: (670237, 51)
No NaNs. Ready for scaling.
Fixed shapes: (670237, 51) (223413, 51) (223413, 51)

--- C tuning ---

Training with C=0.01 (pre-validated)...
Done.

--- Val set results ---
Accuracy: 0.9991898412357383
Macro F1: 0.9991023939546126

Classification report:
                 precision    recall  f1-score   support

            bmw       1.00      1.00      1.00     60000
       mercedes       1.00      1.00      1.00     60000
unknown_article       1.00      1.00      1.00     43413
            vag       1.00      1.00      1.00     60000

       accuracy                           1.00    223413
      macro avg       1.00      1.00      1.00    223413
   weighted avg       1.00      1.00      1.00    223413


Confusion matrix (MB / BMW / VAG):
[[60000     0     0     0]
 [    0 59977     0    23]
 [    0     0 59843   157]
 [    0     0     1 43412]]

--- TEST set results (final) ---
Accuracy: 0.9992659334953651
Macro F1: 0.9991881954567304

Classification report:
                 precision    recall  f1-score   support

            bmw       1.00      1.00      1.00     60000
       mercedes       1.00      1.00      1.00     60000
unknown_article       1.00      1.00      1.00     43413
            vag       1.00      1.00      1.00     60000

       accuracy                           1.00    223413
      macro avg       1.00      1.00      1.00    223413
   weighted avg       1.00      1.00      1.00    223413


Confusion matrix (MB / BMW / VAG / UNK):
[[60000     0     0     0]
 [    0 59985     0    15]
 [    0     2 59851   147]
 [    0     0     0 43413]]

Model saved → /Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Models/linearcvs/linearsvc_atom_1117k_4cls.pkl
Saved 51 features to /Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Scripts/models_scr/../features_scr/feature_order.npy
Traceback (most recent call last):
  File "/Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Scripts/models_scr/classic.py", line 177, in <module>
    main()
  File "/Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Scripts/models_scr/classic.py", line 174, in main
    print(lost_articles.sample(20).tolist())
          ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/pythonProject/lib/python3.11/site-packages/pandas/core/generic.py", line 6000, in sample
    sampled_indices = sample.sample(obj_len, size, replace, weights, rs)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/pythonProject/lib/python3.11/site-packages/pandas/core/sample.py", line 161, in sample
    return random_state.choice(obj_len, size=size, replace=replace, p=weights).astype(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "numpy/random/mtrand.pyx", line 1025, in numpy.random.mtrand.RandomState.choice
ValueError: Cannot take a larger sample than population when 'replace=False'

=== LOST BMW ARTICLES ===
Total lost: 15

First 2 digits distribution:
article
51    10
52     1
36     1
65     1
32     1
64     1
Name: count, dtype: int64

Length distribution:
article
11    15
Name: count, dtype: int64

Sample (20):

Process finished with exit code 1
"""

#PREDICTOR 0.7 (FORGOT TO SWITCH MODEL LOL)
"""
Model loaded: /Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Models/linearcvs/linearsvc_atom_1184k_4cls.pkl
Reading input file: /Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Data/original/1M_parts_numbers.csv
Loaded 999,999 rows
Extracting features...
Features shape: (999999, 51)
Features fixed to model schema: (999999, 52)
Running predictions...
  vag                 138,629  (13.9%)
  bmw                  41,481  (4.1%)
  mercedes             22,150  (2.2%)
  manual_check        103,133  (10.3%)
  unknown_article     694,606  (69.5%)
Saved → /Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Data/processed/1M_parts_numbers_labeled.csv
Total: 999,999 | manual_check: 103,133 | unknown_article: 694,606 | threshold: 0.7"""

#PREDICTOR 0.85 MANUAL_CHECK IS DEAD (DEAR GOD)
"""
Model loaded: /Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Models/linearcvs/linearsvc_atom_1117k_4cls.pkl
Reading input file: /Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Data/original/1M_parts_numbers.csv
Loaded 999,999 rows
Extracting features...
Features shape: (999999, 51)
Features fixed to model schema: (999999, 51)
Running predictions...
  vag                 245,302  (24.5%)
  bmw                  39,182  (3.9%)
  mercedes             22,150  (2.2%)
  manual_check              0  (0.0%)
  unknown_article     693,365  (69.3%)
Saved → /Users/nikitadackov/PycharmProjects/FsML_project-1/Main/Phase#2/Data/processed/1M_parts_numbers_labeled.csv
Total: 999,999 | manual_check: 0 | unknown_article: 693,365 | threshold: 0.85

Process finished with exit code 0"""
