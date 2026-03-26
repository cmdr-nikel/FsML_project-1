import pandas as pd

df_new = pd.read_csv('1M_labeled_report.csv')

df_new.index = range(1, len(df_new) + 1)
df_new.index.name = 'index'

with pd.ExcelWriter('1M_labeled_report_fixed.xlsx') as num_index:
    df_new.to_excel(num_index, index=True)

