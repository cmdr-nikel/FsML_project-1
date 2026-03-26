import pandas as pd
import joblib
from Util import predict_brand

#Some wierd behavior from csv file, done all i could, asked AI 'friend' to help, recommended os
#gonna fix it myself nad forever later
import os

# Delete Old File
if os.path.exists("fathers_file_fixed.csv"):
    os.remove("fathers_file_fixed.csv")
# To Create New
###

df = pd.read_csv("1M_parts_numbers.csv")
df.to_csv("fathers_file_fixed.csv", index=False)
print(f'Lines after format fix: {len(df)}')
print(df.head())


model = joblib.load("mercedes_model.pkl")
result = predict_brand(
    csv_input_path="fathers_file_fixed.csv",
    model=model,
    output_path="1M_labeled_report.csv"
)

"""==REPORTS=="""
print(f"\nTotal Lines: {len(result)}")
print(f"\nDistribution: ")
print(result["decision"].value_counts().to_string())

print(f"\nPercentage Ratio+L: ")
print((result["decision"].value_counts(normalize=True) * 100).round(2).to_string())

print(f"\nExamples Mercedes:")
print(result[result["decision"] == "mercedes"]["article"].head(20).to_string())

print(f"\nExamples manual_review ():")
print(result[result["decision"] == "manual_review"][["article", "prob_mercedes"]].head(20).to_string())




