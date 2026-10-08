import pandas as pd
df = pd.read_csv("data/raw/monthly_reports.csv", dtype=str)
#print(df.head())
# print(df.shape)
print(df.isna().sum())