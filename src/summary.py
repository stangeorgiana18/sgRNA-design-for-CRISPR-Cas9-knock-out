import pandas as pd

df = pd.read_csv("tp53_sgrnas.csv")
print(df["Efficiency"].describe())
df["Efficiency"].hist()