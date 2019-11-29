import pandas as pd

df = pd.read_csv('./datafile.csv')
print(df)

print(df['A'])

print(df['A'][:2])

# df['E'] = df['A'] + df['B']
# print(df['E'])

print(df.describe())