import pandas as pd

a = pd.DataFrame([1, 3, 5, 7, 9])
print(a)

b = pd.DataFrame({'Name': ['Cho', 'Kim', 'Lee'], 'Age': [28, 31, 38]})
print(b)

c = pd.DataFrame([['apple', 7000], ['banana', 5000], ['orange', 4000]])
print(c)

d = pd.DataFrame([['apple', 7000], ['banana', 5000], ['orange', 4000]], columns=['name', 'price'])
print(d)
