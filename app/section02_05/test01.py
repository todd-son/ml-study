import pandas as pd
import numpy as np

a = pd.Series([1, 3, 5, 7, 10])
print(a)

data = np.array(['a', 'b', 'c', 'd'])
b = pd.DataFrame(data)
print(b)

c = pd.Series(np.arange(10, 30, 5))
print(c)

d = pd.Series(['a', 'b', 'c'], index=[10, 20, 30])
print(d)

e = pd.Series({'a': 10, 'b': 20, 'c': 30})
print(e)
