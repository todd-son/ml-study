import matplotlib.pyplot as plt
import numpy as np

x = [1, 3, 5, 7, 9]
y = [100, 200, 300, 400, 500]

plt.plot(x, y)
plt.show()

x = np.linspace(-np.pi, np.pi, 128)
y = np.cos(x)
plt.plot(x, y)
plt.show()
