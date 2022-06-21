import numpy as np

D = 0.1

x_min = -1
x_max = 1
y_min = 0
y_max = 20

n_x = int(x_max - x_min) * 10 + 1
n_y = int(y_max - y_min) * 10 + 1

x = np.linspace(x_min, x_max, n_x)
y = np.linspace(y_min, y_max, n_y)
