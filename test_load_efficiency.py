import numpy as np

mask = np.array([
    [1, 0, 1, 0],
    [0, 0, 0, 1]
])

a = np.random.randint(0, 10, (2, 4, 3))
print(a[mask == 1])
