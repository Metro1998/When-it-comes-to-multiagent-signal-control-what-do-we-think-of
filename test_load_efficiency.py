import numpy as np

mask = np.array([
    [[1, 3], [1, 21], [7, 3], [1, 1]],
    [[0, 3], [1, 4], [4, 3], [12, 3]]
])  # (2, 4, 2)

mean = np.array([
    2, 1
])

mask = np.array([mask[i] - mean[i] for i in range(2)])

print(mask)