import random
import numpy as np
np.random.seed(1337)
coordinates = np.random.randint(0, 100, size=(2, 5, 4))
print(coordinates)

print(coordinates.max(axis=(0,1)))
print(coordinates.min(axis=(0,1)))