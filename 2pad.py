import numpy as np

proportion = 0.2

mask = np.random.choice([0, 1], size=(128, 128), p=[1-proportion, proportion])


print(mask)