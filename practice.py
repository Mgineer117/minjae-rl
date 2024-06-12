import numpy as np

obs = np.arange(10)
print(obs)
obs = np.delete(obs, [], axis=-1)
print(obs)