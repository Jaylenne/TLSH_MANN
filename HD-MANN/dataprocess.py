import numpy as np

data = np.load('./inference/5way1shot_inference.npy', allow_pickle=True)

print(data.shape)