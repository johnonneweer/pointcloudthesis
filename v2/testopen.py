import numpy as np
import os
import torch

path = '/Users/john/AI/Thesis/pointcloudthesis/pointcloudthesis/v2/data/stanford_indoor3d'
filen = 'Area_1_conferenceRoom_1.npy'

room_path = os.path.join(path, filen)
test = np.load(room_path)

print(test[0][6])
print(test)


print(test.shape)
print(np.amin(test, axis=0)[0:3])

print(torch.cuda.is_available())