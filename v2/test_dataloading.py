import argparse
import os
from data_utils.AHN3DataLoader import AHN3Dataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
from sklearn.model_selection import train_test_split

root = 'data/ahn3_set/'
rooms = sorted(os.listdir(root))
roomz = [room for room in rooms if 'azo' in room]
rooms = [room for room in rooms if 'ams' in room]

xTrain, xTest, yTrain, yTest = train_test_split(rooms, rooms, test_size = 0.2, random_state = 0, shuffle=True)
aTrain, aTest, bTrain, bTest = train_test_split(roomz, roomz, test_size = 0.2, random_state = 4, shuffle=True)

# with open('ams_test.txt', 'w') as f:
#     for s in xTest:
#         f.write(str(s) + '\n')

# with open('avo_test.txt', 'w') as f:
#     for s in aTest:
#         f.write(str(s) + '\n')

# sys.exit()

TRAIN_DATASET = AHN3Dataset(split='train', data_root=root, num_point=2048, train_area='azo', block_size=10.0, sample_rate=0.1, transform=None)

for i in range(len(TRAIN_DATASET)):
    sample = TRAIN_DATASET[i]
    # print(i, len(sample), sample[1])
print(len(TRAIN_DATASET))
