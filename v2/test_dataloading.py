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

root = 'data/ahn3_set/'
TRAIN_DATASET = AHN3Dataset(split='train', data_root=root, num_point=1024, test_area=5, block_size=10.0, sample_rate=1.0, transform=None)

for i in range(len(TRAIN_DATASET)):
    sample = TRAIN_DATASET[i]
    print(i, len(sample), sample[1])
