#IMPORT FILES
import glob, os, sys
import pandas as pd
import numpy as np
import csv

from pathlib import Path

dirpath = Path('/Users/john/AI/Thesis/Data')
namepath = os.path.join(dirpath, 'stats')
classes = np.loadtxt(os.path.join(namepath, 'classes.txt'))
filenames = np.genfromtxt(os.path.join(namepath, 'filenames.txt'),dtype='str')
max_x = np.loadtxt(os.path.join(namepath, 'max_x.txt'))
max_y = np.loadtxt(os.path.join(namepath, 'max_y.txt'))
min_x = np.loadtxt(os.path.join(namepath, 'min_x.txt'))
min_y = np.loadtxt(os.path.join(namepath, 'min_y.txt'))

a = 315752
b = a + 25
k = 234759
l = k + 25

# c = filenames[((min_x > a) & (min_x < b) & (min_y > k) & (min_y < l)) | ((max_x > a) & (max_x < b) & (max_y > k) & (max_y < l))]
# d = min_x[((min_x > a) & (min_x < b) & (min_y > k) & (min_y < l)) | ((max_x > a) & (max_x < b) & (max_y > k) & (max_y < l))]
# e = filenames[((min_x > a) & (min_x < b) & (min_y > k) & (min_y < l))]
# f = filenames[((max_x > a) & (max_x < b) & (max_y > k) & (max_y < l))]
# idxs = ((min_x > a) & (min_x < b) & (min_y > k) & (min_y < l)) | ((max_x > a) & (max_x < b) & (max_y > k) & (max_y < l))
# g = ((max_x > a) & (max_x < b) & (max_y > k) & (max_y < l))

def define_idxs_in_tile(lower_x, lower_y, marge, min_x, min_y, max_x, max_y, upperbound = False):
    upper_x = lower_x + marge
    upper_y = lower_y + marge
    if upperbound:
        idx = ((min_x > lower_x) & (min_x < upper_x) & (min_y > lower_y) & (min_y < upper_y)) | ((max_x > lower_x) & (max_x < upper_x) & (max_y > lower_y) & (max_y < upper_y))
    else:
        idx = ((min_x > lower_x) & (min_x < upper_x) & (min_y > lower_y) & (min_y < upper_y))
    return idx

def make_tile(file_names, file_classes, path, lower_x, lower_y, marge, save=False):
    first = np.zeros((4))
    for i, f in enumerate(file_names):
        print(f)
        data = np.loadtxt(os.path.join(path, f), skiprows=1, delimiter=',', usecols=(0,1,2)).astype(np.float32)
        idx = define_idxs_in_tile(lower_x, lower_y, marge, data[:,0], data[:,1], data[:,0], data[:,1])
        data = data[idx]
        label = np.ones((data.shape[0],1)) * file_classes[i]
        test = np.hstack((data, label))
        first = np.vstack((first, test))
    tile = first[1:,]
    if save:
        save_location = path.joinpath('dub')
        save_location.mkdir(exist_ok=True)
        save_to = save_location.joinpath('dubd_'+str(lower_x)+'_'+str(lower_y)+'.npy')
        print(save_to)
        np.save(save_to, first)
    return tile

# save_location = dirpath.joinpath('dub')
# save_location.mkdir(exist_ok=True)
# save_to = save_location.joinpath('dubd_'+str(a)+'_'+str(a+25)+'.npy')
# print(save_to)
# np.save(save_to, first)

#make the indexes
indexes = define_idxs_in_tile(a, k, 25, min_x, min_y, max_x, max_y, upperbound = True)

files = filenames[indexes]
label_num = classes[indexes] 
# make tile
make_tile(files, classes, dirpath, a, k, 25,save=True)