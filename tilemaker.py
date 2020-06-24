#IMPORT FILES
import glob, os, sys
import pandas as pd
import numpy as np
import csv

from pathlib import Path

path = os.path.dirname(os.path.abspath(__file__))
data_root = 'data/dub'
tile_root = 'one'
# NW x = 315500 y = 234750 d g
# NE x = 315750 y = 234750 d g h
# SE x = 315750 y = 234500 d d
# SW x = 315500 y = 234500 d d
#TNW x = 316000 y = 233750 d d
#TSE x = 316250 y = 233500 d d
#TSW x = 316000 y = 233500 d d
#TNWPARK x = 316000 y = 234000 d d
a = 315750
b = a + 25
k = 234750
l = k + 25

# path = os.path.join(path, Path(data_root), Path(tile_root))
# print(path)


dirpath = Path('/Users/john/AI/Thesis/Data')
dirpath = os.path.join(path, Path(data_root), Path(tile_root))
# dirpath = os.path.join(path, Path(data_root))
# fl = os.listdir(dirpath)
# import glob
# import pathlib

# # for path, subdirs, files in os.walk(dirpath):
# #     if 'tiles' in path:
# #         for name in files:
#             # print(pathlib.PurePath(path, name))

# p = [Path(path).joinpath(name) for path, subdirs, files in os.walk(dirpath) if 'tiles' in path for name in files]
# print(len(p))
# s = [str(path+'\\'+name) for path, subdirs, files in os.walk(dirpath) if 'tiles' in path for name in files]
# print(len(s))
# print(s)
# r = [i for i in s if 'tnw' in i]
# print(len(r))
# test = np.load(p[0])
# print(test.shape)
 
# print(os.walk(dirpath))
# print(fl)
# sys.exit()
namepath = os.path.join(dirpath, 'stats')
classes = np.loadtxt(os.path.join(namepath, 'classes.txt'))
filenames = np.genfromtxt(os.path.join(namepath, 'filenames.txt'),dtype='str')
max_x = np.loadtxt(os.path.join(namepath, 'max_x.txt'))
max_y = np.loadtxt(os.path.join(namepath, 'max_y.txt'))
min_x = np.loadtxt(os.path.join(namepath, 'min_x.txt'))
min_y = np.loadtxt(os.path.join(namepath, 'min_y.txt'))



area_length = 250
tile_size = 25

x = int(area_length / tile_size)
t = np.zeros([2,x,x])

for j in range(x):
    for i in range(x):
        t[0][i][j] = i
        t[1][j][i] = i
tiles = np.stack((t[0],t[1]), axis=2).reshape(1,x*x,2).astype(int)[0]*tile_size
coordinates = [[a, k]]

tiles_2 = np.zeros((len(coordinates),tiles.shape[0],tiles.shape[1])).astype(int)

for i in range(len(coordinates)):
    tiles_2[i] = coordinates[i] + tiles

# print(tiles_2)
total = tiles_2[0]
# c = filenames[((min_x > a) & (min_x < b) & (min_y > k) & (min_y < l)) | ((max_x > a) & (max_x < b) & (max_y > k) & (max_y < l))]
# d = min_x[((min_x > a) & (min_x < b) & (min_y > k) & (min_y < l)) | ((max_x > a) & (max_x < b) & (max_y > k) & (max_y < l))]
# e = filenames[((min_x > a) & (min_x < b) & (min_y > k) & (min_y < l))]
# f = filenames[((max_x > a) & (max_x < b) & (max_y > k) & (max_y < l))]
# idxs = ((min_x > a) & (min_x < b) & (min_y > k) & (min_y < l)) | ((max_x > a) & (max_x < b) & (max_y > k) & (max_y < l))
# g = ((max_x > a) & (max_x < b) & (max_y > k) & (max_y < l))

def define_idxs_in_tile(lower_x, lower_y, marge, min_x, min_y, max_x, max_y):
    upper_x = lower_x + marge
    upper_y = lower_y + marge
    idx = np.logical_or(np.logical_or(np.logical_or(min_x > upper_x, max_x < lower_x),min_y > upper_y),max_y < lower_y)
    # idx = idx1 == idx2 == idx3 == idx4 == False
    return np.invert(idx)

def make_tile(file_names, file_classes, path, lower_x, lower_y, marge, save=False):
    first = np.zeros((4))
    for i, f in enumerate(file_names):
        print(f)
        data = np.loadtxt(os.path.join(path, f), skiprows=1, delimiter=',', usecols=(0,1,2)).astype(np.float32)
        try:
            idx = define_idxs_in_tile(lower_x, lower_y, marge, data[:,0], data[:,1], data[:,0], data[:,1])
        except:
            idx = define_idxs_in_tile(lower_x, lower_y, marge, data[0], data[1], data[0], data[1])
        data = data[idx]
        label = np.ones((data.shape[0],1)) * file_classes[i]
        test = np.hstack((data, label))
        first = np.vstack((first, test))
    tile = first[1:,]
    if save:
        save_location = Path(dirpath).joinpath('tiles')
        save_location.mkdir(exist_ok=True)
        save_to = save_location.joinpath('dubd_'+str(lower_x)+'_'+str(lower_y)+'.npy')
        print(save_to)
        np.save(save_to, tile)
    return tile

# save_location = dirpath.joinpath('dub')
# save_location.mkdir(exist_ok=True)
# save_to = save_location.joinpath('dubd_'+str(a)+'_'+str(a+25)+'.npy')
# print(save_to)
# np.save(save_to, first)

#make the indexes

for j in range(len(total)):
    indexes = define_idxs_in_tile(total[j][0], total[j][1], 25, min_x, min_y, max_x, max_y)
    print(indexes)
    files = filenames[indexes]
    label_num = classes[indexes] 
    # make tile
    make_tile(files, classes, dirpath, total[j][0], total[j][1], 25,save=True)