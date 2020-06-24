#IMPORT FILES
import glob, os, sys
import pandas as pd
import numpy as np
import csv

from pathlib import Path
dirpath = os.getcwd()
# print(dirpath)
dirpath = Path('/Users/john/AI/Thesis/Data')
files = os.listdir(dirpath)
# print(files)
files = sorted([f for f in files if '.txt' in f])

min_x = np.zeros(len(files))
min_y = np.zeros(len(files))
max_x = np.zeros(len(files))
max_y = np.zeros(len(files))
# filename = np.zeros(len(files)).astype(str)
fileclass = np.zeros(len(files)).astype(str)
filename = []

e = 0
for f in files:
    print(f)
    data = np.loadtxt(os.path.join(dirpath, f), skiprows=1, delimiter=',', usecols=(0,1)).astype(np.float32)
    min_x[e] = np.min(data[:,0])
    min_y[e] = np.min(data[:,1])
    max_x[e] = np.max(data[:,0])
    max_y[e] = np.max(data[:,1])
    filename.append[f]
    if 'building' in f:
        if 'roof' in f:
            if 'window' in f:
                fileclass[e] = 6
            elif 'door' in f:
                fileclass[e] = 7
            else:
                fileclass[e] = 3
        elif 'window' in f:
            fileclass[e] = 4
        elif 'door' in f:
            fileclass[e] = 5
        else:
            fileclass[e] = 0
    elif 'ground' in f:
        if 'street' in f:
            fileclass[e] = 8
        elif 'sidewalk' in f:
            fileclass[e] = 9
        elif 'grass' in f:
            fileclass[e] = 10
        else:
            fileclass[e] = 2
    elif 'vegetation' in f:
        fileclass[e] = 1
    else:
        fileclass[e] = 11
    e += 1
# frame = pd.read_csv(first)
# print(frame.describe())
print('min x')
print(min_x)
print('min y')
print(min_y)
print('max x')
print(max_x)
print('max y')
print(max_y)
print('filename')
print(filename)
print('classes are')
print(fileclass)

print(np.min(min_x))
print(np.max(max_x))
print(np.max(max_x) - np.min(min_x))
print(np.min(min_y))
print(np.max(max_y))
print(np.max(max_y) - np.min(min_y))

statpath = os.path.join(dirpath, 'stats')
printname = os.path.join(statpath, 'min_x.txt')
with open(printname, 'w') as saver:
    print(min_x)
    np.savetxt(printname, min_x)
    saver.close()
    # for row in min_x:
    #     np.savetxt(printname, row)
    # saver.close()
printname = os.path.join(statpath, 'min_y.txt')
with open(printname, 'w') as saver:
    np.savetxt(printname, min_y)
    saver.close()
printname = os.path.join(statpath, 'max_x.txt')
with open(printname, 'w') as saver:
    np.savetxt(printname, max_x)
    saver.close()
printname = os.path.join(statpath, 'max_y.txt')
with open(printname, 'w') as saver:
    np.savetxt(printname, max_y)
    saver.close()
printname = os.path.join(statpath, 'filenames.txt')
with open(printname, 'w') as saver:
    for i in filename:
        saver.write(i + '\n')
    saver.close()
printname = os.path.join(statpath, 'classes.txt')
with open(printname, 'w') as saver:
    for i in fileclass:
        saver.write(i + '\n')
    saver.close()
sys.exit()
print(files)
print(os.path)
print(dirpath)

# for f in files:
#     print(f)

names = []
for file in files:
    names.append(file)


print('the filenames are: ',names)
#IMPORT FRAMEWORKS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#DATA

data = []
for f in all_files:
    frame = pd.read_csv(f)
    frame['filename'] = os.path.basename(f)
    data.append(frame)

testytest = np.loadtxt('building_01.txt', skiprows=1, delimiter=',', usecols=(0,1,2)).astype(np.float32)

print(testytest)
print('length is: ', len(testytest))

bigframe = pd.concat(data, ignore_index=True)

bigframe.loc[(bigframe['filename'].str.contains('roof')) & (bigframe['filename'].str.contains('door')), 'filename'] = 'roofdoor'
bigframe.loc[(bigframe['filename'].str.contains('roof')) & (bigframe['filename'].str.contains('window')), 'filename'] = 'roofwindow'
bigframe.loc[(bigframe['filename'].str.contains('roof')) & (bigframe['filename'].str.contains('building')), 'filename'] = 'roof'
bigframe.loc[(bigframe['filename'].str.contains('window')) & (bigframe['filename'].str.contains('building')), 'filename'] = 'window'
bigframe.loc[(bigframe['filename'].str.contains('door')) & (bigframe['filename'].str.contains('building')), 'filename'] = 'door'
bigframe.loc[bigframe['filename'].str.contains('building'), 'filename'] = 'building'

#Describing data set
print(bigframe.columns)
print(bigframe.describe())


cloud = bigframe.copy()
cloud = cloud[["//X","Y","Z", "filename"]]
cloud.columns = ['x', 'y', 'z', 'type']

labels = set(cloud['type'])
#print(labels)

train, validate, test = np.split(cloud.sample(frac=1), [int(.6*len(cloud)), int(.97*len(cloud))])

# #ALL FIGURE SETTINGS

# plt.rcParams['figure.figsize'] = [100, 50]

# color_dict = { 'building':'blue','door':'red','roof':'green','window':'yellow', 'roofdoor':'orange','roofwindow':'violet'}
# marker_dict = { 'building':'.','door':'o','roof':'o','window':'o', 'roofdoor':'o','roofwindow':'o'}

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # for kind in marker_dict:
# #     d = cloud[cloud.type==kind]
# #     ax.scatter(cloud['x'],cloud['y'],cloud['z'], c=[color_dict[i] for i in cloud['type']], marker = marker_dict[kind])
# #     print(kind)

# # plt.show()

# for kind in marker_dict:
#     d = test[test.type==kind]
#     ax.scatter(test['x'],test['y'],test['z'], c=[color_dict[i] for i in test['type']], marker = marker_dict[kind])
#     #print(kind)

# plt.show()

