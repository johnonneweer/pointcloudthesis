#IMPORT FILES
import glob, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

dirpath = os.getcwd()

os.chdir(dirpath + "/pointcloudthesis/B1/")

path = "/Users/john/AI/Thesis/pointcloudthesis/pointcloudthesis/B1/"
path2 = "/Users/john/AI/Thesis/pointcloudthesis/pointcloudthesis/B1/B/"

files = [f for f in glob.glob(path + "**/*.txt", recursive=True)]




def load_building_data_insert_class(f):
    point_file = np.loadtxt(f, skiprows=1, delimiter=',', usecols=(0,1,2)).astype(np.float32)
    if 'building' in f:
        c = np.ones((np.size(point_file, 0),1), dtype=int)
    if 'door' in f:
        c = np.ones((np.size(point_file, 0),1), dtype=int) * 2
    if 'window' in f:
        c = np.ones((np.size(point_file, 0),1), dtype=int) * 3
    if 'roof' in f:
        c = np.ones((np.size(point_file, 0),1), dtype=int) * 4
    if 'roof*window' in f:
        c = np.ones((np.size(point_file, 0),1), dtype=int) * 5
    if 'roof*door' in f:
        c = np.ones((np.size(point_file, 0),1), dtype=int) * 6
    point_file = np.append(point_file, c, axis=1)
    return point_file

def concatenate_pointfiles(files):
    point_cloud = []
    firstiteration = True
    for f in files:
        point_file = load_building_data_insert_class(f)
        if firstiteration:
            point_cloud = point_file
            firstiteration = False
        else:
            point_cloud = np.concatenate(([point_cloud, point_file]))
    return point_cloud

# pc = concatenate_pointfiles(files)

def merge_data_files(files):
    number = sorted(list(set([('building_'+f.split('_')[1]) for f in files])))
    for N in number:
        print(N)
        a = []
        for f in files:
            if N in f:
                a.append(f)
        pc = concatenate_pointfiles(a)
        np.savetxt(path2+'%s.txt'%N,pc)

def create_train_test(files, save=False):
    ttlist = list(set([('shape_data/B/building_'+f.split('_')[1]) for f in files]))
    x_train, x_test = train_test_split(ttlist, test_size=0.2)
    if save:
        savepath = "/Users/john/AI/Thesis/pointcloudthesis/pointcloudthesis/dublincity/train_test_split/"
        # np.savetxt(savepath+'shuffled_train_file_list.txt',x_train)
        # np.savetxt(savepath+'shuffled_test_file_list.txt',x_test)
        outF = open("/Users/john/AI/Thesis/pointcloudthesis/pointcloudthesis/dublincity/train_test_split/shuffled_train_file_list.json", "w")
        outF.write(json.dumps(x_train))
        outF.close()
        outF = open("/Users/john/AI/Thesis/pointcloudthesis/pointcloudthesis/dublincity/train_test_split/shuffled_test_file_list.json", "w")
        outF.write(json.dumps(x_test))
        outF.close()
    return x_train, x_test

x_train, x_test = create_train_test(files, save=True)
print('x_train is: ')
print(x_train)
print('x_test is: ')
print(x_test)

    