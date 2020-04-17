#IMPORT FILES
import glob, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

# dirpath = os.getcwd()

# os.chdir(dirpath + "/pointcloudthesis/B1/")

# path = "/Users/john/AI/Thesis/pointcloudthesis/pointcloudthesis/B1/"
# path2 = "/Users/john/AI/Thesis/pointcloudthesis/pointcloudthesis/B1/B/"
# path3 = "/Users/john/AI/Thesis/pointcloudthesis/pointcloudthesis/dublincity/"

# buildingfiles = [f for f in glob.glob(path3 + "**/*.txt", recursive=True) if 'building' in f]
# groundfiles = [f for f in glob.glob(path3 + "**/*.txt", recursive=True) if 'ground' in f]
# vegetationfiles = [f for f in glob.glob(path3 + "**/*.txt", recursive=True) if 'vegetation' in f.lower()]

# files = buildingfiles + groundfiles + vegetationfiles


# files=sorted(np.concatenate((buildingfiles, groundfiles,vegetationfiles), axis=None))

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
    if 'street' in f:
        c = np.ones((np.size(point_file, 0),1), dtype=int)
    if 'sidewalk' in f:
        c = np.ones((np.size(point_file, 0),1), dtype=int) * 2
    if 'grass' in f:
        c = np.ones((np.size(point_file, 0),1), dtype=int) * 3
    if 'vegetation' in f.lower():
        c = np.ones((np.size(point_file, 0),1), dtype=int)
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

def merge_building_data_files(files, type):
    if type == 'building':
        number = sorted(list(set([('building_'+f.split('_')[1]) for f in files])))
        for N in number:
            print(N)
            a = []
            for f in files:
                if N in f:
                    a.append(f)
            pc = concatenate_pointfiles(a)
            np.savetxt(path2+'%s.txt'%N,pc)

def save_non_building_files(files,type):
    if type == 'ground':
        for f in files:
            point_file = load_building_data_insert_class(f)
            name = ('ground_'+f.split('_')[1]+'_'+f.split('_')[2])
            np.savetxt(path2+'%s.txt'%name,point_file)
    if type == 'vegetation':
        for f in files:
            point_file = load_building_data_insert_class(f)
            name = ('vegetation_'+f.split('_')[1]+'_'+f.split('_')[2])
            np.savetxt(path2+'%s.txt'%name,point_file)

# save_non_building_files(groundfiles,type='ground')
# save_non_building_files(vegetationfiles,type='vegetation')


def create_train_test(buildingfiles,vegetationfiles, groundfiles, save=False):
    buildinglist = list(set([('shape_data/B/'+f.split('/')[10]) for f in buildingfiles]))
    vegetationlist = list(set([('shape_data/V/'+f.split('/')[10]) for f in vegetationfiles]))
    groundlist = list(set([('shape_data/G/'+f.split('/')[10]) for f in groundfiles]))

    xb_train, xb_test = train_test_split(buildinglist, test_size=0.2)
    xv_train, xv_test = train_test_split(vegetationlist, test_size=0.2)
    xg_train, xg_test = train_test_split(groundlist, test_size=0.2)

    x_train = xb_train + xv_train + xg_train
    x_test = xb_test + xv_test + xg_test

    if save:
        outF = open("/Users/john/AI/Thesis/pointcloudthesis/pointcloudthesis/dublincity/train_test_split/shuffled_train_file_list.json", "w")
        outF.write(json.dumps(x_train))
        outF.close()
        outF = open("/Users/john/AI/Thesis/pointcloudthesis/pointcloudthesis/dublincity/train_test_split/shuffled_test_file_list.json", "w")
        outF.write(json.dumps(x_test))
        outF.close()
    return x_train, x_test

def create_ahn3_train_test(save=False):
    dataset_folder = 'ahn3'
    folder = 'almere'
    file_path = os.path.join(dataset_folder, folder, '')

    files = glob.glob(file_path + "**/*.txt", recursive=True)
    file_list = list(set([(f.split('/')[1]+'/'+f.split('/')[2]) for f in files]))
    file_list = list(set([(f.split('.')[0]) for f in file_list]))

    train, test = train_test_split(file_list, test_size=0.2)

    if save:
        outF = open(r"C:\Users\Sustainables\Documents\Thesis\Code\pointcloudthesis\ahn3\train_test_split\shuffled_train_file_list.json", "w")
        outF.write(json.dumps(train))
        outF.close()
        outF = open(r"C:\Users\Sustainables\Documents\Thesis\Code\pointcloudthesis\ahn3\train_test_split\shuffled_test_file_list.json", "w")
        outF.write(json.dumps(test))
        outF.close()
    return train, test

def create_ahn3_windows_2_train_test(save=False):
    dataset_folder = 'ahn3'
    folder = 'almere'
    file_path = r"C:\Users\Sustainables\Documents\Thesis\Code\pointcloudthesis\ahn3\almere"
    print(file_path)

    files = glob.glob(file_path + "**/*.txt", recursive=True)
    file_list = list(set([(f.split('\\')[8]+'/'+f.split('\\')[9]) for f in files]))
    train = list(set([(f.split('.')[0]) for f in file_list]))

    folder = 'gein'
    file_path = r"C:\Users\Sustainables\Documents\Thesis\Code\pointcloudthesis\ahn3\gein"

    files = glob.glob(file_path + "**/*.txt", recursive=True)
    file_list = list(set([(f.split('\\')[8]+'/'+f.split('\\')[9]) for f in files]))
    test = list(set([(f.split('.')[0]) for f in file_list]))

    #train, test = train_test_split(file_list, test_size=0.2)

    if save:
        outF = open(r"C:\Users\Sustainables\Documents\Thesis\Code\pointcloudthesis\ahn3\train_test_split\shuffled_train_file_list.json", "w")
        outF.write(json.dumps(train))
        outF.close()
        outF = open(r"C:\Users\Sustainables\Documents\Thesis\Code\pointcloudthesis\ahn3\train_test_split\shuffled_test_file_list.json", "w")
        outF.write(json.dumps(test))
        outF.close()
    return train, test

# train, test = create_ahn3_windows_2_train_test(save=True)

def create_mapping(grid, save=False):
    file_path = r"C:\Users\Sustainables\Documents\Thesis\Code\pointcloudthesis\ahn3"
    files = glob.glob(file_path + "/**/*", recursive=True)
    file_list = list(set([(f.split('\\')[8]) for f in files]))
    file_list = list(set([(f.split('.')[0]) for f in file_list]))

    if save:
        with open(grid +'_mapping.txt','w') as f:
            f.write('\n'.join(file_list))

# create_mapping('utrecht')

file_path = r"C:\Users\Sustainables\Documents\Thesis\Code\pointcloudthesis\ahn3"
files = glob.glob(file_path + "/**/*", recursive=True)

print(files)

#x_train, x_test = create_train_test(buildingfiles, vegetationfiles,groundfiles)
#print('x_train is: ')
#print(x_train)
#print('x_test is: ')
#print(x_test)