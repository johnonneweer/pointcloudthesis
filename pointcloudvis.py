#IMPORT FILES
import glob, os

dirpath = os.getcwd()

os.chdir(dirpath + "/pointcloudthesis/B1/")
all_files = glob.glob("*.txt")

names = []
for file in all_files:
    names.append(file)

print(names)
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
print(labels)

train, validate, test = np.split(cloud.sample(frac=1), [int(.6*len(cloud)), int(.97*len(cloud))])

#ALL FIGURE SETTINGS

plt.rcParams['figure.figsize'] = [100, 50]

color_dict = { 'building':'blue','door':'red','roof':'green','window':'yellow', 'roofdoor':'orange','roofwindow':'violet'}
marker_dict = { 'building':'.','door':'o','roof':'o','window':'o', 'roofdoor':'o','roofwindow':'o'}

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# for kind in marker_dict:
#     d = cloud[cloud.type==kind]
#     ax.scatter(cloud['x'],cloud['y'],cloud['z'], c=[color_dict[i] for i in cloud['type']], marker = marker_dict[kind])
#     print(kind)

# plt.show()

for kind in marker_dict:
    d = test[test.type==kind]
    ax.scatter(test['x'],test['y'],test['z'], c=[color_dict[i] for i in test['type']], marker = marker_dict[kind])
    print(kind)

plt.show()