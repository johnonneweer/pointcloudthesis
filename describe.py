import pandas as pd
import glob
import sys

path = "/Users/john/AI/Thesis/Data/AHN3"
path_pijp = "/pijp/"
path_utrecht = "/utrecht/"
path_almere = "/almere/"

files = [f for f in glob.glob(path + path_almere + "**/*.txt", recursive=True)]

print(files)

# sys.exit()

for f in files:
    pc = pd.read_csv(f, names=['x', 'y', 'z', 'R', 'G', 'B', 'c', 'i', 'a'])
    print(pc.c.unique())


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# for kind in marker_dict:
#     d = cloud[cloud.type==kind]
#     ax.scatter(cloud['x'],cloud['y'],cloud['z'], c=[color_dict[i] for i in cloud['type']], marker = marker_dict[kind])
#     print(kind)

# plt.show()

