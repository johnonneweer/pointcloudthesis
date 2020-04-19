import pandas as pd
import numpy as np
import glob
import sys

path = "/Users/john/AI/Thesis/pointcloudthesis/pointcloudthesis/ahn3"
path_pijp = "/pijp/"
path_utrecht = "/utrecht/"
path_almere = "/almere/"
path_hoogkarspel = "/hoogkarspel/"
path_gein = "/gein/"

win = 0

if win == 1:
    path = r"C:\Users\Sustainables\Documents\Thesis\Data\AHN3"
    path_pijp = r"\pijp\\"
    path_utrecht = r"\utrecht\\"
    path_almere = r"\almere\\"
    path_hoogkarspel = r"\hoogkarspel\\"
    path_test = r"\test\\"
    
    files = [f for f in glob.glob(path + path_test + "**/*.txt", recursive=True)]
else:
    path = "/Users/john/AI/Thesis/Data/AHN3"
    path_pijp = "/pijp/"
    path_utrecht = "/utrecht/"
    path_almere = "/almere/"

    path_location = path_utrecht
    files = [f for f in glob.glob(path + path_location + "**/*.txt", recursive=True)]

# sys.exit()

sc_1 = 0
sc_2 = 0
sc_6 = 0
sc_9 = 0
sc_26 = 0

for f in files:
    # pc = pd.read_csv(f, names=['x', 'y', 'z', 'R', 'G', 'B', 'c', 'i', 'a'])
    sc = np.loadtxt(f, delimiter=',', usecols=(6)).astype(np.int64)
    count = np.count_nonzero(sc == 1)
    sc_1 += count
    count = np.count_nonzero(sc == 2)
    sc_2 += count
    count = np.count_nonzero(sc == 6)
    sc_6 += count
    count = np.count_nonzero(sc == 9)
    sc_9 += count
    count = np.count_nonzero(sc == 26)
    sc_26 += count

sc_t = sc_1 + sc_2 + sc_6 + sc_9 + sc_26

print(" ")
print("Details of fileset: " + path_location)
print("-------------")
print("Amount of Point Cloud Files are: " + str(len(files)))
print("Amount with Class 1 --Vegetation/Other--: "+ str(sc_1))
print("Amount with Class 2 --Ground--: "+ str(sc_2))
print("Amount with Class 6 --Building--: "+ str(sc_6))
print("Amount with Class 9 --Water--: "+ str(sc_9))
print("Amount with Class 26 --Bridges/Art/Highway--: "+ str(sc_26))
print("Total Points in dataset: " + str(sc_t))


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# for kind in marker_dict:
#     d = cloud[cloud.type==kind]
#     ax.scatter(cloud['x'],cloud['y'],cloud['z'], c=[color_dict[i] for i in cloud['type']], marker = marker_dict[kind])
#     print(kind)

# plt.show()

