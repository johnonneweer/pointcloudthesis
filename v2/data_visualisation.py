import os
import sys
import numpy as np

data_root = 'data/ahn3_set/'

rooms = sorted(os.listdir(data_root))

town_name =['almere', 'utrecht']
# town_name =['test']

def pc_statistics(data_root, town_name):
    densities = np.zeros(len(town_name))
    densities_std = np.zeros(len(town_name))
    densities_mean = np.zeros(len(town_name))
    number_files = np.zeros(len(town_name))
    number_points = np.zeros(len(town_name))
    min_z = np.ones((len(town_name), 3)) * 10000000 #initialize high
    np_label = np.zeros((len(town_name), 3))
    mean_z = np.zeros((len(town_name), 3))
    max_z = np.zeros((len(town_name), 3))
    mean_R = np.zeros((len(town_name), 3))
    mean_G = np.zeros((len(town_name), 3))
    mean_B = np.zeros((len(town_name), 3))
    min_R = np.ones((len(town_name), 3)) * 10000000 #initialize high
    min_G = np.ones((len(town_name), 3)) * 10000000 #initialize high
    min_B = np.ones((len(town_name), 3)) * 10000000 #initialize high
    max_R = np.zeros((len(town_name), 3))
    max_G = np.zeros((len(town_name), 3))
    max_B = np.zeros((len(town_name), 3))

    for i in range(len(town_name)):
        file_names = [room for room in rooms if town_name[i] in room]
        number_files[i] = len(file_names)
        std_d = []
        mean_d = []
        numpo = []

        for  j in file_names:
            room_path = os.path.join(data_root, j)
            room_data = np.load(room_path) #xyzrgbl N*7
            room_density = room_data.shape[0] / (20*20)
            densities[i] += room_density
            std_d.append(room_density)
            mean_d.append(room_density)
            numpo.append(room_data.shape[0])
            tmp, _ = np.histogram(room_data[:,6], range(4))
            minz = np.ones(3) * 1000000
            maxz = np.zeros(3)
            meanz = np.zeros(3)
            meanR = np.zeros(3)
            meanG = np.zeros(3)
            meanB = np.zeros(3)

            minR = np.ones(3) * 1000000
            minG = np.ones(3) * 1000000
            minB = np.ones(3) * 1000000
            maxR = np.zeros(3)
            maxG = np.zeros(3)
            maxB = np.zeros(3)

            numpoi = np.zeros(3)
            
            for k in range(3):
                if k in room_data[:,6]:
                    minz[k] = np.min(room_data[:,2][room_data[:,6] == k])
                    maxz[k] = np.amax(room_data[:,2][room_data[:,6] == k])
                    meanz[k] = np.mean(room_data[:,2][room_data[:,6] == k])
                    meanR[k] = np.mean(room_data[:,3][room_data[:,6] == k])
                    meanG[k] = np.mean(room_data[:,4][room_data[:,6] == k])
                    meanB[k] = np.mean(room_data[:,5][room_data[:,6] == k])

                    minR[k] = np.min(room_data[:,3][room_data[:,6] == k])
                    maxR[k] = np.amax(room_data[:,3][room_data[:,6] == k])
                    minG[k] = np.min(room_data[:,4][room_data[:,6] == k])
                    maxG[k] = np.amax(room_data[:,4][room_data[:,6] == k])
                    minB[k] = np.min(room_data[:,5][room_data[:,6] == k])
                    maxB[k] = np.amax(room_data[:,5][room_data[:,6] == k])

                    numpoi[k] = room_data[room_data[:,6] == k].shape[0]
    
                    min_z[i][k] = np.min([min_z[i][k], minz[k]])
                    max_z[i][k] = np.max([max_z[i][k],maxz[k]])

                    min_R[i][k] = np.min([min_R[i][k], minR[k]])
                    max_R[i][k] = np.max([max_R[i][k],maxR[k]])
                    min_G[i][k] = np.min([min_G[i][k], minG[k]])
                    max_G[i][k] = np.max([max_G[i][k],maxG[k]])
                    min_B[i][k] = np.min([min_B[i][k], minB[k]])
                    max_B[i][k] = np.max([max_B[i][k],maxB[k]])

                    mean_z[i][k] = (mean_z[i][k] * np_label[i][k] + meanz[k] * numpoi[k]) / (np_label[i][k] + numpoi[k])
                    mean_R[i][k] = (mean_R[i][k] * np_label[i][k] + meanR[k] * numpoi[k]) / (np_label[i][k] + numpoi[k])
                    mean_G[i][k] = (mean_G[i][k] * np_label[i][k] + meanG[k] * numpoi[k]) / (np_label[i][k] + numpoi[k])
                    mean_B[i][k] = (mean_B[i][k] * np_label[i][k] + meanB[k] * numpoi[k]) / (np_label[i][k] + numpoi[k])

                    np_label[i][k] += numpoi[k]

        # print(std_d)
        densities_std[i] = np.std(std_d)
        densities_mean[i] = np.mean(mean_d)
        number_points[i] = np.sum(numpo)
    return min_z, max_z, mean_z, min_R, max_R, mean_R, min_G, max_G, mean_G, min_B, max_B, mean_B, np_label, densities_mean, number_points

def collect_pc(data_root, town_name):
    for i in range(len(town_name)):
            file_names = [room for room in rooms if town_name[i] in room]
            print(file_names)
            pc_0 = np.zeros((0,7))
            print(pc_0)
            print(pc_0.shape)
            for  j in file_names:
                room_path = os.path.join(data_root, j)
                room_data = np.load(room_path) #xyzrgbl N*7
                print(room_data.shape)
                pc_0 = np.concatenate((pc_0, room_data), axis=0)
    return pc_0

# pocl  = collect_pc(data_root, town_name)


min_z, max_z, mean_z, min_R, max_R, mean_R, min_G, max_G, mean_G, min_B, max_B, mean_B, np_label, densities, number_points = pc_statistics(data_root, town_name)
print(town_name)
print(min_z)
print(max_z)
print(mean_z)
print(min_R)
print(max_R)
print(mean_R)
print(min_G)
print(max_G)
print(mean_G)
print(min_B)
print(max_B)
print(mean_B)
print(np_label)
print
sys.exit()


import open3d as o3d
import random
file_names = [room for room in rooms if town_name[0] in room]

choose_file = file_names[0]
choose_file = random.choice(rooms)
print(choose_file)
room_path = os.path.join(data_root, choose_file)

room_data = np.load(room_path) #xyzrgbl N*7
xyz = room_data[:,0:3]
rgb = room_data[:,3:6]

# xyz = pocl[:,0:3]
# rgb = pocl[:,3:6][pocl[:,6] == 1]

# xyz = rgb/255.0
# print(xyz)
# Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
pcd = o3d.geometry.PointCloud()
# pcd = o3d.PointCloud.colors()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb/255.0)
# o3d.io.write_point_cloud("../../TestData/sync.ply", pcd)

### THIS IS OPEN3D
o3d.visualization.draw_geometries([pcd])
sys.exit()
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)

r= rgb[:,0]
g= rgb[:,1]
b = rgb[:,2]
ax.scatter(r,g,b)

plt.show()

#cluster on color?

import pandas as pd

df = pd.DataFrame({'red': r,
'blue': b,
'green': g})

from scipy.cluster.vq import whiten
df['scaled_red'] = whiten(df['red'])
df['scaled_blue'] = whiten(df['blue'])
df['scaled_green'] = whiten(df['green'])
df.sample(n = 10)

from scipy.cluster.vq import kmeans

cluster_centers, distortion = kmeans(df[['scaled_red', 'scaled_green', 'scaled_blue']], 5)

print(cluster_centers)

colors = []
r_std, g_std, b_std = df[['red', 'green', 'blue']].std()

for cluster_center in cluster_centers:
    scaled_r, scaled_g, scaled_b = cluster_center
    colors.append((
    scaled_r * r_std / 255,
    scaled_g * g_std / 255,
    scaled_b * b_std / 255
    ))
plt.imshow([colors])
plt.show()


sys.exit()
print('minz')
print(min_z)
print('maxz')
print(max_z)
print('meanz')
print(mean_z)
print('minR')
print(min_R)
print('maxR')
print(max_R)
print('minG')
print(min_G)
print('maxG')
print(max_G)
print('minB')
print(min_B)
print('maxB')
print(max_B)
print('meanR')
print(mean_R)
print('meanG')
print(mean_G)
print('meanB')
print(mean_B)
print('number points per label')
print(np_label)

print('summary')
print(str(town_name[0])+' '+str(town_name[1]))
print(densities)
print('the number of files per city are')
print(str(town_name[0])+' '+str(town_name[1]))
print(number_files)
file_densities = densities / number_files
print(file_densities)
print(densities_std)
print(densities_mean)




    
# room_name = rooms[1]
# room_path = os.path.join(data_root, room_name)
# room_data = np.load(room_path) #xyzrgbl N*7

# print(room_data.shape)

# density = room_data.shape[0] / (20*20)
# print(density)