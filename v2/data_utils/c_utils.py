import numpy as np
import sys

def dub_downgrade_classes(labels):
    labels = labels.astype(int)
    labels[labels == 3 ] = 0
    labels[labels == 4 ] = 0
    labels[labels == 5 ] = 0
    labels[labels == 6 ] = 0
    labels[labels == 7 ] = 0
    labels[labels == 1 ] = 1
    labels[labels == 11 ] = 1
    labels[labels == 8 ] = 2
    labels[labels == 9 ] = 2
    labels[labels == 10 ] = 2
    return labels

# tile = np.load('/Users/john/AI/Thesis/Data/dub/dubd_315752_234759.npy')[1:,:]
# npoint = 20 * 25 * 25
# xyz = tile[:,:3]
# labels = tile[:,3].astype(int)
# labelset = set(labels)
# print(labelset)

# def farthest_point_sample(xyz, npoint):
#     centroids = np.zeros(npoint).astype(int)
#     distance = np.ones(xyz.shape[0]) * 1e10
#     farthest = np.random.randint(0,xyz.shape[0],(1,))
#     for i in range(npoint):
#         centroids[i] = farthest
#         centroid = xyz[farthest, :]
#         dist = np.sum((xyz - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = np.argmax(distance)
#     return centroids



# labels = dub_downgrade_classes(labels)
# labelset = set(labels)

# sys.exit()

# centroids = farthest_point_sample(xyz, npoint)
# new_tile = tile[centroids]
# print(tile.shape)
# print(new_tile.shape)