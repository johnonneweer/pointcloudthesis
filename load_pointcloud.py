import numpy as np
import torch.utils.data as data
import torch._six
import os

def prepare_data(point_file, number_of_points=None, point_cloud_class=None, segmentation_label_file=None, segmentation_classes_offset=None):
    point_cloud = np.loadtxt(point_file).astype(np.float32)
    if number_of_points:
        sampling_indices = np.random.choice(point_cloud.shape[0], number_of_points)
        point_cloud = point_cloud[sampling_indices, :]
    point_cloud = torch.from_numpy(point_cloud)
    if segmentation_label_file:
        segmentation_classes = np.loadtxt(segmentation_label_file).astype(np.int64)
        if number_of_points:
            segmentation_classes = segmentation_classes[sampling_indices]
        segmentation_classes = segmentation_classes + segmentation_classes_offset -1
        segmentation_classes = torch.from_numpy(segmentation_classes)
        return point_cloud, segmentation_classes
    elif point_cloud_class is not None:
        point_cloud_class = torch.tensor(point_cloud_class)
        return point_cloud, point_cloud_class
    else:
        return point_cloud

dirpath = os.getcwd()
os.chdir(dirpath + "/pointcloudthesis/B1/")

pc = prepare_data('building_01.txt')

print(pc)