import json
import os
import csv
import torch.utils.data as data
import torch
from torchvision.datasets import MNIST

import numpy as np

from PIL import Image

from utils import transform_2d_img_to_point_cloud

# DEFINE DATASET

class AHN3Dataset(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 1
    NUM_SEGMENTATION_CLASSES = 3

    POINT_DIMENSION = 3

    PER_CLASS_NUM_SEGMENTATION_CLASSES = {
        'almere': 5,
    }

    def __init__(self, dataset_folder, number_of_points=2048, task='classification', train=True):

        #define the dataset folder where the dataset is in first with 'dataset_folder'
        # 'ahn3' is the dataset folder
        self.dataset_folder = dataset_folder
        self.number_of_points = number_of_points
        assert task in ['classification', 'segmentation']
        self.task = task
        self.train = train
        
        category_file = os.path.join(self.dataset_folder, 'mappingtocategory.txt')

        # print(category_file)
        # print('--------')

        self.folders_to_classes_mapping = {}
        self.segmentation_classes_offset = {}

        with open(category_file, 'r') as fid:
            reader = csv.reader(fid, delimiter='\t')
            # print(reader)
            # print('-------')


            offset_seg_class = 0
            for k, row in enumerate(reader):
                self.folders_to_classes_mapping[row[1]] = k
                # print('folders to classes mapping is')
                # print(self.folders_to_classes_mapping)
                self.segmentation_classes_offset[row[1]] = offset_seg_class
                # print('segmentation classes offset is')
                # print(self.segmentation_classes_offset)
                #offset_seg_class += self.PER_CLASS_NUM_SEGMENTATION_CLASSES[row[0]]
        
        if self.train:
            filelist = os.path.join(self.dataset_folder, 'train_test_split', 'shuffled_train_file_list.json')
        else:
            filelist = os.path.join(self.dataset_folder, 'train_test_split', 'shuffled_test_file_list.json')

        with open(filelist, 'r') as fid:
            filenames = json.load(fid)

        self.files = [(f.split('/')[0], f.split('/')[1]) for f in filenames]

    def __getitem__(self, index):
        folder, file = self.files[index]
        point_file = os.path.join(self.dataset_folder,
                                  folder,
                                  '%s.txt' % file)
        # print(point_file)
        segmentation_label_file = os.path.join(self.dataset_folder,
                                               folder,
                                               '%s.txt' % file)
        point_cloud_class = self.folders_to_classes_mapping[folder]
        if self.task == 'classification':
            return self.prepare_data(point_file,
                                     self.number_of_points,
                                     point_cloud_class=point_cloud_class)
        elif self.task == 'segmentation':
            return self.prepare_data(point_file,
                                     self.number_of_points,
                                     point_cloud_class=point_cloud_class,
                                     segmentation_label_file=segmentation_label_file,
                                     segmentation_classes_offset=self.segmentation_classes_offset[folder])

    def __len__(self):
        return len(self.files)

    
    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     point_cloud_class=None,
                     segmentation_label_file=None,
                     segmentation_classes_offset=None):
        point_cloud = np.loadtxt(point_file, delimiter=',', usecols=(0,1,2)).astype(np.float32)
        # print('the pointcloud is: ')
        # print(point_cloud)
        if number_of_points:
            sampling_indices = np.random.choice(point_cloud.shape[0], number_of_points)
            # print(sampling_indices)
            # print(sampling_indices.shape)
            # print('-----')
            # print(point_cloud.shape[0])
            point_cloud = point_cloud[sampling_indices, :]
        point_cloud = torch.from_numpy(point_cloud)
        if segmentation_label_file:
            segmentation_classes = np.loadtxt(segmentation_label_file, delimiter=',', usecols=(6)).astype(np.int64)
            segmentation_classes[segmentation_classes == 6] = 0
            print("voor weghalen: " + str(set(segmentation_classes)))
            segmentation_classes = segmentation_classes[segmentation_classes != 9]
            print("na weghalen: " + str(set(segmentation_classes)))
            segmentation_classes[segmentation_classes == 26] = 2
            if number_of_points:
                segmentation_classes = segmentation_classes[sampling_indices]
            # not necessary in ahn3 set, I guess 
            # #segmentation_classes = segmentation_classes + segmentation_classes_offset -1
            segmentation_classes = torch.from_numpy(segmentation_classes)
            return point_cloud, segmentation_classes
        elif point_cloud_class is not None:
            point_cloud_class = torch.tensor(point_cloud_class)
            return point_cloud, point_cloud_class
        else:
            return point_cloud


class DublinCityDataset(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 16
    NUM_SEGMENTATION_CLASSES = 50

    POINT_DIMENSION = 3

    PER_CLASS_NUM_SEGMENTATION_CLASSES = {
        'Building': 6,
        'Vegetation': 1,
        'Ground': 3,
        'Other': 1,
    }

    def __init__(self, dataset_folder, number_of_points=2500, task='classification', train=True):

        #define the dataset folder where the dataset is in first with 'dataset_folder'
        # 'dublincity' is the dataset folder
        self.dataset_folder = dataset_folder
        self.number_of_points = number_of_points
        assert task in ['classification', 'segmentation']
        self.task = task
        self.train = train
        
        category_file = os.path.join(self.dataset_folder, 'mappingtocategory.txt')

        # print(category_file)
        # print('--------')

        self.folders_to_classes_mapping = {}
        self.segmentation_classes_offset = {}

        with open(category_file, 'r') as fid:
            reader = csv.reader(fid, delimiter='\t')
            # print(reader)
            # print('-------')


            offset_seg_class = 0
            for k, row in enumerate(reader):
                self.folders_to_classes_mapping[row[1]] = k
                # print('folders to classes mapping is')
                # print(self.folders_to_classes_mapping)
                self.segmentation_classes_offset[row[1]] = offset_seg_class
                # print('segmentation classes offset is')
                # print(self.segmentation_classes_offset)
                offset_seg_class += self.PER_CLASS_NUM_SEGMENTATION_CLASSES[row[0]]
        
        if self.train:
            filelist = os.path.join(self.dataset_folder, 'train_test_split', 'shuffled_train_file_list.json')
        else:
            filelist = os.path.join(self.dataset_folder, 'train_test_split', 'shuffled_test_file_list.json')

        with open(filelist, 'r') as fid:
            filenames = json.load(fid)

        self.files = [(f.split('/')[1], f.split('/')[2]) for f in filenames]

    def __getitem__(self, index):
        folder, file = self.files[index]
        point_file = os.path.join(self.dataset_folder,
                                  folder,
                                  'points',
                                  '%s.txt' % file)
        # print(point_file)
        segmentation_label_file = os.path.join(self.dataset_folder,
                                               folder,
                                               'points',
                                               '%s.txt' % file)
        point_cloud_class = self.folders_to_classes_mapping[folder]
        if self.task == 'classification':
            return self.prepare_data(point_file,
                                     self.number_of_points,
                                     point_cloud_class=point_cloud_class)
        elif self.task == 'segmentation':
            return self.prepare_data(point_file,
                                     self.number_of_points,
                                     point_cloud_class=point_cloud_class,
                                     segmentation_label_file=segmentation_label_file,
                                     segmentation_classes_offset=self.segmentation_classes_offset[folder])

    def __len__(self):
        return len(self.files)

    
    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     point_cloud_class=None,
                     segmentation_label_file=None,
                     segmentation_classes_offset=None):
        point_cloud = np.loadtxt(point_file, usecols=(0,1,2)).astype(np.float32)
        # print('the pointcloud is: ')
        # print(point_cloud)
        if number_of_points:
            sampling_indices = np.random.choice(point_cloud.shape[0], number_of_points)
            print(sampling_indices)
            print(sampling_indices.shape)
            print('000000')
            print(point_cloud.shape[0])
            point_cloud = point_cloud[sampling_indices, :]
        point_cloud = torch.from_numpy(point_cloud)
        if segmentation_label_file:
            segmentation_classes = np.loadtxt(segmentation_label_file, usecols=(3)).astype(np.int64)
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
                            
class ShapeNetDataset(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 16
    NUM_SEGMENTATION_CLASSES = 50

    POINT_DIMENSION = 3

    PER_CLASS_NUM_SEGMENTATION_CLASSES = {
        'Airplane': 4,
        'Bag': 2,
        'Cap': 2,
        'Car': 4,
        'Chair': 4,
        'Earphone': 3,
        'Guitar': 3,
        'Knife': 2,
        'Lamp': 4,
        'Laptop': 2,
        'Motorbike': 6,
        'Mug': 2,
        'Pistol': 3,
        'Rocket': 3,
        'Skateboard': 3,
        'Table': 3,
    }

    def __init__(self,
                 dataset_folder,
                 number_of_points=2500,
                 task='classification',
                 train=True):
        self.dataset_folder = dataset_folder
        self.number_of_points = number_of_points
        assert task in ['classification', 'segmentation']
        self.task = task
        self.train = train

        category_file = os.path.join(self.dataset_folder, 'synsetoffset2category.txt')
        print(category_file)
        print('--------')

        self.folders_to_classes_mapping = {}
        self.segmentation_classes_offset = {}

        with open(category_file, 'r') as fid:
            reader = csv.reader(fid, delimiter='\t')
            print(reader)
            print('-------')

            offset_seg_class = 0
            for k, row in enumerate(reader):
                self.folders_to_classes_mapping[row[1]] = k
                self.segmentation_classes_offset[row[1]] = offset_seg_class
                offset_seg_class += self.PER_CLASS_NUM_SEGMENTATION_CLASSES[row[0]]

        if self.train:
            filelist = os.path.join(self.dataset_folder, 'train_test_split', 'shuffled_train_file_list.json')
        else:
            filelist = os.path.join(self.dataset_folder, 'train_test_split', 'shuffled_test_file_list.json')

        with open(filelist, 'r') as fid:
            filenames = json.load(fid)

        self.files = [(f.split('/')[1], f.split('/')[2]) for f in filenames]

    def __getitem__(self, index):
        folder, file = self.files[index]
        point_file = os.path.join(self.dataset_folder,
                                  folder,
                                  'points',
                                  '%s.pts' % file)
        segmentation_label_file = os.path.join(self.dataset_folder,
                                               folder,
                                               'points_label',
                                               '%s.seg' % file)
        point_cloud_class = self.folders_to_classes_mapping[folder]
        if self.task == 'classification':
            return self.prepare_data(point_file,
                                     self.number_of_points,
                                     point_cloud_class=point_cloud_class)
        elif self.task == 'segmentation':
            return self.prepare_data(point_file,
                                     self.number_of_points,
                                     point_cloud_class=point_cloud_class,
                                     segmentation_label_file=segmentation_label_file,
                                     segmentation_classes_offset=self.segmentation_classes_offset[folder])

    def __len__(self):
        return len(self.files)

    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     point_cloud_class=None,
                     segmentation_label_file=None,
                     segmentation_classes_offset=None):
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


class PointMNISTDataset(MNIST):

    NUM_CLASSIFICATION_CLASSES = 10

    POINT_DIMENSION = 2

    def __init__(self, *args, **kwargs):
        kwargs.pop('task')
        self.number_of_points = kwargs.pop('number_of_points')
        kwargs['download'] = True
        super(PointMNISTDataset, self).__init__(*args, **kwargs)
        self.transform = transform_2d_img_to_point_cloud

    def __getitem__(self, index):
        img, target = super(PointMNISTDataset, self).__getitem__(index)
        sampling_indices = np.random.choice(img.shape[0], self.number_of_points)
        img = img[sampling_indices, :].astype(np.float32)
        img = torch.tensor(img)
        return img, target

    @staticmethod
    def prepare_data(image_file):
        img = Image.open(image_file)
        return transform_2d_img_to_point_cloud(img)
