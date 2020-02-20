#IMPORT FILES
import glob, os

dirpath = os.getcwd()

os.chdir(dirpath + "/pointcloudthesis/B1/")
# os.chdir(dirpath + "/B1/")

all_files = glob.glob("*.txt")

names = []
for file in all_files:
    names.append(file)

# print(names)
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

# #Describing data set
# print(bigframe.columns)
# print(bigframe.describe())


cloud = bigframe.copy()
cloud = cloud[["//X","Y","Z", "filename"]]
cloud.columns = ['x', 'y', 'z', 'type']
cloud['buildingclass'] = 0
cloud.loc[cloud['type'].str.contains('building'), 'buildingclass'] = 1

labels = set(cloud['type'])

train, validate, test = np.split(cloud.sample(frac=1), [int(.6*len(cloud)), int(.99*len(cloud))])

train_x = test[['x','y','z']]
train_y = test[['buildingclass']]

print(train_x.describe())
print(train_y.describe())



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from fastprogress import master_bar, progress_bar
from torch.autograd import Variable

# pointnet transformation network 

class TransformationNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(TransformationNet, self).__init__()
        self.output_dim = output_dim
        self.conv_1 = nn.Conv1d(input_dim, 64, 1)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 1024, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(1024)
        self.bn_4 = nn.BatchNorm1d(512)
        self.bn_5 = nn.BatchNorm1d(256)

        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, self.output_dim*self.output_dim)

    def forward(self, x):
        num_points = x.shape[1]
        print(num_points)
        # x = x.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        x = nn.MaxPool1d(num_points)(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn_4(self.fc_1(x)))
        x = F.relu(self.bn_5(self.fc_2(x)))
        x = self.fc_3(x)

        identity_matrix = torch.eye(self.output_dim)

        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        return x

net = TransformationNet(input_dim=3,output_dim=3)
# print(net)

# model = TransformationNet(3,3)
# learning_rate = 0.1
# epochs = 1

# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# mb = master_bar(range(epochs))

# for epoch in mb:
#     epoch_train_loss = []
#     epoch_train_acc = []
#     batch_number = 0
    
#     points = train_x
#     targets = train_y

#     optimizer.zero_grad()
#     model = model.train()
#     preds = model(points)

batch_size=1
# loader = train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),batch_size=batch_size, shuffle=True)
# train_iter = iter(train_loader)

# print(type(train_iter))

# data, target = train_iter.next()

class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(3 * 83695, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return F.log_softmax(x)


# net = Net()

# data = torch.tensor(train_x.values)
# target = torch.tensor(train_y.values)

# print(data.shape)
# print(target.shape)

train_dataloader = torch.utils.data.DataLoader(train_x, batch_size=batch_size, shuffle=True, num_workers=4)

for data in train_dataloader:
    points, targets = data
    print(points.shape)
    print(points)
    print(targets)
    print(targets.shape)


epochs = 10
learning_rate = 0.1
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.NLLLoss()

for epoch in range(epochs):
    # data, target = Variable(data, volatile=True), Variable(target)

    # data = data.view(-1, 3 * 83695)
    net_out = net(data)

    print('---------------')
    print(target)
    print('---------------')
    print(net_out)
    loss = criterion(net_out, target)
    loss.backward()
    optimizer.step()

    print('---------------')
    print(loss.data.item())


# for batch_idx, (data, target) in enumerate(loader[[0]]):
#     data, target = Variable(data), Variable(target)

#     print(data)
#     print('000000000')
#     print(target)