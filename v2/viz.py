import numpy as np
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_root = 'data'
filename = 'utr_ams_p1_azo_p2m_bad.npy'
data = np.load(os.path.join(data_root, filename))
data = data[1:,:]
np.savetxt(os.path.join(data_root, 'utr_ams_p1_azo_p2m_bad.txt'),data)
xyz = data[:,0:3]
label = data[:,6].astype(int)
pred_p1 = data[:,7].astype(int)
pred_p2 = data[:,8].astype(int)
print(xyz.shape)

fig = plt.figure()
# ax = Axes3D(fig)
ax1 = Axes3D(fig)

color_dict = {0:'blue',1:'green',2:'red'}
correct_dict = {0:'red',1:'green'}

# ax =fig.add_subplot(1,2,1,projection='3d')
# ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=[color_dict[i] for i in label], marker='.')
# ax =fig.add_subplot(1,2,2,projection='3d')
# ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=[color_dict[i] for i in pred_data], marker='.')

ax1.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=[color_dict[i] for i in label], marker='.')
plt.show()
