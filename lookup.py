import os
import numpy as np
import sys
from dedenser import farthest_point_sample

data_root = 'data\\tiles_d\\'
# data_root = 'data/dub/onw/'
rooms = [os.path.join(path, name) for path, subdirs, files in os.walk(data_root) if 'tiles_tsw' in path for name in files if 'dub' in name]

for room_name in rooms:
    print(room_name)
    rn = np.load(room_name)
    nrs = farthest_point_sample(rn[:,0:3],20*25*25)
    rs = rn[nrs]
    elements = room_name.split('\\')
    elements2 = elements[-1].split('_')
    out_filename = elements[0]+'\\tiles_s\\'+elements[2]+'\\dubs_'+elements2[1]+'_'+elements2[2]
    print(out_filename)
    np.save(out_filename, rs)
    xyz_min = np.amin(rs, axis=0)[0:3]
    rs[:, 0:3] -= xyz_min
    out_filename2 = elements[0]+'\\tiles_s_norm\\'+elements[2]+'\\dubs_'+elements2[1]+'_'+elements2[2]
    print(out_filename2)
    np.save(out_filename2, rs)
    xyz_min_d = np.amin(rn, axis=0)[0:3]
    rn[:, 0:3] -= xyz_min_d
    out_filename3 = elements[0]+'\\tiles_d_norm\\'+elements[2]+'\\dubd_'+elements2[1]+'_'+elements2[2]
    print(out_filename3)
    np.save(out_filename3, rn)