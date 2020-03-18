
path = "/Users/john/AI/Thesis/Data/AHN3/"

filename = "C_26AZ2.LAZ"

totalpath = "/Users/john/AI/Thesis/Data/AHN3/C_26AZ2.LAZ"

import numpy as np
import pylas

# # Directly read and write las
# las = pylas.read(path + filename)
# las = pylas.convert(las, point_format_id=2)
# las.write('converted.las')

# # Open data to inspect header and then read
# with pylas.open(path + filename) as f:
#     print(f.header.point_count)
# #     if f.header.point_count < 10 ** 8:
# #         las = f.read()
# # print(las.vlrs)

with pylas.open(path+filename) as fh:
    print('Points from Header:', fh.header.point_count)
    las = fh.read()
    print(las)
    print('Points from data:', len(las.points))
    ground_pts = las.classification == 2
    bins, counts = np.unique(las.return_number[ground_pts], return_counts=True)
    print('Ground Point Return Number distribution:')
    for r,c in zip(bins,counts):
        print('    {}:{}'.format(r,c))