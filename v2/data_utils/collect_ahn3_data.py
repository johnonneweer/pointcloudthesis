import os
import sys
from outdoor3d_util import DATA_PATH, collect_point_label

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

sys.path.append(BASE_DIR)

anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_pathsv3.txt'))]
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]
anno_paths = [os.path.join(DATA_PATH, 'utr')]

output_folder = os.path.join(ROOT_DIR, 'data/ahn3_set')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for anno_path in anno_paths:
    print(anno_path)
    try:
        element = anno_path.split('/')[-1]
        collect_point_label(anno_path, os.path.join(output_folder,element), 'numpy')
    except:
        pass
    # print(anno_path, 'ERROR!!')
    #     continue
