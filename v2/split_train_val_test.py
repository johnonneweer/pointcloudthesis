import os, sys
import random
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)

root = 'data/dubd/'

files = sorted(os.listdir(root))
random.shuffle(files)

print(files)

ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

train = ['tiles_one', 'tiles_tsw', 'tiles_tse', 'tiles_tnpw']
test = ['tiles_tnw', 'tiles_osw', 'tiles_ose', 'tiles_onw']


sys.exit()
test_set = []
for i in files:
    p = [os.path.join(path, name) for path, subdirs, files in os.walk(root) if i in path for name in files if 'dub' in name]
    r = random.sample(p, int(len(p)*0.2))
    test_set.append(r)
print(test_set)
printname = os.path.join(root, 'dubd_test.txt')
print(root)
print(printname)
with open(printname, 'w') as saver:
    for i in test_set:
        for j in i:
            print(j)
            saver.write(j + '\n')
    saver.close()