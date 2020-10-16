import glob
import shutil
import os

talksumm = []
for file in glob.glob('/home/sajad/datasets/longsumm/new-abs-set/splits/train/raw/*.json'):
    try:

        int(file.split('/')[-1].replace('.json', '').strip())
    except:
        # talksumm
        shutil.copy(file, '/home/sajad/datasets/tsum/' + file.split('/')[-1])