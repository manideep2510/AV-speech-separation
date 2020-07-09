import os
from os.path import join
from glob import glob 
import random
import shutil
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
from scipy import signal
import math
from PIL import Image
import skvideo.io
import time
import glob
import subprocess
import random
from pathlib import Path
import shutil
home = str(Path.home())

from audio_utils import compare_lengths, compute_spectrograms, audios_sum, ibm
from file_utils import pair_files, gen_comb_folders_3comb
import glob
import cv2

'''from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('-dataset_type', action="store", dest="dataset_type", default="train")
parser.add_argument('-each_comb', action="store", dest="combination_no", type=int)
parser.add_argument('-num_combs', action="store", dest="count", type=int)

args = parser.parse_args()

dataset_type = args.dataset_type
combination_no = args.combination_no
count = args.count'''

dataset_type = 'train'
'''combination_no = 1
count = 2'''

'''# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

if dataset_type == "train":
    files = sorted(glob.glob('/data/lrs2/mvlrs_v1/pretrain/*/*_lips.mp4'), key=numericalSort)
    
    try:
        os.mkdir('/data/lrs2/voxceleb_2comb')
    except OSError:
        pass
    
    dest_folder = '/data/lrs2/voxceleb_2comb'
    
elif dataset_type == "val":
    files = sorted(glob.glob('/data/lrs2/mvlrs_v1/main/*/*_lips.mp4'), key=numericalSort)
    
    try:
        os.mkdir('/data/lrs2/val')
    except OSError:
        pass
    
    dest_folder = '/data/lrs2/val'

print('Total files: ', len(files))

import cv2

def get_frames(fi):
    video = cv2.VideoCapture(fi)
    fps = video.get(cv2.CAP_PROP_FRAME_COUNT)
    video.release()
    return fps

files_req = []
for item in files:
    frames = get_frames(item)
    #sh = frames.shape[0]
    if frames>=501 and frames<550:
        files_req.append(item)

print('Files > 3 secs and < 21 secs', len(files_req))

# Make combinations
a = time.time()
combinations_list = pair_files(files_req, combination_no = combination_no, count = count)
b = time.time()'''

dest_folder = '/data/lrs2/val_3comb_new'
combinations_list = np.loadtxt('/data/val_3comb_new_fromTrain.txt', dtype='object')

with open("/data/AV-speech-separation/data_preparation/log_val_3comb_new.txt", "w") as myfile:
    myfile.write(str(len(combinations_list)) + ' pairs generated')

print(len(combinations_list), 'pairs generated')

#np.savetxt('/data/lrs2/combinations_list_20s.txt', combinations_list, fmt='%s')

# Create training folders
c = 0
times = []
start = time.time()
times.append(start)
for combination in combinations_list:
    
    gen_comb_folders_3comb(combination, dest_folder = dest_folder)
    c = c+1
    if c%100 == 0:
        b = time.time()
        times.append(b)
        print(c, '/', len(combinations_list), 'folders created in ', times[-1] - times[-2], 'seconds')
        with open("/data/AV-speech-separation/data_preparation/log_val_3comb_new.txt", "a") as myfile:
            myfile.write(str(c) + ' / ' + str(len(combinations_list)) + ' folders created in ' + str(times[-1] - times[-2]) + ' seconds \n')
