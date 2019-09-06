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

from audio_utils import compare_lengths, compute_spectrograms, audios_sum
from file_utils import pair_files, gen_comb_folders
import glob

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('-dataset_type', action="store", dest="dataset_type", default="train")

args = parser.parse_args()

dataset_type = args.dataset_type

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

if dataset_type == "train":
    files = sorted(glob.glob('/data/lrs2/mvlrs_v1/pretrain/*/*.mp4'), key=numericalSort)
    
    try:
        os.mkdir('/data/lrs2/train')
    except OSError:
        pass
    
    dest_folder = '/data/lrs2/train'
    
elif dataset_type == "val":
    files = sorted(glob.glob('/data/lrs2/mvlrs_v1/main/*/*.mp4'), key=numericalSort)
    
    try:
        os.mkdir('/data/lrs2/val')
    except OSError:
        pass
    
    dest_folder = '/data/lrs2/val'
    
# Make combinations
combinations_list = pair_files(files_audio_7to10, combination_no=1, count=5)

# Create training folders

for combination in combinations_list:
    
    gen_comb_folders(combination, dest_folder = dest_folder)