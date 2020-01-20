import os
from os.path import join
from glob import glob 
import random
import shutil
import numpy as np
from pydub import AudioSegment
import tensorflow as tf
from scipy.io import wavfile
from scipy import signal
import math
from PIL import Image
import dlib
import skvideo.io
import time
import glob
import subprocess
import random
from PIL import Image
from scipy import signal
from scipy.io import wavfile
import math
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import shutil
home = str(Path.home())
# Avoid printing TF log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from video_utils import get_frames_mouth, get_video_frames, get_cropped_video_
import glob

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

'''files = sorted(glob.glob('/data/lrs2/mvlrs_v1/pretrain/*/*.mp4'), key=numericalSort) + sorted(glob.glob('/data/lrs2/mvlrs_v1/main/*/*.mp4'), key=numericalSort)
files_lips = sorted(glob.glob('/home/gpudata/avinash/lrs2/mvlrs_v1/pretrain/*/*_lips.mp4'), key=numericalSort) + sorted(glob.glob('/home/gpudata/avinash/lrs2/mvlrs_v1/main/*/*_lips.mp4'), key=numericalSort)
files = [x for x in files if x not in files_lips]'''

files = sorted(glob.glob('/data/voxceleb_parts/part5/*.mp4'), key=numericalSort) + sorted(glob.glob('/data/voxceleb_parts/part6/*.mp4'), key=numericalSort) + sorted(glob.glob('/data/voxceleb_parts/part7/*.mp4'), key=numericalSort)
print('No. of file:', len(files))

#files = np.loadtxt('/data/AV-speech-separation/data_preparation/files_4_to_10secs.txt', dtype='object')
#files = files.tolist()
'''files = files[:38000]
files = files[33600:]'''
# Calculate Lip crops for all the videos

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/data/AV-speech-separation/shape_predictor_68_face_landmarks.dat')

c = 0
times = []
start = time.time()
times.append(start)
for video in files:
    output_path = video[:-4]+'_lips.mp4'
    get_cropped_video_(video, output_path, detector = detector, predictor = predictor)

    c = c+1
    if c%100 == 0:
        #print(c, '/', len(files), 'lip crops created')
        b = time.time()
        times.append(b)
        print(c, '/', len(files), 'lip crops created in', times[-1]-times[-2], 'seconds')
        with open("/data/AV-speech-separation/data_preparation/log_create_crops_vox.txt", "a") as myfile:
            myfile.write(str(c) + ' / ' +  str(len(files)) + ' lip crops created in ' +  str(times[-1]-times[-2]) + ' seconds \n')
