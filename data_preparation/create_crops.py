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
import shutil
home = str(Path.home())
# Avoid printing TF log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from video_utils import get_frames_mouth, get_video_frames, get_cropped_video
import glob

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

files = sorted(glob.glob('/data/lrs2/mvlrs_v1/pretrain/*/*.mp4'), key=numericalSort) + sorted(glob.glob('/data/lrs2/mvlrs_v1/main/*/*.mp4'), key=numericalSort)

# Calculate Lip crops for all the videos

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')

for video in files:
    output_path = video[:-4]+'_lips.mp4'
    get_cropped_video(video, output_path, detector = detector, predictor = predictor)