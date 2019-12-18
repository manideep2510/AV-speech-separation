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

from audio_utils import get_wav
import glob

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Read all the .mp4 files

files = sorted(glob.glob('/data/lrs2/mvlrs_v1/pretrain/*/*.mp4'), key=numericalSort) + sorted(glob.glob('/data/lrs2/mvlrs_v1/main/*/*.mp4'), key=numericalSort)

c = 0
for item in files:
    get_wav(item)
    c = c+1
    if c%100 == 0:
#        print('.wav files extracted for ' +  str(c) + '/' + str(len(files)) + ' .mp4 files \n')
        with open("/data/AV-speech-separation/data_preparation/log_create_wav.txt", "a") as myfile:
            myfile.write('.wav files extracted for ' +  str(c) + '/' + str(len(files)) + ' .mp4 files \n')
