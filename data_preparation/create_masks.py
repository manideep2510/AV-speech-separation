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

from audio_utils import downsampling, ltass_speaker, compute_tbm, save_target_binary_mask_speaker
import glob

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Read all the .wav files

files = sorted(glob.glob('/data/lrs2/mvlrs_v1/pretrain/*/*.wav'), key=numericalSort) + sorted(glob.glob('/data/lrs2/mvlrs_v1/main/*/*.wav'), key=numericalSort)


# Calculate Binary Masks for all the audios

for audio in files:
    output_path = audio[:-4]+'_mask.npy'
    save_target_binary_mask_speaker(audio, mask_file = output_path)