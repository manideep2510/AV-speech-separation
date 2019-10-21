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
from file_utils import pair_files, gen_comb_folders_crm
import glob
import cv2

combinations_list = np.loadtxt('/data/lrs2/combinations_list.txt', dtype='object')

print(len(combinations_list), 'pairs generated')

# Create training folders
c = 0
times = []
start = time.time()
times.append(start)
for combination in combinations_list:
    if c<400:
        gen_comb_folders_crm(combination, dest_folder = '/data/lrs2/train')
        c = c+1
        if c%100 == 0:
            b = time.time()
            times.append(b)
            print(c, '/', len(combinations_list), 'CRMs created in ', times[-1] - times[-2], 'seconds')
