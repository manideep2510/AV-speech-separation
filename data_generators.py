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

from data_preparation.video_utils import get_video_frames

# DataGenerator 

def DataGenerator(lips_filelist, masks_filelist, spects_filelist, batch_size):

    L = len(files)

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            
            X_lips = np.asarray([get_video_frames(fname) for fname in lips_filelist[batch_start:limit]])
            
            X_mask = np.asarray([np.load(fname) for fname in masks_filelist[batch_start:limit]])
            
            X_spect = np.asarray([np.load(fname) for fname in spects_filelist[batch_start:limit]])
            
            spect_len = X_spect.shape[1]
            mask_len = X_mask.shape[1]
            fps = 25
            frames_10s = fps*10
            frames = X_lips.shape[0]
            
            # Pad or crop spectrogram to 10 seconds
            if spect_len > 1000:
                X_spect = X_spect[:, :1000]
                
            elif spect_len < 1000:
                pad_len = 1000 - spect_len
                X_spect = np.pad(X_spect, ((0,0),(0,pad_len)), 'constant')
                
            elif spect_len == 1000:
                X_spect = X_spect
                
            # Pad or crop mask to 10 seconds
            if mask_len > 1000:
                X_mask = X_mask[:, :1000]
                
            elif mask_len < 1000:
                pad_len = 1000 - mask_len
                X_mask = np.pad(X_mask, ((0,0),(0,pad_len)), 'constant')
                
            elif mask_len == 1000:
                X_mask = X_mask
                
            # Delete or add frames to make the video to 10 seconds
            if frames > frames_10s:
                X_lips = X_lips[:frames_10s, :, :, :]
                
            elif frames < 1000:
                pad_len = frames_10s - frames
                X_lips = np.pad(X_lips, ((0,pad_len),(0,0), (0,0), (0,0)), 'constant')
                
            elif frames == 1000:
                X_lips = X_lips
            
            #X = seq.augment_images(X)
            
            yield X_lips, X_spect, X_mask

            batch_start += batch_size
            batch_end += batch_size

