import os
from os.path import join
import glob
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

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def crop_pad_frames(frames, req_frames, fps, seconds):

    req_frames = fps*seconds

    num_frames = frames.shape[0]

    # Delete or add frames to make the video to 10 seconds
    if num_frames > req_frames:
        frames = frames[:req_frames, :, :, :]

    elif num_frames < req_frames:
        pad_len = req_frames - num_frames
        frames = np.pad(frames, ((0,pad_len),(0,0), (0,0), (0,0)), 'constant')

    elif num_frames == req_frames:
        frames = frames

    return frames

# DataGenerator 

def DataGenerator(lips_filelist, masks_filelist, spects_filelist, batch_size):

    L = len(files)

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            
            lips = lips_filelist[batch_start:limit]
            
            X_mask = np.asarray([cv2.imread(fname, cv2.IMREAD_UNCHANGED) for fname in masks_filelist[batch_start:limit]])
            
            X_spect = np.asarray([np.load(fname) for fname in spects_filelist[batch_start:limit]])
            
            #spect_len = X_spect.shape[1]
            #mask_len = X_mask.shape[1]
            #frames = X_lips.shape[0]
            
            '''# Pad or crop spectrogram to 10 seconds
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
                X_lips = X_lips'''

            X_lips = []

            for i in range(len(lips)):

                x_lips = cv2.imread(lips[i], cv2.IMREAD_UNCHANGED)
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 5)
                X_lips.append(x_lips)

            X_lips = np.asarray(X_lips)
            
            #X = seq.augment_images(X)
            
            yield X_lips, X_spect, X_mask

            batch_start += batch_size
            batch_end += batch_size

# DataGenerator 

def DataGenerator_siamese(folderlist, batch_size):

    L = len(folderlist)

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips1 = []
            lips2 = []
            mask1 = []
            mask2 = []
            spect = []
            
            for folder in folders_batch:

                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                masks_ = sorted(glob.glob(folder + '/*_mask.png'), key=numericalSort)
                spect_ = folder + '/mixed_spectrogram.npy'

                lips1.append(lips_[0])
                lips2.append(lips_[1])

                mask1.append(mask_[0])
                mask2.append(mask_[1])

                spect.append(spect_)
            
            X_lips1 = [get_video_frames(fname) for fname in lips1]
            X_lips2 = [get_video_frames(fname) for fname in lips2]
            
            X_mask1 = np.asarray([cv2.imread(fname, cv2.IMREAD_UNCHANGED) for fname in mask1])
            X_mask2 = np.asarray([cv2.imread(fname, cv2.IMREAD_UNCHANGED) for fname in mask2])
            
            X_spect = np.asarray([np.load(fname) for fname in spect])
            
            '''spect_len = X_spect.shape[1]
            mask_len = X_mask.shape[1]
            
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
                X_mask = X_mask'''
            
            X_lips1 = []
            X_lips2 = []

            for i in range(len(lips1)):

                x_lips1 = cv2.imread(lips1[i], cv2.IMREAD_UNCHANGED)
                x_lips1 = crop_pad_frames(frames = x_lips1, fps = 25, seconds = 5)
                X_lips1.append(x_lips1)

                x_lips2 = cv2.imread(lips2[i], cv2.IMREAD_UNCHANGED)
                x_lips2 = crop_pad_frames(frames = x_lips2, fps = 25, seconds = 5)
                X_lips2.append(x_lips2)

            X_lips1 = np.asarray(X_lips1)
            X_lips2 = np.asarray(X_lips2)

            #X = seq.augment_images(X)

            yield X_lips1, X_lips2, X_spect, X_mask1, X_mask2

            batch_start += batch_size
            batch_end += batch_size
    
            
# DataGenerator 

def DataGenerator_train(folderlist, batch_size):

    L = len(folderlist)

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips = []
            mask = []
            spect = []
            
            for folder in folders_batch:

                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                masks_ = sorted(glob.glob(folder + '/*_mask.png'), key=numericalSort)
                spect_ = folder + '/mixed_spectrogram.npy'

                lips.append(lips_[0])
                lips.append(lips_[1])

                mask.append(mask_[0])
                mask.append(mask_[1])

                spect.append(spect_)
                spect.append(spect_)
            
            X_mask = np.asarray([cv2.imread(fname, cv2.IMREAD_UNCHANGED) for fname in mask])
            
            X_spect = np.asarray([np.load(fname) for fname in spect])
            
            X_lips = []
            
            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i])
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 5)
                X_lips.append(x_lips)


            X_lips = np.asarray(X_lips)

            #X = seq.augment_images(X)

            yield X_lips, X_spect, X_mask

            batch_start += batch_size
            batch_end += batch_size
