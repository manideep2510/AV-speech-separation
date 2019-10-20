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
import cv2

home = str(Path.home())
# Avoid printing TF log messages
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from data_preparation.video_utils import get_video_frames

def to_onehot(arr):
    arr = (np.arange(arr.max()+1) == arr[...,None]).astype(int)
    return arr

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def crop_pad_frames(frames, fps, seconds):

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


def DataGenerator_train_softmask(folderlist, batch_size):

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
            phase = []
            samples = []
            for folder in folders_batch:

                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                masks_ = sorted(glob.glob(folder + '/*_softmask.npy'), key=numericalSort)
                samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
                spect_ = folder + '/mixed_spectrogram.npy'
                phase_ = folder + '/phase_spectrogram.npy'

                lips.append(lips_[0])
                lips.append(lips_[1])

                samples.append(samples_[0])
                samples.append(samples_[1])

                mask.append(masks_[0])
                mask.append(masks_[1])

                spect.append(spect_)
                spect.append(spect_)

                phase.append(phase_)
                phase.append(phase_)

            zipped = list(zip(lips, samples, mask, spect, phase))
            random.shuffle(zipped)
            lips, samples, mask, spect, phase = zip(*zipped)
            
            #X_mask = np.asarray([to_onehot(cv2.imread(fname, cv2.IMREAD_UNCHANGED)) for fname in mask])
            X_mask = np.asarray([np.load(fname).reshape(257, 500, 1) for fname in mask])
            #print(X_mask.shape)
#            print('mask', X_mask.shape)
            
            X_spect = [np.load(fname) for fname in spect]
            
            X_phase = [np.load(fname) for fname in phase]

            X_samples = np.asarray([np.pad(np.load(fname), (0, 128500), mode='constant')[:128500] for fname in samples])
            
            X_spect_phase = []
            for i in range(len(X_spect)):
                x_spect_phase = np.stack([X_spect[i], X_phase[i]], axis=-1)
                X_spect_phase.append(x_spect_phase)

            X_spect_phase = np.asarray(X_spect_phase)

#            print("X_spect_phase", X_spect_phase.shape)
            
            X_lips = []
            
            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i])
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 5)
                X_lips.append(x_lips)


            X_lips = np.asarray(X_lips)
           # print(X_lips.shape)
            #X = seq.augment_images(X)

            yield [X_spect_phase, X_lips, X_samples], X_mask

            batch_start += batch_size
            batch_end += batch_size

def DataGenerator_test_softmask(folderlist, batch_size):

    L = len(folderlist)

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips = []
           # mask = []
            spect = []
            phase = []
            samples = []
            for folder in folders_batch:

                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                masks_ = sorted(glob.glob(folder + '/*_softmask.npy'), key=numericalSort)
                samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
                spect_ = folder + '/mixed_spectrogram.npy'
                phase_ = folder + '/phase_spectrogram.npy'

                lips.append(lips_[0])
                lips.append(lips_[1])

                samples.append(samples_[0])
                samples.append(samples_[1])

              #  mask.append(masks_[0])
             #   mask.append(masks_[1])

                spect.append(spect_)
                spect.append(spect_)

                phase.append(phase_)
                phase.append(phase_)
            
            #X_mask = np.asarray([to_onehot(cv2.imread(fname, cv2.IMREAD_UNCHANGED)) for fname in mask])
            #print(X_mask.shape)
#            print('mask', X_mask.shape)
            
            X_spect = [np.load(fname) for fname in spect]
            
            X_phase = [np.load(fname) for fname in phase]

            X_samples = np.asarray([np.pad(np.load(fname), (0, 128500), mode='constant')[:128500] for fname in samples])
            
            X_spect_phase = []
            for i in range(len(X_spect)):
                x_spect_phase = np.stack([X_spect[i], X_phase[i]], axis=-1)
                X_spect_phase.append(x_spect_phase)

            X_spect_phase = np.asarray(X_spect_phase)

#            print("X_spect_phase", X_spect_phase.shape)
            
            X_lips = []
            
            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i])
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 5)
                X_lips.append(x_lips)


            X_lips = np.asarray(X_lips)
#            print(X_samples.shape)
            #X = seq.augment_images(X)

            yield [X_spect_phase, X_lips, X_samples]

            batch_start += batch_size
            batch_end += batch_size

def DataGenerator_sampling_softmask(folderlist_all, folders_per_epoch, batch_size):
    
    epoch_number = 0
    L = folders_per_epoch

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:

            if batch_start==0:
                epoch_number += 1

                if epoch_number%3 == 1:
                    indices = []
                    for ind in range(len(folderlist_all)):
                        indices.append(ind)

                pick_indices = random.sample(indices, L)

                for item in pick_indices:
                    indices.remove(item)

                folderlist = []
                for index in pick_indices:
                    folderlist.append(folderlist_all[index])

            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips = []
            mask = []
            spect = []
            phase = []
            samples = []
            for folder in folders_batch:

                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                masks_ = sorted(glob.glob(folder + '/*_softmask.npy'), key=numericalSort)
                samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
                spect_ = folder + '/mixed_spectrogram.npy'
                phase_ = folder + '/phase_spectrogram.npy'

                lips.append(lips_[0])
                lips.append(lips_[1])

                samples.append(samples_[0])
                samples.append(samples_[1])

                mask.append(masks_[0])
                mask.append(masks_[1])

                spect.append(spect_)
                spect.append(spect_)

                phase.append(phase_)
                phase.append(phase_)

            zipped = list(zip(lips, samples, mask, spect, phase))
            random.shuffle(zipped)
            lips, samples, mask, spect, phase = zip(*zipped)
            
            #X_mask = np.asarray([to_onehot(cv2.imread(fname, cv2.IMREAD_UNCHANGED)) for fname in mask])
            X_mask = np.asarray([np.load(fname).reshape(257, 500, 1) for fname in mask])
            #print(X_mask.shape)
#            print('mask', X_mask.shape)
            
            X_spect = [np.load(fname) for fname in spect]
            
            X_phase = [np.load(fname) for fname in phase]

            X_samples = np.asarray([np.pad(np.load(fname), (0, 128500), mode='constant')[:128500] for fname in samples])
            
            X_spect_phase = []
            for i in range(len(X_spect)):
                x_spect_phase = np.stack([X_spect[i], X_phase[i]], axis=-1)
                X_spect_phase.append(x_spect_phase)

            X_spect_phase = np.asarray(X_spect_phase)

#            print("X_spect_phase", X_spect_phase.shape)
            
            X_lips = []
            
            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i])
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 5)
                X_lips.append(x_lips)


            X_lips = np.asarray(X_lips)
           # print(X_lips.shape)
            #X = seq.augment_images(X)

            yield [X_spect_phase, X_lips, X_samples], X_mask

            batch_start += batch_size
            batch_end += batch_size
