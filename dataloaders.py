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
import sys
sys.path.append('/data/AV-speech-separation/LipNet')
from lipnet.lipreading.helpers import text_to_labels
from lipnet.lipreading.aligns import Align
import json


'''import imgaug as ia
import imgaug.augmenters as iaa

sometimes1 = lambda aug: iaa.Sometimes(0.35, aug)
sometimes2 = lambda aug: iaa.Sometimes(0.35, aug)

seq = iaa.Sequential(
    [
        sometimes1(iaa.Affine(rotate=(-10, 10))),
        iaa.Fliplr(0.35),
        sometimes2(iaa.Affine(translate_px={"x": (-10,10), "y": (-5, 5)}, mode='constant', cval=0))
    ]
)'''

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


def DataGenerator_train_crm(folderlist, batch_size):

    L = len(folderlist)

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips = []
            crm = []
            spect = []
            phase = []
            samples = []
            #phase_mask = []
            for folder in folders_batch:

                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                crms_ = sorted(glob.glob(folder + '/*_crm.npy'), key=numericalSort)
                samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
                #phase_mask_ = sorted(glob.glob(folder + '/*_phasemask.npy'), key=numericalSort)
                spect_ = folder + '/mixed_spectrogram.npy'
                phase_ = folder + '/phase_spectrogram.npy'

                lips.append(lips_[0])
                lips.append(lips_[1])

                samples.append(samples_[0])
                samples.append(samples_[1])

                crm.append(crms_[0])
                crm.append(crms_[1])

                spect.append(spect_)
                spect.append(spect_)

                phase.append(phase_)
                phase.append(phase_)
                
                #phase_mask.append(phase_mask_[0])
                #phase_mask.append(phase_mask_[1])

            zipped = list(zip(lips, samples, crm, spect, phase))
            random.shuffle(zipped)
            lips, samples, crm, spect, phase = zip(*zipped)
            
            #X_mask = np.asarray([to_onehot(cv2.imread(fname, cv2.IMREAD_UNCHANGED)) for fname in mask])
            #X_mask = np.asarray([np.load(fname) for fname in mask])
            #X_phasemask = np.asarray([np.load(fname) for fname in phase_mask])
            X_crm = np.asarray([np.load(fname) for fname in crm])
            X_crm = np.asarray(X_crm, dtype='float32')

            Cx = 0.9999999*(X_crm>0.9999999)+X_crm*(X_crm<=0.9999999)
            X_crm = -0.9999999*(Cx<-0.9999999)+Cx*(Cx>=-0.9999999)

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

                x_lips = get_video_frames(lips[i], fmt= 'grey')
                #x_lips = seq.augment_images(x_lips)
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)
                X_lips.append(x_lips)


            X_lips = np.asarray(X_lips)
           # print(X_lips.shape)
            #X = seq.augment_images(X)
            
            #X_mag_phase_mask = np.stack([X_mask,X_phasemask], axis=-1)

            X_spect_phase = X_spect_phase[:,:,:200,:]
            X_samples = X_samples[:,:257*200]
            X_crm=X_crm[:,:,:200,:]

            #X_attns = np.random.rand(batch_size, 200, 200)

            yield [X_spect_phase, X_lips, X_samples], X_crm

            batch_start += batch_size
            batch_end += batch_size

            
def DataGenerator_sampling_crm(folderlist_all, folders_per_epoch, batch_size):
    
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
                random.seed(100*epoch_number)
                pick_indices = random.sample(indices, L)

                for item in pick_indices:
                    indices.remove(item)

                folderlist = []
                for index in pick_indices:
                    folderlist.append(folderlist_all[index])

            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips = []
            crm = []
            spect = []
            phase = []
            samples = []
            #phase_mask = []
            for folder in folders_batch:
                #print(folder)

                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                crms_ = sorted(glob.glob(folder + '/*_crm.npy'), key=numericalSort)
                samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
                #phase_mask_ = sorted(glob.glob(folder + '/*_phasemask.npy'), key=numericalSort)
                spect_ = folder + '/mixed_spectrogram.npy'
                phase_ = folder + '/phase_spectrogram.npy'
                for i in range(len(lips_)):
                    lips.append(lips_[i])
                for i in range(len(samples_)):
                    samples.append(samples_[i])
                for i in range(len(crms_)):
                    crm.append(crms_[i])
                for i in range(len(lips_)):
                    spect.append(spect_)
                for i in range(len(lips_)):
                    phase.append(phase_)
                
                #phase_mask.append(phase_mask_[0])
                #phase_mask.append(phase_mask_[1])
            
            zipped = list(zip(lips, samples, crm, spect, phase))
            random.shuffle(zipped)
            lips, samples, crm, spect, phase = zip(*zipped)
            
            #X_mask = np.asarray([to_onehot(cv2.imread(fname, cv2.IMREAD_UNCHANGED)) for fname in mask])
            X_crm = np.asarray([np.load(fname) for fname in crm])
            X_crm = np.asarray(X_crm, dtype='float32')

            Cx = 0.9999999*(X_crm>0.9999999)+X_crm*(X_crm<=0.9999999)
            X_crm = -0.9999999*(Cx<-0.9999999)+Cx*(Cx>=-0.9999999)

            #X_phasemask = np.asarray([np.load(fname) for fname in phase_mask])
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

                x_lips = get_video_frames(lips[i], fmt= 'grey')
                #x_lips = seq.augment_images(x_lips)
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)
                X_lips.append(x_lips)


            X_lips = np.asarray(X_lips)
           # print(X_lips.shape)
            #X = seq.augment_images(X)
            
            #X_mag_phase_mask = np.stack([X_mask,X_phasemask], axis=-1)

            '''print('X_spect_phase:', X_spect_phase.shape)
            print('X_lips', X_lips.shape)
            print('X_samples')'''

            X_spect_phase = X_spect_phase[:,:,:200,:]
            X_samples = X_samples[:,:257*200]
            X_crm=X_crm[:,:,:200,:]

            #X_attns = np.random.rand(batch_size, 200, 200)

            yield [X_spect_phase, X_lips, X_samples], X_crm

            batch_start += batch_size
            batch_end += batch_size

            
def DataGenerator_test_crm(folderlist, batch_size):

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

            X_samples = np.asarray([np.pad(np.load(fname), (0, 257*100*5), mode='constant')[:257*100*5] for fname in samples])
            
            X_spect_phase = []
            for i in range(len(X_spect)):
                x_spect_phase = np.stack([X_spect[i], X_phase[i]], axis=-1)
                X_spect_phase.append(x_spect_phase)

            X_spect_phase = np.asarray(X_spect_phase)

#            print("X_spect_phase", X_spect_phase.shape)
            
            X_lips = []
            
            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i], fmt='grey')
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds =2)
                X_lips.append(x_lips)


            X_lips = np.asarray(X_lips)
#            print(X_samples.shape)
            #X = seq.augment_images(X)

            X_spect_phase = X_spect_phase[:,:,:200,:]
            X_samples = X_samples[:,:257*200]

            yield [X_spect_phase, X_lips, X_samples]

            batch_start += batch_size
            batch_end += batch_size


# Data generators for time samples prediction

            
def DataGenerator_sampling_samples(folderlist_all, folders_per_epoch, batch_size):
    
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
                random.seed(100*(epoch_number+9))
                pick_indices = random.sample(indices, L)

                for item in pick_indices:
                    indices.remove(item)

                folderlist = []
                for index in pick_indices:
                    folderlist.append(folderlist_all[index])

            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips = []
            samples = []
            samples_mix = []
            #phase_mask = []
            for folder in folders_batch:
                #print(folder)

                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
                #samples_mix_ = sorted(glob.glob(folder + '/mix_samples.npy'), key=numericalSort)
                samples_mix_ = '/data/mixed_audio_files/' +folder.split('/')[-1]+'.wav'
                for i in range(len(lips_)):
                    lips.append(lips_[i])
                for i in range(len(samples_)):
                    samples.append(samples_[i])
                for i in range(len(lips_)):
                    samples_mix.append(samples_mix_)
          
            zipped = list(zip(lips, samples, samples_mix))
            random.shuffle(zipped)
            lips, samples, samples_mix = zip(*zipped)

            #X_phasemask = np.asarray([np.load(fname) for fname in phase_mask])
            #print(X_mask.shape)
#            print('mask', X_mask.shape)
            
            #X_spect = [np.load(fname) for fname in spect]
            
            #X_phase = [np.load(fname) for fname in phase]

            X_samples = np.asarray([np.pad(np.load(fname), (0, 32000), mode='constant')[:32000] for fname in samples])
            X_samples_mix = np.asarray([np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[:32000] for fname in samples_mix])
            
            '''X_spect_phase = []
            for i in range(len(X_spect)):
                x_spect_phase = np.stack([X_spect[i], X_phase[i]], axis=-1)
                X_spect_phase.append(x_spect_phase)

            X_spect_phase = np.asarray(X_spect_phase)'''

#            print("X_spect_phase", X_spect_phase.shape)
            
            X_lips = []
            
            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i], fmt= 'grey')
                #x_lips = seq.augment_images(x_lips)
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)
                X_lips.append(x_lips)


            X_lips = np.asarray(X_lips)
           # print(X_lips.shape)
            #X = seq.augment_images(X)
            
            #X_mag_phase_mask = np.stack([X_mask,X_phasemask], axis=-1)

            '''print('X_spect_phase:', X_spect_phase.shape)
            print('X_lips', X_lips.shape)
            print('X_samples')'''

            #X_spect_phase = X_spect_phase[:,:,:200,:]
            #X_samples = X_samples[:,:32000]
            X_samples_targ = X_samples.reshape(X_samples.shape[0], 32000, 1).astype('float32')
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], 32000, 1).astype('float32')
            X_samples_targ = X_samples_targ/1350.0
            X_samples_mix = X_samples_mix/1350.0
            #print(X_samples_targ.shape)

            #X_attns = np.random.rand(batch_size, 200, 200)

            yield (X_lips, X_samples_mix), X_samples_targ
            #print(yes)
            batch_start += batch_size
            batch_end += batch_size

            
def DataGenerator_sampling_samples_attention(folderlist_all, folders_per_epoch, batch_size):
    
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
                random.seed(100*(epoch_number+9))
                pick_indices = random.sample(indices, L)

                for item in pick_indices:
                    indices.remove(item)

                folderlist = []
                for index in pick_indices:
                    folderlist.append(folderlist_all[index])

            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips = []
            samples = []
            samples_mix = []
            #phase_mask = []
            for folder in folders_batch:
                #print(folder)

                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
                #samples_mix_ = sorted(glob.glob(folder + '/mix_samples.npy'), key=numericalSort)
                samples_mix_ = '/data/mixed_audio_files/' +folder.split('/')[-1]+'.wav'
                for i in range(len(lips_)):
                    lips.append(lips_[i])
                for i in range(len(samples_)):
                    samples.append(samples_[i])
                for i in range(len(lips_)):
                    samples_mix.append(samples_mix_)
          
            zipped = list(zip(lips, samples, samples_mix))
            random.shuffle(zipped)
            lips, samples, samples_mix = zip(*zipped)

            #X_phasemask = np.asarray([np.load(fname) for fname in phase_mask])
            #print(X_mask.shape)
#            print('mask', X_mask.shape)
            
            #X_spect = [np.load(fname) for fname in spect]
            
            #X_phase = [np.load(fname) for fname in phase]

            X_samples = np.asarray([np.pad(np.load(fname), (0, 32000), mode='constant')[:32000] for fname in samples])
            X_samples_mix = np.asarray([np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[:32000] for fname in samples_mix])
            
            '''X_spect_phase = []
            for i in range(len(X_spect)):
                x_spect_phase = np.stack([X_spect[i], X_phase[i]], axis=-1)
                X_spect_phase.append(x_spect_phase)

            X_spect_phase = np.asarray(X_spect_phase)'''

#            print("X_spect_phase", X_spect_phase.shape)
            
            X_lips = []
            
            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i], fmt= 'grey')
                #x_lips = seq.augment_images(x_lips)
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)
                X_lips.append(x_lips)


            X_lips = np.asarray(X_lips)
           # print(X_lips.shape)
            #X = seq.augment_images(X)
            
            #X_mag_phase_mask = np.stack([X_mask,X_phasemask], axis=-1)

            '''print('X_spect_phase:', X_spect_phase.shape)
            print('X_lips', X_lips.shape)
            print('X_samples')'''

            #X_spect_phase = X_spect_phase[:,:,:200,:]
            #X_samples = X_samples[:,:32000]
            X_samples_targ = X_samples.reshape(X_samples.shape[0], 32000, 1).astype('float32')
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], 32000, 1).astype('float32')
            X_samples_targ = X_samples_targ/1350.0
            X_samples_mix = X_samples_mix/1350.0
            #print(X_samples_targ.shape)
            placeholder_1 = tf.zeros(shape=(X_lips.shape[0], 256))
            placeholder_2 = tf.zeros(shape=(X_lips.shape[0], 256))
            #X_attns = np.random.rand(batch_size, 200, 200)

            yield (X_lips, X_samples_mix,placeholder_1,placeholder_2), X_samples_targ
            #print(yes)
            batch_start += batch_size
            batch_end += batch_size



def DataGenerator_val_samples(folderlist, batch_size):
    
    L = len(folderlist)

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:

            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips = []
            samples = []
            samples_mix = []

            for folder in folders_batch:
                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
                samples_mix_ = '/data/mixed_audio_files/' +folder.split('/')[-1]+'.wav'
                for i in range(len(lips_)):
                    lips.append(lips_[i])
                for i in range(len(samples_)):
                    samples.append(samples_[i])
                for i in range(len(lips_)):
                    samples_mix.append(samples_mix_)
          
            '''zipped = list(zip(lips, samples, samples_mix))
            random.shuffle(zipped)
            lips, samples, samples_mix = zip(*zipped)'''

            #X_phasemask = np.asarray([np.load(fname) for fname in phase_mask])
            #print(X_mask.shape)
#            print('mask', X_mask.shape)
            
            #X_spect = [np.load(fname) for fname in spect]
            
            #X_phase = [np.load(fname) for fname in phase]

            X_samples = np.asarray([np.pad(np.load(fname), (0, 32000), mode='constant')[:32000] for fname in samples])
            X_samples_mix = np.asarray([np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[:32000] for fname in samples_mix])
            
            '''X_spect_phase = []
            for i in range(len(X_spect)):
                x_spect_phase = np.stack([X_spect[i], X_phase[i]], axis=-1)
                X_spect_phase.append(x_spect_phase)

            X_spect_phase = np.asarray(X_spect_phase)'''

#            print("X_spect_phase", X_spect_phase.shape)
            
            X_lips = []
            
            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i], fmt= 'grey')
                #x_lips = seq.augment_images(x_lips)
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)
                X_lips.append(x_lips)


            X_lips = np.asarray(X_lips)
           # print(X_lips.shape)
            #X = seq.augment_images(X)
            
            #X_mag_phase_mask = np.stack([X_mask,X_phasemask], axis=-1)

            '''print('X_spect_phase:', X_spect_phase.shape)
            print('X_lips', X_lips.shape)
            print('X_samples')'''

            #X_spect_phase = X_spect_phase[:,:,:200,:]
            #X_samples = X_samples[:,:32000]
            X_samples_targ = X_samples.reshape(X_samples.shape[0], 32000, 1).astype('float32')
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], 32000, 1).astype('float32')
            X_samples_targ = X_samples_targ
            X_samples_mix = X_samples_mix/1350.0
            #print(X_samples_targ.shape)

            #X_attns = np.random.rand(batch_size, 200, 200)

            yield [X_lips, X_samples_mix], X_samples_targ

            batch_start += batch_size
            batch_end += batch_size


def DataGenerator_train_samples(folderlist, batch_size):
    
    L = len(folderlist)

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:

            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips = []
            samples = []
            samples_mix = []

            for folder in folders_batch:
                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
                samples_mix_ = '/data/mixed_audio_files/' +folder.split('/')[-1]+'.wav'
                for i in range(len(lips_)):
                    lips.append(lips_[i])
                for i in range(len(samples_)):
                    samples.append(samples_[i])
                for i in range(len(lips_)):
                    samples_mix.append(samples_mix_)
          
            zipped = list(zip(lips, samples, samples_mix))
            random.shuffle(zipped)
            lips, samples, samples_mix = zip(*zipped)

            #X_phasemask = np.asarray([np.load(fname) for fname in phase_mask])
            #print(X_mask.shape)
#            print('mask', X_mask.shape)
            
            #X_spect = [np.load(fname) for fname in spect]
            
            #X_phase = [np.load(fname) for fname in phase]

            X_samples = np.asarray([np.pad(np.load(fname), (0, 32000), mode='constant')[:32000] for fname in samples])
            X_samples_mix = np.asarray([np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[:32000] for fname in samples_mix])
            
            '''X_spect_phase = []
            for i in range(len(X_spect)):
                x_spect_phase = np.stack([X_spect[i], X_phase[i]], axis=-1)
                X_spect_phase.append(x_spect_phase)

            X_spect_phase = np.asarray(X_spect_phase)'''

#            print("X_spect_phase", X_spect_phase.shape)
            
            X_lips = []
            
            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i], fmt= 'grey')
                #x_lips = seq.augment_images(x_lips)
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)
                X_lips.append(x_lips)


            X_lips = np.asarray(X_lips)
           # print(X_lips.shape)
            #X = seq.augment_images(X)
            
            #X_mag_phase_mask = np.stack([X_mask,X_phasemask], axis=-1)

            '''print('X_spect_phase:', X_spect_phase.shape)
            print('X_lips', X_lips.shape)
            print('X_samples')'''

            #X_spect_phase = X_spect_phase[:,:,:200,:]
            #X_samples = X_samples[:,:32000]
            X_samples_targ = X_samples.reshape(X_samples.shape[0], 32000, 1).astype('float32')
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], 32000, 1).astype('float32')
            X_samples_targ = X_samples_targ
            X_samples_mix = X_samples_mix/1350.0
            #print(X_samples_targ.shape)

            #X_attns = np.random.rand(batch_size, 200, 200)

            yield [X_lips, X_samples_mix], X_samples_targ

            batch_start += batch_size
            batch_end += batch_size

            
def DataGenerator_val_samples_attention(folderlist, batch_size):
    
    L = len(folderlist)

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:

            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips = []
            samples = []
            samples_mix = []

            for folder in folders_batch:
                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
                samples_mix_ = '/data/mixed_audio_files/' +folder.split('/')[-1]+'.wav'
                for i in range(len(lips_)):
                    lips.append(lips_[i])
                for i in range(len(samples_)):
                    samples.append(samples_[i])
                for i in range(len(lips_)):
                    samples_mix.append(samples_mix_)
          
            '''zipped = list(zip(lips, samples, samples_mix))
            random.shuffle(zipped)
            lips, samples, samples_mix = zip(*zipped)'''

            #X_phasemask = np.asarray([np.load(fname) for fname in phase_mask])
            #print(X_mask.shape)
#            print('mask', X_mask.shape)
            
            #X_spect = [np.load(fname) for fname in spect]
            
            #X_phase = [np.load(fname) for fname in phase]

            X_samples = np.asarray([np.pad(np.load(fname), (0, 32000), mode='constant')[:32000] for fname in samples])
            X_samples_mix = np.asarray([np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[:32000] for fname in samples_mix])
            
            '''X_spect_phase = []
            for i in range(len(X_spect)):
                x_spect_phase = np.stack([X_spect[i], X_phase[i]], axis=-1)
                X_spect_phase.append(x_spect_phase)

            X_spect_phase = np.asarray(X_spect_phase)'''

#            print("X_spect_phase", X_spect_phase.shape)
            
            X_lips = []
            
            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i], fmt= 'grey')
                #x_lips = seq.augment_images(x_lips)
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)
                X_lips.append(x_lips)


            X_lips = np.asarray(X_lips)
           # print(X_lips.shape)
            #X = seq.augment_images(X)
            
            #X_mag_phase_mask = np.stack([X_mask,X_phasemask], axis=-1)

            '''print('X_spect_phase:', X_spect_phase.shape)
            print('X_lips', X_lips.shape)
            print('X_samples')'''

            #X_spect_phase = X_spect_phase[:,:,:200,:]
            #X_samples = X_samples[:,:32000]
            X_samples_targ = X_samples.reshape(X_samples.shape[0], 32000, 1).astype('float32')
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], 32000, 1).astype('float32')
            X_samples_targ = X_samples_targ
            X_samples_mix = X_samples_mix/1350.0
            #print(X_samples_targ.shape)
            np.random.seed(100)
            placeholder_1 = tf.constant(np.random.rand(X_lips.shape[0], 256))
            placeholder_2 = tf.constant(np.random.rand(X_lips.shape[0], 256))
            placeholder_3 = tf.constant(np.random.rand(X_lips.shape[0], 512))

            #X_attns = np.random.rand(batch_size, 200, 200)

            yield [X_lips, X_samples_mix,placeholder_1,placeholder_2, placeholder_3], X_samples_targ

            batch_start += batch_size
            batch_end += batch_size

def DataGenerator_train_samples_attention(folderlist, batch_size):
    
    L = len(folderlist)
    epoch_number = 0

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:

            if batch_start == 0:
                epoch_number += 1

            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips = []
            samples = []
            samples_mix = []

            for folder in folders_batch:
                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
                samples_mix_ = '/data/mixed_audio_files/' +folder.split('/')[-1]+'.wav'
                for i in range(len(lips_)):
                    lips.append(lips_[i])
                for i in range(len(samples_)):
                    samples.append(samples_[i])
                for i in range(len(lips_)):
                    samples_mix.append(samples_mix_)
          
            zipped = list(zip(lips, samples, samples_mix))
            random.shuffle(zipped)
            lips, samples, samples_mix = zip(*zipped)

            #X_phasemask = np.asarray([np.load(fname) for fname in phase_mask])
            #print(X_mask.shape)
#            print('mask', X_mask.shape)
            
            #X_spect = [np.load(fname) for fname in spect]
            
            #X_phase = [np.load(fname) for fname in phase]

            X_samples = np.asarray([np.pad(np.load(fname), (0, 32000), mode='constant')[:32000] for fname in samples])
            X_samples_mix = np.asarray([np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[:32000] for fname in samples_mix])
            
            '''X_spect_phase = []
            for i in range(len(X_spect)):
                x_spect_phase = np.stack([X_spect[i], X_phase[i]], axis=-1)
                X_spect_phase.append(x_spect_phase)

            X_spect_phase = np.asarray(X_spect_phase)'''

#            print("X_spect_phase", X_spect_phase.shape)
            
            X_lips = []
            
            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i], fmt= 'grey')
                #x_lips = seq.augment_images(x_lips)
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)
                X_lips.append(x_lips)


            X_lips = np.asarray(X_lips)
           # print(X_lips.shape)
            #X = seq.augment_images(X)
            
            #X_mag_phase_mask = np.stack([X_mask,X_phasemask], axis=-1)

            '''print('X_spect_phase:', X_spect_phase.shape)
            print('X_lips', X_lips.shape)
            print('X_samples')'''

            #X_spect_phase = X_spect_phase[:,:,:200,:]
            #X_samples = X_samples[:,:32000]
            X_samples_targ = X_samples.reshape(X_samples.shape[0], 32000, 1).astype('float32')
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], 32000, 1).astype('float32')
            X_samples_targ = X_samples_targ
            X_samples_mix = X_samples_mix/1350.0
            #print(X_samples_targ.shape)
            np.random.seed(100)
            placeholder_1 = tf.constant(np.random.rand(X_lips.shape[0], 256))
            placeholder_2 = tf.constant(np.random.rand(X_lips.shape[0], 256))
            placeholder_3 = tf.constant(np.random.rand(X_lips.shape[0], 512))

            #X_attns = np.random.rand(batch_size, 200, 200)

            yield [X_lips, X_samples_mix,placeholder_1,placeholder_2, placeholder_3], X_samples_targ

            batch_start += batch_size
            batch_end += batch_size

def DataGenerator_test_samples(folderlist, batch_size):
    
    L = len(folderlist)

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:

            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips = []
            samples_mix = []

            for folder in folders_batch:
                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                samples_mix_ = '/data/mixed_audio_files/' +folder.split('/')[-1]+'.wav'
                for i in range(len(lips_)):
                    lips.append(lips_[i])
                for i in range(len(lips_)):
                    samples_mix.append(samples_mix_)
          
            '''zipped = list(zip(lips, samples, samples_mix))
            random.shuffle(zipped)
            lips, samples, samples_mix = zip(*zipped)'''

            #X_phasemask = np.asarray([np.load(fname) for fname in phase_mask])
            #print(X_mask.shape)
#            print('mask', X_mask.shape)
            
            #X_spect = [np.load(fname) for fname in spect]
            
            #X_phase = [np.load(fname) for fname in phase]

            X_samples_mix = np.asarray([np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[:32000] for fname in samples_mix])
            
            '''X_spect_phase = []
            for i in range(len(X_spect)):
                x_spect_phase = np.stack([X_spect[i], X_phase[i]], axis=-1)
                X_spect_phase.append(x_spect_phase)

            X_spect_phase = np.asarray(X_spect_phase)'''

#            print("X_spect_phase", X_spect_phase.shape)
            
            X_lips = []
            
            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i], fmt= 'grey')
                #x_lips = seq.augment_images(x_lips)
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)
                X_lips.append(x_lips)


            X_lips = np.asarray(X_lips)
           # print(X_lips.shape)
            #X = seq.augment_images(X)
            
            #X_mag_phase_mask = np.stack([X_mask,X_phasemask], axis=-1)

            '''print('X_spect_phase:', X_spect_phase.shape)
            print('X_lips', X_lips.shape)
            print('X_samples')'''

            #X_spect_phase = X_spect_phase[:,:,:200,:]
            #X_samples = X_samples[:,:32000]
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], 32000, 1).astype('float32')
            X_samples_mix = X_samples_mix/1350.0
            #print(X_samples_targ.shape)

            #X_attns = np.random.rand(batch_size, 200, 200)

            yield [X_lips, X_samples_mix]

            batch_start += batch_size
            batch_end += batch_size

def DataGenerator_test_samples_attention(folderlist, batch_size):
    
    L = len(folderlist)

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:

            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips = []
            samples_mix = []

            for folder in folders_batch:
                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                samples_mix_ = '/data/mixed_audio_files/' +folder.split('/')[-1]+'.wav'
                for i in range(len(lips_)):
                    lips.append(lips_[i])
                for i in range(len(lips_)):
                    samples_mix.append(samples_mix_)
          
            '''zipped = list(zip(lips, samples, samples_mix))
            random.shuffle(zipped)
            lips, samples, samples_mix = zip(*zipped)'''

            #X_phasemask = np.asarray([np.load(fname) for fname in phase_mask])
            #print(X_mask.shape)
#            print('mask', X_mask.shape)
            
            #X_spect = [np.load(fname) for fname in spect]
            
            #X_phase = [np.load(fname) for fname in phase]

            X_samples_mix = np.asarray([np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[:32000] for fname in samples_mix])
            
            '''X_spect_phase = []
            for i in range(len(X_spect)):
                x_spect_phase = np.stack([X_spect[i], X_phase[i]], axis=-1)
                X_spect_phase.append(x_spect_phase)

            X_spect_phase = np.asarray(X_spect_phase)'''

#            print("X_spect_phase", X_spect_phase.shape)
            
            X_lips = []
            
            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i], fmt= 'grey')
                #x_lips = seq.augment_images(x_lips)
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)
                X_lips.append(x_lips)


            X_lips = np.asarray(X_lips)
           # print(X_lips.shape)
            #X = seq.augment_images(X)
            
            #X_mag_phase_mask = np.stack([X_mask,X_phasemask], axis=-1)

            '''print('X_spect_phase:', X_spect_phase.shape)
            print('X_lips', X_lips.shape)
            print('X_samples')'''

            #X_spect_phase = X_spect_phase[:,:,:200,:]
            #X_samples = X_samples[:,:32000]
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], 32000, 1).astype('float32')
            X_samples_mix = X_samples_mix/1350.0
            #print(X_samples_targ.shape)
            placeholder_1 = tf.zeros(shape=(X_lips.shape[0], 256))
            placeholder_2 = tf.zeros(shape=(X_lips.shape[0], 256))
            #X_attns = np.random.rand(batch_size, 200, 200)

            yield [X_lips, X_samples_mix,placeholder_1,placeholder_2]

            batch_start += batch_size
            batch_end += batch_size


def Data_predict_attention(folderlist):

            lips = []
            samples_mix = []

            for folder in folderlist:
                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                samples_mix_ = '/data/mixed_audio_files/' +folder.split('/')[-1]+'.wav'
                for i in range(len(lips_)):
                    lips.append(lips_[i])
                for i in range(len(lips_)):
                    samples_mix.append(samples_mix_)
          
            '''zipped = list(zip(lips, samples, samples_mix))
            random.shuffle(zipped)
            lips, samples, samples_mix = zip(*zipped)'''

            #X_phasemask = np.asarray([np.load(fname) for fname in phase_mask])
            #print(X_mask.shape)
#            print('mask', X_mask.shape)
            
            #X_spect = [np.load(fname) for fname in spect]
            
            #X_phase = [np.load(fname) for fname in phase]

            X_samples_mix = np.asarray([np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[:32000] for fname in samples_mix])
            
            '''X_spect_phase = []
            for i in range(len(X_spect)):
                x_spect_phase = np.stack([X_spect[i], X_phase[i]], axis=-1)
                X_spect_phase.append(x_spect_phase)

            X_spect_phase = np.asarray(X_spect_phase)'''

#            print("X_spect_phase", X_spect_phase.shape)
            
            X_lips = []
            
            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i], fmt= 'grey')
                #x_lips = seq.augment_images(x_lips)
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)
                X_lips.append(x_lips)


            X_lips = np.asarray(X_lips).astype('float32')
           # print(X_lips.shape)
            #X = seq.augment_images(X)
            
            #X_mag_phase_mask = np.stack([X_mask,X_phasemask], axis=-1)

            '''print('X_spect_phase:', X_spect_phase.shape)
            print('X_lips', X_lips.shape)
            print('X_samples')'''

            #X_spect_phase = X_spect_phase[:,:,:200,:]
            #X_samples = X_samples[:,:32000]
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], 32000, 1).astype('float32')
            X_samples_mix = X_samples_mix
            #print(X_samples_targ.shape)
            np.random.seed(100)
            placeholder_1 = tf.constant(np.random.rand(X_lips.shape[0], 256))
            placeholder_2 = tf.constant(np.random.rand(X_lips.shape[0], 256))
            placeholder_3 = tf.constant(np.random.rand(X_lips.shape[0], 512))
            #X_attns = np.random.rand(batch_size, 200, 200)

            return [X_lips, X_samples_mix, placeholder_1, placeholder_2, placeholder_3]

            
def DataGenerator_train_samples_lips(folderlist, batch_size, time=5):
    
    L = len(folderlist)

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:

            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips = []
            samples = []
            samples_mix = []
            transcripts = []
            for folder in folders_batch:
                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
                samples_mix_ = '/data/mixed_audio_files/' +folder.split('/')[-1]+'.wav'
                transcripts_ = sorted(glob.glob(folder + '/*.txt'), key=numericalSort)
                for i in range(len(lips_)):
                    lips.append(lips_[i])
                for i in range(len(samples_)):
                    samples.append(samples_[i])
                for i in range(len(lips_)):
                    samples_mix.append(samples_mix_)
                for i in range(len(lips_)):
                    transcripts.append(transcripts_[i])
          
            zipped = list(zip(lips, samples, samples_mix, transcripts))
            random.shuffle(zipped)
            lips, samples, samples_mix, transcripts = zip(*zipped)

            #X_phasemask = np.asarray([np.load(fname) for fname in phase_mask])
            #print(X_mask.shape)
#            print('mask', X_mask.shape)
            
            #X_spect = [np.load(fname) for fname in spect]
            
            #X_phase = [np.load(fname) for fname in phase]

            X_samples = np.asarray([np.pad(np.load(fname), (0, time*16000), mode='constant')[:time*16000] for fname in samples])
            X_samples_mix = np.asarray([np.pad(wavfile.read(fname)[1], (0, time*16000), mode='constant')[:time*16000] for fname in samples_mix])
            
            '''X_spect_phase = []
            for i in range(len(X_spect)):
                x_spect_phase = np.stack([X_spect[i], X_phase[i]], axis=-1)
                X_spect_phase.append(x_spect_phase)

            X_spect_phase = np.asarray(X_spect_phase)'''

#            print("X_spect_phase", X_spect_phase.shape)
            
            X_lips = []
            
            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i], fmt= 'grey')
                #x_lips = seq.augment_images(x_lips)
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = time)
                X_lips.append(x_lips)

            X_lips = np.asarray(X_lips)
           # print(X_lips.shape)
            #X = seq.augment_images(X)
            
            #X_mag_phase_mask = np.stack([X_mask,X_phasemask], axis=-1)

            '''print('X_spect_phase:', X_spect_phase.shape)
            print('X_lips', X_lips.shape)
            print('X_samples')'''

            align = []
            Y_data = []
            label_length = []
            input_length = []
            source_str = []

            for i in range(len(transcripts)):align.append(Align(128, text_to_labels).from_file(transcripts[i]))
            for i in range(X_lips.shape[0]):
               Y_data.append(align[i].padded_label)
               label_length.append(align[i].label_length)
               input_length.append(X_lips.shape[1])
               source_str.append(align[i].sentence)
            Y_data = np.array(Y_data)

            #X_spect_phase = X_spect_phase[:,:,:200,:]
            #X_samples = X_samples[:,:32000]
            X_samples_targ = X_samples.reshape(X_samples.shape[0], time*16000, 1).astype('float32')
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], time*16000, 1).astype('float32')
            X_samples_targ = X_samples_targ/1350.0
            X_samples_mix = X_samples_mix/1350.0
            #print(X_samples_targ.shape)

            #X_attns = np.random.rand(batch_size, 200, 200)

            yield [X_lips, X_samples_mix, Y_data, np.array(input_length), np.array(label_length)], [X_samples_targ, np.zeros([X_lips.shape[0]])]

            batch_start += batch_size
            batch_end += batch_size

def DataGenerator_val_samples_lips(folderlist, batch_size, time=5):
    
    L = len(folderlist)

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:

            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips = []
            samples = []
            samples_mix = []
            transcripts = []
            for folder in folders_batch:
                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
                samples_mix_ = '/data/mixed_audio_files/' +folder.split('/')[-1]+'.wav'
                transcripts_ = sorted(glob.glob(folder + '/*.txt'), key=numericalSort)
                for i in range(len(lips_)):
                    lips.append(lips_[i])
                for i in range(len(samples_)):
                    samples.append(samples_[i])
                for i in range(len(lips_)):
                    samples_mix.append(samples_mix_)
                for i in range(len(lips_)):
                    transcripts.append(transcripts_[i])
          
            '''zipped = list(zip(lips, samples, samples_mix, transcripts))
            random.shuffle(zipped)
            lips, samples, samples_mix, transcripts = zip(*zipped)'''

            #X_phasemask = np.asarray([np.load(fname) for fname in phase_mask])
            #print(X_mask.shape)
#            print('mask', X_mask.shape)
            
            #X_spect = [np.load(fname) for fname in spect]
            
            #X_phase = [np.load(fname) for fname in phase]

            X_samples = np.asarray([np.pad(np.load(fname), (0, time*16000), mode='constant')[:time*16000] for fname in samples])
            X_samples_mix = np.asarray([np.pad(wavfile.read(fname)[1], (0, time*16000), mode='constant')[:time*16000] for fname in samples_mix])
            
            '''X_spect_phase = []
            for i in range(len(X_spect)):
                x_spect_phase = np.stack([X_spect[i], X_phase[i]], axis=-1)
                X_spect_phase.append(x_spect_phase)

            X_spect_phase = np.asarray(X_spect_phase)'''

#            print("X_spect_phase", X_spect_phase.shape)
            
            X_lips = []
            
            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i], fmt= 'grey')
                #x_lips = seq.augment_images(x_lips)
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = time)
                X_lips.append(x_lips)

            X_lips = np.asarray(X_lips)
           # print(X_lips.shape)
            #X = seq.augment_images(X)
            
            #X_mag_phase_mask = np.stack([X_mask,X_phasemask], axis=-1)

            '''print('X_spect_phase:', X_spect_phase.shape)
            print('X_lips', X_lips.shape)
            print('X_samples')'''

            align = []
            Y_data = []
            label_length = []
            input_length = []
            source_str = []

            for i in range(len(transcripts)):align.append(Align(128, text_to_labels).from_file(transcripts[i]))
            for i in range(X_lips.shape[0]):
               Y_data.append(align[i].padded_label)
               label_length.append(align[i].label_length)
               input_length.append(X_lips.shape[1])
               source_str.append(align[i].sentence)
            Y_data = np.array(Y_data)

            #X_spect_phase = X_spect_phase[:,:,:200,:]
            #X_samples = X_samples[:,:32000]
            X_samples_targ = X_samples.reshape(X_samples.shape[0], time*16000, 1).astype('float32')
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], time*16000, 1).astype('float32')
            X_samples_targ = X_samples_targ/1350.0
            X_samples_mix = X_samples_mix/1350.0
            #print(X_samples_targ.shape)

            #X_attns = np.random.rand(batch_size, 200, 200)

            yield [X_lips, X_samples_mix, Y_data, np.array(input_length), np.array(label_length)], [X_samples_targ, np.zeros([X_lips.shape[0]])]

            batch_start += batch_size
            batch_end += batch_size

'''samples_mix = sorted(glob.glob('/data/mixed_audio_files/*.wav'), key=numericalSort)
samples = [np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[:32000] for fname in samples_mix]
means = []
stds = []
for i in samples:
    means.append(np.mean(i))
    stds.append(np.std(i))'''


def DataGenerator_val_unsync_attention(folderlist, batch_size):
    
    L = len(folderlist)
    #unsync_files = np.loadtxt('/data/AV-speech-separation1/lrs2_1dot5k-unsync_audio_val.json', dtype='object')
    with open('/data/AV-speech-separation1/lrs2_1dot5k-unsync_audio_val.json') as json_file:
        unsync_dict = json.load(json_file)
    
    unsync_files = unsync_dict['folds']
    offsets = unsync_dict['offsets']

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:

            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips = []
            samples = []
            samples_mix = []

            for folder in folders_batch:
                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
                samples_mix_ = '/data/mixed_audio_files/' +folder.split('/')[-1]+'.wav'
                for i in range(len(lips_)):
                    lips.append(lips_[i])
                for i in range(len(samples_)):
                    samples.append(samples_[i])
                for i in range(len(lips_)):
                    samples_mix.append(samples_mix_)
          
            '''zipped = list(zip(lips, samples, samples_mix))
            random.shuffle(zipped)
            lips, samples, samples_mix = zip(*zipped)'''

            #X_samples = np.asarray([np.pad(np.load(fname), (0, 32000), mode='constant')[:32000] for fname in samples])
            
            X_samples_mix = []
            X_samples = []
            X_lips = []
            for i, fname in enumerate(samples_mix):
                fold = fname.split('/')[-1][:-4]
                if fold in unsync_files:
                    offset = abs(offsets[fold])
                    if offset > 12:
                        offset = 13
                    #offset = offset-2
                    aud_offset = int(abs((offset/25)*16000))

                    # Read Mixed Audio with offset
                    '''mix_aud = np.pad(wavfile.read(fname)[1], (aud_offset, 32000), mode='constant')[:32000]
                    aud = np.pad(np.load(samples[i]), (0, 32000), mode='constant')[:32000]
                    aud = np.pad(aud[:-aud_offset], (0, 32000), mode='constant')[:32000]'''

                    mix_aud = np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[aud_offset:32000+aud_offset]
                    aud = np.pad(np.load(samples[i]), (0, 32000), mode='constant')[aud_offset:32000+aud_offset]
                    #aud = np.pad(aud[:-aud_offset], (0, 32000), mode='constant')[:32000]

                    # Read Lips
                    x_lips = get_video_frames(lips[i], fmt= 'grey')
                    x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)


                else:
                    mix_aud = np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[:32000]
                    aud = np.pad(np.load(samples[i]), (0, 32000), mode='constant')[:32000]

                    # Read Lips
                    x_lips = get_video_frames(lips[i], fmt= 'grey')
                    x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)
                
                X_samples_mix.append(mix_aud)
                X_samples.append(aud)
                X_lips.append(x_lips)

            X_samples_mix = np.asarray(X_samples_mix)
            X_samples = np.asarray(X_samples)
            X_lips = np.asarray(X_lips)

            X_samples_targ = X_samples.reshape(X_samples.shape[0], 32000, 1).astype('float32')
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], 32000, 1).astype('float32')
            X_samples_targ = X_samples_targ
            X_samples_mix = X_samples_mix/1350.0
            #print(X_samples_targ.shape)
            np.random.seed(100)
            placeholder_1 = tf.constant(np.random.rand(X_lips.shape[0], 256))
            placeholder_2 = tf.constant(np.random.rand(X_lips.shape[0], 256))
            placeholder_3 = tf.constant(np.random.rand(X_lips.shape[0], 512))

            #X_attns = np.random.rand(batch_size, 200, 200)

            yield [X_lips, X_samples_mix,placeholder_1,placeholder_2, placeholder_3], X_samples_targ

            batch_start += batch_size
            batch_end += batch_size

def DataGenerator_train_unsync_attention(folderlist, batch_size):
    
    L = len(folderlist)
    #unsync_files = np.loadtxt('/data/AV-speech-separation1/lrs2_8k-unsync_audio_train.json', dtype='object')
    with open('/data/AV-speech-separation1/lrs2_8k-unsync_audio_train.json') as json_file:
        unsync_dict = json.load(json_file)

    unsync_files = unsync_dict['folds']
    offsets = unsync_dict['offsets']
    
    epoch_number = 0

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:

            if batch_start == 0:
                epoch_number += 1

            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips = []
            samples = []
            samples_mix = []

            for folder in folders_batch:
                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
                samples_mix_ = '/data/mixed_audio_files/' +folder.split('/')[-1]+'.wav'
                for i in range(len(lips_)):
                    lips.append(lips_[i])
                for i in range(len(samples_)):
                    samples.append(samples_[i])
                for i in range(len(lips_)):
                    samples_mix.append(samples_mix_)
          
            zipped = list(zip(lips, samples, samples_mix))
            random.shuffle(zipped)
            lips, samples, samples_mix = zip(*zipped)

            #X_samples = np.asarray([np.pad(np.load(fname), (0, 32000), mode='constant')[:32000] for fname in samples])
            
            X_samples_mix = []
            X_samples = []
            X_lips = []
            for i, fname in enumerate(samples_mix):
                fold = fname.split('/')[-1][:-4]
                if fold in unsync_files:
                    offset = abs(offsets[fold])
                    if offset > 15:
                        offset = 10
                    elif offset == 14:
                        offset = 14
                    elif offset == 15:
                        offset = 15
                    offset = offset-2
                    aud_offset = int(abs((offset/25)*16000))

                    # Read Mixed Audio with offset
                    '''mix_aud = np.pad(wavfile.read(fname)[1], (aud_offset, 32000), mode='constant')[:32000]
                    aud = np.pad(np.load(samples[i]), (0, 32000), mode='constant')[:32000]
                    aud = np.pad(aud[:-aud_offset], (0, 32000), mode='constant')[:32000]'''

                    mix_aud = np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[aud_offset:32000+aud_offset]
                    aud = np.pad(np.load(samples[i]), (0, 32000), mode='constant')[aud_offset:32000+aud_offset]
                    #aud = np.pad(aud[:-aud_offset], (0, 32000), mode='constant')[:32000]

                    # Read Lips
                    x_lips = get_video_frames(lips[i], fmt= 'grey')
                    x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)


                else:
                    mix_aud = np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[:32000]
                    aud = np.pad(np.load(samples[i]), (0, 32000), mode='constant')[:32000]

                    # Read Lips
                    x_lips = get_video_frames(lips[i], fmt= 'grey')
                    x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)
                
                X_samples_mix.append(mix_aud)
                X_samples.append(aud)
                X_lips.append(x_lips)

            X_samples_mix = np.asarray(X_samples_mix)
            X_samples = np.asarray(X_samples)
            X_lips = np.asarray(X_lips)

            
            X_samples_targ = X_samples.reshape(X_samples.shape[0], 32000, 1).astype('float32')
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], 32000, 1).astype('float32')
            X_samples_targ = X_samples_targ
            X_samples_mix = X_samples_mix/1350.0
            #print(X_samples_targ.shape)
            np.random.seed(100)
            placeholder_1 = tf.constant(np.random.rand(X_lips.shape[0], 256))
            placeholder_2 = tf.constant(np.random.rand(X_lips.shape[0], 256))
            placeholder_3 = tf.constant(np.random.rand(X_lips.shape[0], 512))

            #X_attns = np.random.rand(batch_size, 200, 200)

            yield [X_lips, X_samples_mix,placeholder_1,placeholder_2, placeholder_3], X_samples_targ

            batch_start += batch_size
            batch_end += batch_size


def DataGenerator_val_unsync_attention_easy(folderlist, batch_size):
    
    L = len(folderlist)
    #unsync_files = np.loadtxt('/data/AV-speech-separation1/lrs2_1dot5k-unsync_audio_val.json', dtype='object')
    with open('/data/AV-speech-separation1/lrs2_1dot5k-unsync_audio_val.json') as json_file:
        unsync_dict = json.load(json_file)
    
    unsync_files = unsync_dict['folds']
    offsets = unsync_dict['offsets']

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:

            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips = []
            samples = []
            samples_mix = []

            for folder in folders_batch:
                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
                samples_mix_ = '/data/mixed_audio_files/' +folder.split('/')[-1]+'.wav'
                for i in range(len(lips_)):
                    lips.append(lips_[i])
                for i in range(len(samples_)):
                    samples.append(samples_[i])
                for i in range(len(lips_)):
                    samples_mix.append(samples_mix_)
          
            '''zipped = list(zip(lips, samples, samples_mix))
            random.shuffle(zipped)
            lips, samples, samples_mix = zip(*zipped)'''

            #X_samples = np.asarray([np.pad(np.load(fname), (0, 32000), mode='constant')[:32000] for fname in samples])
            
            X_samples_mix = []
            X_samples = []
            X_lips = []
            for i, fname in enumerate(samples_mix):
                fold = fname.split('/')[-1][:-4]
                if fold in unsync_files:
                    offset = abs(offsets[fold])
                    if offset > 12:
                        offset = 10
                    offset = offset-2
                    aud_offset = int(abs((offset/25)*16000))

                    # Read Mixed Audio with offset
                    '''mix_aud = np.pad(wavfile.read(fname)[1], (aud_offset, 32000), mode='constant')[:32000]
                    aud = np.pad(np.load(samples[i]), (0, 32000), mode='constant')[:32000]
                    aud = np.pad(aud[:-aud_offset], (0, 32000), mode='constant')[:32000]'''

                    mix_aud = np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[aud_offset:32000+aud_offset]
                    aud = np.pad(np.load(samples[i]), (0, 32000), mode='constant')[aud_offset:32000+aud_offset]
                    #aud = np.pad(aud[:-aud_offset], (0, 32000), mode='constant')[:32000]

                    # Read Lips
                    x_lips = get_video_frames(lips[i], fmt= 'grey')
                    x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)


                else:
                    mix_aud = np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[:32000]
                    aud = np.pad(np.load(samples[i]), (0, 32000), mode='constant')[:32000]

                    # Read Lips
                    x_lips = get_video_frames(lips[i], fmt= 'grey')
                    x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)
                
                X_samples_mix.append(mix_aud)
                X_samples.append(aud)
                X_lips.append(x_lips)

            X_samples_mix = np.asarray(X_samples_mix)
            X_samples = np.asarray(X_samples)
            X_lips = np.asarray(X_lips)

            X_samples_targ = X_samples.reshape(X_samples.shape[0], 32000, 1).astype('float32')
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], 32000, 1).astype('float32')
            X_samples_targ = X_samples_targ
            X_samples_mix = X_samples_mix/1350.0
            #print(X_samples_targ.shape)

            np.random.seed(100)
            placeholder_1 = tf.constant(np.random.rand(X_lips.shape[0], 256))
            placeholder_2 = tf.constant(np.random.rand(X_lips.shape[0], 256))
            placeholder_3 = tf.constant(np.random.rand(X_lips.shape[0], 512))

            #X_attns = np.random.rand(batch_size, 200, 200)

            yield [X_lips, X_samples_mix,placeholder_1,placeholder_2, placeholder_3], X_samples_targ

            batch_start += batch_size
            batch_end += batch_size


def Data_predict_attention(folderlist_dict):

    folderlist = list(folderlist_dict.keys())
    lips = []
    samples = []
    samples_mix = []

    for folder in folderlist:
        lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
        samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
        samples_mix_ = '/data/mixed_audio_files/' +folder.split('/')[-1]+'.wav'
        for i in range(len(lips_)):
            lips.append(lips_[i])
        for i in range(len(samples_)):
            samples.append(samples_[i])
        for i in range(len(lips_)):
            samples_mix.append(samples_mix_)
    
    
    X_samples_mix = []
    X_samples = []
    X_lips = []
    for i, fname in enumerate(samples_mix):
        fold = fname.split('/')[-1][:-4]
        offset = abs(folderlist_dict[lips[i][:-35]])
        #offset = abs(offsets[fold])
        aud_offset = int(abs((offset/25)*16000))

        # Read Mixed Audio with offset
        mix_aud = np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[aud_offset:32000+aud_offset]
        aud = np.pad(np.load(samples[i]), (0, 32000), mode='constant')[aud_offset:32000+aud_offset]

        # Read Lips
        x_lips = get_video_frames(lips[i], fmt= 'grey')
        x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)
        
        X_samples_mix.append(mix_aud)
        X_samples.append(aud)
        X_lips.append(x_lips)

    X_samples_mix = np.asarray(X_samples_mix)
    X_samples = np.asarray(X_samples)
    X_lips = np.asarray(X_lips).astype('float32')

    X_samples_targ = X_samples.reshape(X_samples.shape[0], 32000, 1).astype('float32')
    X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], 32000, 1).astype('float32')
    X_samples_targ = X_samples_targ
    X_samples_mix = (X_samples_mix/1350.0).astype('float32')
    #print(X_samples_targ.shape)

    np.random.seed(100)
    placeholder_1 = np.random.rand(X_lips.shape[0], 256).astype('float32')
    placeholder_2 = np.random.rand(X_lips.shape[0], 256).astype('float32')
    placeholder_3 = np.random.rand(X_lips.shape[0], 512).astype('float32')

    #X_attns = np.random.rand(batch_size, 200, 200)

    return [X_lips, X_samples_mix,placeholder_1,placeholder_2, placeholder_3]
