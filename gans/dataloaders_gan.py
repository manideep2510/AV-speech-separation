import numpy as np
import os
import glob
from scipy import signal
from scipy.io import wavfile
import cv2
import random
import time

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

def get_video_frames(path, fmt='rgb'):

    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if fmt == 'rgb':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif fmt == 'grey':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = frame.reshape(frame.shape[0], frame.shape[1], 1)
            frames.append(frame)

        # Break the loop
        else: 
            break

    cap.release()
    return np.asarray(frames)

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def DataGenerator_train(folderlist, batch_size, norm=1):
    
    L = len(folderlist)
    #print('L len:', L)
    steps_per_epoch = int(np.ceil(L/float(batch_size)))
    
    steps = 0
    #this line is just to make the generator infinite, keras needs that
    while steps < steps_per_epoch:
        #print('steps_per_epoch:', steps_per_epoch)

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            #print('batch_start:', batch_start)
            #print('steps:', steps)

            steps += 1

            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips = []
            samples = []
            samples_mix = []

            for folder in folders_batch:
                lips_ = folder
                samples_ = folder[:-9] + '_samples.npy'
                samples_mix_ = '/home/ubuntu/lrs2/mixed_audios/' + folder.split('/')[-2] + '.wav'

                lips.append(lips_)
                samples.append(samples_)
                samples_mix.append(samples_mix_)
          
            zipped = list(zip(lips, samples, samples_mix))
            random.shuffle(zipped)
            lips, samples, samples_mix = zip(*zipped)

            X_samples = np.asarray([np.pad(np.load(fname), (0, 32000), mode='constant')[:32000] for fname in samples])
            X_samples_mix = np.asarray([np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[:32000] for fname in samples_mix])
            
            X_lips = []
            
            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i], fmt= 'grey')
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)
                X_lips.append(x_lips)

            X_lips = np.asarray(X_lips).astype('float32')

            X_samples_targ = X_samples.reshape(X_samples.shape[0], 32000, 1).astype('float32')
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], 32000, 1).astype('float32')
            X_samples_targ = X_samples_targ
            if norm != 1:
                X_samples_mix = X_samples_mix/norm

            yield [X_lips, X_samples_mix], X_samples_targ

            batch_start += batch_size
            batch_end += batch_size


def DataGenerator_val(folderlist, batch_size, norm=1):
    
    L = len(folderlist)
    #print('L len:', L)
    steps_per_epoch = int(np.ceil(L/float(batch_size)))
    
    steps = 0
    #this line is just to make the generator infinite, keras needs that
    while steps < steps_per_epoch:
        #print('steps_per_epoch:', steps_per_epoch)

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            #print('batch_start:', batch_start)
            #print('steps:', steps)

            steps += 1

            limit = min(batch_end, L)

            folders_batch = folderlist[batch_start:limit]

            lips = []
            samples = []
            samples_mix = []

            for folder in folders_batch:
                lips_ = folder
                samples_ = folder[:-9] + '_samples.npy'
                samples_mix_ = '/home/ubuntu/lrs2/mixed_audios/' + folder.split('/')[-2] + '.wav'

                lips.append(lips_)
                samples.append(samples_)
                samples_mix.append(samples_mix_)

            X_samples = np.asarray([np.pad(np.load(fname), (0, 32000), mode='constant')[:32000] for fname in samples])
            X_samples_mix = np.asarray([np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[:32000] for fname in samples_mix])
            
            X_lips = []
            
            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i], fmt= 'grey')
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)
                X_lips.append(x_lips)

            X_lips = np.asarray(X_lips).astype('float32')

            X_samples_targ = X_samples.reshape(X_samples.shape[0], 32000, 1).astype('float32')
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], 32000, 1).astype('float32')
            X_samples_targ = X_samples_targ
            if norm != 1:
                X_samples_mix = X_samples_mix/norm

            yield [X_lips, X_samples_mix], X_samples_targ

            batch_start += batch_size
            batch_end += batch_size