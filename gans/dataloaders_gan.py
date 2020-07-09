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

def crop_pad_frames1(frames, fps, seconds):

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

    mid_ht=frames.shape[1]//2
    mid_wd=frames.shape[2]//2

    frames=frames[:,mid_ht-35:mid_ht+35, mid_wd-35:mid_wd+35,:]
    #frames=tf.image.resize(frames,[112,112])
    frames=np.asarray(list(map( lambda x: cv2.resize(x, (112, 112),interpolation = cv2.INTER_CUBIC), frames)))
    frames = frames.reshape(frames.shape[0], frames.shape[1], frames.shape[2], 1)

    '''frames1 = []
    for frame in frames:
        frame = cv2.resize(frame, (112, 112),interpolation = cv2.INTER_CUBIC)
        frames1.append(frame)
    frames = np.asarray(frames1)'''
    #print(frames.shape)

    return frames


def DataGenerator_train(folderlist, train_vids, batch_size, norm, epoch):
    
    L = len(folderlist)
    #print('L len:', L)
    steps_per_epoch = int(np.ceil(L/float(batch_size)))

    random.seed(125*(epoch+1))
    zipped = list(zip(folderlist, train_vids))
    random.shuffle(zipped)
    folderlist, train_vids = zip(*zipped)
    
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
            vid_batch = train_vids[batch_start:limit]

            lips_disc = []
            lips = []
            samples = []
            samples_mix = []

            for i, folder in enumerate(folders_batch):
                #lips_disc_ = vid_batch[i]
                lips_disc_ = folder
                lips_ = folder[:-9] + '_embedding.npy'
                samples_ = folder[:-9] + '_samples.npy'
                samples_mix_ = '/data/mixed_audio_files/' + folder.split('/')[-2] + '.wav'

                lips_disc.append(lips_disc_)
                lips.append(lips_)
                samples.append(samples_)
                samples_mix.append(samples_mix_)
          
            zipped = list(zip(lips, samples, samples_mix, lips_disc))
            random.shuffle(zipped)
            lips, samples, samples_mix, lips_disc = zip(*zipped)

            X_samples = np.asarray([np.pad(np.load(fname), (0, 32000), mode='constant')[:32000] for fname in samples])
            X_samples_mix = np.asarray([np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[:32000] for fname in samples_mix])
            
            X_lips = []
            X_lips_disc = []
            
            for i in range(len(lips)):

                x_lips_disc = get_video_frames(lips_disc[i], fmt= 'grey')
                #x_lips = seq.augment_images(x_lips)
                x_lips_disc = crop_pad_frames(frames = x_lips_disc, fps = 25, seconds = 2)

                x_lips = np.load(lips[i])
                x_lips = np.pad(x_lips, ((0, 50), (0,0)), mode='constant')[:50]

                X_lips.append(x_lips)
                X_lips_disc.append(x_lips_disc)

            X_lips = np.asarray(X_lips).astype('float32')
            X_lips_disc = np.asarray(X_lips_disc).astype('float32')
            X_lips_disc = X_lips_disc/255.0

            X_samples_targ = X_samples.reshape(X_samples.shape[0], 32000, 1).astype('float32')
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], 32000, 1).astype('float32')
            X_samples_targ = X_samples_targ
            if norm != 1:
                X_samples_mix = X_samples_mix/norm
                X_samples_targ = X_samples_targ/norm

            yield [X_lips, X_samples_mix, X_lips_disc], X_samples_targ

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

            lips_disc = []
            lips = []
            samples = []
            samples_mix = []

            for folder in folders_batch:
                #lips_disc_ = '/data/lrs2/mvlrs_v1/main/' + folder.split('/')[-1][:-15] + '/' + folder.split('/')[-1][-14:-9]+'.mp4'
                lips_disc_ = folder
                lips_ = folder[:-9] + '_embedding.npy'
                samples_ = folder[:-9] + '_samples.npy'
                samples_mix_ = '/data/mixed_audio_files/' + folder.split('/')[-2] + '.wav'

                lips_disc.append(lips_disc_)
                lips.append(lips_)
                samples.append(samples_)
                samples_mix.append(samples_mix_)

            X_samples = np.asarray([np.pad(np.load(fname), (0, 32000), mode='constant')[:32000] for fname in samples])
            X_samples_mix = np.asarray([np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[:32000] for fname in samples_mix])
            
            X_lips = []
            X_lips_disc = []
            
            for i in range(len(lips)):

                x_lips_disc = get_video_frames(lips_disc[i], fmt= 'grey')
                #x_lips = seq.augment_images(x_lips)
                x_lips_disc = crop_pad_frames(frames = x_lips_disc, fps = 25, seconds = 2)

                x_lips = np.load(lips[i])
                x_lips = np.pad(x_lips, ((0, 50), (0,0)), mode='constant')[:50]

                X_lips.append(x_lips)
                X_lips_disc.append(x_lips_disc)

            X_lips = np.asarray(X_lips).astype('float32')
            X_lips_disc = np.asarray(X_lips_disc).astype('float32')
            X_lips_disc = X_lips_disc/255.0

            X_samples_targ = X_samples.reshape(X_samples.shape[0], 32000, 1).astype('float32')
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], 32000, 1).astype('float32')
            X_samples_targ = X_samples_targ
            if norm != 1:
                X_samples_mix = X_samples_mix/norm
                X_samples_targ = X_samples_targ/norm

            yield [X_lips, X_samples_mix, X_lips_disc], X_samples_targ

            batch_start += batch_size
            batch_end += batch_size


def DataGenerator_train_old(folderlist, batch_size, norm, epoch):
    
    L = len(folderlist)
    #print('L len:', L)
    steps_per_epoch = int(np.ceil(L/float(batch_size)))

    random.seed(125*(epoch+1))
    random.shuffle(folderlist)
    
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
                lips_ = folder[:-9] + '_embedding.npy'
                samples_ = folder[:-9] + '_samples.npy'
                samples_mix_ = '/data/mixed_audio_files/' + folder.split('/')[-2] + '.wav'

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

                '''x_lips = get_video_frames(lips[i], fmt= 'grey')
                #x_lips = seq.augment_images(x_lips)
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)'''

                x_lips = np.load(lips[i])
                x_lips = np.pad(x_lips, ((0, 50), (0,0)), mode='constant')[:50]

                X_lips.append(x_lips)

            X_lips = np.asarray(X_lips).astype('float32')

            X_samples_targ = X_samples.reshape(X_samples.shape[0], 32000, 1).astype('float32')
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], 32000, 1).astype('float32')
            X_samples_targ = X_samples_targ
            if norm != 1:
                X_samples_mix = X_samples_mix/norm
                X_samples_targ = X_samples_targ/norm

            yield [X_lips, X_samples_mix], X_samples_targ

            batch_start += batch_size
            batch_end += batch_size


def DataGenerator_val_old(folderlist, batch_size, norm=1):
    
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
                lips_ = folder[:-9] + '_embedding.npy'
                samples_ = folder[:-9] + '_samples.npy'
                samples_mix_ = '/data/mixed_audio_files/' + folder.split('/')[-2] + '.wav'

                lips.append(lips_)
                samples.append(samples_)
                samples_mix.append(samples_mix_)

            X_samples = np.asarray([np.pad(np.load(fname), (0, 32000), mode='constant')[:32000] for fname in samples])
            X_samples_mix = np.asarray([np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[:32000] for fname in samples_mix])
            
            X_lips = []
            
            for i in range(len(lips)):

                '''x_lips = get_video_frames(lips[i], fmt= 'grey')
                #x_lips = seq.augment_images(x_lips)
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)'''

                x_lips = np.load(lips[i])
                x_lips = np.pad(x_lips, ((0, 50), (0,0)), mode='constant')[:50]

                X_lips.append(x_lips)

            X_lips = np.asarray(X_lips).astype('float32')

            X_samples_targ = X_samples.reshape(X_samples.shape[0], 32000, 1).astype('float32')
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], 32000, 1).astype('float32')
            X_samples_targ = X_samples_targ
            if norm != 1:
                X_samples_mix = X_samples_mix/norm
                X_samples_targ = X_samples_targ/norm

            yield [X_lips, X_samples_mix], X_samples_targ

            batch_start += batch_size
            batch_end += batch_size
