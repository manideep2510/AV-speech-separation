import os
from os.path import join
import glob
import random
import shutil
import numpy as np
import numpy
numpy.random.bit_generator = numpy.random._bit_generator
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
sys.path.append('/data/AV-speech-separation1/LipNet')
from LipNet.lipnet.lipreading.aligns import Align, Align_1
from LipNet.lipnet.lipreading.helpers import text_to_labels,text_to_labels_original,pad
from tensorflow.keras import backend as K

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

'''import imgaug as ia
import imgaug.augmenters as iaa

sometimes = lambda aug: iaa.Sometimes(1, aug)

seq1_1 = iaa.Sequential(
    [
        sometimes(iaa.Affine(rotate=(10), mode='reflect')),
        #iaa.Fliplr(1),
        #sometimes(iaa.Affine(translate_px={"x": (-10,10), "y": (-5, 5)}, mode='constant', cval=0))
    ]
)

seq1_2 = iaa.Sequential(
    [
        sometimes(iaa.Affine(rotate=(-10), mode='reflect')),
        #iaa.Fliplr(1),
        #sometimes(iaa.Affine(translate_px={"x": (-10,10), "y": (-5, 5)}, mode='constant', cval=0))
    ]
)

seq2 = iaa.Sequential(
    [
        #sometimes(iaa.Affine(rotate=(-10, 10))),
        iaa.Fliplr(1),
        #sometimes(iaa.Affine(translate_px={"x": (-10,10), "y": (-5, 5)}, mode='constant', cval=0))
    ]
)

seq3 = iaa.Sequential(
    [
        #sometimes(iaa.Affine(rotate=(-10, 10))),
        #iaa.Fliplr(1),
        sometimes(iaa.Affine(translate_px={"x": (10), "y": (-5)}, mode='constant', cval=0))
    ]
)

seq4 = iaa.Sequential(
    [
        #sometimes(iaa.Affine(rotate=(-10, 10))),
        #iaa.Fliplr(1),
        sometimes(iaa.Affine(translate_px={"x": (-10), "y": (5)}, mode='constant', cval=0))
    ]
)

seq5 = iaa.Sequential(
    [
        #sometimes(iaa.Affine(rotate=(-10, 10))),
        #iaa.Fliplr(1),
        sometimes(iaa.Affine(translate_px={"x": (10), "y": (5)}, mode='constant', cval=0))
    ]
)

seq6 = iaa.Sequential(
    [
        #sometimes(iaa.Affine(rotate=(-10, 10))),
        #iaa.Fliplr(1),
        sometimes(iaa.Affine(translate_px={"x": (-10), "y": (-5)}, mode='constant', cval=0))
    ]
)'''

def DataGenerator_train_softmask(folderlist, batch_size, time, augment):

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
            mask = []
            spect = []
            phase = []
            samples = []
            transcripts=[]
            for folder in folders_batch:



                #lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                #masks_ = sorted(glob.glob(folder + '/*_softmask.npy'), key=numericalSort)
                #samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
                #transcripts_ = sorted(glob.glob(folder + '/*.txt'), key=numericalSort)
                #spect_ = folder + '/mixed_spectrogram.npy'
                #phase_ = folder + '/phase_spectrogram.npy'

                lips_ = folder
                transcripts_ = folder[:-9]+'.txt'

                lips.append(lips_)
                #lips.append(lips_[1])

                # samples.append(samples_[0])
                # samples.append(samples_[1])
                #
                # mask.append(masks_[0])
                # mask.append(masks_[1])
                #
                # spect.append(spect_)
                # spect.append(spect_)
                #
                # phase.append(phase_)
                # phase.append(phase_)

                transcripts.append(transcripts_)
                #transcripts.append(transcripts_[1])

            zipped = list(zip(lips, transcripts))
            random.shuffle(zipped)
            lips, transcripts = zip(*zipped)

#             #X_mask = np.asarray([to_onehot(cv2.imread(fname, cv2.IMREAD_UNCHANGED)) for fname in mask])
#             X_mask = np.asarray([np.load(fname).reshape(257, 500, 1) for fname in mask])
#             #print(X_mask.shape)
# #            print('mask', X_mask.shape)
#
#             X_spect = [np.load(fname) for fname in spect]
#
#             X_phase = [np.load(fname) for fname in phase]
#
#             X_samples = np.asarray([np.pad(np.load(fname), (0, 128500), mode='constant')[:128500] for fname in samples])
#
#             X_spect_phase = []
#             for i in range(len(X_spect)):
#                 x_spect_phase = np.stack([X_spect[i], X_phase[i]], axis=-1)
#                 X_spect_phase.append(x_spect_phase)
#
#             X_spect_phase = np.asarray(X_spect_phase)
#
# #            print("X_spect_phase", X_spect_phase.shape)

            X_lips = []

            for i in range(len(lips)):

                #print(lips[i])
                x_lips = get_video_frames(lips[i], fmt = 'grey')
                
                if augment == True:
                    choices = [1,2,3,4]
                    choice = random.choice(choices)
                    if choice == 1:
                        x_lips = seq2.augment_images(x_lips)

                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = time)
                x_lips = x_lips/255.0
                X_lips.append(x_lips)

            align=[]
            Y_data = []
            label_length = []
            input_length = []
            source_str = []

            #X_lips = np.asarray(X_lips)

           #  for i in range(len(transcripts)):
           #      a=(Align(256, text_to_labels).from_file(transcripts[i]))
           #      if(a.label_length<=125):
           #              align.append(a)
           #              X_lip.append(X_lips[i])
           #  for i in range(len(X_lip)):
           #      Y_data.append(align[i].padded_label)
           #      label_length.append(align[i].label_length)
           #      input_length.append(125)
           #      #source_str.append(align[i].sentence)
           #  Y_data = np.array(Y_data)
           # # print(X_lips.shape)
           #  #X = seq.augment_images(X)
           #
           #  #yield [X_spect_phase, X_lips, X_samples], X_mask
           #  yield [np.array(X_lip),Y_data,np.array(input_length),np.array(label_length)],np.zeros([len(X_lip)])

            X_lips = np.asarray(X_lips)
            frame_length = X_lips.shape[1]

            for i in range(len(transcripts)):
                align.append(Align(128, text_to_labels).from_file(transcripts[i]))
            for i in range(X_lips.shape[0]):
               Y_data.append(align[i].padded_label)
               label_length.append(align[i].label_length)
               input_length.append(X_lips.shape[1])
               source_str.append(align[i].sentence)
            Y_data = np.array(Y_data)
            #  inputs = {'the_input': X_lips,
            #       'the_labels': Y_data,
            #       'input_length': input_length,
            #       'label_length': label_length,
            #       'source_str': source_str
            #       }
            # outputs = {'ctc': np.zeros([X_lips.shape[0]])}  # dummy data for dummy loss function
            #
            # yield (inputs,outputs)
            yield [X_lips,Y_data,np.array(input_length),np.array(label_length)],np.zeros([X_lips.shape[0]])
            batch_start += batch_size
            batch_end += batch_size


def DataGenerator_train(folderlist, batch_size):

    L = len(folderlist)
    epoch_number = 0

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:

            if batch_start == 0:
                epoch_number += 1

            '''if epoch_number<=5:
                rule = 1
                time = 1
                augment = False
            elif epoch_number > 5 and epoch_number <= 10:
                rule = 2
                time = 2
                augment = False
            elif epoch_number > 10 and epoch_number <= 15:
                rule = 2
                time = 2
                augment = True
            elif epoch_number > 15 and epoch_number <= 25:
                rule = 3
                time = 3
                augment = True
            else:
                rule = 4
                time = 4
                augment = True'''

            limit = min(batch_end, L)
            rule = 2
            time = 2
            augment = False

            folders_batch = folderlist[batch_start:limit]

            lips = []
            mask = []
            spect = []
            phase = []
            samples = []
            transcripts=[]
            for folder in folders_batch:

                lips_ = folder
                transcripts_ = folder[:-4]+'.txt'

                lips.append(lips_)
                transcripts.append(transcripts_)

            zipped = list(zip(lips, transcripts))
            random.shuffle(zipped)
            lips, transcripts = zip(*zipped)
            X_lips = []

            for i in range(len(lips)):

                #print(lips[i])
                x_lips = get_video_frames(lips[i], fmt = 'grey')
                
                '''if augment == True:
                    choices = [1,2,3,4]
                    choice = random.choice(choices)
                    if choice == 1:
                        x_lips = seq2.augment_images(x_lips)'''

                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = time)
                x_lips = x_lips/255.0
                X_lips.append(x_lips)

            align=[]
            Y_data = []
            label_length = []
            input_length = []
            source_str = []
            
            spliced_lips=[]
            X_lips = np.asarray(X_lips)
            #frame_length = X_lips.shape[1]

            if(rule==1): 
                absolute_max_string_len=32 
                length=1 
                index=0
            elif(rule==2):
                absolute_max_string_len=64
                length=2
                index=0
            elif(rule==3):
                absolute_max_string_len=128
                length=3
                index=0
            elif(rule==4):
                absolute_max_string_len=128
                length=4
                index=0
            else:
                absolute_max_string_len=128
                length=5
                index=0

            for i in range(len(transcripts)):
                #print('Train trans:', transcripts[i])
                a=Align_1(absolute_max_string_len, text_to_labels_original,length,index).from_file(transcripts[i])
                if(a.label_length!=0 and 2*a.label_length <= length*25):
                    align.append(a)
                    start,end=a.video_range()
                    spliced_lips.append(pad(X_lips[i][start:end+1],length*25))
                #else: 
                    #print(transcripts[i])

            for i in range(len(spliced_lips)):
               Y_data.append(align[i].padded_label)
               label_length.append(align[i].label_length)
               input_length.append(length*25)
               source_str.append(align[i].sentence)
            
            Y_data = np.array(Y_data)
            spliced_lips=np.asarray(spliced_lips)

            '''if(spliced_lips.shape[0]==0):
                     spliced_lips=np.zeros([1,time*25,112,112,1])
                     Y_data=np.array([41]*absolute_max_string_len)
                     label_length=[12]
                     input_length=[length*25]'''
            
            #print(spliced_lips.shape[0])
            #if(spliced_lips.shape[0]==0):continue
            yield [spliced_lips,Y_data,np.array(input_length),np.array(label_length)],np.zeros([spliced_lips.shape[0]])
            batch_start += batch_size
            batch_end += batch_size

def DataGenerator_val(folderlist, batch_size):
    
    L = len(folderlist)
    epoch_number = 0

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)

            if batch_start == 0:
                epoch_number += 1

            '''if epoch_number <= 5:
                rule = 1
                time = 1
                augment = False
            elif epoch_number > 5 and epoch_number <= 10:
                rule = 2
                time = 2
                augment = False
            elif epoch_number > 10 and epoch_number <= 15:
                rule = 2
                time = 2
                augment = True
            elif epoch_number > 15 and epoch_number <= 25:
                rule = 3
                time = 3
                augment = True
            else:
                rule = 4
                time = 4
                augment = True'''

            folders_batch = folderlist[batch_start:limit]
            rule = 2
            time = 2
            augment = False

            lips = []
            mask = []
            spect = []
            phase = []
            samples = []
            transcripts=[]
            for folder in folders_batch:

                lips_ = folder
                transcripts_ = folder[:-4]+'.txt'

                lips.append(lips_)
               
                transcripts.append(transcripts_)
                #transcripts.append(transcripts_[1])

            '''zipped = list(zip(lips_, transcripts_))
            random.shuffle(zipped)
            lips, transcripts = zip(*zipped)'''

            X_lips = []

            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i], fmt = 'grey')
#                x_lips = seq.augment_images(x_lips)
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = time)
                x_lips = x_lips/255.0
                X_lips.append(x_lips)

            align=[]
            Y_data = []
            label_length = []
            input_length = []
            source_str = []
            spliced_lips=[]

            X_lips = np.asarray(X_lips)
            #frame_length = X_lips.shape[1]
            if(rule==1): 
                absolute_max_string_len=32 
                length=1 
                index=0
            elif(rule==2):
                absolute_max_string_len=64
                length=2
                index=0
            elif(rule==3):
                absolute_max_string_len=128
                length=3
                index=0
            elif(rule==4):
                absolute_max_string_len=128
                length=4
                index=0
            else:
                absolute_max_string_len=128
                length=5
                index=0

            for i in range(len(transcripts)):
                #print('Val trans:', transcripts[i])
                a=Align_1(absolute_max_string_len, text_to_labels_original,length,index).from_file(transcripts[i])
                if(a.label_length!=0 and 2*a.label_length <= length*25):
                    align.append(a)
                    start,end=a.video_range()
                    spliced_lips.append(pad(X_lips[i][start:end+1],length*25))
                #else:
                    #print(transcripts[i])
                #align.append(a)
                #start,end=a.video_range()
                #spliced_lips.append(pad(X_lips[i][start:end+1],length*25))

            for i in range(len(spliced_lips)):
               Y_data.append(align[i].padded_label)
               label_length.append(align[i].label_length)
               input_length.append(length*25)
               source_str.append(align[i].sentence)
            
            Y_data = np.array(Y_data)
            spliced_lips=np.asarray(spliced_lips)

            '''if(spliced_lips.shape[0]==0):
                     print('hidfafd')
                     spliced_lips=np.zeros([1,time*25,112,112,1])
                     Y_data=np.array([41]*absolute_max_string_len)
                     print(Y_data.shape)
                     label_length=[12]
                     input_length=[length*25]'''
            #print(spliced_lips.shape[0])
            
            #for i in range(len(transcripts)):align.append(Align(frame_length, text_to_labels_original).from_file(transcripts[i]))
            #for i in range(X_lips.shape[0]):
            #   Y_data.append(align[i].padded_label)
            #   label_length.append(align[i].label_length)
            #   input_length.append(X_lips.shape[1])
            #   source_str.append(align[i].sentence)
            #Y_data = np.array(Y_data)
            #  inputs = {'the_input': X_lips,
            #       'the_labels': Y_data,
            #       'input_length': input_length,
            #       'label_length': label_length,
            #       'source_str': source_str
            #       }
            # outputs = {'ctc': np.zeros([X_lips.shape[0]])}  # dummy data for dummy loss function
            #
            # yield (inputs,outputs)
            #batch_start += batch_size
            #batch_end += batch_size
            #if(spliced_lips.shape[0]!=0):
            yield [spliced_lips,Y_data,np.array(input_length),np.array(label_length)],np.zeros([spliced_lips.shape[0]])
            batch_start += batch_size
            batch_end += batch_size


def DataGenerator_val_softmask(folderlist, batch_size, time):

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
            transcripts=[]
            for folder in folders_batch:

                #lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                #masks_ = sorted(glob.glob(folder + '/*_softmask.npy'), key=numericalSort)
                #samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
                #transcripts_ = sorted(glob.glob(folder + '/*.txt'), key=numericalSort)
                #spect_ = folder + '/mixed_spectrogram.npy'
                #phase_ = folder + '/phase_spectrogram.npy'

                lips_ = folder
                transcripts_ = folder[:-9]+'.txt'

                lips.append(lips_)
                #lips.append(lips_[1])

                # samples.append(samples_[0])
                # samples.append(samples_[1])
                #
                # mask.append(masks_[0])
                # mask.append(masks_[1])
                #
                # spect.append(spect_)
                # spect.append(spect_)
                #
                # phase.append(phase_)
                # phase.append(phase_)

                transcripts.append(transcripts_)
                #transcripts.append(transcripts_[1])

            '''zipped = list(zip(lips_, transcripts_))
            random.shuffle(zipped)
            lips, transcripts = zip(*zipped)'''

#             #X_mask = np.asarray([to_onehot(cv2.imread(fname, cv2.IMREAD_UNCHANGED)) for fname in mask])
#             X_mask = np.asarray([np.load(fname).reshape(257, 500, 1) for fname in mask])
#             #print(X_mask.shape)
# #            print('mask', X_mask.shape)
#
#             X_spect = [np.load(fname) for fname in spect]
#
#             X_phase = [np.load(fname) for fname in phase]
#
#             X_samples = np.asarray([np.pad(np.load(fname), (0, 128500), mode='constant')[:128500] for fname in samples])
#
#             X_spect_phase = []
#             for i in range(len(X_spect)):
#                 x_spect_phase = np.stack([X_spect[i], X_phase[i]], axis=-1)
#                 X_spect_phase.append(x_spect_phase)
#
#             X_spect_phase = np.asarray(X_spect_phase)
#
# #            print("X_spect_phase", X_spect_phase.shape)

            X_lips = []

            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i], fmt = 'grey')
#                x_lips = seq.augment_images(x_lips)
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = time)
                x_lips = x_lips/255.0
                X_lips.append(x_lips)

            align=[]
            Y_data = []
            label_length = []
            input_length = []
            source_str = []

            #X_lips = np.asarray(X_lips)

           #  for i in range(len(transcripts)):
           #      a=(Align(256, text_to_labels).from_file(transcripts[i]))
           #      if(a.label_length<=125):
           #              align.append(a)
           #              X_lip.append(X_lips[i])
           #  for i in range(len(X_lip)):
           #      Y_data.append(align[i].padded_label)
           #      label_length.append(align[i].label_length)
           #      input_length.append(125)
           #      #source_str.append(align[i].sentence)
           #  Y_data = np.array(Y_data)
           # # print(X_lips.shape)
           #  #X = seq.augment_images(X)
           #
           #  #yield [X_spect_phase, X_lips, X_samples], X_mask
           #  yield [np.array(X_lip),Y_data,np.array(input_length),np.array(label_length)],np.zeros([len(X_lip)])

            X_lips = np.asarray(X_lips)
            frame_length = X_lips.shape[1]

            for i in range(len(transcripts)):
                align.append(Align(128, text_to_labels_original).from_file(transcripts[i]))
            for i in range(X_lips.shape[0]):
               Y_data.append(align[i].padded_label)
               label_length.append(align[i].label_length)
               input_length.append(X_lips.shape[1])
               source_str.append(align[i].sentence)
            Y_data = np.array(Y_data)
            #  inputs = {'the_input': X_lips,
            #       'the_labels': Y_data,
            #       'input_length': input_length,
            #       'label_length': label_length,
            #       'source_str': source_str
            #       }
            # outputs = {'ctc': np.zeros([X_lips.shape[0]])}  # dummy data for dummy loss function
            #
            # yield (inputs,outputs)
            yield [X_lips,Y_data,np.array(input_length),np.array(label_length)],np.zeros([X_lips.shape[0]])
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
           # print(folders_batch)
            lips = []
            mask = []
            spect = []
            phase = []
            samples = []
            transcripts=[]
            for folder in folders_batch:

                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                #masks_ = sorted(glob.glob(folder + '/*_softmask.npy'), key=numericalSort)
                #samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
                transcripts_ = sorted(glob.glob(folder + '/*.txt'), key=numericalSort)
                #spect_ = folder + '/mixed_spectrogram.npy'
                #phase_ = folder + '/phase_spectrogram.npy'

                lips.append(lips_[0])
                lips.append(lips_[1])

                # samples.append(samples_[0])
                # samples.append(samples_[1])
                #
                # mask.append(masks_[0])
                # mask.append(masks_[1])
                #
                # spect.append(spect_)
                # spect.append(spect_)
                #
                # phase.append(phase_)
                # phase.append(phase_)

                transcripts.append(transcripts_[0])
                transcripts.append(transcripts_[1])

            zipped = list(zip(lips, transcripts))
            random.shuffle(zipped)
            lips, transcripts = zip(*zipped)

#             #X_mask = np.asarray([to_onehot(cv2.imread(fname, cv2.IMREAD_UNCHANGED)) for fname in mask])
#             X_mask = np.asarray([np.load(fname).reshape(257, 500, 1) for fname in mask])
#             #print(X_mask.shape)
# #            print('mask', X_mask.shape)
#
#             X_spect = [np.load(fname) for fname in spect]
#
#             X_phase = [np.load(fname) for fname in phase]
#
#             X_samples = np.asarray([np.pad(np.load(fname), (0, 128500), mode='constant')[:128500] for fname in samples])
#
#             X_spect_phase = []
#             for i in range(len(X_spect)):
#                 x_spect_phase = np.stack([X_spect[i], X_phase[i]], axis=-1)
#                 X_spect_phase.append(x_spect_phase)
#
#             X_spect_phase = np.asarray(X_spect_phase)
#
# #            print("X_spect_phase", X_spect_phase.shape)

            X_lips = []

            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i], fmt = 'grey')
                choices = [0,1,2,3]
                choose = random.choice(choices)
                if choose == 0:
                    choices = [1,2,3]
                    choose = random.choice(choices)
                    if choose == 1:
                        choices = [1,2]
                        choose = random.choice(choices)
                        if choose == 1:
                            x_lips = seq1_1.augment_images(x_lips)
                        elif choose == 2:
                            x_lips = seq1_2.augment_images(x_lips)
                    elif choose == 2:
                        x_lips = seq2.augment_images(x_lips)
                    elif choose == 3:
                        choices = [0,1,2,3]
                        choose = random.choice(choices)
                        if choose == 0:
                            x_lips = seq3.augment_images(x_lips)
                        elif choose == 1:
                            x_lips = seq4.augment_images(x_lips)
                        elif choose == 2:
                            x_lips = seq5.augment_images(x_lips)
                        elif choose == 3:
                            x_lips = seq6.augment_images(x_lips)
                else:
                    x_lips = x_lips
                #x_lips = seq.augment_images(x_lips)
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 5)
                X_lips.append(x_lips)

            align=[]
            Y_data = []
            label_length = []
            input_length = []
            source_str = []


	     #X_lips = np.asarray(X_lips)

            # for i in range(len(transcripts)):
            #     a=(Align(256, text_to_labels).from_file(transcripts[i]))
            #     if(a.label_length<=125):
            #             align.append(a)
            #             X_lip.append(X_lips[i])
            # for i in range(len(X_lip)):
            #     Y_data.append(align[i].padded_label)
            #     label_length.append(align[i].label_length)
            #     input_length.append(125)
            #     #source_str.append(align[i].sentence)
            # Y_data = np.array(Y_data)
           # print(X_lips.shape)
            #X = seq.augment_images(X)

            #yield [X_spect_phase, X_lips, X_samples], X_mask
            #yield [np.array(X_lip),Y_data,np.array(input_length),np.array(label_length)],np.zeros([len(X_lip)])

            X_lips = np.asarray(X_lips)

            for i in range(len(transcripts)):align.append(Align(128, text_to_labels).from_file(transcripts[i]))
            for i in range(X_lips.shape[0]):
               Y_data.append(align[i].padded_label)
               label_length.append(align[i].label_length)
               input_length.append(X_lips.shape[1])
               source_str.append(align[i].sentence)
            Y_data = np.array(Y_data)
           # print(X_lips.shape)
            #X = seq.augment_images(X)

            #yield [X_spect_phase, X_lips, X_samples], X_mask
            yield [X_lips,Y_data,np.array(input_length),np.array(label_length)],np.zeros([X_lips.shape[0]])
            #  inputs = {'the_input': X_lips,
            #       'the_labels': Y_data,
            #       'input_length': input_length,
            #       'label_length': label_length,
            #       'source_str': source_str
            #       }
            # outputs = {'ctc': np.zeros([X_lips.shape[0]])}  # dummy data for dummy loss function
            #
            # yield (inputs,outputs)

            batch_start += batch_size
            batch_end += batch_size

