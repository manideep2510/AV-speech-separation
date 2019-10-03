import glob
import os
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

import math

import tensorflow as tf
from keras.layers import *
from keras import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, ReduceLROnPlateau, EarlyStopping, ReduceLROnPlateau
from callbacks import learningratescheduler, earlystopping, reducelronplateau
from plotting import plot_loss_and_acc
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cv2
from losses import l2_loss, sparse_categorical_crossentropy_loss, cross_entropy_loss, categorical_crossentropy
from models.lipnet import LipNet
from models.cocktail_lipnet import VideoModel
from data_generators import DataGenerator_train, DataGenerator_test

from audio_utils import retrieve_samples
from data_preparation.video_utils import get_video_frames

import scipy

'''from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
set_session(tf.Session(config=config))
'''
from keras.utils import multi_gpu_model
from metrics import sdr_metric, Metrics
from argparse import ArgumentParser

from mir_eval.separation import bss_eval_sources

#parser = ArgumentParser()

#parser.add_argument('-epochs', action="store", dest="fi")

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

# Read training folders
folders_list = sorted(glob.glob('/data/lrs2/train/*'), key=numericalSort)

folders_list_train = folders_list[:24]
import random
random.shuffle(folders_list_train)
folders_list_val = folders_list[12000:13000]
random.seed(20)
folders_list_val = random.sample(folders_list_val, 12)
print(folders_list_val[4])

# Read the mixed spectrogram

#mixed_spect = np.load('/data/lrs2/train/6277199500760946671_00100_6092785631109560111_00060_2/mixed_spectrogram.npy')

# Read mixed phase

#phase_spect = np.load('/data/lrs2/train/6277199500760946671_00100_6092785631109560111_00060_2/phase_spectrogram.npy')

# Mix phase and mag

#mixed_spect_phase = np.stack([mixed_spect, phase_spect], axis=-1)

# Read the lips file

#lips = get_video_frames('/data/lrs2/train/6277199500760946671_00100_6092785631109560111_00060_2/6092785631109560111_00060_lips.mp4')
#lips = crop_pad_frames(lips, fps = 25, seconds = 5)

model = VideoModel(256,96,(257,500,2),(125,50,100,3)).FullModel(lipnet_pretrained = True)
model = multi_gpu_model(model, gpus=2)
#print(model.summary())
#model = load_model('/data/models/Lipnet+cocktail_1in_1out_20k-train_epochs20_lr1e-4_0.322decay5epochs/weights-03-1.0565-0.9780.hdf5')
model.load_weights('/data/models/test_Lipnet+cocktail_1in_1out_20k-train_valSDR_epochs20_lr1e-4_0.322decay5epochs/weights-12-0.4127.hdf5')
print('Weights Loaded')
#mask = model.predict([mixed_spect_phase, lips])

#true_samples = np.load('/data/lrs2/train/6277199500760946671_00100_6092785631109560111_00060_2/6092785631109560111_00060_samples.npy')
#true_samples = np.pad(true_samples, (0, 128500), mode = 'constant')[:128500]
#mask = model.predict([mixed_spect_phase.reshape(1, 257,500,2), lips.reshape(1,125,50,100,3), true_samples.reshape(1, 128500)], batch_size = None, steps=1)
batch_size = 2
masks = np.asarray(model.predict_generator(DataGenerator_test(folders_list_val, batch_size), steps = np.ceil((len(folders_list_val))/float(batch_size))))
mask = masks[9,:,:,:2]
mixed_spect = masks[9,:,:,2]
mixed_phase = masks[9,:,:,3]
true_samples = masks[9,:,:,4]
true_samples = true_samples.reshape(-1,)
print(mask.shape)
mask = np.argmax(mask, axis=2)
pred_samples = retrieve_samples(spec_signal = mixed_spect,phase_spect = mixed_phase,mask = mask,sample_rate=16e3, n_fft=512, window_size=25, step_size=10)

# Save predicted samples

scipy.io.wavfile.write('/data/predict2.wav', 16000, pred_samples)

length_pred = len(pred_samples)

true_samples = np.pad(true_samples, (0,length_pred), mode = 'constant')[:length_pred]

sdr, sir, sar, _ = bss_eval_sources(true_samples, pred_samples, compute_permutation=False)

print('SDR:', sdr)
print('SIR', sir)
print('SAR', sar)
