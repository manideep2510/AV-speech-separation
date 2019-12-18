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
from losses import l2_loss, sparse_categorical_crossentropy_loss, cross_entropy_loss, categorical_crossentropy, mse
from models.lipnet import LipNet
#from models.tasnet_lipnet import TasNet
from models.tasnet_resnetLip import TasNet
#from data_generators import DataGenerator_train_softmask, DataGenerator_sampling_softmask, DataGenerator_test_softmask
#from dataloaders import DataGenerator_train_crm, DataGenerator_sampling_crm, DataGenerator_test_crm

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

#from keras.utils import multi_gpu_model
#from metrics import sdr_metric, Metrics_softmask
from argparse import ArgumentParser
import shutil
import re

#from mir_eval.separation import bss_eval_sources
#from metrics import metric_eval

from data_preparation.audio_utils import compress_crm, inverse_crm, return_samples_complex, compare_lengths, compute_spectrograms, audios_sum, ibm, irm
from data_preparation.video_utils import get_video_frames

import scipy

print('Imports Done')

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

mixed_audio = '/data/cropped_cnn/1women_1man/cnn_debate_2_man_women.wav'
lips = '/data/cropped_cnn/1women_1man/man_high.mp4'
time= 30
print('time:', time)

# Compute the spectrogram of mised audio
s, n, c = compute_spectrograms(mixed_audio)
mixed_spectogram =s[:,:time*100]  # Useful frames
mixed_spectogram = mixed_spectogram.reshape(1, 257, time*100)
print('mixed_spectogram:', mixed_spectogram.shape)

phase=np.angle(c)
phase=phase[:,:time*100]
phase = phase.reshape(1, 257, time*100)
print('phase:', phase.shape)

spect_phase = np.stack((mixed_spectogram, phase), axis=3)
print('spect_phase:', spect_phase.shape)

# Fake X_samples
samples = np.zeros((1, 257*time*100))
print('samples:', samples.shape)

# Read Video frames
x_lips = get_video_frames(lips, fmt='grey')
x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = time)
num_frames = x_lips.shape[0]
x_lips = x_lips.reshape(1, num_frames, 50, 100, 1)
print('x_lips:', x_lips.shape)

# Building the model
tasnet = TasNet(video_ip_shape=(125,50,100,3), time_dimensions=time*100, frequency_bins=257, n_frames=num_frames, lipnet_pretrained='pretrain', train_lipnet=None)
model = tasnet.model
model.load_weights('/data/models/tasnet_ResNetLSTMLip_Lips_crm_236kTrain_epochs20_lr1e-4_0.1decay9epochs_exp1/weights-20-237.6519.hdf5')
print('Weights Loaded')

from io import StringIO

tmp_smry = StringIO()
model.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
summary = tmp_smry.getvalue()
summary_split = summary.split('\n')
summary_params = summary_split[-6:]
summary_params = '\n'.join(summary_params)
print('\n'+summary_params)

batch_size = 1

print('Predicting on the demo example')

val_predict = model.predict([spect_phase, x_lips, samples], batch_size=batch_size, verbose=1)

mixed_spect = val_predict[:,:,:,2]
mixed_phase = val_predict[:,:,:,3]
crms = val_predict[:,:,:,:2]

crm = crms[0]
real = crm[:,:,0]
imaginary = crm[:,:,1]
inverse_mask = inverse_crm(real_part=real,imaginary_part=imaginary,K=1,C=2)
#print('crm', crm.shape)
mixed_spect_ = mixed_spect[0]
#print('mixed_spect_' ,mixed_spect_.shape)
mixed_phase_ = mixed_phase[0]
#print('mixed_phase_', mixed_phase_.shape)
samples_ = return_samples_complex(mixed_mag = mixed_spect_, mixed_phase = mixed_phase_, mask = inverse_mask,sample_rate=16e3, n_fft=512, window_size=25, step_size=10)
samples_ = samples_[256:]

scipy.io.wavfile.write('/data/demo_preds/1women_1man_man_pred.wav', 16000, samples_)
