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
from callbacks import Logger, learningratescheduler, earlystopping, reducelronplateau
from plotting import plot_loss_and_acc
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cv2
from losses import l2_loss, sparse_categorical_crossentropy_loss, cross_entropy_loss, categorical_crossentropy, mse
from models.lipnet import LipNet
from models.tasnet_lipnet import TasNet
from data_generators import DataGenerator_train_softmask, DataGenerator_sampling_softmask, DataGenerator_test_softmask
from dataloaders import DataGenerator_train_crm, DataGenerator_sampling_crm, DataGenerator_test_crm

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

from keras.utils import multi_gpu_model
from metrics import sdr_metric, Metrics_softmask
from argparse import ArgumentParser
import shutil
import re

from mir_eval.separation import bss_eval_sources
from metrics import metric_eval

from data_preparation.audio_utils import retrieve_samples, compress_crm, inverse_crm, return_samples_complex, audios_sum
from data_preparation.video_utils import get_video_frames

import scipy

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
folders_list = np.loadtxt('/data/AV-speech-separation/data_filenames.txt', dtype='object').tolist()

#folders_list_train = folders_list[:24]
import random
#random.shuffle(folders_list_train)
val_folders_pred = folders_list[91500:93000] + folders_list[238089:]
random.seed(200)
val_folders_pred = random.sample(val_folders_pred, 200)
#print(folders_list_val[4])

'''model = VideoModel(256,96,(257,500,2),(125,50,100,3)).FullModel(lipnet_pretrained = 'pretrain', unet_pretrained = 'pretrain')
'''

# Building the model
tasnet = TasNet(video_ip_shape=(125,50,100,3), time_dimensions=500, frequency_bins=257, n_frames=125, lipnet_pretrained='pretrain')
model = tasnet.model
model.load_weights('/data/models/tasnet_crm_236kTrain_epochs14to20_lr1e-5_0.1decay4epochs_exp4/weights-04-267.8097.hdf5')
print('Weights Loaded')

from io import StringIO

tmp_smry = StringIO()
model.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
summary = tmp_smry.getvalue()
summary_split = summary.split('\n')
summary_params = summary_split[-6:]
summary_params = '\n'.join(summary_params)
print('\n'+summary_params)

sdr_list = []

batch_size = 8

val_predict = np.asarray(model.predict_generator(DataGenerator_test_softmask(val_folders_pred, batch_size), steps = np.ceil((len(val_folders_pred))/float(batch_size))))

mixed_spect = val_predict[:,:,:,2]
mixed_phase = val_predict[:,:,:,3]
val_targ = val_predict[:,:,:,4]
batch = val_targ.shape[0]
val_targ = val_targ.reshape(batch, -1)
#       val_targ = val_targ[:, :80000]

crms = val_predict[:,:,:,:2]

samples_pred = []
for i in range(crms.shape[0]):
    crm = crms[i]
    real = crm[:,:,0]
    imaginary = crm[:,:,1]
    inverse_mask = inverse_crm(real_part=real,imaginary_part=imaginary,K=1,C=2)
    #print('crm', crm.shape)
    mixed_spect_ = mixed_spect[i]
    #print('mixed_spect_' ,mixed_spect_.shape)
    mixed_phase_ = mixed_phase[i]
    #print('mixed_phase_', mixed_phase_.shape)
    samples = return_samples_complex(mixed_mag = mixed_spect_, mixed_phase = mixed_phase_, mask = inverse_mask,sample_rate=16e3, n_fft=512, window_size=25, step_size=10)

    #print('samples', samples.shape) 
    samples_pred.append(samples[256:32000])

val_targ1 = []
for i in range(batch):
    length_pred = len(samples_pred[i])
    #print('length_pred', length_pred)
    val_targ_ = val_targ[i, :length_pred]
    #val_targ_ = val_targ_.reshape(1, -1)
    #print('val_targ_', val_targ_.shape)
    val_targ1.append(val_targ_)

val_targ = val_targ1

samples_pred = np.asarray(samples_pred)

val_targ = np.asarray(val_targ)

val_sdr, _, val_snr, _ = metric_eval(target_samples = val_targ, predicted_samples = samples_pred)

print('SDR:', val_sdr)
print('SNR:', val_snr)

'''samples = []

for i, item in enumerate(val_folders_pred):

    items = sorted(glob.glob(item+ '/*_lips.mp4'), key=numericalSort)
    
    samples = []
    for j, item in enumerate(items):
        try:
            os.mkdir('/data/pred_tasnet/'+item[-88:-35])
        except OSError:
            pass

        scipy.io.wavfile.write('/data/pred_tasnet/'+item[-88:-9]+'_pred.wav', 16000, samples_pred[2*i+j])

        shutil.copy2('/data/lrs2/train/'+item[-88:-9]+'_samples.npy','/data/pred_tasnet/'+item[-88:-35])

        samples_ = np.load('/data/lrs2/train/'+item[-88:-9]+'_samples.npy')
        scipy.io.wavfile.write('/data/pred_tasnet/'+item[-88:-9]+'_original.wav', 16000, samples_)
        #'/data/pred_sample/'+item[-88:-35]
        samples.append(samples_)
    sam1 = np.zeros((80000,))
    sam1[:len(samples[0][:80000])] = samples[0][:80000]
    sam2 = np.zeros((80000,))
    sam2[:len(samples[1][:80000])] = samples[1][:80000]
    add_samples = sam1+sam2
    scipy.io.wavfile.write('/data/pred_tasnet/'+item[-88:-35]+'/mixed.wav', 16000, add_samples)


val_folders_pred = sorted(glob.glob('/data/pred_tasnet/*'), key=numericalSort)

for fold in val_folders_pred:
    wavs = sorted(glob.glob(fold + '/*_original.wav'), key=numericalSort)
    file_sum_name = '/data/pred_tasnet/'+fold[-53:]
    su = audios_sum(wavs,file_sum_name, volume_reduction=0)

npys = sorted(glob.glob('pred_tasnet/*/*.npy'),key=numericalSort)
for i in folds:
    os.remove(i)'''