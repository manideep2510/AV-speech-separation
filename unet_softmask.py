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
#from models.lipnet import LipNet
from models.unet import VideoModel
from data_generators import DataGenerator_train_unet, DataGenerator_sampling_unet

'''from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
set_session(tf.Session(config=config))
'''
from keras.utils import multi_gpu_model
from metrics import sdr_metric, Metrics_softmask
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('-epochs', action="store", dest="epochs", type=int)
parser.add_argument('-batch_size', action="store", dest="batch_size", type=int)
parser.add_argument('-lr', action="store", dest="lrate", type=float)

args = parser.parse_args()


# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Read training folders
folders_list = sorted(glob.glob('/data/lrs2/train/*'), key=numericalSort)
import random
folders_list_train = folders_list[:91500] +folders_list[93000:238089]
#folders_list_train = random.sample(folders_list_train, 200)
random.shuffle(folders_list_train)
folders_list_val = folders_list[91500:93000] + folders_list[238089:]
#random.seed(20)
#folders_list_val = random.sample(folders_list_val, 150)

print('Training data:', len(folders_list_train)*2)
print('Validation data:', len(folders_list_val)*2)

#lips_filelist = sorted(glob.glob('/data/lrs2/train/*/*_lips.mp4'), key=numericalSort)

#masks_filelist = sorted(glob.glob('/data/lrs2/train/*/*_mask.png'), key=numericalSort)

#spects_filelist = sorted(glob.glob('/data/lrs2/train/*/mixed_spectrogram.npy'), key=numericalSort)

model = VideoModel(256,96,(257,500,2)).FullModel()
from io import StringIO

tmp_smry = StringIO()
model.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
summary = tmp_smry.getvalue()
summary_split = summary.split('\n')
summary_params = summary_split[-6:]
summary_params = '\n'.join(summary_params)
print('\n'+summary_params)

# Compile the model
lrate = args.lrate

#model.load_weights('/data/models/softmask_unet_Lipnet+cocktail_1in_1out_90k-train_1to3ratio_valSDR_epochs20_lr1e-4_0.1decay10epochs/weights-10-188.9557.hdf5')

#model = multi_gpu_model(model, gpus=2)

#model.load_weights('/data/models/test_Lipnet+cocktail_1in_1out_20k-train_valSDR_epochs20_lr1e-4_0.322decay5epochs/weights-12-0.4127.hdf5')

model.compile(optimizer = Adam(lr=lrate), loss = l2_loss)

batch_size = args.batch_size
epochs = args.epochs

# callcack
#metrics_sdr = Metrics_softmask(model = model, val_folders = folders_list_val, batch_size = batch_size)
learningratescheduler = learningratescheduler()
earlystopping = earlystopping()
reducelronplateau = reducelronplateau()
#logger = Logger('/data/results')

# Path to save model checkpoints

path = 'pretrain_unet_236k-train_1to3ratio_epochs20_lr1e-4_0.32decay5epochs'

try:
    os.mkdir('/data/models/'+ path)
except OSError:
    pass

filepath='/data/models/' +  path+ '/weights-{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint_save_weights = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, mode='min')

# Fit Generator

folders_per_epoch = int(len(folders_list_train)/3)

history = model.fit_generator(DataGenerator_sampling_unet(folders_list_train, folders_per_epoch, batch_size),
                steps_per_epoch = np.ceil((folders_per_epoch)/float(batch_size)),
                epochs=epochs,
                validation_data=DataGenerator_train_unet(folders_list_val, batch_size), 
                validation_steps = np.ceil((len(folders_list_val))/float(batch_size)),
                callbacks=[earlystopping, learningratescheduler, checkpoint_save_weights], verbose = 1)

# Plots
plot_loss_and_acc(history, path)

