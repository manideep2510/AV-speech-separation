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
from callbacks import Metrics, learningratescheduler, earlystopping, reducelronplateau
from plotting import plot_loss_and_acc
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from losses import l2_loss
from models.lipnet import LipNet
from models.cocktail_lipnet import VideoModel
from data_generators import DataGenerator

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
set_session(tf.Session(config=config))

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('-epochs', action="store", dest="epochs", type=int)
parser.add_argument('-batch_size', action="store", dest="batch_size", type=int)
parser.add_argument('-lr', action="store", dest="lrate", type=int)

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

model = VideoModel(256,96,(298,257,2),(500,50,100,3)).FullModel(lipnet_pretrained = True)

# Compile the model
lrate = args.lrate
model.compile(optimizer = Adam(lr=lrate), loss = l2_loss, metrics=['accuracy'])

# callcack
metrics = Metrics()
learningratescheduler = learningratescheduler()
earlystopping = earlystopping()
reducelronplateau = reducelronplateau()

# Path to save model checkpoints

path = ''

try:
    os.mkdir('/data/models/'+ path)
except OSError:
    pass

filepath='/home/manideep/models/' +  path+ '/weights-best.hdf5'
checkpoint_save_weights = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='max')

# Fit Generator

batch_size = args.batch_size
epochs = args.epochs

history = model.fit_generator(DataGenerator(lips_filelist, masks_filelist, spects_filelist, batch_size),
                steps_per_epoch = np.ceil((len(lips_filelist)/float(batch_size)),
                epochs=epochs,
                validation_data=DataGenerator(lips_filelist_val, masks_filelist_val, spects_filelist_val, batch_size), 
                validation_steps = np.ceil((len(lips_filelist_val)/float(batch_size)),
                callbacks=[earlystopping, learningratescheduler, checkpoint_save_weights], verbose = 1)

