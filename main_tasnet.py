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
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, ReduceLROnPlateau, EarlyStopping, ReduceLROnPlateau, CSVLogger
from callbacks import earlystopping, reducelronplateau, LoggingCallback
#from tensorflow.keras.callbacks import CSVLogger
from plotting import plot_loss_and_acc
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cv2
from losses import l2_loss, mse, l1_loss, mag_phase_loss
from models.lipnet import LipNet
from models.tasnet_resnetLip_2branch import TasNet
#from models.tasnet_lipnet import TasNet
from dataloaders import DataGenerator_train_crm, DataGenerator_sampling_crm
#from dataloaders_aug import DataGenerator_train_crm, DataGenerator_sampling_crm, DataGenerator_test_crm

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

from keras.utils import multi_gpu_model
from metrics import sdr_metric, Metrics_crm
from argparse import ArgumentParser
import wandb
from wandb.keras import WandbCallback

parser = ArgumentParser()

parser.add_argument('-epochs', action="store", dest="epochs", type=int)
parser.add_argument('-batch_size', action="store", dest="batch_size", type=int)
parser.add_argument('-lr', action="store", dest="lrate", type=float)

args = parser.parse_args()
os.environ['WANDB_CONFIG_DIR'] = '/data/.config/wandb'
wandb.init(name='tasnet-2branches-5SecVids_RootOfL2Loss', notes='Training with 5 Second clips. TasNet with 2 branches at the end, one for each real and imaginary mask prediction. Loss is Root of TF L2 Loss', project="av-speech-seperation")

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Read training folders
#folders_list = sorted(glob.glob('/data/lrs2/train/*'), key=numericalSort)
folders_list = np.loadtxt('/data/AV-speech-separation/data_filenames.txt', dtype='object').tolist()
folders_list_train = folders_list[:91500] +folders_list[93000:238089]
'''folders_list_train2 = np.loadtxt('/data/AV-speech-separation/data_filenames_3comb.txt', dtype='object').tolist()
folders_list_train2_=[]
for item in folders_list_train2:
    fold = '/data/lrs2/'+item
    folders_list_train2_.append(fold)
folders_list_train = folders_list_train1[:55000] + folders_list_train2_'''
import random
random.seed(10)
random.shuffle(folders_list_train)
folders_list_val = folders_list[91500:93000] + folders_list[238089:]
#random.seed(30)
#folders_list_val = random.sample(folders_list_val, 120)
#folders_list_train = random.sample(folders_list_train, 180)
#folders_list_train = folders_list[:180]
#folders_list_val = folders_list[180:300]

#print('Training data:', len(folders_list_train1[:55000])*2 + len(folders_list_train2)*3)
print('Training data:', len(folders_list_train)*2)
print('Validation data:', len(folders_list_val)*2)

# Building the model
#tasnet = TasNet(video_ip_shape=(125,50,100,3), time_dimensions=500, frequency_bins=257, n_frames=125, lipnet_pretrained='pretrain', train_lipnet=False)
tasnet = TasNet(video_ip_shape=(125,50,100,3), time_dimensions=500, frequency_bins=257, n_frames=125, lipnet_pretrained='pretrain', train_lipnet=None)
model = tasnet.model

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

#model.load_weights('/data/models/tasnet_ResNetLSTMLip_Lips_crm_236kTrain_epochs20_lr1e-4_0.1decay9epochs_exp1/weights-17-249.6407.hdf5')

#model = multi_gpu_model(model, gpus=2)

model.compile(optimizer = Adam(lr=lrate), loss = l2_loss)

batch_size = args.batch_size
epochs = args.epochs

# Path to save model checkpoints

#path = 'test_tasnet_lipnet_crm_236kTrain_epochs20_lr1e-4_0.46decay3epochs_exp1'
#path = 'test_tasnetAttn_ResNetLSTMLip_Lips_crm_236kTrain_2secondsClips_RMSLoss_epochs20_lr1e-4_0.1decay10epochs_exp1'
path = 'tasnet_ResNetLSTMLip_Lips_crm_236kTrain_5secondsClips_RMSLoss_epochs20_lr6e-5_0.1decay10epochs_exp2'

try:
    os.mkdir('/data/models/'+ path)
except OSError:
    pass

try:
    os.mkdir('/data/results/'+ path)
except OSError:
    pass

def log_to_file(msg, file='/data/results/'+path+'/logs.txt'):
    
    with open(file, "a") as myfile:
        
        myfile.write(msg)

# callcack

def step_decay(epoch):
    initial_lrate = 0.00006
    drop = 0.1
    epochs_drop = 10
    lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
    return lrate

def learningratescheduler():
    learningratescheduler = LearningRateScheduler(step_decay)
    return learningratescheduler

metrics_crm = Metrics_crm(model = model, val_folders = folders_list_val, batch_size = batch_size, save_path = '/data/results/'+path+'/logs.txt')
learningratescheduler = learningratescheduler()
earlystopping = earlystopping()
reducelronplateau = reducelronplateau()

filepath='/data/models/' +  path+ '/weights-{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint_save_weights = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, save_weights_only=True, mode='min')

# Fit Generator

folders_per_epoch = int(len(folders_list_train)/3)

history = model.fit_generator(DataGenerator_sampling_crm(folders_list_train, folders_per_epoch, batch_size),
                steps_per_epoch = np.ceil(folders_per_epoch/float(batch_size)),
                epochs=epochs,
                validation_data=DataGenerator_train_crm(folders_list_val, batch_size), 
                validation_steps = np.ceil((len(folders_list_val))/float(batch_size)),
                callbacks=[earlystopping, learningratescheduler, checkpoint_save_weights, LoggingCallback(print_fcn=log_to_file), metrics_crm, WandbCallback(save_model=False, data_type="image")], verbose = 1)

#, WandbCallback(save_model=False, data_type="image")

# Plots
plot_loss_and_acc(history, path)


