from metrics import sdr_metric, Metrics_crm, Metrics_samples, Metrics_wandb
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
from tensorflow.keras.layers import *
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, ReduceLROnPlateau, EarlyStopping, ReduceLROnPlateau, CSVLogger
from callbacks import earlystopping, LoggingCallback, save_weights
#from tensorflow.keras.callbacks import CSVLogger
from plotting import plot_loss_and_acc
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cv2
from losses import l2_loss, mse, l1_loss, mag_phase_loss, snr_loss, snr_acc
#from models.lipnet import LipNet
from models.tdavss_cotrain import TasNet_cotrain
#from models.tasnet_lipnet import TasNet
from dataloaders import DataGenerator_train_samples_lips, DataGenerator_val_samples_lips
from LipNet.lipnet.lipreading.callback import Metrics_cotrain
#from LipNet.lipnet.lipreading.generator import DataGenerator_train_softmask, DataGenerator_sampling_softmask
import random
import sys
#from dataloaders_aug import DataGenerator_train_crm, DataGenerator_sampling_crm, DataGenerator_test_crm

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

from tensorflow.keras.utils import multi_gpu_model
from argparse import ArgumentParser
import wandb
from wandb.keras import WandbCallback
import pickle

parser = ArgumentParser()

parser.add_argument('-epochs', action="store", dest="epochs", type=int)
parser.add_argument('-batch_size', action="store", dest="batch_size", type=int)
parser.add_argument('-lr', action="store", dest="lrate", type=float)

args = parser.parse_args()
#os.environ['WANDB_CONFIG_DIR'] = '/data/.config/wandb'
#os.environ['WANDB_MODE'] = 'dryrun'
#wandb.init(name='tdavss_25kTrain_NoLSTM_BahdanauAttention_freezeLip_norm', notes='Remove LSTM Layers in Lip Embidder, Bahdanau Attention, 25K training folders, random seed 35, freeze LipNet, Normalize input with 1350, Batch size = 8, Predict time samples directly.Mish Activation Function, 2 Second clips. 0.35 lr decay if no Val_loss Dec for 3epochs. TasNet with Resnet without LSTM Lipnet. Loss is Si-SNR', project="av-speech-seperation", dir='/data/AV-speech-separation1/wandb')

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

'''random.seed(25)
folders_list_train = random.sample(folders_list_train, 50000)'''
'''random.seed(35)
folders_list_train = random.sample(folders_list_train, 25000)
np.savetxt('/data/AV-speech-separation/lrs2_25k_split.txt', folders_list_train, fmt='%s')'''

'''folders_list_train2 = np.loadtxt('/data/AV-speech-separation/data_filenames_3comb.txt', dtype='object').tolist()
folders_list_train2_=[]
for item in folders_list_train2:
    fold = '/data/lrs2/'+item
    folders_list_train2_.append(fold)
folders_list_train = folders_list_train1[:55000] + folders_list_train2_'''

'''random.seed(10)
random.shuffle(folders_list_train)
folders_list_val = folders_list[91500:93000] + folders_list[238089:]'''

folders_list_train= np.loadtxt('/data/AV-speech-separation1/lrs2_25k_split.txt', dtype='object').tolist()

folders_list_val = np.loadtxt('/data/AV-speech-separation1/lrs2_3k_val_split.txt', dtype='object').tolist()

random.seed(10)
random.shuffle(folders_list_train)

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

# Compile the model
lrate = args.lrate
batch_size = args.batch_size
epochs = args.epochs

#mirrored_strategy = tf.distribute.MirroredStrategy()
#strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
#with strategy.scope():

time = 5
tasnet = TasNet_cotrain(time_dimensions=100*time, frequency_bins=257, n_frames=25*time, absolute_max_string_len=128, attention=False, lipnet_pretrained=True,  train_lipnet=True)
model = tasnet.model
#model.load_weights('/data/models/tdavss_freezeLip_batchsize8_Normalize_ResNetLSTMLip_236kTrain_2secondsClips_epochs7to20_lr1e-4_0.35decayNoValDec2epochs_exp3/weights-03--13.8362.hdf5')
#model.compile(optimizer=Adam(lr=lrate), loss=snr_loss, metrics=[snr_acc])
#model.compile(loss={'speech': snr_loss, 'ctc': lambda outv_classes, y_pred: y_pred}, optimizer=Adam(lr=lrate))
model.compile(loss={'speech_out': snr_loss, 'ctc': lambda y_true,
                    y_pred: y_pred}, optimizer=Adam(lr=lrate), metrics={'speech_out':[snr_acc]},
                    loss_weights={'speech_out': 1, 'ctc': 0.1})


from io import StringIO
tmp_smry = StringIO()
model.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
summary = tmp_smry.getvalue()
summary_split = summary.split('\n')
summary_params = summary_split[-6:]
summary_params = '\n'.join(summary_params)
print('\n'+summary_params)



#model.load_weights('/data/models/tasnet_ResNetLSTMLip_Lips_crm_236kTrain_epochs20_lr1e-4_0.1decay9epochs_exp1/weights-17-249.6407.hdf5')

#model = multi_gpu_model(model, gpus=2)

# Path to save model checkpoints

#path = 'test_tasnet_lipnet_crm_236kTrain_epochs20_lr1e-4_0.46decay3epochs_exp1'
path = 'tdavss_CoTrain_epochs40_lr1e-4'
print('Model weights path:', path + '\n')
#path = 'tasnet_ResNetLSTMLip_Lips_crm_236kTrain_5secondsClips_RMSLoss_epochs20_lr6e-5_0.1decay10epochs_exp2'

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
    initial_lrate = 0.0001
    drop = 0.1
    epochs_drop = 10
    lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
    return lrate

def learningratescheduler():
    learningratescheduler = LearningRateScheduler(step_decay)
    return learningratescheduler

#metrics_crm = Metrics_samples(model = model, val_folders = folders_list_val, batch_size = batch_size, save_path = '/data/results/'+path+'/logs.txt')
metrics_wandb = Metrics_wandb()
metrics_wer = Metrics_cotrain(model=model, val_folders=folders_list_val,
                              batch_size=batch_size, save_path='/data/results/'+path+'/logs_lips.txt')
save_weights = save_weights(model, path)
learningratescheduler = learningratescheduler()
earlystopping = earlystopping()
reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.35, patience=3, min_lr = 0.00000001)

filepath = '/data/models/' + path + '/weights-{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint_save_weights = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, save_weights_only=True, mode='min')

# Fit Generator

#folders_per_epoch = int(len(folders_list_train)/3)

history = model.fit_generator(DataGenerator_train_samples_lips(folders_list_train, int(batch_size), time=time),
                steps_per_epoch = int(np.ceil(len(folders_list_train)/float(batch_size))),
                epochs=int(epochs),
                validation_data=DataGenerator_val_samples_lips(folders_list_val, int(batch_size), time=time), 
                validation_steps = int(np.ceil((len(folders_list_val))/float(batch_size))),
    callbacks=[reducelronplateau, checkpoint_save_weights, LoggingCallback(print_fcn=log_to_file)], verbose=1)

#, WandbCallback(save_model=False, data_type="image")

# Plots
plot_loss_and_acc(history, path)


