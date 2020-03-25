from metrics import Metrics_wandb, Metrics_unsync, Metrics_3speak
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
from models.tdavss import TasNet
from models.tdavss_sepconv import TasNet as TasNetSepCon
from dataloaders import DataGenerator_train_samples, DataGenerator_val_samples, DataGenerator_val_unsync_attention, DataGenerator_train_unsync_attention
import random
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

parser = ArgumentParser()

parser.add_argument('-epochs', action="store", dest="epochs", type=int)
parser.add_argument('-batch_size', action="store", dest="batch_size", type=int)
parser.add_argument('-lr', action="store", dest="lrate", type=float)

args = parser.parse_args()
os.environ['WANDB_CONFIG_DIR'] = '/data/.config/wandb'
os.environ['WANDB_MODE'] = 'dryrun'
wandb.init(name='tdavss_3speakers_SepConv_1600Frames', notes='3 speakers baseline training,Batch = 5, 20K training folders. TasNet with Resnet without LSTM Lipnet.', 
           project="av-speech-seperation", dir='/data/wandb')

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Read training folders
#folders_list = sorted(glob.glob('/data/lrs2/train/*'), key=numericalSort)
#folders_list = np.loadtxt('/data/AV-speech-separation/data_filenames.txt', dtype='object').tolist()

'''folders_list_train2 = np.loadtxt('/data/AV-speech-separation/data_filenames_3comb.txt', dtype='object').tolist()
folders_list_train2_=[]
for item in folders_list_train2:
    fold = '/data/lrs2/'+item
    folders_list_train2_.append(fold)
folders_list = folders_list_train2_

folders_list_val = folders_list[-2000:]
random.seed(1234)
folders_list_train = random.sample(folders_list[:-3000], 16666)'''

'''folders_list_train = np.loadtxt(
    '/data/AV-speech-separation1/lrs2_3comb_20k_split_train.txt', dtype='object').tolist()
    
folders_list_train = np.loadtxt(
    '/data/AV-speech-separation1/lrs2_3comb_train_all_45k.txt', dtype='object').tolist()

folders_list_val = np.loadtxt(
    '/data/AV-speech-separation1/lrs2_3comb_2k_split_val.txt', dtype='object').tolist()
#folders_list_train = folders_list_train[:16666]'''

folders_list_train = np.loadtxt(
    '/home/manideepkolla/lrs2_train_comb3.txt', dtype='object').tolist()

folders_list_val = np.loadtxt(
    '/home/manideepkolla/lrs2_val_comb3.txt', dtype='object').tolist()

'''random.seed(123)
folders_list_train = random.sample(folders_list_train_all, 50000)
random.seed(1234)
folders_list_val = random.sample(folders_list_val_all, 5000)'''
random.seed(12345)
random.shuffle(folders_list_train)

#random.seed(30)
#folders_list_val = random.sample(folders_list_val_, 120)
#folders_list_train = random.sample(folders_list_train, 180)
#folders_list_train = folders_list[:180]
#folders_list_val = folders_list[180:300]

print('Training data:', len(folders_list_train))
print('Validation data:', len(folders_list_val))

# Compile the model
lrate = args.lrate
batch_size = args.batch_size
epochs = args.epochs

tasnet = TasNetSepCon(time_dimensions=200, frequency_bins=257, n_frames=50, 
                    attention=False, lstm = False, lipnet_pretrained=True, train_lipnet=True)
model = tasnet.model
#model.load_weights('/data/models/tdavss_freezeLip_batchsize8_Normalize_ResNetLSTMLip_236kTrain_2secondsClips_epochs7to20_lr1e-4_0.35decayNoValDec2epochs_exp3/weights-03--13.8362.hdf5')
model.compile(optimizer=Adam(lr=lrate), loss=snr_loss, metrics=[snr_acc])


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
path = 'tdavss_3speakers_SepConv_400Frames_epochs40_5lr1e-4_exp1'
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

# callcacks


metrics_unsync = Metrics_3speak(model=model, val_folders=folders_list_val,
                                batch_size=batch_size, save_path='/data/results/'+path+'/logs.txt')
metrics_wandb = Metrics_wandb()
save_weights = save_weights(model, path)
earlystopping = earlystopping()
reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr = 0.00000001)

filepath = '/data/models/' + path + '/weights-{epoch:02d}-{val_loss:.4f}.tf'
checkpoint_save_weights = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, save_weights_only=True, mode='min')

# Fit Generator

#folders_per_epoch = int(len(folders_list_train)/3)

try:
    history = model.fit_generator(DataGenerator_train_samples(folders_list_train, int(batch_size), norm=1),
                steps_per_epoch = int(np.ceil(len(folders_list_train)/float(batch_size))),
                epochs=int(epochs),
                validation_data=DataGenerator_val_samples(folders_list_val, int(batch_size), norm=1), 
                validation_steps = int(np.ceil((len(folders_list_val))/float(batch_size))),
        callbacks=[reducelronplateau, checkpoint_save_weights, metrics_wandb, LoggingCallback(print_fcn=log_to_file), metrics_unsync], verbose=1)

except KeyboardInterrupt:

    model.save_weights('/data/models/' + path + '/final_freez.tf')
    print('Final model saved')

    for layer in model.layers:
        layer.trainable = True

    model.save_weights('/data/models/' + path + '/final_unfreez.tf')
    print('Final model saved')

except Exception as e:
    print(e)

#, WandbCallback(save_model=False, data_type="image")

model.save_weights('/data/models/' + path + '/final_freez.tf')
print('Final model saved')

for layer in model.layers:
    layer.trainable = True

model.save_weights('/data/models/' + path + '/final_unfreez.tf')
print('Final unfreezed model saved')

# Plots
plot_loss_and_acc(history, path)


