from metrics import sdr_metric, Metrics_crm, Metrics_samples, Metrics_wandb, Metrics_3speak
import glob
import os
#from skimage import io, transform
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
#from models.tdavss import TasNet
#from models.tdavss_attn1 import TasNet as TasNetSepCon
from models.tdavss_sepconv import TasNet as TasNetSepCon
#from models.tasnet_lipnet import TasNet
from dataloaders import DataGenerator_val_samples, DataGenerator_train_samples
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

'''gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6400)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)'''

from tensorflow.keras.utils import multi_gpu_model
from argparse import ArgumentParser
import wandb
from wandb.keras import WandbCallback

parser = ArgumentParser()

parser.add_argument('-epochs', action="store", dest="epochs", type=int)
parser.add_argument('-batch_size', action="store", dest="batch_size", type=int)
parser.add_argument('-lr', action="store", dest="lrate", type=float)

args = parser.parse_args()
#os.environ['WANDB_CONFIG_DIR'] = '/home/ubuntu/.config/wandb'
#os.environ['WANDB_MODE'] = 'dryrun'
wandb.init(name='tdavss_PerInputMeanSTDNorm_baseline_2speakers', notes='Batch size = 8, TDAVSS exact baseline, 256 dim SepNet, Per example mean and std input norm, lr = 5e-4',
                resume='3drke3ko', project="av-speech-seperation", dir='/home/ubuntu/wandb') #resume = '2fb8mx5o', 

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Read training folders
#folders_list = sorted(glob.glob('/home/ubuntu/lrs2/train/*'), key=numericalSort)
#folders_list = np.loadtxt('/home/ubuntu/AV-speech-separation/data_filenames.txt', dtype='object').tolist()

'''folders_list_train2 = np.loadtxt('/home/ubuntu/AV-speech-separation/data_filenames_3comb.txt', dtype='object').tolist()
folders_list_train2_=[]
for item in folders_list_train2:
    fold = '/data/lrs2/'+item
    folders_list_train2_.append(fold)
folders_list_train = folders_list_train1[:55000] + folders_list_train2_'''

'''random.seed(10)
random.shuffle(folders_list_train)
folders_list_val = folders_list[91500:93000] + folders_list[238089:]'''

'''folders_list_train= np.loadtxt('/data/AV-speech-separation1/lrs2_25k_split.txt', dtype='object').tolist()

folders_list_val = np.loadtxt('/data/AV-speech-separation1/lrs2_3k_val_split.txt', dtype='object').tolist()'''


folders_list_train = np.loadtxt(
    '/home/ubuntu/lrs2_train_comb2.txt', dtype='object').tolist()

folders_list_val = np.loadtxt(
    '/home/ubuntu/lrs2_val_comb2.txt', dtype='object').tolist()

#random.seed(123)
#folders_list_train = random.sample(folders_list_train_all, 50000)
#random.seed(1234)
#folders_list_val = random.sample(folders_list_val_all, 5000)
random.seed(12345)
random.shuffle(folders_list_train)


'''random.seed(30)
folders_list_val = random.sample(folders_list_val_, 120)
folders_list_train = random.sample(folders_list_train, 180)'''

print('Training data:', len(folders_list_train))
print('Validation data:', len(folders_list_val))
# Building the model
#tasnet = TasNet(video_ip_shape=(125,50,100,3), time_dimensions=500, frequency_bins=257, n_frames=125, lipnet_pretrained='pretrain', train_lipnet=False)

# Compile the model
lrate = args.lrate
batch_size = args.batch_size
epochs = args.epochs

#mirrored_strategy = tf.distribute.MirroredStrategy()
#strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
#with mirrored_strategy.scope():

tasnet = TasNetSepCon(time_dimensions=200, frequency_bins=257, n_frames=50,
                      attention=False, lstm=False, lipnet_pretrained=True,  train_lipnet=True)
model = tasnet.model
model.load_weights(
    '/home/ubuntu/models/tdavss_PerInputMeanSTDNorm_baseline_2speakers_epochs80_from7_lr5e-4/weights-44--11.5932.tf')
model.compile(optimizer=Adam(lr=lrate), loss=snr_loss, metrics=[snr_acc])
#parallel_model=tf.keras.utils.multi_gpu_model(model, gpus=2)
#parallel_model.compile(optimizer=Adam(lr=lrate), loss=snr_loss, metrics=[snr_acc])

from io import StringIO
tmp_smry = StringIO()
model.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
summary = tmp_smry.getvalue()
summary_split = summary.split('\n')
summary_params = summary_split[-6:]
summary_params = '\n'.join(summary_params)
print('\n'+summary_params)


#path = 'tdavss_baseline_256dimSepNet_2speakers_epochs40_lr1e-3_exp4'
path='tdavss_PerInputMeanSTDNorm_baseline_2speakers_epochs80_from51_lr5e-4'
print('Model weights path:', path + '\n')

try:
    os.mkdir('/home/ubuntu/models/'+ path)
except OSError:
    pass

try:
    os.mkdir('/home/ubuntu/results/'+ path)
except OSError:
    pass

def log_to_file(msg, file='/home/ubuntu/results/'+path+'/logs.txt'):
    
    with open(file, "a") as myfile:
        
        myfile.write(msg)

# callcack


metrics_unsync = Metrics_3speak(model=model, val_folders=folders_list_val,
                                batch_size=batch_size, save_path='/home/ubuntu/results/'+path+'/logs.txt')
metrics_wandb = Metrics_wandb()
save_weights = save_weights(model, path)
earlystopping = earlystopping()
reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr = 0.00000001)

filepath = '/home/ubuntu/models/' + path + '/weights-{epoch:02d}-{val_loss:.4f}.tf'
checkpoint_save_weights = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, save_weights_only=True, mode='min')

# Fit Generator

#folders_per_epoch = int(len(folders_list_train)/3)
#gn = tf.data.Dataset.from_generator(
#    DataGenerator_train_samples(folders_list_train, int(batch_size), norm=1), ([tf.float32, tf.float32], tf.float32))
    

try:
    history = model.fit(DataGenerator_train_samples(folders_list_train, int(batch_size), norm=1),
                steps_per_epoch = int(np.ceil(len(folders_list_train)/float(batch_size))),
                epochs=int(epochs),
                validation_data=DataGenerator_val_samples(folders_list_val, int(batch_size), norm=1),
                validation_steps = int(np.ceil((len(folders_list_val))/float(batch_size))),
        callbacks=[reducelronplateau, checkpoint_save_weights, metrics_wandb, LoggingCallback(print_fcn=log_to_file), metrics_unsync], verbose=1)

except KeyboardInterrupt:
    for layer in model.layers:
        layer.trainable = True

    model.save_weights('/home/ubuntu/models/' + path + '/final.tf')
    print('Final model saved')

except Exception as e:
    print(e)

#, WandbCallback(save_model=False, data_type="image")


for layer in model.layers:
    layer.trainable = True

model.save_weights('/home/ubuntu/models/' + path + '/final.tf')
print('Final model saved')

# Plots
plot_loss_and_acc(history, path)


