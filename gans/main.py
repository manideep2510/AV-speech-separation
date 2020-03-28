# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import glob
import random
import wandb
from argparse import ArgumentParser
from nets import Generator, Discriminator
from train_utils import fit
import soundfile
import math
# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

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
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7400)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)'''

# Input arguments
parser = ArgumentParser()

parser.add_argument('-epochs', action="store", dest="epochs", type=int)
parser.add_argument('-batch_size', action="store", dest="batch_size", type=int)
parser.add_argument('-lr', action="store", dest="lrate", type=float)

args = parser.parse_args()

lrate = args.lrate
batch_size = args.batch_size
epochs = args.epochs

#os.environ['WANDB_CONFIG_DIR'] = '/data/.config/wandb'
#os.environ['WANDB_MODE'] = 'dryrun'
wandb.init(name='tdavss_LSGAN_PhaseShuffle_InstanceNoise_Lambda100', notes='BS=6, PhaseShuffle=2, LAMBDA=100 to 1, LSGAN with Instance Noise for 20 epochs, Lr(D) = 2*Lr(G), 0.5 lr after 20 epochs',
                resume='2z9u44ut', project="av-speech-seperation", dir='/home/manideepkolla/wandb')


# Read training folders
folders_list_train = np.loadtxt(
    '/home/manideepkolla/lrs2_train_comb2.txt', dtype='object').tolist()

folders_list_val = np.loadtxt(
    '/home/manideepkolla/lrs2_val_comb2.txt', dtype='object').tolist()

'''random.seed(123)
folders_list_train = random.sample(folders_list_train_all, 50000)
random.seed(1234)
folders_list_val = random.sample(folders_list_val_all, 5000)'''
random.seed(12345)
random.shuffle(folders_list_train)

'''random.seed(30)
folders_list_val = random.sample(folders_list_val, 40)
folders_list_train = random.sample(folders_list_train, 80)'''

print('Training data:', len(folders_list_train))
print('Validation data:', len(folders_list_val))

# Build and compile models

print('------------Building Generator------------')
generator = Generator(time_dimensions=200, frequency_bins=257, n_frames=50,
                      lstm=False, lipnet_pretrained=True,  train_lipnet=True)

generator.load_weights('/home/manideepkolla/models/tdavss_LSGAN_PhaseShuffle_InstanceNoise_Lambda100lr5e-4_exp1/generator-2-5.6655.tf')
print('Generator weights loaded')

print('----------Building Discriminator----------')
discriminator = Discriminator(time_dimensions=200, frequency_bins=257, n_frames=50,
                      phaseshuffle_rad=2, lstm=False, lipnet_pretrained=True,  train_lipnet=True)

discriminator.load_weights('/home/manideepkolla/models/tdavss_LSGAN_PhaseShuffle_InstanceNoise_Lambda100lr5e-4_exp1/discriminator-2-5.6655.tf')
print('Discriminator weights loaded')

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lrate/1.414, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lrate*1.414, beta_1=0.5)

from io import StringIO

print('\n------------Generator Parameters------------')
tmp_smry = StringIO()
generator.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
summary = tmp_smry.getvalue()
summary_split = summary.split('\n')
summary_params = summary_split[-6:]
summary_params = '\n'.join(summary_params)
print('\n'+summary_params)

print('----------Discriminator Parameters----------')
tmp_smry = StringIO()
discriminator.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
summary = tmp_smry.getvalue()
summary_split = summary.split('\n')
summary_params = summary_split[-6:]
summary_params = '\n'.join(summary_params)
print('\n'+summary_params)

path = 'tdavss_LSGAN_PhaseShuffle_InstanceNoise_Lambda100_epoch4t040_lr5e-4_exp1'
print('Model weights path:', path + '\n')

try:
    os.mkdir('/home/manideepkolla/models/' + path)
except OSError:
    pass

try:
    os.mkdir('/home/manideepkolla/results/' + path)
except OSError:
    pass

# Training
fit(folders_list_train, folders_list_val, batch_size=batch_size,
    generator=generator, discriminator=discriminator, generator_optimizer = generator_optimizer, 
    discriminator_optimizer = discriminator_optimizer, epochs=epochs, save_path=path)



# Learning Rate schedular
'''def step_decay_gen(epoch):
    initial_lrate = 0.0005/1.414
    drop = 0.5
    epochs_drop = 1
    lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
    return lrate

#learningratescheduler_gen = tf.keras.optimizers.schedules.LearningRateSchedule(step_decay_gen)

def step_decay_disc(epoch):
    initial_lrate = 0.0005*1.414
    drop = 0.5
    epochs_drop = 1
    lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
    return lrate

#learningratescheduler_disc = tf.keras.optimizers.schedules.LearningRateSchedule(step_decay_disc)

boundaries_gen = [1]
values_gen = [lrate/1.414, lrate*0.5/1.414]
learning_rate_fn_gen = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries_gen, values_gen)

boundaries_disc = [1]
values_disc = [lrate*1.414, lrate*0.5*1.414]
learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries_disc, values_disc)'''
