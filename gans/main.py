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

# Input arguments
parser = ArgumentParser()

parser.add_argument('-epochs', action="store", dest="epochs", type=int)
parser.add_argument('-batch_size', action="store", dest="batch_size", type=int)
parser.add_argument('-lr', action="store", dest="lrate", type=float)

args = parser.parse_args()

lrate = args.lrate
batch_size = args.batch_size
epochs = args.epochs

os.environ['WANDB_CONFIG_DIR'] = '/data/.config/wandb'
os.environ['WANDB_MODE'] = 'dryrun'
wandb.init(name='tdavss_gan_basic', notes='Basic gan',
                project="av-speech-seperation", dir='/data/wandb')


# Read training folders
folders_list_train_all = np.loadtxt(
    '/data/AV-speech-separation1/lrs2_comb2_train_snr_filter.txt', dtype='object').tolist()

folders_list_val_all = np.loadtxt(
    '/data/AV-speech-separation1/lrs2_comb2_val_snr_filter.txt', dtype='object').tolist()

random.seed(123)
folders_list_train = random.sample(folders_list_train_all, 50000)
random.seed(1234)
folders_list_val = random.sample(folders_list_val_all, 5000)
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
                      attention=False, lstm=False, lipnet_pretrained=True,  train_lipnet=True)

print('----------Building Discriminator----------')
discriminator = Discriminator(time_dimensions=200, frequency_bins=257, n_frames=50,
                      attention=False, lstm=False, lipnet_pretrained=True,  train_lipnet=True)

generator_optimizer = tf.keras.optimizers.Adam(lrate/1.414, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(lrate*1.414, beta_1=0.5)

from io import StringIO

print('\n------------Generator Parameters------------', end='')
tmp_smry = StringIO()
generator.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
summary = tmp_smry.getvalue()
summary_split = summary.split('\n')
summary_params = summary_split[-6:]
summary_params = '\n'.join(summary_params)
print('\n'+summary_params)

print('----------Discriminator Parameters----------', end='')
tmp_smry = StringIO()
discriminator.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
summary = tmp_smry.getvalue()
summary_split = summary.split('\n')
summary_params = summary_split[-6:]
summary_params = '\n'.join(summary_params)
print('\n'+summary_params)

path = 'tdavss_LSGAN_Lambda10_InstanceNoise_lr5e-4_exp1'
print('Model weights path:', path + '\n')

try:
    os.mkdir('/data/models/' + path)
except OSError:
    pass

try:
    os.mkdir('/data/results/' + path)
except OSError:
    pass

# Training
fit(folders_list_train, folders_list_val, batch_size=batch_size,
    generator=generator, discriminator=discriminator, generator_optimizer = generator_optimizer, 
    discriminator_optimizer = discriminator_optimizer, epochs=epochs, save_path=path)
