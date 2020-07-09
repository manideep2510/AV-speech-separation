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
from keras.layers import *
from keras import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, ReduceLROnPlateau, EarlyStopping, ReduceLROnPlateau
from callbacks import learningratescheduler, earlystopping, reducelronplateau,LoggingCallback
from plotting import plot_loss_and_acc
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cv2
from losses import l2_loss, sparse_categorical_crossentropy_loss, cross_entropy_loss, categorical_crossentropy, mse
from LipNet.lipnet.lipreading.callback import Metrics
from LipNet.lipnet.lipreading.generator import DataGenerator_train, DataGenerator_val
#from LipNet.lipnet.model2 import LipNet
from models.resnet_lstm_lipread_ctc import lipreading
#from models.resnet_lstm_lipread_initial import lipreading
import numpy as np
import datetime
import pickle
import random


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

from keras.utils import multi_gpu_model
#from metrics import sdr_metric, Metrics_softmask
from argparse import ArgumentParser

print('Imports Done')

parser = ArgumentParser()

parser.add_argument('-epochs', action="store", dest="epochs", type=int)
parser.add_argument('-batch_size', action="store", dest="batch_size", type=int)
parser.add_argument('-lr', action="store", dest="lrate", type=float)

args = parser.parse_args()

print('Args Done')

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
print('numaricalsort Done')
# Read training folders
#folders_list = sorted(glob.glob('/data/lrs2/train/*'), key=numericalSort)
print('Folders_list Done')

'''with open("/data/AV-speech-separation/folder_filter_1.txt", "rb") as fp:  
       folders_list = pickle.load(fp) '''

time = 2

folders_list = np.loadtxt('/data/av-speech-separation/all_lipvideos.txt', dtype='object').tolist()

'''folders_list_1s = np.loadtxt('/data/AV-speech-separation1/lipreading_trainset_above_1sec.txt', dtype='object').tolist()
preprocess_collapse_repeated=Truerandom.seed(123)
random.shuffle(folders_list_1s)
folders_list_2s = np.loadtxt('/data/AV-speech-separation1/lipreading_trainset_above_2sec.txt', dtype='object').tolist()
random.seed(1234)
random.shuffle(folders_list_2s)
folders_list_3s = np.loadtxt('/data/AV-speech-separation1/lipreading_trainset_above_3sec.txt', dtype='object').tolist()
random.seed(1235)
random.shuffle(folders_list_3s)

if time == 1:
    folders_list_train = folders_list[:-1000] #+ folders_list_1s
if time == 2:
    folders_list_train = folders_list[:-1000] + folders_list_2s
if time == 3:
    folders_list_train = folders_list[:-1000] + folders_list_3s
if time == 5:
    folders_list_train = folders_list[:-1000]'''

random.seed(123)
random.shuffle(folders_list)
print(len(folders_list))
#folders_list=folders_list[:88000]
folders_list_train = folders_list[0:88000]
folders_list_val=folders_list[88000:96000]

#random.seed(30)
#folders_list_val = random.sample(folders_list_val, 120)
#folders_list_train = random.sample(folders_list_train, 180)

#folders_list_train = folders_list[:91500] +folders_list[93000:238089]
#folders_list_train=folders_list[0:192000]
#print(folders_list_train[34])

#folders_list_train=folders_list[:256]
#folders_list_val=folders_list[256:320]
#folders_list_val = folders_list[192000:204000]
#folders_list_train = folders_list[:9150] + folders_list[9300:23751]
#folders_list_val = folders_list[9150:9300] + folders_list[23751:]
import random
random.seed(10)
random.shuffle(folders_list_train)

print('Training data:', len(folders_list_train))
print('Validation data:', len(folders_list_val))

if(time == 1):
    absolute_max_string_len=32
elif(time == 2):
    absolute_max_string_len=64
elif(time == 3):
    absolute_max_string_len=128
elif(time == 4):
    absolute_max_string_len=128
else:
    absolute_max_string_len=128

#lip=LipNet(pretrained=True,weights_path='/data/models/lip_net_236k-train_1to3ratio_valSDR_epochs10-20_lr1e-4_0.1decay10epochs/weights-04-125.3015.hdf5')
lip = lipreading(mode='temporalConv', inputDim=256, hiddenDim=512, nClasses=43,
                 frameLen=time*25, AbsoluteMaxStringLen=absolute_max_string_len, every_frame=True)
model = lip.model
#model.load_weights(
#    '/data/models/combResnetLSTM_CTCloss_seperableConv_3sTrain_NoAug_epochs40_5lr1e-4/weights-04-110.0346.hdf5')
#model.load_weights('/data/models/combResnetLSTM_CTCloss_seperableConv_236ktrain_1to3ratio_valWER_epochs20_lr1e-4_0.1decay9epochs/weights-10-116.9441.hdf5')
#print('Weights after 7 epochs loaded')
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
adam = Adam(lr=lrate)
#model = multi_gpu_model(lip.model, gpus=2)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

model.load_weights('/data/models/FullConvs_NoGRUs_CombResNetLSTMs/weights-02-28.0938.hdf5')

#model.compile(optimizer = Adam(lr=lrate), loss = l2_loss)

batch_size = args.batch_size
epochs = args.epochs


# spell = Spell(path=PREDICT_DICTIONARY)
# decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
#                   postprocessors=[labels_to_text, spell.sentence])
#
# metrics_error_rates  = Statistics(lip,DataGenerator_train_softmask(folders_list_val, batch_size) , decoder, 256, output_dir='./results'))

# Path to save model checkpoints

path = 'FullConvs_NoGRUs_CombResNetLSTMs'

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
metrics_wer = Metrics(model= lip, val_folders = folders_list_val, 
                            batch_size = batch_size, save_path = '/data/results/'+path+'/logs.txt', 
                            rule = time, time=time)

learningratescheduler = learningratescheduler()
earlystopping = earlystopping()
reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr = 0.00000001)

filepath='/data/models/' +  path+ '/weights-{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint_save_weights = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, save_weights_only=True, mode='min')

# Fit Generator

#folders_per_epoch = int(len(folders_list_train)/3)

history = model.fit_generator(DataGenerator_train(folders_list_train, batch_size),
                steps_per_epoch = np.ceil(len(folders_list_train)/float(batch_size)),
                epochs=epochs,
                validation_data=DataGenerator_val(folders_list_val, batch_size),
                validation_steps = np.ceil((len(folders_list_val))/float(batch_size)),
                callbacks=[LoggingCallback(print_fcn=log_to_file), checkpoint_save_weights, reducelronplateau, metrics_wer], verbose = 1)

# Plots
plot_loss_and_acc(history, path)

# Logs
#command = "kubectl logs pods/train | egrep -E -i -e 'val|epoch' > /data/results/" + path + "/logs.txt"
#os.system(command)
