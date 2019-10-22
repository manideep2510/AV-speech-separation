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
from callbacks import Logger, learningratescheduler, earlystopping, reducelronplateau,LoggingCallback
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cv2
from losses import l2_loss, sparse_categorical_crossentropy_loss, cross_entropy_loss, categorical_crossentropy, mse
from models.lipnet import LipNet
from models.cocktail_lipnet_unet_softmask import VideoModel
from data_generators import DataGenerator_train_softmask, DataGenerator_sampling_softmask

#from keras.optimizers import Adam
#from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
#from lipnet.lipreading.generators import BasicGenerator
#from lipnet.lipreading.callbacks import Statistics, Visualize
#from lipnet.lipreading.curriculums import Curriculum
#from lipnet.core.decoders import Decoder
#from lipnet.lipreading.helpers import labels_to_text
#from lipnet.utils.spell import Spell
from LipNet.lipnet.lipreading.callback import Metrics_softmask, Decoder
from LipNet.lipnet.lipreading.generator import DataGenerator_train_softmask, DataGenerator_sampling_softmask
from LipNet.lipnet.lipreading.helpers import text_to_labels
from LipNet.lipnet.lipreading.aligns import Align
#from LipNet.lipnet.model2 import LipNet
from models.resnet_lstm_lipread_ctc import lipreading
import numpy as np
import datetime
import pickle
from data_preparation.video_utils import get_video_frames, crop_pad_frames

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

from keras.utils import multi_gpu_model
#from metrics import sdr_metric, Metrics_softmask
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('-video', action="store", dest="video_file")
#parser.add_argument('-batch_size', action="store", dest="batch_size", type=int)
#parser.add_argument('-lr', action="store", dest="lrate", type=float)

args = parser.parse_args()

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

with open("/data/AV-speech-separation/folder_filter_1.txt", "rb") as fp:  
       folders_list = pickle.load(fp) 

#folders_list_train=folders_list[0:192000]
#folders_list_val=folders_list[192000:204000]
#import random
#random.seed(10)
#random.shuffle(folders_list_train)
#folders_list_val = folders_list[91500:93000] + folders_list[238089:]
#folders_list_val=folders_list[512:768]
#random.seed(20)
#folders_list_train = random.sample(folders_list_train, 180)
#folders_list_val = random.sample(folders_list_val, 100)

#print('Training data:', len(folders_list_train)*2)
#print('Validation data:', len(folders_list_val)*2)

video_file = args.video_file
transcript_file = video_file[:-9]+'.txt'
lips = get_video_frames(video_file, fmt='grey')
lips = crop_pad_frames(frames = lips, fps = 25, seconds = 5)
lips = lips.reshape(1, 125,50,100,1)

# Read text

trans=(Align(128, text_to_labels).from_file(transcripts_path))
y_data=(trans.padded_label)
y_data = y_data.reshape(1, 128)
label_length=(trans.label_length)
input_length=125

lip = lipreading(mode='backendGRU', inputDim=256, hiddenDim=512, nClasses=29, frameLen=125, AbsoluteMaxStringLen=128, every_frame=True)
model = lip.model
model.load_weights('/data/models/combResnetLSTM_CTCloss_236k-train_1to3ratio_valWER_epochs20_lr1e-4_0.1decay9epochs/weights-07-117.3701.hdf5')

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

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#model = multi_gpu_model(lip.model, gpus=2)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

batch_size = args.batch_size
epochs = args.epochs


spell = Spell(path=PREDICT_DICTIONARY)
decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                  postprocessors=[labels_to_text, spell.sentence])

#metrics_error_rates  = Statistics(lip,DataGenerator_train_softmask(folders_list_val, batch_size) , decoder, 256, output_dir='./results'))

# callcack
metrics_wer = Metrics_softmask(model = lip, val_folders = folders_list_val, batch_size = batch_size, save_path = '/data/results/'+path+'/logs.txt')

# Fit Generator

pred = model.predict([lips, y_data, np.asarray(input_length), np.asarray(128)], batch_size=1)

#decode_res=decoder.decode(pred, 125)



