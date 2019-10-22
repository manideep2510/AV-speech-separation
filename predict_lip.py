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
from plotting import plot_loss_and_acc
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
<<<<<<< HEAD
from LipNet.lipnet.lipreading.callback import Metrics_softmask, Decoder
from LipNet.lipnet.lipreading.generator import DataGenerator_train_softmask, DataGenerator_sampling_softmask, crop_pad_frames
from LipNet.lipnet.lipreading.helpers import text_to_labels
from LipNet.lipnet.lipreading.aligns import Align
=======
from LipNet.lipnet.lipreading.callback import Metrics_softmask
from LipNet.lipnet.lipreading.generator import DataGenerator_train_softmask, DataGenerator_sampling_softmask
>>>>>>> parent of 8cef436... Update predict_lip.py
#from LipNet.lipnet.model2 import LipNet
from models.resnet_lstm_lipread import lipreading
import numpy as np
import datetime
import pickle
<<<<<<< HEAD
from data_preparation.video_utils import get_video_frames
=======

>>>>>>> parent of 8cef436... Update predict_lip.py

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

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

with open("/data/AV-speech-separation/folder_filter_1.txt", "rb") as fp:  
       folders_list = pickle.load(fp) 

#folders_list_train = folders_list[:91500] +folders_list[93000:238089]
#print(folders_list_train[34])

#folders_list_train=folders_list[:256]
#folders_list_val=folders_list[256:320]

folders_list_train=folders_list[0:192000]
folders_list_val=folders_list[192000:204000]
import random
random.seed(10)
random.shuffle(folders_list_train)
#folders_list_val = folders_list[91500:93000] + folders_list[238089:]
#folders_list_val=folders_list[512:768]
#random.seed(20)
#folders_list_train = random.sample(folders_list_train, 180)
#folders_list_val = random.sample(folders_list_val, 100)

print('Training data:', len(folders_list_train)*2)
print('Validation data:', len(folders_list_val)*2)

#lips_filelist = sorted(glob.glob('/data/lrs2/train/*/*_lips.mp4'), key=numericalSort)

<<<<<<< HEAD
video_file = args.video_file
transcript_file = video_file[:-9]+'.txt'
lips = get_video_frames(video_file, fmt='rgb')
lips = crop_pad_frames(frames = lips, fps = 25, seconds = 5)
lips = lips.reshape(1, 125,50,100,3)
print('lips shape:', lips.shape)
=======
#masks_filelist = sorted(glob.glob('/data/lrs2/train/*/*_mask.png'), key=numericalSort)
>>>>>>> parent of 8cef436... Update predict_lip.py

#spects_filelist = sorted(glob.glob('/data/lrs2/train/*/mixed_spectrogram.npy'), key=numericalSort)

<<<<<<< HEAD
trans=(Align(128, text_to_labels).from_file(transcript_file))
y_data=(trans.padded_label)
y_data = y_data.reshape(1, 128)
print('y_data shape:',y_data.shape)
label_length=(trans.label_length)
input_length=125

#lip = lipreading(mode='backendGRU', inputDim=256, hiddenDim=512, nClasses=29, frameLen=125, AbsoluteMaxStringLen=128, every_frame=True)
#model = lip
model=LipNet(input_shape=(125,50,100,3), pretrained='pretrain', output_size = 29, absolute_max_string_len=128)
#model.load_weights('/data/models/combResnetLSTM_CTCloss_236k-train_1to3ratio_valWER_epochs20_lr1e-4_0.1decay9epochs/weights-07-117.3701.hdf5')
=======
#model = VideoModel(256,96,(257,500,2),(125,50,100,3)).FullModel(lipnet_pretrained = True)

#lip=LipNet(pretrained=True,weights_path='/data/models/lip_net_236k-train_1to3ratio_valSDR_epochs10-20_lr1e-4_0.1decay10epochs/weights-04-125.3015.hdf5')
lip = lipreading(mode='backendGRU', inputDim=256, hiddenDim=512, nClasses=29, frameLen=125, AbsoluteMaxStringLen=128, every_frame=True)
model = lip.model
model.load_weights('/data/models/combResnetLSTM_CTCloss_236k-train_1to3ratio_valWER_epochs20_lr1e-4_0.1decay9epochs/weights-07-117.3701.hdf5')
>>>>>>> parent of 8cef436... Update predict_lip.py

from io import StringIO
tmp_smry = StringIO()
model.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
summary = tmp_smry.getvalue()
summary_split = summary.split('\n')
summary_params = summary_split[-6:]
summary_params = '\n'.join(summary_params)
print('\n'+summary_params)

# Compile the model
#lrate = args.lrate

<<<<<<< HEAD
#adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
=======
#model.load_weights('/data/models/softmask_unet_Lipnet+cocktail_1in_1out_90k-train_1to3ratio_valSDR_epochs20_lr1e-4_0.1decay10epochs/weights-10-188.9557.hdf5')
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
>>>>>>> parent of 8cef436... Update predict_lip.py
#model = multi_gpu_model(lip.model, gpus=2)
#model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

<<<<<<< HEAD
#batch_size = args.batch_size
#epochs = args.epochs


#spell = Spell(path=PREDICT_DICTIONARY)
#decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
#                  postprocessors=[labels_to_text, spell.sentence])
=======
#model.load_weights('/data/models/test_Lipnet+cocktail_1in_1out_20k-train_valSDR_epochs20_lr1e-4_0.322decay5epochs/weights-12-0.4127.hdf5')

#model.compile(optimizer = Adam(lr=lrate), loss = l2_loss)

batch_size = args.batch_size
epochs = args.epochs


# spell = Spell(path=PREDICT_DICTIONARY)
# decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
#                   postprocessors=[labels_to_text, spell.sentence])
#
# metrics_error_rates  = Statistics(lip,DataGenerator_train_softmask(folders_list_val, batch_size) , decoder, 256, output_dir='./results'))
>>>>>>> parent of 8cef436... Update predict_lip.py

# Path to save model checkpoints

path = 'combResnetLSTM_CTCloss_236k-train_1to3ratio_valWER_epochs8to9_lr1e-4_0.1decay9epochs'

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
<<<<<<< HEAD
#metrics_wer = Metrics_softmask(model = lip, val_folders = folders_list_val, batch_size = batch_size, save_path = '/data/results/'+path+'/logs.txt')
=======
metrics_wer = Metrics_softmask(model = lip, val_folders = folders_list_val, batch_size = batch_size, save_path = '/data/results/'+path+'/logs.txt')
learningratescheduler = learningratescheduler()
#earlystopping = earlystopping()
#reducelronplateau = reducelronplateau()
#logger = Logger('/data/results')
>>>>>>> parent of 8cef436... Update predict_lip.py

filepath='/data/models/' +  path+ '/weights-{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint_save_weights = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, save_weights_only=True, mode='min')

<<<<<<< HEAD
pred = model.predict(lips, batch_size=1)

def labels_to_text(labels):
    # 26 is space, 27 is CTC blank char
    text = ''
    for c in labels:
        c1=int(c)
        if c1 >= 0 and c1 < 26:
            text += chr(c1 + ord('a'))
        elif c1 == 26:
            text += ' '
    return text

def _decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    """Decodes the output of a softmax.
    Can use either greedy search (also known as best path)
    or a constrained dictionary search.
    # Arguments
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, )` containing the sequence length for
            each batch item in `y_pred`.
        greedy: perform much faster best-path search if `true`.
            This does not use a dictionary.
        beam_width: if `greedy` is `false`: a beam search decoder will be used
            with a beam of this width.
        top_paths: if `greedy` is `false`,
            how many of the most probable paths will be returned.
    # Returns
        Tuple:
            List: if `greedy` is `true`, returns a list of one element that
                contains the decoded sequence.
                If `false`, returns the `top_paths` most probable
                decoded sequences.
                Important: blank labels are returned as `-1`.
            Tensor `(top_paths, )` that contains
                the log probability of each decoded sequence.
    """
    decoded = K.ctc_decode(y_pred=y_pred, input_length=input_length,
                           greedy=greedy, beam_width=beam_width, top_paths=top_paths)
    paths = [path.eval(session=K.get_session()) for path in decoded[0]]
    logprobs  = decoded[1].eval(session=K.get_session())

    return (paths, logprobs)

def decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1, **kwargs):
    language_model = kwargs.get('language_model', None)

    paths, logprobs = _decode(y_pred=y_pred, input_length=input_length,
                              greedy=greedy, beam_width=beam_width, top_paths=top_paths)
    if language_model is not None:
        # TODO: compute using language model
        raise NotImplementedError("Language model search is not implemented yet")
    else:
        # simply output highest probability sequence
        # paths has been sorted from the start
        result = paths[0]
    return result

class Decoder(object):
    def __init__(self, greedy=True, beam_width=100, top_paths=1, **kwargs):
        self.greedy         = greedy
        self.beam_width     = beam_width
        self.top_paths      = top_paths
        self.language_model = kwargs.get('language_model', None)
        self.postprocessors = kwargs.get('postprocessors', [])

    def decode(self, y_pred, input_length):
        decoded = decode(y_pred, input_length, greedy=self.greedy, beam_width=self.beam_width,
                         top_paths=self.top_paths, language_model=self.language_model)
        preprocessed = []
        for output in decoded:
            out = output
            for postprocessor in self.postprocessors:
                out = postprocessor(out)
            preprocessed.append(out)

        return preprocessed

PREDICT_GREEDY      = True
PREDICT_BEAM_WIDTH  = 200



decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                  postprocessors=[labels_to_text])

out = decoder.decode(pred, [125])


pred = np.argmax(pred, axis=2)
pred = pred.reshape(125,)
#print(pred)
letters = labels_to_text(pred.tolist())

#out = ''.join(letters)
print('Raw Output:', letters)
print('Prediction:', out)
print('Transcript:', trans.sentence)
=======
# Fit Generator
>>>>>>> parent of 8cef436... Update predict_lip.py

folders_per_epoch = int(len(folders_list_train)/3)

history = model.fit_generator(DataGenerator_sampling_softmask(folders_list_train, folders_per_epoch, batch_size),
                steps_per_epoch = np.ceil((folders_per_epoch)/float(batch_size)),
                epochs=epochs,
                validation_data=DataGenerator_train_softmask(folders_list_val, batch_size),
                validation_steps = np.ceil((len(folders_list_val))/float(batch_size)),
                callbacks=[LoggingCallback(print_fcn=log_to_file), checkpoint_save_weights, learningratescheduler, metrics_wer], verbose = 1)

# Plots
plot_loss_and_acc(history, path)

# Logs
#command = "kubectl logs pods/train | egrep -E -i -e 'val|epoch' > /data/results/" + path + "/logs.txt"
#os.system(command)
