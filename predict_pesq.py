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
from models.tdavss_sepconv import TasNet as TasNetSepCon
from gans.nets import Generator
#from models.tdavss_sepconv1 import TasNet as TasNetSepCon
#from models.tasnet_lipnet import TasNet
from dataloaders import DataGenerator_val_samples, DataGenerator_train_samples
from gans.dataloaders_gan import DataGenerator_val
import random
#from pypesq import pesq
#from pesq import pesq
from pypesq import pesq
from mir_eval.separation import bss_eval_sources

#from dataloaders_aug import DataGenerator_train_crm, DataGenerator_sampling_crm, DataGenerator_test_crm

from argparse import ArgumentParser
import time

def si_snr(x, s, remove_dc=True):
    """
    Compute SI-SNR
    Arguments:
        x: vector, enhanced/separated signal
        s: vector, reference signal(ground truth)
    """

    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    # zero mean, seems do not hurt results
    if remove_dc:
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)
        t = np.inner(x_zm, s_zm) * s_zm / vec_l2norm(s_zm)**2
        n = x_zm - t
    else:
        t = np.inner(x, s) * s / vec_l2norm(s)**2
        n = x - t
    return 20 * np.log10(vec_l2norm(t) / vec_l2norm(n))


def vec_l2norm(x):

    nr = K.sqrt(tf.math.reduce_sum(K.square(x), axis=1))
    nr = tf.reshape(nr, (-1, 1))
    #nr = tf.broadcast_to(nr, (int(x.shape[1]), int(x.shape[0])))
    return nr


def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def snr_acc(s, x):
    """
    Compute SI-SNR
    Arguments:
        x: vector, enhanced/separated signal
        s: vector, reference signal(ground truth)
    """

    x = x[:,:,0]
    #s = x[:,:,1]

    x = tf.reshape(x, (-1, 32000))
    s = tf.reshape(s, (-1, 32000))

    '''print('Pred:', x.shape)
    print('GT:', s.shape)'''

    # zero mean, seems do not hurt results
    x_zm = x - tf.reshape(tf.math.reduce_mean(x, axis=1), (-1, 1))
    s_zm = s - tf.reshape(tf.math.reduce_mean(s, axis=1), (-1, 1))
    t = tf.reshape(tf.math.reduce_sum(x_zm*s_zm, axis=1), (-1, 1)) * s_zm / vec_l2norm(s_zm)**2
    n = x_zm - t

    snr_loss_batch = 20 * log10(vec_l2norm(t) / vec_l2norm(n))

    return tf.reduce_mean(snr_loss_batch)

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

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

folders_list_val = np.loadtxt(
    '/home/ubuntu/lrs2_val_comb3.txt', dtype='object').tolist()

#folders_list_val = folders_list_val[:200]

print('Validation data:', len(folders_list_val))

parser = ArgumentParser()
parser.add_argument('-batch_size', action="store", dest="batch_size", type=int)
args = parser.parse_args()
batch_size = args.batch_size

'''print('------------Building Generator------------')
generator = Generator(time_dimensions=200, frequency_bins=257, n_frames=50,
                      lstm=False, lipnet_pretrained=True,  train_lipnet=True)
generator.load_weights(
    '/home/ubuntu/models/tdavss_baseline_256dimSepNet_2speakers_epochs80_lr5e-4_exp1/weights-51--11.1873.tf')
print('Generator weights loaded')'''

tasnet = TasNetSepCon(time_dimensions=200, frequency_bins=257, n_frames=50,
                      attention=False, lstm=False, lipnet_pretrained=True,  train_lipnet=True)
generator = tasnet.model
generator.load_weights(
    '/home/ubuntu/models/tdavss_baseline_3speakers_best/weights-31--8.0022.tf')
print('Model weights loaded')

from io import StringIO
tmp_smry = StringIO()
generator.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
summary = tmp_smry.getvalue()
summary_split = summary.split('\n')
summary_params = summary_split[-6:]
summary_params = '\n'.join(summary_params)
print('\n'+summary_params)


pesqs=[]
sdrs = []
snrs = []

pesqs_mix=[]
sdrs_mix = []
snrs_mix = []

pred_means = []
target_means = []
count=0
start = time.time()
for inputs, target in DataGenerator_val(folders_list_val, int(batch_size), norm=1350.0):
    #inputs=inputs.astype('float32')
    #print(type(inputs))
    count=count+1
    if(count%25==0):print(count)
    if count<= np.ceil(len(folders_list_val)/batch_size):
        pred = generator(inputs)
        mixed = inputs[1]

        pred = np.asarray(pred)
        #pred_means.append(np.mean(pred, axis=1))
        #target_means.append(np.mean(target, axis=1))
        #pred = (pred-np.mean(pred, axis=1, keepdims=True))/np.mean(np.std(pred, axis=1))
        #target = target-np.mean(target, axis=1, keepdims=True)
        #pred = pred/np.mean(np.std(pred, axis=1))
        #pred = pred*1350
        #pred = pred.astype('int16')
            #pred = pred.tolist()
            #pred_audios = pred_audios + pred
            #true_audios= true_audios + target

        for i in range(pred.shape[0]):

            pesqs.append(pesq(target[i].reshape(32000),pred[i].reshape(32000),16000))
            sdr, sir, sar, _ = bss_eval_sources(target[i].reshape(32000),pred[i].reshape(32000), compute_permutation=False)
            sdrs.append(sdr)

            pesqs_mix.append(pesq(target[i].reshape(32000),mixed[i].reshape(32000),16000))
            sdr, sir, sar, _ = bss_eval_sources(target[i].reshape(32000),mixed[i].reshape(32000), compute_permutation=False)
            sdrs_mix.append(sdr)

            snr = snr_acc(target[i].reshape(1, 32000, 1), pred[i].reshape(1, 32000, 1))
            snrs.append(snr.numpy())

            snr = snr_acc(target[i].reshape(1, 32000, 1), mixed[i].reshape(1, 32000, 1))
            snrs_mix.append(snr.numpy())

            #snr = si_snr(pred[i], target[i], remove_dc=True)
            #snrs.append(snr)
            #score1.append(pesq(16000,target[i].reshape(32000),pred[i].reshape(32000),'nb'))
    else:
        break
    #pred_audios = np.asarray(pred_audios)
    #true_audios= np.asarray(true_audios)
    #pred_audios = pred_audios/np.mean(np.std(pred_audios, axis=1))
    #pred_audios = pred_audios*1350
    #pred_audios = pred_audios.astype('int16')
    #true_audios = true_audios.astype('int16')
    #score = pesq(, 16000)
print('Len:', len(pesqs))
print('PESQ:', np.mean(np.asarray(pesqs)))
print('SDR:', np.mean(sdrs))
print('SNR:', np.mean(snrs))

print('PESQ Noisy:', np.mean(np.asarray(pesqs_mix)))
print('SDR Noisy:', np.mean(sdrs_mix))
print('SNR Noisy:', np.mean(snrs_mix))

#print('Targets means:', np.mean(target_means[:-1]))
#print('Preds means', np.mean(pred_means[:-1]))

end = time.time()
print('Total time:', end-start)
#print(np.mean(np.asarray(score1)))

path = 'tdavss_baseline_3speakers_best'

try:
    os.mkdir('/home/ubuntu/metric_results/' + path)
except OSError:
    pass

np.savetxt('/home/ubuntu/metric_results/' + path + '/pesq_val.txt', pesqs)
np.savetxt('/home/ubuntu/metric_results/' + path + '/sdr_val.txt', sdrs)
np.savetxt('/home/ubuntu/metric_results/' + path + '/snr_val.txt', snrs)
np.savetxt('/home/ubuntu/metric_results/' + path + '/pesq_mix.txt', pesqs_mix)
np.savetxt('/home/ubuntu/metric_results/' + path + '/sdr_mix.txt', sdrs_mix)
np.savetxt('/home/ubuntu/metric_results/' + path + '/snr_mix.txt', snrs_mix)

'''area = np.pi*1
# Plot
plt.scatter(sdr_mixed, sdr_imp, s=area, c='green', alpha=0.5)
plt.title('SDR Improvement Vs Input SDR')
plt.xlabel('Input SDR (dB)')
plt.ylabel('SDR Improvement (dB)')
#plt.xticks(np.arange(min(sdr_mixed), max(sdr_mixed)+1, 0.5))
#plt.xticks(np.arange(min(sdr_imp), max(sdr_imp)+1, 0.5))
plt.savefig('sdr_improv_4.png')'''
