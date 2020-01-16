from metrics import metric_eval, si_snr
from losses import snr_loss, snr_acc
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
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, ReduceLROnPlateau, EarlyStopping, ReduceLROnPlateau
from callbacks import learningratescheduler, earlystopping, reducelronplateau
from plotting import plot_loss_and_acc
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cv2
from models.lipnet import LipNet
#from models.tasnet_lipnet import TasNet
from models.tdavss import TasNet
from data_generators import DataGenerator_train_softmask, DataGenerator_sampling_softmask, DataGenerator_test_softmask
from dataloaders import DataGenerator_val_samples, DataGenerator_sampling_samples

import shutil
import re

from mir_eval.separation import bss_eval_sources

from data_preparation.audio_utils import retrieve_samples, compress_crm, inverse_crm, return_samples_complex, audios_sum
from data_preparation.video_utils import get_video_frames

import scipy

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def crop_pad_frames(frames, fps, seconds):

    req_frames = fps*seconds

    num_frames = frames.shape[0]

    # Delete or add frames to make the video to 10 seconds
    if num_frames > req_frames:
        frames = frames[:req_frames, :, :, :]

    elif num_frames < req_frames:
        pad_len = req_frames - num_frames
        frames = np.pad(frames, ((0,pad_len),(0,0), (0,0), (0,0)), 'constant')

    elif num_frames == req_frames:
        frames = frames

    return frames


# Read training folders
folders_list = np.loadtxt('/data/AV-speech-separation/data_filenames.txt', dtype='object').tolist()

#folders_list_train = folders_list[:24]
import random
#random.shuffle(folders_list_train)
val_folders_pred_all = folders_list[91500:93000] + folders_list[238089:]
random.seed(200)
val_folders_pred_all = random.sample(val_folders_pred_all, 200)
#print(folders_list_val[4])

'''val_folders_pred_all = sorted(glob.glob('/data/lrs2/train_20s/*'), key=numericalSort)
print('Pred folders:', len(val_folders_pred_all))

time = 20'''

'''model = VideoModel(256,96,(257,500,2),(125,50,100,3)).FullModel(lipnet_pretrained = 'pretrain', unet_pretrained = 'pretrain')
'''

# Building the model
tasnet = TasNet(time_dimensions=200, frequency_bins=257, n_frames=50, attention=False, lipnet_pretrained=True,  train_lipnet=None)
model = tasnet.model
model.compile(optimizer=Adam(lr=0.0001), loss=snr_loss, metrics=[snr_acc])
model.load_weights('/data/models/tdavss_ResNetLSTMLip_236kTrain_2secondsClips_epochs20_lr1e-4_0.35decayNoValDec2epochs_exp2/weights-05-nan.hdf5')
print('Weights Loaded')

from io import StringIO

tmp_smry = StringIO()
model.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
summary = tmp_smry.getvalue()
summary_split = summary.split('\n')
summary_params = summary_split[-6:]
summary_params = '\n'.join(summary_params)
print('\n'+summary_params)

sdr_list = []

batch_size = 20


#pred = model.evaluate_generator(DataGenerator_val_samples(val_folders_pred_all, int(batch_size)),
#                                steps = int(np.ceil((len(val_folders_pred_all))/float(batch_size))),
#                                verbose=1)

print('Predicting on the data')
num = len(val_folders_pred_all)
num_100s = int(num/200)
sdr_list = []
snr_list = []
for n in range(num_100s):
    val_folders_pred = val_folders_pred_all[n*200:(n+1)*200]
    val_predict = model.predict_generator(DataGenerator_val_samples(val_folders_pred_all, int(batch_size)),
                                steps = int(np.ceil((len(val_folders_pred_all))/float(batch_size))),
                                verbose=1)

    mixed_spect = val_predict[:,:,:,2]
    mixed_phase = val_predict[:,:,:,3]
    val_targ = val_predict[:,:,:,4]
    batch = val_targ.shape[0]
    val_targ = val_targ.reshape(batch, -1)
    #       val_targ = val_targ[:, :80000]

    crms = val_predict[:,:,:,:2]

    samples_pred = []
    for i in range(crms.shape[0]):
        crm = crms[i]
        real = crm[:,:,0]
        imaginary = crm[:,:,1]
        inverse_mask = inverse_crm(real_part=real,imaginary_part=imaginary,K=1,C=2)
        #print('crm', crm.shape)
        mixed_spect_ = mixed_spect[i]
        #print('mixed_spect_' ,mixed_spect_.shape)
        mixed_phase_ = mixed_phase[i]
        #print('mixed_phase_', mixed_phase_.shape)
        samples = return_samples_complex(mixed_mag = mixed_spect_, mixed_phase = mixed_phase_, mask = inverse_mask,sample_rate=16e3, n_fft=512, window_size=25, step_size=10)

        #print('samples', samples.shape) 
        samples_pred.append(samples[256:])

    val_targ1 = []
    for i in range(batch):
        length_pred = len(samples_pred[i])
        #print('length_pred', length_pred)
        val_targ_ = val_targ[i, :length_pred]
        #val_targ_ = val_targ_.reshape(1, -1)
        #print('val_targ_', val_targ_.shape)
        val_targ1.append(val_targ_)

    val_targ = val_targ1

    samples_pred = np.asarray(samples_pred)

    val_targ = np.asarray(val_targ)

    val_sdr, val_sdr_list, val_snr, val_snr_list, val_pesq, val_pesq_list = metric_eval(target_samples = val_targ, predicted_samples = samples_pred)

    print('SDR:', val_sdr)
    print('SNR:', val_snr)
    print('PESQ:', val_pesq)

'''samples = []

    try:
        os.mkdir('/data/pred_tasnet_20s')
    except OSError:
        pass

    for i, item in enumerate(val_folders_pred):

        items = sorted(glob.glob(item+ '/*_lips.mp4'), key=numericalSort)
        length = len(samples_pred[i])
        
        samples = []
        for j, item in enumerate(items):
            try:
                os.mkdir('/data/pred_tasnet_20s/'+item[-88:-35])
            except OSError:
                pass

            scipy.io.wavfile.write('/data/pred_tasnet_20s/'+item[-88:-9]+'_pred.wav', 16000, samples_pred[2*i+j])

            shutil.copy2('/data/lrs2/train_20s/'+item[-88:-9]+'_samples.npy','/data/pred_tasnet_20s/'+item[-88:-35])
            shutil.copy2('/data/lrs2/mvlrs_v1/pretrain/'+item[-34:-15]+'/'+item[-14:-9]+'.mp4','/' '/data/pred_tasnet_20s/'+item[-88:-35])

            samples_ = np.load('/data/lrs2/train_20s/'+item[-88:-9]+'_samples.npy')
            scipy.io.wavfile.write('/data/pred_tasnet_20s/'+item[-88:-9]+'_original.wav', 16000, samples_[:length])
            #'/data/pred_sample/'+item[-88:-35]
            samples.append(samples_)
        sam1 = np.zeros((length,))
        sam1[:len(samples[0][:length])] = samples[0][:length]
        sam2 = np.zeros((length,))
        sam2[:len(samples[1][:length])] = samples[1][:length]
        add_samples = sam1+sam2
        scipy.io.wavfile.write('/data/pred_tasnet_20s/'+item[-88:-35]+'/mixed.wav', 16000, add_samples)


    val_folders_pred = sorted(glob.glob('/data/pred_tasnet_20s/*'), key=numericalSort)

    for fold in val_folders_pred:
        wavs = sorted(glob.glob(fold + '/*_original.wav'), key=numericalSort)
        file_sum_name = '/data/pred_tasnet_20s/'+fold[-53:]+'/mixed.wav'
        su = audios_sum(wavs,file_sum_name, volume_reduction=0)

    npys = sorted(glob.glob('/data/pred_tasnet_20s/*/*.npy'),key=numericalSort)
    for i in npys:
        os.remove(i)
'''

'''try: 
        os.mkdir('/data/pred_tasnet_6000')
    except OSError:
        pass

    sdr_mixed = []
    snr_mixed = []
    for i, fold in enumerate(val_folders_pred):

        items = sorted(glob.glob(fold+ '/*_lips.mp4'), key=numericalSort)

        true = val_predict[i,:,:,4]
        true = true.reshape(-1,)    
        true = true[:80000]

        samples = []
        wavs=[]
        for j, item in enumerate(items):
            try:
                os.mkdir('/data/pred_tasnet_6000/'+item[-88:-35])
            except OSError:
                pass

            scipy.io.wavfile.write('/data/pred_tasnet_6000/'+item[-88:-9]+'_pred.wav', 16000, samples_pred[2*i+j])

            shutil.copy2('/data/lrs2/train/'+item[-88:-9]+'_samples.npy','/data/pred_tasnet_6000/'+item[-88:-35])

            samples_ = np.load('/data/lrs2/train/'+item[-88:-9]+'_samples.npy')
            scipy.io.wavfile.write('/data/pred_tasnet_6000/'+item[-88:-9]+'_original.wav', 16000, samples_)
            #'/data/pred_sample/'+item[-88:-35]
            wavs.append('/data/pred_tasnet_6000/'+item[-88:-9]+'_original.wav')
            samples.append(samples_)

        file_sum_name = '/data/mixed.wav'
        su = audios_sum(wavs,file_sum_name, volume_reduction=0)
        su = su[:80000]

        sam1 = np.zeros((80000,))
        sam1[:len(su)] = su

        mix = sam1[:80000]
        print('mix:',mix)
        print('true',true)

        val_sdr_, val_sdr_list_, val_snr_, val_snr_list_ = metric_eval(target_samples = np.asarray([true]), predicted_samples = np.asarray([mix]))

        sdr_mixed.append(val_sdr_)
        snr_mixed.append(val_snr_)

    print(np.max(np.asarray(sdr_mixed)))
    print(np.min(np.asarray(sdr_mixed)))

    sdr_imp = val_sdr - np.asarray(sdr_mixed)
    snr_imp = val_snr - np.asarray(snr_mixed)

    area = np.pi*3

    # Plot
    plt.scatter(sdr_mixed, sdr_imp, s=area, c='green', alpha=0.5)
    plt.title('SDR Improvement Vs Input SDR')
    plt.xlabel('Input SDR')
    plt.ylabel('SDR Improvement')
    plt.savefig('sdr_improv.png')

    # Plot
    plt.scatter(snr_mixed, snr_imp, s=area, c='green', alpha=0.5)
    plt.title('SNR Improvement Vs Input SNR')
    plt.xlabel('Input SNR')
    plt.ylabel('SNR Improvement')
    plt.savefig('snr_improv.png')'''
