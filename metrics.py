from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, ReduceLROnPlateau, EarlyStopping, ReduceLROnPlateau
from mir_eval.separation import bss_eval_sources

from data_preparation.audio_utils import retrieve_samples, compress_crm, inverse_crm, return_samples_complex
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from dataloaders import DataGenerator_test_samples, DataGenerator_val_unsync_attention, DataGenerator_val_unsync_attention_easy, Data_predict_attention, DataGenerator_val_samples
from pesq import pesq
import wandb
import random
import json
import glob
from pydub import AudioSegment
from scipy.io import wavfile
from scipy import signal
from data_preparation.video_utils import get_video_frames
from seaborn import heatmap

# To read the images in numerical order
import re
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


def metric_eval(target_samples, predicted_samples):
	
    sdr_batch=[]
    snr_batch=[]
    pesq_batch=[]
    batch_size=(target_samples.shape)[0]
    for i in range(batch_size):
        sdr, sir, sar, _ = bss_eval_sources(target_samples[i], predicted_samples[i], compute_permutation=False)
        snr = si_snr(predicted_samples[i], target_samples[i], remove_dc=True)
        pesq_ = pesq(16000, target_samples[i], predicted_samples[i], 'wb')
        sdr_batch.append(sdr[0])
        snr_batch.append(snr)
        pesq_batch.append(pesq_)
    
    sdr_batch = np.asarray(sdr_batch)
    snr_batch = np.asarray(snr_batch)
    pesq_batch = np.asarray(pesq_batch)

    return np.mean(sdr_batch), sdr_batch, np.mean(snr_batch), snr_batch, np.mean(pesq_batch), pesq_batch

def sdr(y_true, mask, mixed_spect, mixed_phase):
    
    mask = tf.keras.backend.argmax(mask, axis=3)

    y_pred = retrieve_samples(spec_signal = mixed_spect,phase_spect = mixed_phase,mask = mask,sample_rate=16e3, n_fft=512, window_size=25, step_size=10)

    return metric_eval(target_samples = y_true, predicted_samples = y_pred)


def sdr_metric(y_true, y_pred):

    mixed_spect = y_pred[:,:,:,2]
    mixed_phase = y_pred[:,:,:,3]
    samples = y_pred[:,:,:,4]
    shape_samples = K.int_shape(samples)
    samples = tf.reshape(samples, [shape_samples[0], -1])
    samples = samples[:, :80000]
        
    y_true = samples

    mask = tf.keras.backend.argmax(y_pred[:,:,:,:2], axis=3)

    y_pred = retrieve_samples(spec_signal = mixed_spect,phase_spect = mixed_phase,mask = mask,sample_rate=16e3, n_fft=512, window_size=25, step_size=10)

    return metric_eval(target_samples = y_true, predicted_samples = y_pred)

class Metrics(Callback):

    def __init__(self, model, val_folders, batch_size):
        self.model = model
        self.val_folders = val_folders
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.val_sdr = []
 
    def on_epoch_end(self, epoch, logs={}):
        num = len(self.val_folders)
        num_100s = int(num/100)
        sdr_list = []
        for n in range(num_100s):
            val_folders_100 = self.val_folders[n*100:(n+1)*100]
            val_predict = np.asarray(self.model.predict_generator(DataGenerator_test(val_folders_100, self.batch_size), steps = np.ceil((len(val_folders_100))/float(self.batch_size))))
            mixed_spect = val_predict[:,:,:,2]
            mixed_phase = val_predict[:,:,:,3]
            val_targ = val_predict[:,:,:,4]
            batch = val_targ.shape[0]
            val_targ = val_targ.reshape(batch, -1)
#           val_targ = val_targ[:, :80000]

            masks = np.argmax(val_predict[:,:,:,:2], axis = 3)
     
            samples_pred = []
            for i in range(masks.shape[0]):
                mask = masks[i]
                #print('mask', mask.shape)
                mixed_spect_ = mixed_spect[i]
                #print('mixed_spect_' ,mixed_spect_.shape)
                mixed_phase_ = mixed_phase[i]
                #print('mixed_phase_', mixed_phase_.shape)
                samples = retrieve_samples(spec_signal = mixed_spect_,phase_spect = mixed_phase_,mask = mask,sample_rate=16e3, n_fft=512, window_size=25, step_size=10)
            
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
            #print('samples_pred', samples_pred.shape)
            val_targ = np.asarray(val_targ)
            #print('val_targ', val_targ.shape)
         
            _val_sdr1, _, _, _ = metric_eval(target_samples = val_targ, predicted_samples = samples_pred)
            sdr_list.append(_val_sdr1)

        sdr_list = np.asarray(sdr_list)
        _val_sdr = np.mean(sdr_list)
        self.val_sdr.append(_val_sdr)

        print('Validation SDR: ', _val_sdr)

        return

class Metrics_softmask(Callback):

    def __init__(self, model, val_folders, batch_size):
        self.model = model
        self.val_folders = val_folders
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.val_sdr = []
 
    def on_epoch_end(self, epoch, logs={}):
        num = len(self.val_folders)
        num_100s = int(num/100)
        sdr_list = []
        for n in range(num_100s):
            val_folders_100 = self.val_folders[n*100:(n+1)*100]
            val_predict = np.asarray(self.model.predict_generator(DataGenerator_test(val_folders_100, self.batch_size), steps = np.ceil((len(val_folders_100))/float(self.batch_size))))
            mixed_spect = val_predict[:,:,:,1]
            mixed_phase = val_predict[:,:,:,2]
            val_targ = val_predict[:,:,:,3]
            batch = val_targ.shape[0]
            val_targ = val_targ.reshape(batch, -1)
#           val_targ = val_targ[:, :80000]

            masks = val_predict[:,:,:,0]
     
            samples_pred = []
            for i in range(masks.shape[0]):
                mask = masks[i]
                #print('mask', mask.shape)
                mixed_spect_ = mixed_spect[i]
                #print('mixed_spect_' ,mixed_spect_.shape)
                mixed_phase_ = mixed_phase[i]
                #print('mixed_phase_', mixed_phase_.shape)
                samples = retrieve_samples(spec_signal = mixed_spect_,phase_spect = mixed_phase_,mask = mask,sample_rate=16e3, n_fft=512, window_size=25, step_size=10)
            
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
            #print('samples_pred', samples_pred.shape)
            val_targ = np.asarray(val_targ)
            #print('val_targ', val_targ.shape)
         
            _val_sdr1, _, _, _ = metric_eval(target_samples = val_targ, predicted_samples = samples_pred)
            sdr_list.append(_val_sdr1)

        sdr_list = np.asarray(sdr_list)
        _val_sdr = np.mean(sdr_list)
        self.val_sdr.append(_val_sdr)

        print('Validation SDR: ', _val_sdr)

        return

class Metrics_crm(Callback):

    def __init__(self, model, val_folders, batch_size, save_path):
        self.model = model
        self.val_folders = val_folders
        self.batch_size = batch_size
        self.save_path = save_path

    def on_train_begin(self, logs={}):
        self.val_sdr = []
        self.val_snr = []
        self.val_pesq = []
 
    def on_epoch_end(self, epoch, logs={}):
        '''print(len(self.val_folders[:1500]))
        print(len(self.val_folders[1500:]))'''
        val_folders_samp = [self.val_folders[200], self.val_folders[305],self.val_folders[400],self.val_folders[500],self.val_folders[600],self.val_folders[800],self.val_folders[1200],self.val_folders[1400],self.val_folders[-100], self.val_folders[-305],self.val_folders[-400],self.val_folders[-500],self.val_folders[-600],self.val_folders[-800],self.val_folders[-1200],self.val_folders[-1400]]
        #val_folders_samp = random.sample(self.val_folders[:1500], 8) + random.sample(self.val_folders[1500:], 8)

        attn_states_out_model = Model(inputs=self.model.input, outputs=self.model.get_layer('attention_layer').output)
        atten_outs = attn_states_out_model.predict_generator(DataGenerator_test_crm(val_folders_samp, self.batch_size), steps = int(np.ceil((len(val_folders_samp))/float(self.batch_size))), verbose=0)
        #print('atten_outs len:', len(atten_outs))
        #print('atten_outs:', atten_outs.shape)
        atten_states = atten_outs[1]
        np.save(self.save_path[:-8]+'atten_states_'+str(epoch)+'.npy', atten_states)
        atten_states = atten_states*10000
        '''print('atten_states:', atten_states.shape)
        print('atten_states max:', np.max(atten_states))
        print('atten_states min:', np.min(atten_states))'''

        wandb.log({"Attention Weights Alignment": [wandb.Image(i, caption="Attention_align") for i in atten_states]}, commit=False)

        num = len(self.val_folders)
        num_100s = int(num/100)
        sdr_list = []
        snr_list = []
        pesq_list =[]
        for n in range(num_100s):
            val_folders_100 = self.val_folders[n*100:(n+1)*100]
            val_predict = np.asarray(self.model.predict_generator(DataGenerator_test_crm(val_folders_100, self.batch_size), steps = int(np.ceil((len(val_folders_100))/float(self.batch_size))), verbose=0))

            mixed_spect = val_predict[:,:,:,2]
            mixed_phase = val_predict[:,:,:,3]
            val_targ = val_predict[:,:,:,4]
            batch = val_targ.shape[0]
            val_targ = val_targ.reshape(batch, -1)
#           val_targ = val_targ[:, :80000]

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
            #print('samples_pred', samples_pred.shape)
            val_targ = np.asarray(val_targ)

            _val_sdr1, _, _val_snr1, _, _val_pesq1, _ = metric_eval(target_samples = val_targ, predicted_samples = samples_pred)
            sdr_list.append(_val_sdr1)
            snr_list.append(_val_snr1)
            pesq_list.append(_val_pesq1)

        # SDR
        sdr_list = np.asarray(sdr_list)
        _val_sdr = np.mean(sdr_list)
        self.val_sdr.append(_val_sdr)
        #SNR
        snr_list = np.asarray(snr_list)
        _val_snr = np.mean(snr_list)
        self.val_snr.append(_val_snr)
        #PESQ
        pesq_list = np.asarray(pesq_list)
        _val_pesq = np.mean(pesq_list)
        self.val_pesq.append(_val_pesq)
        print('Val SDR:', _val_sdr, ' -  Val Si_SNR:', _val_snr, ' -  Val PESQ:', _val_pesq)
        #print(logs['val_loss'])
        wandb.log({'loss':logs['loss'], 'val_loss':logs['val_loss'], 'lr':logs['lr'], 'sdr': _val_sdr, 'snr': _val_snr, 'pesq':_val_pesq})
        with open(self.save_path, "a") as myfile:
            myfile.write(', Val_SDR: ' + str(_val_sdr) + ',  Val_Si-SNR: '+ str(_val_snr) + '\n')
        return


class Metrics_samples(Callback):

    def __init__(self, model, val_folders, batch_size, save_path):
        self.model = model
        self.val_folders = val_folders
        self.batch_size = batch_size
        self.save_path = save_path

    def on_train_begin(self, logs={}):
        self.val_sdr = []
        self.val_snr = []
        self.val_pesq = []
 
    def on_epoch_end(self, epoch, logs={}):
        '''print(len(self.val_folders[:1500]))
        print(len(self.val_folders[1500:]))'''
        #val_folders_samp = [self.val_folders[200], self.val_folders[305],self.val_folders[400],self.val_folders[500],self.val_folders[600],self.val_folders[800],self.val_folders[1200],self.val_folders[1400],self.val_folders[-100], self.val_folders[-305],self.val_folders[-400],self.val_folders[-500],self.val_folders[-600],self.val_folders[-800],self.val_folders[-1200],self.val_folders[-1400]]
        #val_folders_samp = [self.val_folders[20], self.val_folders[30],self.val_folders[40],self.val_folders[50],self.val_folders[60],self.val_folders[80],self.val_folders[12],self.val_folders[14],self.val_folders[-1], self.val_folders[-3],self.val_folders[-4],self.val_folders[-5],self.val_folders[-6],self.val_folders[-8],self.val_folders[-12],self.val_folders[-10]]

        #attn_states_out_model = Model(inputs=self.model.input, outputs=self.model.get_layer('attention_layer').output)
        #atten_outs = attn_states_out_model.predict_generator(DataGenerator_test_samples(val_folders_samp, self.batch_size), steps = int(np.ceil((len(val_folders_samp))/float(self.batch_size))), verbose=0)
        #print('atten_outs len:', len(atten_outs))
        #print('atten_outs:', atten_outs.shape)
        #atten_states = atten_outs[1]
        #np.save(self.save_path[:-8]+'atten_states_'+str(epoch)+'.npy', atten_states)
        #atten_states = atten_states*10000
        '''print('atten_states:', atten_states.shape)
        print('atten_states max:', np.max(atten_states))
        print('atten_states min:', np.min(atten_states))'''

        #wandb.log({"Attention Weights Alignment": [wandb.Image(i, caption="Attention_align") for i in atten_states]}, commit=False)

        num = len(self.val_folders)
        num_100s = int(num/100)
        sdr_list = []
        snr_list = []
        pesq_list =[]
        for n in range(num_100s):
            val_folders_100 = self.val_folders[n*100:(n+1)*100]
            val_predict = np.asarray(self.model.predict_generator(DataGenerator_test_samples(val_folders_100, int(self.batch_size)), steps = int(np.ceil((len(val_folders_100))/float(self.batch_size))), verbose=0))

            val_targ = val_predict[:,:,1]

            samples_pred = val_predict[:,:,0]

            _val_sdr1, _, _val_snr1, _, _val_pesq1, _ = metric_eval(target_samples = val_targ, predicted_samples = samples_pred)
            sdr_list.append(_val_sdr1)
            snr_list.append(_val_snr1)
            pesq_list.append(_val_pesq1)

        # SDR
        sdr_list = np.asarray(sdr_list)
        _val_sdr = np.mean(sdr_list)
        self.val_sdr.append(_val_sdr)
        #SNR
        snr_list = np.asarray(snr_list)
        _val_snr = np.mean(snr_list)
        self.val_snr.append(_val_snr)
        #PESQ
        pesq_list = np.asarray(pesq_list)
        _val_pesq = np.mean(pesq_list)
        self.val_pesq.append(_val_pesq)
        print('Val SDR:', _val_sdr, ' -  Val Si_SNR:', _val_snr, ' -  Val PESQ:', _val_pesq)
        #print(logs['val_loss'])
        wandb.log({'loss':logs['loss'], 'val_loss':logs['val_loss'], 'lr':logs['lr'], 'sdr': _val_sdr, 'snr': _val_snr, 'pesq':_val_pesq})
        with open(self.save_path, "a") as myfile:
            myfile.write(', Val_SDR: ' + str(_val_sdr) + ',  Val_Si-SNR: '+ str(_val_snr) + '\n')
        return


class Metrics_wandb(Callback):

    def __init__(self):
        pass

    def on_train_begin(self, logs={}):
        self.val_sdr = []

 
    def on_epoch_end(self, epoch, logs={}):
        
        self.val_sdr.append(0)
        #print(logs.keys())

        wandb.log({'loss':logs['loss'], 'val_loss':logs['val_loss'], 'lr':logs['lr'], 'SNR':logs['snr_acc'], 'Val_SNR':logs['val_snr_acc']}, commit=False)
        
        return


class Metrics_unsync(Callback):

    def __init__(self, model, val_folders, batch_size, save_path):
        self.model = model
        self.val_folders = val_folders
        self.batch_size = batch_size
        self.save_path = save_path

    def on_train_begin(self, logs={}):
        self.val_snr = []
 
    def on_epoch_end(self, epoch, logs={}):

        with open('/data/AV-speech-separation1/lrs2_1dot5k-unsync_audio_val.json') as json_file:
            unsync_dict = json.load(json_file)
    
        unsync_files = unsync_dict['folds']
        offsets = unsync_dict['offsets']

        val_folders_pred_all = []
        for item in unsync_files:
            path = '/data/lrs2/train/' + item
            val_folders_pred_all.append(path)

        val_folders_samp_dict = {'/data/lrs2/train/5954958412463581171_00074_6048924136462145896_00033_2':13,  
                                '/data/lrs2/train/6265355699075927737_00014_5942747390944340934_00018_2':10, 
                                '/data/lrs2/train/6265243600299093383_00043_5854842724793826670_00018_2':12,
                                '/data/lrs2/train/5954958412463581171_00078_6049708826987126376_00037_2':8,
                                '/data/lrs2/train/5954958412463581171_00085_6092785631109560111_00001_2':13,
                                '/data/lrs2/train/6265243600299093383_00045_5942747390944340934_00018_2':8,
                                self.val_folders[10]:0, self.val_folders[20]:0, self.val_folders[50]:0, 
                                self.val_folders[100]:0, self.val_folders[1600]:0, self.val_folders[1610]:0, 
                                self.val_folders[1620]:0, self.val_folders[1630]:0, self.val_folders[60]:0, 
                                self.val_folders[1750]:0}
        val_folders_samp = list(val_folders_samp_dict.keys())

        folderlist = list(val_folders_samp_dict.keys())
        lips = []
        samples = []
        samples_mix = []

        for folder in folderlist:
            lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
            samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
            samples_mix_ = '/data/mixed_audio_files/' +folder.split('/')[-1]+'.wav'
            for i in range(len(lips_)):
                lips.append(lips_[i])
            for i in range(len(samples_)):
                samples.append(samples_[i])
            for i in range(len(lips_)):
                samples_mix.append(samples_mix_)

        attn_states_out_model = Model(inputs=self.model.input, outputs=self.model.get_layer('attention_layer').output)
        mask_out_model = Model(inputs=self.model.input, outputs=self.model.get_layer('mask').output)

        pred_weights = attn_states_out_model.predict(Data_predict_attention(val_folders_samp_dict))
        pred_weights = pred_weights[1]

        preds_mask = mask_out_model.predict(Data_predict_attention(val_folders_samp_dict))

        preds_audio = self.model.predict(Data_predict_attention(val_folders_samp_dict))
        preds_audio = preds_audio*3500
        preds_audio = preds_audio.astype('int16')
        
        # Log all things to wandb
        offset = []
        aud_offset = []
        save_name_base = []
        for i, item in enumerate(lips):

            offset_ = val_folders_samp_dict[val_folders_samp[i//2]]
            aud_offset_ = int(abs((offset_/25)*16000))
            save_name_base_ = str(i//2) + '_' + item.split('/')[-1][:-9] + '_' + 'off' + str(offset_)
            offset.append(offset_)
            aud_offset.append(aud_offset_)
            save_name_base.append(save_name_base_)

        wandb.log({"Mask Predictions": [wandb.Image(
            img*100, caption=save_name_base[i]) for i, img in enumerate(preds_mask)]}, commit=False)
        wandb.log({"Attention Weights Alignment": [wandb.Image(
            img*1000000, caption=save_name_base[i]) for i, img in enumerate(pred_weights)]}, commit=False)
        wandb.log({"Predicted Audio": [wandb.Audio(
            aud, caption=save_name_base[i], sample_rate=16000) for i, aud in enumerate(preds_audio)]}, commit=False)
        wandb.log({"True Audio": [wandb.Audio(
            np.pad(np.load(item), (0, 32000), mode='constant')[aud_offset[i]:32000+aud_offset[i]], 
            caption=save_name_base[i], sample_rate=16000) for i, item in enumerate(samples)]}, commit=False)
        wandb.log({"Mixed Audio": [wandb.Audio(
            np.pad(wavfile.read(item)[1], (0, 32000), mode='constant')[aud_offset[i]:32000+aud_offset[i]], 
            caption=save_name_base[i], sample_rate=16000) for i, item in enumerate(samples_mix)]}, commit=False)


        #val_folders_pred_all = val_folders_pred_all[:100]

        _, snr = self.model.evaluate_generator(DataGenerator_val_unsync_attention(val_folders_pred_all, int(self.batch_size)), 
                            steps=int(np.ceil((len(val_folders_pred_all))/float(self.batch_size))), 
                            verbose=0)

        _, snr_easy = self.model.evaluate_generator(DataGenerator_val_unsync_attention_easy(val_folders_pred_all, int(self.batch_size)), 
                            steps=int(np.ceil((len(val_folders_pred_all))/float(self.batch_size))), 
                            verbose=0)

        '''_, snr_vox = self.model.evaluate_generator(DataGenerator_val_unsync_attention_easy(val_folders_pred_all, int(self.batch_size)), 
                            steps=int(np.ceil((len(val_folders_pred_all))/float(self.batch_size))), 
                            verbose=0)'''

        #SNR
        self.val_snr.append(snr)

        print('\nUnsync SNR Hard:', snr, ' - Unsync SNR Easy:', snr_easy)

        wandb.log({'UnSync-SNR-hard':snr, 'UnSync-SNR-easy':snr_easy})

        with open(self.save_path, "a") as myfile:
            myfile.write(', Unsync SNR Hard: ' + str(snr) + ', Unsync SNR Easy: ' + str(snr_easy) + '\n')
        return


class Metrics_3speak(Callback):

    def __init__(self, model, val_folders, batch_size, save_path):
        self.model = model
        self.val_folders = val_folders
        self.batch_size = batch_size
        self.save_path = save_path

    def on_train_begin(self, logs={}):
        self.val_snr = []
 
    def on_epoch_end(self, epoch, logs={}):

        with open('/data/AV-speech-separation1/lrs2_1dot5k-unsync_audio_val.json') as json_file:
            unsync_dict = json.load(json_file)
    
        unsync_files = unsync_dict['folds']
        offsets = unsync_dict['offsets']

        val_folders_pred_all = []
        for item in unsync_files:
            path = '/data/lrs2/train/' + item
            val_folders_pred_all.append(path)

        '''val_folders_samp_dict = {'/data/lrs2/train/5954958412463581171_00074_6048924136462145896_00033_2':13,  
                                '/data/lrs2/train/6265355699075927737_00014_5942747390944340934_00018_2':10, 
                                '/data/lrs2/train/6265243600299093383_00043_5854842724793826670_00018_2':12,
                                '/data/lrs2/train/5954958412463581171_00078_6049708826987126376_00037_2':8,
                                '/data/lrs2/train/5954958412463581171_00085_6092785631109560111_00001_2':13,
                                '/data/lrs2/train/6265243600299093383_00045_5942747390944340934_00018_2':8,
                                self.val_folders[10]:0, self.val_folders[20]:0, self.val_folders[50]:0, 
                                self.val_folders[100]:0, self.val_folders[1600]:0, self.val_folders[1610]:0, 
                                self.val_folders[1620]:0, self.val_folders[1630]:0, self.val_folders[60]:0, 
                                self.val_folders[1750]:0}'''

        val_folders_samp_dict = {self.val_folders[10]:0, self.val_folders[30]:0, self.val_folders[50]:0, 
                                self.val_folders[100]:0, self.val_folders[1600]:0, self.val_folders[1620]:0, 
                                self.val_folders[1640]:0, self.val_folders[1660]:0, self.val_folders[70]:0, 
                                self.val_folders[1750]:0}
        val_folders_samp = list(val_folders_samp_dict.keys())

        folderlist = list(val_folders_samp_dict.keys())
        lips = []
        samples = []
        samples_mix = []

        for folder in folderlist:
            '''lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
            samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
            samples_mix_ = '/data/mixed_audio_files/' +folder.split('/')[-1]+'.wav'
            for i in range(len(lips_)):
                lips.append(lips_[i])
            for i in range(len(samples_)):
                samples.append(samples_[i])
            for i in range(len(lips_)):
                samples_mix.append(samples_mix_)'''

            lips_ = folder
            samples_ = folder[:-9] + '_samples.npy'
            samples_mix_ = '/data/mixed_audio_files/' + folder.split('/')[-2] + '.wav'

            lips.append(lips_)
            samples.append(samples_)
            samples_mix.append(samples_mix_)

        #attn_states_out_model = Model(inputs=self.model.input, outputs=self.model.get_layer('attention_layer').output)
        mask_out_model = Model(inputs=self.model.input, outputs=self.model.get_layer('mask').output)

        #pred_weights = attn_states_out_model.predict(Data_predict_attention(val_folders_samp_dict))
        #pred_weights = pred_weights[1]

        preds_mask = mask_out_model.predict(Data_predict_attention(val_folders_samp_dict))

        preds_audio = self.model.predict(Data_predict_attention(val_folders_samp_dict))
        #preds_audio = preds_audio/np.mean(np.std(preds_audio, axis=1))
        #preds_audio = preds_audio*1350
        preds_audio = preds_audio.astype('int16')
        '''print('Pred max', np.max(preds_audio))
        print('Pred min', np.min(preds_audio))'''
        
        # Log all things to wandb
        offset = []
        aud_offset = []
        save_name_base = []
        for i, item in enumerate(lips):

            offset_ = val_folders_samp_dict[val_folders_samp[i]]
            aud_offset_ = int(abs((offset_/25)*16000))
            save_name_base_ = str(i) + '_' + item.split('/')[-1][:-9] + '_' + 'off' + str(offset_)
            offset.append(offset_)
            aud_offset.append(aud_offset_)
            save_name_base.append(save_name_base_)

        wandb.log({"Mask Predictions": [wandb.Image(
            img*100, caption=save_name_base[i]) for i, img in enumerate(preds_mask)]}, commit=False)
        '''wandb.log({"Attention Weights Alignment": [wandb.Image(
            img*1000000, caption=save_name_base[i]) for i, img in enumerate(pred_weights)]}, commit=False)'''
        wandb.log({"Predicted Audio": [wandb.Audio(
            aud, caption=save_name_base[i], sample_rate=16000) for i, aud in enumerate(preds_audio)]}, commit=False)
        wandb.log({"True Audio": [wandb.Audio(
            np.pad(np.load(item), (0, 32000), mode='constant')[aud_offset[i]:32000+aud_offset[i]], 
            caption=save_name_base[i], sample_rate=16000) for i, item in enumerate(samples)]}, commit=False)
        wandb.log({"Mixed Audio": [wandb.Audio(
            np.pad(wavfile.read(item)[1], (0, 32000), mode='constant')[aud_offset[i]:32000+aud_offset[i]], 
            caption=save_name_base[i], sample_rate=16000) for i, item in enumerate(samples_mix)]}, commit=True)


        #val_folders_pred_all = val_folders_pred_all[:100]

        '''_, snr = self.model.evaluate_generator(DataGenerator_val_unsync_attention(val_folders_pred_all, int(self.batch_size)), 
                            steps=int(np.ceil((len(val_folders_pred_all))/float(self.batch_size))), 
                            verbose=0)

        _, snr_easy = self.model.evaluate_generator(DataGenerator_val_unsync_attention_easy(val_folders_pred_all, int(self.batch_size)), 
                            steps=int(np.ceil((len(val_folders_pred_all))/float(self.batch_size))), 
                            verbose=0)'''

        '''_, snr_vox = self.model.evaluate_generator(DataGenerator_val_unsync_attention_easy(val_folders_pred_all, int(self.batch_size)), 
                            steps=int(np.ceil((len(val_folders_pred_all))/float(self.batch_size))), 
                            verbose=0)'''
        
        '''folders_list_val_all = np.loadtxt('/data/AV-speech-separation1/lrs2_comb2_val_snr_filter.txt', dtype='object').tolist()
        random.seed(1234)
        val_folders_pred_all1 = random.sample(folders_list_val_all, 5000)
        _, snr_2speak = self.model.evaluate_generator(DataGenerator_val_samples(val_folders_pred_all1, int(self.batch_size), norm=1),
                            steps=int(np.ceil((len(val_folders_pred_all1))/float(self.batch_size))), 
                            verbose=0)

        print('\nSNR 2 speak:', snr_2speak)'''

        #SNR
        self.val_snr.append(0)

        '''print('\nUnsync SNR Hard:', snr, ' - Unsync SNR Easy:', snr_easy)

        wandb.log({'UnSync-SNR-hard':snr, 'UnSync-SNR-easy':snr_easy})'''

        '''with open(self.save_path, "a") as myfile:
            myfile.write(', SNR 2 speak: ' + str(snr_2speak) + '\n')'''
        return

#[np.random.rand(12, 50, 50, 100, 1), np.random.rand(12, 32000, 1), np.random.rand(12, 256), np.random.rand(12, 256), np.random.rand(12, 512)]

#print('Preds:', pred_weights.shape)
#print('preds_mask:', preds_mask.shape)
#print('preds_audio:', preds_audio.shape)

#wandb.log({"Attention Weights Alignment": [wandb.Image(i, caption="Attention_align") for i in atten_states]}, commit=False)


'''mix_aud = np.pad(wavfile.read(samples_mix[i])[1], (0, 32000), mode='constant')[aud_offset:32000+aud_offset]
wandb.log({"Mixed Audio": wandb.Audio(mix_aud, caption=save_name_base, sample_rate=16000)}, commit=False)
print('done1')

aud = np.pad(np.load(samples[i]), (0, 32000), mode='constant')[aud_offset:32000+aud_offset]
wandb.log({"True Audio": wandb.Audio(aud, caption=save_name_base, sample_rate=16000)}, commit=False)
print('done2')

aud_pred = preds_audio[i]
wandb.log({"Predicted Audio": wandb.Audio(aud_pred, caption=save_name_base, sample_rate=16000)}, commit=False)
print('done3')

#vid = get_video_frames(lips[i], fmt= 'grey')
#vid = crop_pad_frames(frames = vid, fps = 25, seconds = 2)
#wandb.log({"Lip Videos": wandb.Video(vid, fps=25, format="mp4", caption=save_name_base)}, commit=False)

mask = preds_mask[i]
#h_mask = heatmap(mask, cbar=False)
wandb.log({"Mask Predictions": wandb.Image(mask*10000, caption=save_name_base)}, commit=False)
print('done4')

weights = pred_weights[i]
#h_weights = heatmap(weights, cbar=False)
wandb.log({"Attention Weights Alignment": wandb.Image(weights*1000000, caption=save_name_base)}, commit=False)

print('one loop done')'''
