from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, ReduceLROnPlateau, EarlyStopping, ReduceLROnPlateau
from mir_eval.separation import bss_eval_sources

from data_preparation.audio_utils import retrieve_samples, compress_crm, inverse_crm, return_samples_complex
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from dataloaders import DataGenerator_test_samples
from pesq import pesq
import wandb
import random

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

        wandb.log({'loss':logs['loss'], 'val_loss':logs['val_loss'], 'lr':logs['lr'], 'SNR':logs['snr_acc'], 'Val_SNR':logs['val_snr_acc']})
        
        return