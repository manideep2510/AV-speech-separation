#pip3 install mir_eval
#target_samples and predicted_samples shoud have same size.

#returns mean_sdr,sdr_values
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, ReduceLROnPlateau, EarlyStopping, ReduceLROnPlateau
from mir_eval.separation import bss_eval_sources

from audio_utils import retrieve_samples
import numpy as np

import tensorflow as tf
from keras import backend as K
from data_generators import DataGenerator_test

def metric_eval(target_samples, predicted_samples):
	
    sdr_batch=[]
	
    batch_size=(target_samples.shape)[0]
    for i in range(batch_size):
        sdr, sir, sar, _ = bss_eval_sources(target_samples[i], predicted_samples[i], compute_permutation=False)
        sdr_batch.append(sdr[0])
    
    sdr_batch = np.asarray(sdr_batch)

    return np.mean(sdr_batch),sdr_batch


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
        #self.val_f1s_weigh = []
        #self.val_recalls = []
        #self.val_precisions = []
 
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
            #val_predict = val_predict1
            #val_targ = val_targ1
            #_val_f1 = f1_score(val_targ, val_predict)
            #_val_f1_weigh = f1_score(val_targ, val_predict, average='weighted')
            #_val_recall = recall_score(val_targ, val_predict)
            #_val_precision = precision_score(val_targ, val_predict)
         
            _val_sdr1, _ = metric_eval(target_samples = val_targ, predicted_samples = samples_pred)
            sdr_list.append(_val_sdr1)

        sdr_list = np.asarray(sdr_list)
        _val_sdr = np.mean(sdr_list)
        self.val_sdr.append(_val_sdr)
        #self.val_f1s_weigh.append(_val_f1_weigh)
        #self.val_recalls.append(_val_recall)
        #self.val_precisions.append(_val_precision)
#        print '\n'
        print('Validation SDR: ', _val_sdr)
        #print('Weighted validation f1: ', _val_f1_weigh)
        #, '_val_precision: ', _val_precision, '_val_recall', _val_recall
        return

class Metrics_softmask(Callback):

    def __init__(self, model, val_folders, batch_size):
        self.model = model
        self.val_folders = val_folders
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.val_sdr = []
        #self.val_f1s_weigh = []
        #self.val_recalls = []
        #self.val_precisions = []
 
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
            #val_predict = val_predict1
            #val_targ = val_targ1
            #_val_f1 = f1_score(val_targ, val_predict)
            #_val_f1_weigh = f1_score(val_targ, val_predict, average='weighted')
            #_val_recall = recall_score(val_targ, val_predict)
            #_val_precision = precision_score(val_targ, val_predict)
         
            _val_sdr1, _ = metric_eval(target_samples = val_targ, predicted_samples = samples_pred)
            sdr_list.append(_val_sdr1)

        sdr_list = np.asarray(sdr_list)
        _val_sdr = np.mean(sdr_list)
        self.val_sdr.append(_val_sdr)
        #self.val_f1s_weigh.append(_val_f1_weigh)
        #self.val_recalls.append(_val_recall)
        #self.val_precisions.append(_val_precision)
#        print '\n'
        print('Validation SDR: ', _val_sdr)
        #print('Weighted validation f1: ', _val_f1_weigh)
        #, '_val_precision: ', _val_precision, '_val_recall', _val_recall
        return


