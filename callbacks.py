from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, ReduceLROnPlateau, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import glob
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve, average_precision_score, accuracy_score
import sys


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_f1s_weigh = []
        #self.val_recalls = []
        #self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.asarray(self.model.predict(x_val))
        val_targ = y_val
        #pred = val_predict
        pred_all_binary = []
        for i in range(val_predict.shape[0]):
            pred1 = np.argmax(val_predict[i])
            pred_all_binary.append(pred1)
            val_predict1 = np.asarray(pred_all_binary, dtype = 'int32')

        targ_all_binary = []
        for i in range(val_targ.shape[0]):
            targ1 = np.argmax(val_targ[i])
            targ_all_binary.append(targ1)
            val_targ1 = np.asarray(targ_all_binary, dtype = 'int32')

        val_predict = val_predict1
        val_targ = val_targ1
        _val_f1 = f1_score(val_targ, val_predict)
        _val_f1_weigh = f1_score(val_targ, val_predict, average='weighted')
        #_val_recall = recall_score(val_targ, val_predict)
        #_val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_f1s_weigh.append(_val_f1_weigh)
        #self.val_recalls.append(_val_recall)
        #self.val_precisions.append(_val_precision)
#        print '\n'
        print ('Validation f1: ', _val_f1)
        print ('Weighted validation f1: ', _val_f1_weigh)
        #, '_val_precision: ', _val_precision, '_val_recall', _val_recall
        return
    
#metrics = Metrics()

class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """

    def __init__(self):
        Callback.__init__(self)


    def on_epoch_end(self, epoch, logs={}):
        msg = "Epoch: %i, %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.iteritems()))
        print(msg)
        
#loggingcallback = LoggingCallback(

def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.1
    epochs_drop = 10
    lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
    return lrate

def learningratescheduler():
    learningratescheduler = LearningRateScheduler(step_decay)
    return learningratescheduler

def earlystopping():
    earlystopping = EarlyStopping(monitor='val_loss', patience=10)
    return earlystopping

def reducelronplateau():
    reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr = 0.00000001)
    return reducelronplateau

    def __init__(self, logsdir):
        self.terminal = sys.stdout
        self.log = open(os.path.join(logsdir, 'log.txt'), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class LearningRateSchedulerPerBatch(LearningRateScheduler):
    """ Callback class to modify the default learning rate scheduler to operate each batch"""
    def __init__(self, schedule, verbose=0):
        super(LearningRateSchedulerPerBatch, self).__init__(schedule, verbose)
        self.count = 0  # Global batch index (the regular batch argument refers to the batch index within the epoch)

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        super(LearningRateSchedulerPerBatch, self).on_epoch_begin(self.count, logs)

    def on_batch_end(self, batch, logs=None):
        super(LearningRateSchedulerPerBatch, self).on_epoch_end(self.count, logs)
        self.count += 1

class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """


    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn


    def on_epoch_end(self, epoch, logs={}):

        msg = "{Epoch: %i} %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.print_fcn(msg)
