# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from keras.layers import *
from keras import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, ReduceLROnPlateau, EarlyStopping, ReduceLROnPlateau
from callbacks import Metrics, learningratescheduler, earlystopping, reducelronplateau
from plotting import plot_loss_and_acc
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def LipNet(input_shape, pretrained=None, output_size = 28, absolute_max_string_len=32):
        
        '''if K.image_data_format() == 'channels_first':
            input_shape = (img_c, frames_n, img_w, img_h)
        else:
            input_shape = (frames_n, img_w, img_h, img_c)'''

        input_data = Input(name='the_input', shape=input_shape, dtype='float32')

        zero1 = ZeroPadding3D(padding=(1, 2, 2), name='zero1')(input_data)
        conv1 = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv1')(zero1)
        batc1 = BatchNormalization(name='batc1')(conv1)
        actv1 = Activation('relu', name='actv1')(batc1)
        drop1 = SpatialDropout3D(0.5)(actv1)
        maxp1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(drop1)

        zero2 = ZeroPadding3D(padding=(1, 2, 2), name='zero2')(maxp1)
        conv2 = Conv3D(64, (3, 5, 5), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv2')(zero2)
        batc2 = BatchNormalization(name='batc2')(conv2)
        actv2 = Activation('relu', name='actv2')(batc2)
        drop2 = SpatialDropout3D(0.5)(actv2)
        maxp2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(drop2)

        zero3 = ZeroPadding3D(padding=(1, 1, 1), name='zero3')(maxp2)
        conv3 = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv3')(zero3)
        batc3 = BatchNormalization(name='batc3')(conv3)
        actv3 = Activation('relu', name='actv3')(batc3)
        drop3 = SpatialDropout3D(0.5)(actv3)
        maxp3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(drop3)

        resh1 = TimeDistributed(Flatten())(maxp3)

        gru_1 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat')(resh1)
        gru_2 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'), merge_mode='concat')(gru_1)

        # transforms RNN output to character activations:
        dense1 = Dense(output_size, kernel_initializer='he_normal', name='dense1')(gru_2)

        y_pred = Activation('softmax', name='softmax')(dense1)

        #labels = Input(name='the_labels', shape=[absolute_max_string_len], dtype='float32')
        #input_length = Input(name='input_length', shape=[1], dtype='int64')
        #label_length = Input(name='label_length', shape=[1], dtype='int64')

        #loss_out = CTC('ctc', [y_pred, labels, input_length, label_length])

        model = Model(inputs=input_data, outputs=y_pred)
        
        if pretrained == True:
            model.load_weights('/Users/manideepkolla/Downloads/unseen-weights178.h5')
        return model