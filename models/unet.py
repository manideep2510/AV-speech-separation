# Ignore warnings
import os
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from keras.layers import *
from keras import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers.core import Lambda
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, ReduceLROnPlateau, EarlyStopping, ReduceLROnPlateau
#from callbacks import Metrics, learningratescheduler, earlystopping, reducelronplateau
#from plotting import plot_loss_and_acc
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# Model

class VideoModel():

    def __init__(self, filters,filters_audio, audio_ip_shape, pretrain):

        self.filters = filters
        self.filters_audio=filters_audio
        self.audio_ip_shape = audio_ip_shape
        self.pretrain = pretrain

        self.conv_transpose = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]), name='lambda3_unet')

    def FullModel(self):
      #  import tensorflow as tf
        ip = Input(shape = (self.audio_ip_shape[0], self.audio_ip_shape[1], 2), name = 'spect_input_unet')#; print("input_audio", ip.shape)
        input_spects = Lambda(lambda x : x, name='lambda_input_spects_unet')(ip)
        print('input_spects', input_spects.shape)
        
#        ip_magnitude = Lambda(lambda x : x[:,:,:,0],name="ip_mag")(ip)#; print("ip_mag ", ip_magnitude.shape)  #takes magnitude from stack[magnitude,phase]
#        ip_phase = Lambda(lambda x : tf.expand_dims(x[:,:,:,1], axis = -1),name="ip_phase")(ip)#; print("ip_phase ", ip_phase.shape)  #takes phase from stack[magnitude,phase]

        conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(input_spects)
        conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(pool1)
        conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(pool2)
        conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(266, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(pool3)
        conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        # Bottom of the U-Net
        conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(pool4)
        conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv5)
        conv5 = BatchNormalization()(conv5)
        drop5 = Dropout(0.5)(conv5)

        # Upsampling Starts, right side of the U-Net
        up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(merge6)
        conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv6)
        conv6 = BatchNormalization()(conv6)
        print('conv6', conv6.shape)

        up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(UpSampling2D(size = (2,2))(conv6))
        print('up7', up7.shape)
        # Padding to match concat chape
        up7_pad = ZeroPadding2D(padding=((0, 0), (0, 1)))(up7)
        merge7 = concatenate([conv3,up7_pad], axis = 3)
        conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(merge7)
        conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(merge8)
        conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(UpSampling2D(size = (2,2))(conv8))
        deconv9 = Conv2DTranspose(16, (2,1), strides=(1, 1), padding='valid', activation = 'relu')(up9)
        print('deconv9', deconv9.shape)
        merge9 = concatenate([conv1,deconv9], axis = 3)
        conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(merge9)
        conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv9)    ## Shape = (None, 256, 500, 16)
        print('conv9', conv9.shape)
        #conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv9)
        #deconv9 = Conv2DTranspose(16, (2,1), strides=(1, 1), padding='valid', activation = 'relu')(conv9)
        #print('deconv9', deconv9.shape)
        conv9 = BatchNormalization()(deconv9)

        # Output layer of the U-Net with 8 channels
        conv10 = Conv2D(2, 1, activation = 'sigmoid', padding = 'same')(conv9)
        print('conv10', conv10.shape)

        model = Model(inputs = ip, outputs = conv10)

        if self.pretrain == 'pretrain':
            model.load_weights('/data/models/pretrain_unet_236k-train_1to3ratio_epochs20_lr1e-4_0.32decay5epochs/weights-16-397.1351.hdf5')
            print('Loaded pretrained weights for UNet')

        return model
