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
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from .lipnet import LipNet
from .unet import VideoModel as VideoModel_unet

'''
Usage: 

model = VideoModel(256,96,(257,500,2),(125,50,100,3)).FullModel(lipnet_pretrained = True)

'''


# Model

class VideoModel():

    def __init__(self, filters,filters_audio, audio_ip_shape, video_ip_shape):
        
        self.filters = filters
        self.filters_audio=filters_audio       
        self.audio_ip_shape = audio_ip_shape
        self.video_ip_shape = video_ip_shape

        self.conv1 = Conv2D(filters = filters, kernel_size = (7), padding = "same", dilation_rate = (1,1),
                      activation = "relu")
        self.bn1 = BatchNormalization(axis=-1)

        self.conv2 = Conv2D(filters = filters, kernel_size = (5), padding = "same", dilation_rate = (1,1),
                      activation = "relu")
        self.bn2 = BatchNormalization(axis=-1)

        self.conv3 = Conv2D(filters = filters, kernel_size = (5), padding = "same", dilation_rate = (2,2),
                      activation = "relu")
        self.bn3 = BatchNormalization(axis=-1)

        self.conv4 = Conv2D(filters = filters, kernel_size = (5), padding = "same", dilation_rate = (4,4),
                      activation = "relu")
        self.bn4 = BatchNormalization(axis=-1)

        self.conv5 = Conv2D(filters = filters, kernel_size = (5), padding = "same", dilation_rate = (8,8),
                      activation = "relu")
        self.bn5 = BatchNormalization(axis=-1)

        self.conv6 = Conv2D(filters = filters, kernel_size = (5), padding = "same", dilation_rate = (16,16),
                      activation = "relu")
        self.bn6 = BatchNormalization(axis=-1)
    
        #import tensorflow as tf
        self.conv7 = Lambda(lambda x : tf.expand_dims(x, axis = -1), name='lambda1')

        self.conv8 = Lambda(lambda x: tf.image.resize_nearest_neighbor(x, size = (500, x.shape[-2])), name='lambda2')
        self.conv_transpose = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]), name='lambda3')
        
    def FullModel(self, lipnet_pretrained, unet_pretrained):
      #  import tensorflow as tf
        #ip = Input(shape = (self.audio_ip_shape[0], self.audio_ip_shape[1], 2), name = 'spect_input')#; print("input_audio", ip.shape) 
        #input_spects = Lambda(lambda x : x, name='lambda_input_spects')(ip)
        #print('input_spects', input_spects.shape)
        #ip_embeddings_1 = Input(shape = (int(self.video_ip_shape[0]), int(self.video_ip_shape[1]),int(self.video_ip_shape[2]), int(self.video_ip_shape[3])))#; print("ip video", ip_embeddings_1.shape)  #[75, 512]
        ip_samples = Input(shape = (128500,))
        input_samples = Lambda(lambda x : x, name='lambda_input_samples')(ip_samples)
        print('input_samples', input_samples.shape)
        input_samples = Reshape([self.audio_ip_shape[0], self.audio_ip_shape[1], 1])(input_samples)
        print('input_samples_reshape', input_samples.shape)
#        ip_magnitude = Lambda(lambda x : x[:,:,:,0],name="ip_mag")(ip)#; print("ip_mag ", ip_magnitude.shape)  #takes magnitude from stack[magnitude,phase]
#        ip_phase = Lambda(lambda x : tf.expand_dims(x[:,:,:,1], axis = -1),name="ip_phase")(ip)#; print("ip_phase ", ip_phase.shape)  #takes phase from stack[magnitude,phase]

        #ip_embeddings_1_expanded = Lambda(lambda x : tf.expand_dims(x, axis = -1))(ip_embeddings_1)

        unet = VideoModel_unet(256,96,(257,500,2), pretrain=unet_pretrained).FullModel()
        input_spects = unet.input
        print('input_spects', input_spects.shape)
        conv9 = unet.layers[-2].output
        print('Conv9- pretrained UNet output:', conv9.shape)
        
        # Output layer of the U-Net with 8 channels
        conv10 = Conv2D(8, 1, activation = 'relu', padding = 'same')(conv9) 
        print('conv10', conv10.shape)

        audio_stream = BatchNormalization(axis=-1)(conv10)
        audio_stream = self.conv_transpose(audio_stream)
        print('audio_stream', audio_stream.shape)

        
        lipnet_model = LipNet(input_shape = self.video_ip_shape, pretrained=lipnet_pretrained)
        x = lipnet_model.output
        print("lipnet_model ", x.shape)
        x = Dense(128, kernel_initializer='he_normal', name='dense2')(x)
        x = Dense(256, kernel_initializer='he_normal', name='dense3')(x)
        x = self.conv7(x)
        video_stream_1 = self.conv8(x)
        print("video_stream_1 ", video_stream_1.shape)

        audio_flatten = TimeDistributed(Flatten())(audio_stream) 
        print(audio_flatten.shape)
        video_flatten_1 = TimeDistributed(Flatten())(video_stream_1)
        print("video_flatten_1 ", video_flatten_1.shape)

        concated = concatenate([audio_flatten, video_flatten_1], axis = 2) 
        print("concat shape ", concated.shape)

        lstm = Bidirectional(LSTM(units = 64, return_sequences = True, activation = "tanh"))(concated)   

        flatten = Flatten()(lstm) 

        dense = Dense(100, activation = "relu")(flatten)

        dense = Dense(self.audio_ip_shape[0] * self.audio_ip_shape[1], activation = 'sigmoid')(dense) 

        mask = Reshape([self.audio_ip_shape[0], self.audio_ip_shape[1], 1])(dense)
        print("mask", mask.shape)
        output_mask_specs = concatenate([mask, input_spects], axis=3, name='concat1')
        print('output_mask_specs', output_mask_specs.shape)
        output_mask_specs_samples = concatenate([output_mask_specs, input_samples], axis=3, name='concat2') 
        print('output_mask_specs_samples', output_mask_specs_samples.shape)
#        mask = Lambda(lambda x : x[:,0], name='lambda_out')(combo_mask) 

#        output_mag_1 = Lambda(lambda x : tf.multiply(x[0], x[1]), name = "mask_multiply_1")([ip_magnitude, mask_1])#; print("output_mag_1", output_mag_1.shape)

#        output_mag_1 = Lambda(lambda x : tf.expand_dims(x, axis= -1), name= "expand_dim_1")(output_mag_1)#; print("output_mag_expand_1", output_mag_1.shape)

#        output_final_1 = Lambda(lambda x : tf.concat(values=[x[0], x[1]], axis = -1),name="concat_mag_phase_1")([output_mag_1, ip_phase]) 
        
        model = Model(inputs = [unet.input, lipnet_model.input, ip_samples], outputs = output_mask_specs_samples)

        return model
    
    
