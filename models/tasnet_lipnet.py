# Ignore warnings
import os
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import keras.backend as K

from keras.layers import *
from keras.activations import *
from keras.models import Model
from keras.models import load_model
from keras.layers.core import Lambda

from .lipnet import LipNet

def custom_tanh(x,K=1,C=2):
    
    Cx=K*tf.math.divide(1-tf.math.exp(-1*C*x),1+tf.math.exp(-1*C*x))
    Cy=tf.keras.backend.switch(Cx>0.9999999,tf.constant(0.9999999, dtype=tf.float32),Cx)
    
    return Cy

def Conv_Block(inputs,dialation_rate=1,stride=1,filters=512,kernel_size=3):
    
        x = Conv1D(filters,1)(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = SeparableConv1D(filters,kernel_size,dilation_rate=dialation_rate,padding='same',strides=stride)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(int(inputs.shape[-1]),1)(x)
        x = Add()([inputs,x])
        return x


def Conv_Block_Audio(inputs,dialation_rate=1,stride=1,filters=512,kernel_size=3):
    
    x = Conv1D(filters,1)(inputs)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = SeparableConv1D(filters,kernel_size,dilation_rate=dialation_rate,padding='same',strides=stride)(x)
    x = Add()([inputs,x])
    
    return x

def Conv_Block_Video(inputs,dialation_rate=1,stride=1,filters=512,kernel_size=3):
    
    
    x = Activation('relu')(inputs)
    x = BatchNormalization()(x)
    x = SeparableConv1D(filters,kernel_size,dilation_rate=dialation_rate,padding='same',strides=stride)(x)
    x = Add()([inputs,x])
    
    return x

class TasNet(object):
    
    def __init__(self,video_ip_shape,time_dimensions=500,frequency_bins=257,n_frames=125, lipnet_pretrained=None):
        
        self.video_ip_shape=video_ip_shape
        self.t=time_dimensions
        self.f=frequency_bins
        self.frames=n_frames
        self.lipnet_pretrained=lipnet_pretrained
        self.build()
        
    def build(self):
        
        self.ip_samples = Input(shape = (128500,))
        self.input_samples = Lambda(lambda x : x, name='lambda_input_samples')(self.ip_samples)
        print('input_samples', self.input_samples.shape)
        self.input_samples = Reshape([self.f, self.t, 1])(self.input_samples)
        print('input_samples_reshape', self.input_samples.shape)
        
        #self.video_input_data=Input(shape=(self.frames,256))#video_shape=(125,256)
        
        self.audio_input_data=Input(shape=(self.f,self.t,2),dtype='float32')#audio_shape=(257,500,2)
        self.audio_input = Lambda(lambda x : tf.transpose(x, [0, 2, 1, 3]))(self.audio_input_data) # Transpose
        self.audio_magnitude = Lambda(lambda x : x[:,:,:,0])(self.audio_input)
        self.audio_phase = Lambda(lambda x : x[:,:,:,1])(self.audio_input)
        self.audio=concatenate([self.audio_magnitude,self.audio_phase],axis=-1)
        self.audio=Conv1D(256,1)(self.audio)
    
        #video_processing

        self.lipnet_model = LipNet(input_shape = self.video_ip_shape, pretrained=self.lipnet_pretrained)
        self.outv = self.lipnet_model.output
        self.outv = Dense(128, kernel_initializer='he_normal', name='dense2')(self.outv)
        self.outv = Dense(256, kernel_initializer='he_normal', name='dense3')(self.outv)
        #self.outv = GlobalAveragePooling2D(self.outv)

        self.video_data=Conv1D(512,1)(self.outv)
        self.outv=Conv_Block_Video(self.video_data,dialation_rate=1)
        self.outv=Conv_Block_Video(self.outv,dialation_rate=2)
        self.outv=Conv_Block_Video(self.outv,dialation_rate=4)
        self.outv=Conv_Block_Video(self.outv,dialation_rate=8)
        self.outv=Conv_Block_Video(self.outv,dialation_rate=16)
        self.outv=Conv1D(256,1)(self.outv)
    
        #audio_processing
        #self.audio_data=Conv1D(256,1)(self.input_data)
        self.outa=Conv_Block(self.audio,dialation_rate=1)
        self.outa=Conv_Block(self.outa,dialation_rate=2)
        self.outa=Conv_Block(self.outa,dialation_rate=4)
        self.outa=Conv_Block(self.outa,dialation_rate=8)
        self.outa=Conv_Block(self.outa,dialation_rate=16)
        self.outa=Conv_Block(self.outa,dialation_rate=32)
        self.outa=Conv_Block(self.outa,dialation_rate=64)
        self.outa=Conv_Block(self.outa,dialation_rate=128)
        
        #fusion_process
        
        self.outv=UpSampling1D(size=4)(self.outv)
        self.fusion=concatenate([self.outv,self.outa],axis=-1)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=1,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=2,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=4,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=8,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=16,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=32,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=64,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=128,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=1,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=2,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=4,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=8,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=16,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=32,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=64,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=128,filters=512)
        
        #prediction
        self.mag=Conv1D(257,1)(self.fusion)
        self.mag=Activation(custom_tanh)(self.mag)
        self.mag=Reshape([self.t,self.f,1], name='reshape_mag')(self.mag)
        self.mag = Lambda(lambda x : tf.transpose(x, [0, 2, 1, 3]))(self.mag)
        self.phase=Conv1D(257,1)(self.fusion)
        self.phase=Activation('tanh')(self.phase)
        self.phase=Reshape([self.t,self.f,1], name='reshape_phase')(self.phase)
        self.phase = Lambda(lambda x : tf.transpose(x, [0, 2, 1, 3]))(self.phase)

        self.output_concats = concatenate([self.mag, self.phase, self.audio_input_data, self.input_samples], axis=3, name='concat_maps')

        self.model=Model(inputs=[self.audio_input_data, self.lipnet_model.input, self.ip_samples],outputs=self.output_concats)
        
        return self.model
    
    def summary(self):
        Model(inputs=[self.audio_input_data, self.lipnet_model.input, self.ip_samples],outputs=self.output_concats).summary()
