# Ignore warnings
import os
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda

from .resnet_lstm_lipread import lipreading
from .attention_layers import AttentionLayer
from .mish import Mish

def custom_tanh(x):
    
    #Cx=K*tf.math.divide(1-tf.math.exp(-1*C*x),1+tf.math.exp(-1*C*x))
    Cx = tf.math.tanh(x)
  
    Cx = 0.9999999*tf.dtypes.cast((Cx>0.9999999), dtype=tf.float32)+Cx*tf.dtypes.cast((Cx<=0.9999999), dtype=tf.float32)
    Cy = -0.9999999*tf.dtypes.cast((Cx<-0.9999999), dtype=tf.float32)+Cx*tf.dtypes.cast((Cx>=-0.9999999), dtype=tf.float32)
    
    return Cy

def Conv_Block(inputs,dialation_rate=1,stride=1,filters=512,kernel_size=3):
    
        x = Conv1D(filters,1)(inputs)
        #x = Mish('Mish')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = SeparableConv1D(filters,kernel_size,dilation_rate=dialation_rate,padding='same',strides=stride)(x)
        #x = Mish('Mish')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(int(inputs.shape[-1]),1)(x)
        x = Add()([inputs,x])
        return x


def Conv_Block_Audio(inputs,dialation_rate=1,stride=1,filters=512,kernel_size=3):
    
    x = Conv1D(filters,1)(inputs)
    #x = Mish('Mish')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = SeparableConv1D(filters,kernel_size,dilation_rate=dialation_rate,padding='same',strides=stride)(x)
    x = Add()([inputs,x])
    
    return x

def Conv_Block_Video(inputs,dialation_rate=1,stride=1,filters=512,kernel_size=3):
    
    
    #x = Mish('Mish')(inputs)
    x = Activation('relu')(inputs)
    x = BatchNormalization()(x)
    x = SeparableConv1D(filters,kernel_size,dilation_rate=dialation_rate,padding='same',strides=stride)(x)
    x = Add()([inputs,x])
    
    return x


class TasNet(object):
    
    def __init__(self,time_dimensions=500,frequency_bins=257,n_frames=125, attention=None, lstm = False,lipnet_pretrained=None, train_lipnet=None):
        
        self.t=time_dimensions
        self.f=frequency_bins
        self.frames=n_frames
        self.lipnet_pretrained = lipnet_pretrained
        self.train_lipnet = train_lipnet
        self.attention = attention
        self.lstm = lstm
        self.build()
        
    def build(self):
        
        
        #self.video_input_data=Input(shape=(self.frames,))#video_shape=(125,256)
        self.audio_input_data=Input(shape=(self.t*160,1))#audio_shape=(80000,1)
        
        #audio_encoding
        self.audio=Conv1D(256,40,padding='same',strides=20, activation='relu')(self.audio_input_data)
        self.audio=Conv1D(256,16,padding='same',strides=8, activation='relu')(self.audio)
        
        #video_processing

        self.lipnet_model = lipreading(mode='backendGRU', inputDim=256, hiddenDim=512, nClasses=29, frameLen=self.frames, AbsoluteMaxStringLen=128, every_frame=True, pretrain=self.lipnet_pretrained)

        if self.train_lipnet == False:
            for layer in self.lipnet_model.layers:
                layer.trainable = False

        if self.lstm == True:
            self.outv = self.lipnet_model.output
            self.outv = Dense(128, kernel_initializer='he_normal', name='dense2')(self.outv)
            self.outv = Dense(256, kernel_initializer='he_normal', name='dense3')(self.outv)
            #self.outv = GlobalAveragePooling2D(self.outv)
        else:
            self.outv = self.lipnet_model.layers[-4].output
        
        #video_processing
        self.video_data=Conv1D(512,1)(self.outv)
        self.outv=Conv_Block_Video(self.video_data,dialation_rate=1)
        self.outv=Conv_Block_Video(self.outv,dialation_rate=2)
        self.outv=Conv_Block_Video(self.outv,dialation_rate=4)
        self.outv=Conv_Block_Video(self.outv,dialation_rate=8)
        self.outv=Conv_Block_Video(self.outv,dialation_rate=16)
        self.outv=Conv1D(256,1)(self.outv)
        
        #audio_processing
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

        print('outv:', self.outv.shape)
        print('outa:', self.outa.shape)

        if self.attention == True:
            self.attn_layer = AttentionLayer(name='attention_layer')
            self.attn_out, self.attn_states = self.attn_layer([self.outv, self.outa], verbose=False)
            print('attn_out:', self.attn_out.shape)
            print('attn_states:', self.attn_states.shape)

            self.fusion=concatenate([self.attn_out, self.outv, self.outa],axis=-1)
            self.fusion=Conv1D(512,1)(self.fusion)
            
        else:
            self.fusion=concatenate([self.outv,self.outa],axis=-1)
        
        print('fusion:', self.fusion.shape)
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
        
        #Decoding
        self.fusion=Conv1D(256,1,activation='relu')(self.fusion)
        self.fusion=Multiply()([self.audio,self.fusion])
        self.decode = Lambda(lambda x: K.expand_dims(x, axis=2))(self.fusion)
        self.decode=Conv2DTranspose(256,(16,1),strides=(8,1),padding='same',data_format='channels_last')(self.decode)
        self.decode=Conv2DTranspose(1,(40,1),strides=(20,1),padding='same',data_format='channels_last')(self.decode)
        self.out = Lambda(lambda x: K.squeeze(x, axis=2), dtype='float32')(self.decode)
  
        
        
        self.model=Model(inputs=[self.lipnet_model.input,self.audio_input_data],outputs=[self.out])
