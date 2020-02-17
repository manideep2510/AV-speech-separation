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
        x = Mish('Mish')(x)
        x = BatchNormalization()(x)
        x = SeparableConv1D(filters,kernel_size,dilation_rate=dialation_rate,padding='same',strides=stride)(x)
        x = Mish('Mish')(x)
        x = BatchNormalization()(x)
        x = Conv1D(int(inputs.shape[-1]),1)(x)
        x = Add()([inputs,x])
        return x


def Conv_Block_Audio(inputs,dialation_rate=1,stride=1,filters=512,kernel_size=3):
    
    x = Conv1D(filters,1)(inputs)
    x = Mish('Mish')(x)
    x = BatchNormalization()(x)
    x = SeparableConv1D(filters,kernel_size,dilation_rate=dialation_rate,padding='same',strides=stride)(x)
    x = Add()([inputs,x])
    
    return x

def Conv_Block_Video(inputs,dialation_rate=1,stride=1,filters=512,kernel_size=3):
    
    
    x = Mish('Mish')(inputs)
    x = BatchNormalization()(x)
    x = SeparableConv1D(filters,kernel_size,dilation_rate=dialation_rate,padding='same',strides=stride)(x)
    x = Add()([inputs,x])
    
    return x


# Actual loss calculation
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # From Keras example image_ocr.py:
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # y_pred = y_pred[:, 2:, :]
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# CTC Layer implementation using Lambda layer
# (because Keras doesn't support extra prams on loss function)
def CTC(name, args):
    return Lambda(ctc_lambda_func, output_shape=(1,), name=name)(args)

## Co-learning model

class TasNet_cotrain(object):
    
    def __init__(self,time_dimensions=500,frequency_bins=257,n_frames=125, absolute_max_string_len=128, attention=None,lipnet_pretrained=None, train_lipnet=None):
        
        self.t=time_dimensions
        self.f=frequency_bins
        self.frames=n_frames
        self.lipnet_pretrained = lipnet_pretrained
        self.train_lipnet = train_lipnet
        self.attention = attention
        self.absolute_max_string_len = absolute_max_string_len
        self.build()
        
    def build(self):
        
        
        #self.video_input_data=Input(shape=(self.frames,))#video_shape=(125,256)
        self.audio_input_data=Input(shape=(self.t*160,1))#audio_shape=(80000,1)
        
        #audio_encoding
        self.audio=Conv1D(256,40,padding='same',strides=40, activation='relu')(self.audio_input_data)
        self.audio=Conv1D(256,16,padding='same',strides=16, activation='relu')(self.audio)
        
        #video_processing

        self.lipnet_model = lipreading(mode='backendGRU', inputDim=256, hiddenDim=512, nClasses=29, frameLen=self.frames, AbsoluteMaxStringLen=128, every_frame=True, pretrain=self.lipnet_pretrained)

        if self.train_lipnet == False:
            for layer in self.lipnet_model.layers:
                layer.trainable = False

        self.outv1 = self.lipnet_model.layers[-4].output
        self.outv = Dense(128, kernel_initializer='he_normal', name='dense2')(self.outv1)
        self.outv = Dense(256, kernel_initializer='he_normal', name='dense3')(self.outv)
        #self.outv = GlobalAveragePooling2D(self.outv)
        
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
        
        #self.outv=UpSampling1D(size=4)(self.outv)

        print('outv:', self.outv.shape)
        print('outa:', self.outa.shape)

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
        self.mask=Conv1D(256,1,activation='relu')(self.fusion)
        
        self.outv1_gru = Bidirectional(tf.keras.layers.GRU(512, return_sequences=True, kernel_initializer='Orthogonal', reset_after=False, name='gru1'), merge_mode='concat')(self.outv1)
        print('GRU1:', self.outv1_gru.shape)
        self.outv1_gru = Bidirectional(tf.keras.layers.GRU(512, return_sequences=True, kernel_initializer='Orthogonal', reset_after=False, name='gru2'), merge_mode='concat')(self.outv1_gru)
        print('GRU2:', self.outv1_gru.shape)
        self.outv1 = Dense(256)(self.outv1_gru) 
        print('Outv1:', self.outv1.shape)
        
        ## Apply some kind of attention b/w mask and outv1 and output a new mask
        
        self.outv1 = concatenate([self.mask,self.outv1],axis=-1)
        self.mask_new = Conv1D(256,1,activation='relu')(self.outv1)
        
        print('New mask:', self.mask_new.shape)
        
        self.outv_classes = Dense(29)(self.outv1) # shape = (200,29)
        print('Outv classes:', self.outv_classes.shape)

        self.labels = Input(name='the_labels', shape=[self.absolute_max_string_len], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')

        self.loss_out = CTC('ctc', [self.outv_classes, self.labels, self.input_length, self.label_length])
        
        self.mul=Multiply()([self.audio,self.mask_new])
        self.decode = Lambda(lambda x: K.expand_dims(x, axis=2))(self.mul)
        self.decode=Conv2DTranspose(256,(16,1),strides=(16,1),padding='same',data_format='channels_last')(self.decode)
        self.decode=Conv2DTranspose(1,(40,1),strides=(40,1),padding='same',data_format='channels_last')(self.decode)
        self.out = Lambda(lambda x: K.squeeze(x, axis=2), name='speech_out')(self.decode)

        self.model=Model(inputs=[self.lipnet_model.input,self.audio_input_data, self.labels, self.input_length, self.label_length],outputs=[self.out, self.loss_out])
        #self.model=Model(inputs=[self.lipnet_model.input,self.audio_input_data],outputs=[self.out])
        return self.model
    
    def summary(self):
        Model(inputs=[self.lipnet_model.input, self.audio_input_data, self.labels, self.input_length, self.label_length], outputs=[self.out, self.loss_out]).summary()
        
    def predict(self, input_batch):
        return self.test_function([input_batch, 0])[0]  # the first 0 indicates test

    @property
    def test_function(self):
        # captures output of softmax so we can decode the output during visualization
        return K.function([[self.lipnet_model.input, self.audio_input_data], K.learning_phase()], [self.out, self.outv_classes])