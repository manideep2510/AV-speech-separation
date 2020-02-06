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
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

from .resnet_lstm_lipread import lipreading
from .attention_layers import AttentionLayer,Luong,MinimalRNN
from .layers import Attention
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


class TasNet(object):
    
    def __init__(self,time_dimensions=500,frequency_bins=257,n_frames=125, attention=None,lipnet_pretrained=None, train_lipnet=None):
        
        self.t=time_dimensions
        self.f=frequency_bins
        self.frames=n_frames
        self.lipnet_pretrained = lipnet_pretrained
        self.train_lipnet = train_lipnet
        self.attention = attention
        self.build()
        
    def build(self):
        
        
        #self.video_input_data=Input(shape=(self.frames,))#video_shape=(125,256)
        self.audio_input_data=Input(shape=(self.t*160,1))#audio_shape=(80000,1)
        self.initial_cell_state = Input(shape=(256,), name='cell_state')
        self.initial_hidden_state = Input(shape=(256,), name='hidden_state')
        hidden_state, cell_state = self.initial_hidden_state, self.initial_cell_state
        
        #audio_encoding
        self.audio=Conv1D(256,40,padding='same',strides=20, activation='relu')(self.audio_input_data)
        self.audio=Conv1D(256,16,padding='same',strides=8, activation='relu')(self.audio)
        
        #video_processing

        self.lipnet_model = lipreading(mode='backendGRU', inputDim=256, hiddenDim=512, nClasses=29, frameLen=self.frames, AbsoluteMaxStringLen=128, every_frame=True, pretrain=self.lipnet_pretrained)

        if self.train_lipnet == False:
            for layer in self.lipnet_model.layers:
                layer.trainable = False

        self.outv = self.lipnet_model.output
        self.outv = Dense(128, kernel_initializer='he_normal', name='dense2')(self.outv)
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
        
        self.outv=UpSampling1D(size=4)(self.outv)

        print('outv:', self.outv.shape)
        print('outa:', self.outa.shape)
        print('hidden',hidden_state.shape)
        if self.attention == True:
            

            decoder_recurrent_layer =LSTM(units=256,return_state=True)
            #decoder_recurrent_layer = MinimalRNN(256)

            #attention_layer=Attention(context='many-to-many', alignment_type='global',score_function='general')
            attention_layer=Luong(name='attention_layer')
            outputs = []
            weights = []
            for timestep in range(self.t):
        
    
                #context_vector, attn_states = attention_layer([self.outa,hidden_state,timestep])
                context_vector, attn_states = attention_layer([self.outa,hidden_state])
                current_word=Lambda(lambda x:K.expand_dims(x[:,timestep,:],axis=1))(self.outv)
                #current_word=Lambda(lambda x:x[:,timestep,:])(self.outv)
                #decoder_input = Concatenate(axis=2)([K.expand_dims(context_vector,axis=1), current_word])
                #decoder_input = Concatenate(axis=1)([context_vector, current_word])
                #print('dec',decoder_input.shape)
                #hidden_state=decoder_recurrent_layer(current_word,states=[hidden_state,cell_state])
                output,hidden_state, cell_state= decoder_recurrent_layer(current_word,initial_state=[hidden_state, cell_state])
                #print('hid',hidden_state.shape)
                #print('cell',cell_state[0].shape)
                #print('cell',cell_state[1].shape)
                outputs.append(context_vector)
                weights.append(attn_states)
        
       
            self.attn_out=tf.stack(outputs,axis=1)
#            self.attn_weights=K.squeeze(tf.stack(weights,axis=1),axis=-1)
            self.attn_weights=tf.stack(weights,axis=1)

#            alphas=Lambda(lambda x:x,name='attention_weights')(self.attn_weights)
            print('attn_out:', self.attn_out.shape)
            print('attn_states:', self.attn_weights.shape)

            self.fusion1=concatenate([self.attn_out, self.outa],axis=-1)
            #self.fusion=Conv1D(512,1)(self.fusion)
            
        else:
            self.fusion=concatenate([self.outv,self.outa],axis=-1)
        
        print('fusion:', self.fusion1.shape)
        self.fusion=Conv_Block_Audio(self.fusion1,dialation_rate=1,filters=512)
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
        self.out = Lambda(lambda x: K.squeeze(x, axis=2))(self.decode)
  
        
        
        self.model=Model(inputs=[self.lipnet_model.input,self.audio_input_data,self.initial_hidden_state, self.initial_cell_state],outputs=[self.out])
