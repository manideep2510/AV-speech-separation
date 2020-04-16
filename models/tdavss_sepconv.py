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

from .sep_resnet_lstm_lipread import lipreading
from .attention_layers import AttentionLayer, multi_head_self_attention
from .mish import Mish

def custom_tanh(x):
    
    #Cx=K*tf.math.divide(1-tf.math.exp(-1*C*x),1+tf.math.exp(-1*C*x))
    Cx = tf.math.tanh(x)
  
    Cx = 0.9999999*tf.dtypes.cast((Cx>0.9999999), dtype=tf.float32)+Cx*tf.dtypes.cast((Cx<=0.9999999), dtype=tf.float32)
    Cy = -0.9999999*tf.dtypes.cast((Cx<-0.9999999), dtype=tf.float32)+Cx*tf.dtypes.cast((Cx>=-0.9999999), dtype=tf.float32)
    
    return Cy


def Conv_Block(inputs, dialation_rate=1, stride=1, filters=512, kernel_size=3):

    x = Conv1D(filters, 1)(inputs)
    #x = Mish('Mish')(x)
    x = Activation('relu')(x)
    x = GlobalLayerNorm()(x)
    x = SeparableConv1D(filters, kernel_size, dilation_rate=dialation_rate, padding='same', strides=stride)(x)
    #x = Mish('Mish')(x)
    x = Activation('relu')(x)
    x = GlobalLayerNorm()(x)
    x = Conv1D(int(inputs.shape[-1]), 1)(x)
    x = Add()([inputs, x])
    return x


def Conv_Block_Audio(inputs,dialation_rate=1,stride=1,filters=512,kernel_size=3):
    
    x = Conv1D(filters,1)(inputs)
    #x = Mish('Mish')(x)
    x = Activation('relu')(x)
    #x = BatchNormalization()(x)
    x= GlobalLayerNorm()(x)
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

def vec_l2norm(x):

    nr = K.sqrt(tf.math.reduce_sum(K.square(x), axis=1))
    nr = tf.reshape(nr, (-1, 1, 1))
    #nr = tf.broadcast_to(nr, (int(x.shape[1]), int(x.shape[0])))
    return nr

class GlobalLayerNorm(Layer):
    """Global Layer Normalization (gLN)"""
    def __init__(self,**kwargs):
        super(GlobalLayerNorm, self).__init__(**kwargs)
      
    def build(self,input_shape):
        
        self.gamma = self.add_weight(name='gamma',shape=(1,1,input_shape[2]),initializer='Ones',trainable='True')
        self.beta = self.add_weight(name='beta',shape=(1,1,input_shape[2]),initializer='Zeros',trainable='True')
        super(GlobalLayerNorm, self).build(input_shape)
        
    def call(self,inputs):
        
        mean = tf.math.reduce_mean((tf.math.reduce_mean(inputs,axis=2,keepdims=True)),axis=1,keepdims=True)
        var = tf.math.square((inputs-mean))
        var = tf.math.reduce_mean((tf.math.reduce_mean(var,axis=2,keepdims=True)),axis=1,keepdims=True)
        gln = self.gamma*(inputs-mean)*tf.math.rsqrt(var+1e-8)+self.beta
        return gln

class ChannelwiseLayerNorm(Layer):
    """Channel-wise Layer Normalization (cLN)"""
    def __init__(self,**kwargs):
        super(ChannelwiseLayerNorm, self).__init__(**kwargs)
    
    def build(self,input_shape):
        
        self.gamma=self.add_weight(name='gamma',shape=(1,1,input_shape[2]),initializer='Ones',trainable='True')
        self.beta=self.add_weight(name='beta',shape=(1,1,input_shape[2]),initializer='Zeros',trainable='True')
        super(ChannelwiseLayerNorm, self).build(input_shape)
        
    def call(self,inputs):
        
        mean = tf.math.reduce_mean(inputs,axis=2,keepdims=True)
        var = tf.math.square(tf.math.reduce_std(inputs,axis=2,keepdims=True))
        cln = self.gamma*(inputs-mean)*tf.math.rsqrt(var+1e-8)+self.beta
        return cln

class TasNet(object):
    
    def __init__(self,time_dimensions=500,frequency_bins=257,n_frames=125, attention=None, lstm = False,lipnet_pretrained=None, train_lipnet=None):
        
        self.t=time_dimensions
        self.f=frequency_bins
        self.frames=n_frames
        self.lipnet_pretrained = lipnet_pretrained
        self.train_lipnet = train_lipnet
        self.attention = attention
        self.lstm = lstm
        #self.positions=self.GetPosEncodingMatrix(self.t,512)
        self.build()
    
    '''def GetPosEncodingMatrix(self,max_len, d_emb):
        pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
            for pos in range(max_len)
            ])
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
        return pos_enc'''

    def build(self):
        
        
        #self.video_input_data=Input(shape=(self.frames,))#video_shape=(125,256)
        self.audio_input_data=Input(shape=(self.t*160,1))#audio_shape=(80000,1)

        '''self.norm = vec_l2norm(self.audio_input_data)
        self.audio_normalized = self.audio_input_data/self.norm'''
        
        #audio_encoding
        self.audio=Conv1D(256,40,padding='same',strides=20, activation='relu')(self.audio_input_data)
        #self.audio=Conv1D(256,16,padding='same',strides=8, activation='relu')(self.audio)
        self.audio1 = ChannelwiseLayerNorm()(self.audio)
        
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
        self.outv=Conv_Block_Video(self.video_data)
        self.outv=Conv_Block_Video(self.outv)
        self.outv=Conv_Block_Video(self.outv)
        self.outv=Conv_Block_Video(self.outv)
        self.outv=Conv_Block_Video(self.outv)
        self.outv=Conv1D(256,1)(self.outv)
        
        #audio_processing
        '''self.outa = Conv_Block(self.audio, dialation_rate=1)
        self.outa = Conv_Block(self.outa, dialation_rate=2)
        self.outa = Conv_Block(self.outa, dialation_rate=4)
        self.outa = Conv_Block(self.outa, dialation_rate=8)
        self.outa = Conv_Block(self.outa, dialation_rate=16)
        self.outa = Conv_Block(self.outa, dialation_rate=32)
        self.outa = Conv_Block(self.outa, dialation_rate=64)
        self.outa = Conv_Block(self.outa, dialation_rate=128)'''

        self.outa = Conv_Block_Audio(self.audio1, dialation_rate=1, filters=256)
        self.outa = Conv_Block_Audio(self.outa, dialation_rate=2, filters=256)
        self.outa = Conv_Block_Audio(self.outa, dialation_rate=4, filters=256)
        self.outa = Conv_Block_Audio(self.outa, dialation_rate=8, filters=256)
        self.outa = Conv_Block_Audio(self.outa, dialation_rate=16, filters=256)
        self.outa = Conv_Block_Audio(self.outa, dialation_rate=32, filters=256)
        self.outa = Conv_Block_Audio(self.outa, dialation_rate=64, filters=256)
        self.outa = Conv_Block_Audio(self.outa, dialation_rate=128, filters=256)
        
        #fusion_process
        
        self.outv=UpSampling1D(size=4*8)(self.outv)

        print('outv:', self.outv.shape)
        print('outa:', self.outa.shape)

        if self.attention == True:
            #self.outa1=Conv1D(256,16,padding='same',strides=8, activation='relu')(self.outa)
            self.attn_layer = AttentionLayer(name='attention_layer')
            self.attn_out, self.attn_states = self.attn_layer([self.outv, self.outa], verbose=False)
            print('attn_out:', self.attn_out.shape)
            print('attn_states:', self.attn_states.shape)

            
            self.fusion=concatenate([self.attn_out, self.outv, self.outa],axis=-1)  # B, 200, 512
            #self.fusion = Conv1D(512, 1)(self.fusion)
            #self.fusion = Lambda(lambda x: K.expand_dims(x, axis=2))(self.fusion)
            #self.fusion=Conv2DTranspose(256,(16,1),strides=(8,1),padding='same',data_format='channels_last')(self.fusion) # B, 1600, 256
            #self.fusion = Lambda(lambda x: K.squeeze(x, axis=2), name='fusion_out')(self.fusion)
            
            #self.fusion=concatenate([self.fusion, self.outa],axis=-1)  # B, 200, 512
            self.fusion=Conv1D(512,1)(self.fusion)
        else:
            self.fusion=concatenate([self.outv,self.outa],axis=-1)
        
        print('fusion:', self.fusion.shape)

        '''self.fusion=multi_head_self_attention(8,512)(self.fusion)
        self.fusion=multi_head_self_attention(8,512)(self.fusion)
        self.fusion=multi_head_self_attention(8,512)(self.fusion)
        self.fusion=multi_head_self_attention(8,512)(self.fusion)
        self.fusion=multi_head_self_attention(8,512)(self.fusion)
        self.fusion=multi_head_self_attention(8,512)(self.fusion)
        print('self attention fusion shape',self.fusion.shape)'''

        '''self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=1,filters=512)
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

        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=1,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=2,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=4,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=8,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=16,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=32,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=64,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=128,filters=512)'''

        self.fusion=Conv1D(256,1)(self.fusion)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=1,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=2,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=4,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=8,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=16,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=32,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=64,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=128,filters=256)

        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=1,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=2,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=4,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=8,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=16,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=32,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=64,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=128,filters=256)

        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=1,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=2,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=4,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=8,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=16,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=32,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=64,filters=256)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=128,filters=256)
        
        #Decoding
        #self.mask=Conv1D(256,1,activation='relu', name='mask')(self.fusion)
        self.mask = Activation('relu', name='mask')(self.fusion)
        self.fusion=Multiply()([self.audio,self.mask])
        self.decode = Lambda(lambda x: K.expand_dims(x, axis=2))(self.fusion)

        #self.decode=Conv2DTranspose(256,(16,1),strides=(8,1),padding='same',data_format='channels_last')(self.decode)
        self.decode=Conv2DTranspose(1,(40,1),strides=(20,1),padding='same',data_format='channels_last')(self.decode)
        self.out = Lambda(lambda x: K.squeeze(x, axis=2), name='out')(self.decode)
        print('Out:', self.out.shape)
        
        '''self.out_norm = vec_l2norm(self.out)
        self.out_normalized = self.out/self.out_norm
        self.out_denormalized = self.out_normalized*self.norm'''
        
        self.model=Model(inputs=[self.lipnet_model.input,self.audio_input_data],outputs=[self.out])
