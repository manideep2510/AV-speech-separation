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

class TasNet(object):
    
    def __init__(self,video_ip_shape,time_dimensions=500,frequency_bins=257,n_frames=125, lipnet_pretrained=None, train_lipnet=None):
        
        self.video_ip_shape=video_ip_shape
        self.t=time_dimensions
        self.f=frequency_bins
        self.frames=n_frames
        self.lipnet_pretrained=lipnet_pretrained
        self.train_lipnet=train_lipnet
        self.build()
        
    def build(self):
        
        self.ip_samples = Input(shape = ((self.t/100)*16000,))
        self.input_samples = Lambda(lambda x : x, name='lambda_input_samples')(self.ip_samples)
        print('input_samples', self.input_samples.shape)
        self.input_samples = Reshape([int((self.t/100)*16000), 1])(self.input_samples)
        print('input_samples_reshape', self.input_samples.shape)
        
        #self.video_input_data=Input(shape=(self.frames,256))#video_shape=(125,256)
        
        self.audio_input_data=Input(shape=(self.f,self.t,2),dtype='float32')#audio_shape=(257,500,2)
        self.audio_input = Lambda(lambda x : tf.transpose(x, [0, 2, 1, 3]))(self.audio_input_data) # Transpose
        self.audio_magnitude = Lambda(lambda x : x[:,:,:,0])(self.audio_input)
        self.audio_phase = Lambda(lambda x : x[:,:,:,1])(self.audio_input)
        self.audio=concatenate([self.audio_magnitude,self.audio_phase],axis=-1)
        self.audio=Conv1D(256,1)(self.audio)
    
        #video_processing

        self.lipnet_model = lipreading(mode='backendGRU', inputDim=256, hiddenDim=512, nClasses=29, frameLen=self.frames, AbsoluteMaxStringLen=128, every_frame=True, pretrain=True)

        if self.train_lipnet == False:
            for layer in self.lipnet_model.layers:
                layer.trainable = False

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
        
        print('outv:', self.outv.shape)
        print('outa:', self.outa.shape)
        self.attn_layer = AttentionLayer(name='attention_layer')
        self.attn_out, self.attn_states = self.attn_layer([self.outv, self.outa], verbose=False)
        print('attn_out:', self.attn_out.shape)
        print('attn_states:', self.attn_states.shape)

        self.fusion=concatenate([self.attn_out, self.outv,self.outa],axis=-1)
        self.fusion=Conv1D(512,1)(self.fusion)
        print('fusion:', self.fusion.shape)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=1,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=2,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=4,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=8,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=16,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=32,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=64,filters=512)
        self.fusion=Conv_Block_Audio(self.fusion,dialation_rate=128,filters=512)

        self.fusion1=Conv_Block_Audio(self.fusion,dialation_rate=1,filters=512)
        self.fusion1=Conv_Block_Audio(self.fusion1,dialation_rate=2,filters=512)
        self.fusion1=Conv_Block_Audio(self.fusion1,dialation_rate=4,filters=512)
        self.fusion1=Conv_Block_Audio(self.fusion1,dialation_rate=8,filters=512)
        self.fusion1=Conv_Block_Audio(self.fusion1,dialation_rate=16,filters=512)
        self.fusion1=Conv_Block_Audio(self.fusion1,dialation_rate=32,filters=512)
        self.fusion1=Conv_Block_Audio(self.fusion1,dialation_rate=64,filters=512)
        self.fusion1=Conv_Block_Audio(self.fusion1,dialation_rate=128,filters=512)

        self.fusion2=Conv_Block_Audio(self.fusion,dialation_rate=1,filters=512)
        self.fusion2=Conv_Block_Audio(self.fusion2,dialation_rate=2,filters=512)
        self.fusion2=Conv_Block_Audio(self.fusion2,dialation_rate=4,filters=512)
        self.fusion2=Conv_Block_Audio(self.fusion2,dialation_rate=8,filters=512)
        self.fusion2=Conv_Block_Audio(self.fusion2,dialation_rate=16,filters=512)
        self.fusion2=Conv_Block_Audio(self.fusion2,dialation_rate=32,filters=512)
        self.fusion2=Conv_Block_Audio(self.fusion2,dialation_rate=64,filters=512)
        self.fusion2=Conv_Block_Audio(self.fusion2,dialation_rate=128,filters=512)
        
        #prediction
        self.mag=Conv1D(257,1)(self.fusion1)
        self.mag=Mish('Mish')(self.mag)
        self.mag=Reshape([self.t,self.f,1], name='reshape_mag')(self.mag)
        #self.mag = Lambda(lambda x : tf.transpose(x, [0, 2, 1, 3]))(self.mag)
        self.phase=Conv1D(257,1)(self.fusion2)
        self.phase=Mish('Mish')(self.phase)
        self.phase=Reshape([self.t,self.f,1], name='reshape_phase')(self.phase)
        #self.phase = Lambda(lambda x : tf.transpose(x, [0, 2, 1, 3]))(self.phase)
        print('Phase', self.phase.shape)

        self.mag_phase = concatenate([self.mag, self.phase], axis=3, name='concat_phase_mag')
        self.mag_phase = Conv2D(1, 3, padding='same')(self.mag_phase)

        #self.mag_phase_mul_spect = Lambda(lambda x : x, name='mag_phase_mul_spect')([self.audio_input, self.mag_phase])
        self.mag_phase_mul_spect = tf.keras.layers.multiply([self.audio_input, self.mag_phase], name='mul_masks_spects')
        self.mag_phase = Conv2D(1, 3, padding='same')(self.mag_phase_mul_spect)

        print('mag_phase:', self.mag_phase.shape)

        self.mag_phase = Reshape([self.t,self.f])(self.mag_phase)

        self.up1 = Conv1D(256, 26, padding='valid')(self.mag_phase)
        self.up1 = UpSampling1D(4)(self.up1)
        print('up1:', self.up1.shape)

        self.up2 = Conv1D(64, 26, padding='valid')(self.up1)
        self.up2 = UpSampling1D(4)(self.up2)
        print('up2:', self.up2.shape)

        self.up3 = Conv1D(16, 26, padding='valid')(self.up2)
        self.up3 = UpSampling1D(4)(self.up3)
        print('up3:', self.up3.shape)

        self.up4 = Conv1D(4, 26, padding='valid')(self.up3)
        self.up4 = UpSampling1D(3)(self.up4)
        print('up4:', self.up4.shape)

        self.out_samples = Conv1D(1, 26, padding='valid', name='output_layer')(self.up4)
        print('out_samples:', self.out_samples.shape)

        '''#self.mag_phase_mul_spect = 

        self.trans_filters1 = tf.Variable(tf.random_normal([8, 64, 256]))
        self.trans_filters2 = tf.Variable(tf.random_normal([20, 1, 64]))
        self.out_samples = tf.compat.v1.nn.conv1d_transpose(self.mag_phase, filters=self.trans_filters1, strides=8, padding='VALID', name='transpose_conv1')
        self.out_samples = tf.compat.v1.nn.conv1d_transpose(self.out_samples, filters=self.trans_filters2, strides=20, padding='VALID', name='transpose_conv2')'''

        #self.out_samples = Reshape([(self.t/100)*16000, 1])(self.out_samples)
        self.output_concats = concatenate([self.out_samples, self.input_samples], axis=2, name='concat_maps')
        print('output_concats:', self.output_concats.shape)
        
        self.model=Model(inputs=[self.audio_input_data, self.lipnet_model.input, self.ip_samples],outputs=self.output_concats)
        
        return self.model
    
    def summary(self):
        Model(inputs=[self.audio_input_data, self.lipnet_model.input, self.ip_samples],outputs=self.output_concats).summary()
