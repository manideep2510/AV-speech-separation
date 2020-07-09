import sys
#sys.path.append('/data/AV-speech-separation1/LipNet')
#sys.path.append('/data/AV-speech-separation1/models')
sys.path.append('/data/av-speech-separation/models/classification_models-master/')

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from sep_classification_models.tfkeras import Classifiers as Separable_Classifiers
from tensorflow.keras.constraints import Constraint

# Spectral Norm constraint
def l2_normalize(x, eps=1e-12):
  '''
  Scale input by the inverse of it's euclidean norm
  '''
  return x / tf.linalg.norm(x + eps)

POWER_ITERATIONS = 1
class Spectral_Norm(Constraint):
    '''
    Uses power iteration method to calculate a fast approximation 
    of the spectral norm (Golub & Van der Vorst)
    The weights are then scaled by the inverse of the spectral norm
    '''
    def __init__(self, power_iters=POWER_ITERATIONS):
        self.n_iters = power_iters

    def __call__(self, w):
      flattened_w = tf.reshape(w, [w.shape[0], -1])
      u = tf.random.normal([flattened_w.shape[0]])
      v = tf.random.normal([flattened_w.shape[1]])
      for i in range(self.n_iters):
        v = tf.linalg.matvec(tf.transpose(flattened_w), u)
        v = l2_normalize(v)
        u = tf.linalg.matvec(flattened_w, v)
        u = l2_normalize(u)
      sigma = tf.tensordot(u, tf.linalg.matvec(flattened_w, v), axes=1)
      return w / sigma

    def get_config(self):
        return {'n_iters': self.n_iters}

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

def Conv_Block_Audio_disc(inputs,dialation_rate=1,stride=1,filters=512,kernel_size=3):
    
    x = Conv1D(filters,1)(inputs)
    #x = Mish('Mish')(x)
    x = LeakyReLU(alpha=0.3)(x)
    #x = BatchNormalization()(x)
    x=GlobalLayerNorm()(x)
    x = SeparableConv1D(filters,kernel_size,dilation_rate=dialation_rate,padding='same',strides=stride)(x)
    x = Add()([inputs,x])
    
    return x

def Conv_Block_Video_disc(inputs,dialation_rate=1,stride=1,filters=512,kernel_size=3):
    
    
    #x = Mish('Mish')(inputs)
    x = LeakyReLU(alpha=0.3)(inputs)
    x = BatchNormalization()(x)
    x = SeparableConv1D(filters,kernel_size,dilation_rate=dialation_rate,padding='same',strides=stride)(x)
    x = Add()([inputs,x])
    
    return x

'''def Conv_Block_disc(inputs,dialation_rate=1,stride=1,filters=512,kernel_size=3):
    
        x = Conv1D(filters,kernel_size,dilation_rate=dialation_rate,padding='same',strides=stride)(inputs)
        #x = Mish('Mish')(x)
        x = LeakyReLU(alpha=0.3)(x)
        #x = BatchNormalization()(x)
        x=GlobalLayerNorm()(x)
        x = Conv1D(int(inputs.shape[-1]),1)(x)
        x = Add()([inputs,x])
        x = LeakyReLU(alpha=0.3)(x)
        return x'''

def Conv_Block_disc(inputs, dialation_rate=1, stride=1, filters=512, kernel_size=3):

    x = Conv1D(filters, 1)(inputs)
    #x = Mish('Mish')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = GlobalLayerNorm()(x)
    x = SeparableConv1D(filters, kernel_size, dilation_rate=dialation_rate, padding='same', strides=stride)(x)
    #x = Mish('Mish')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = GlobalLayerNorm()(x)
    x = Conv1D(int(inputs.shape[-1]), 1)(x)
    x = Add()([inputs, x])
    return x


def Conv_Block_Audio_disc_SpectNorm(inputs,dialation_rate=1,stride=1,filters=512,kernel_size=3):
    
    x = Conv1D(filters,1,kernel_constraint=Spectral_Norm())(inputs)
    #x = Mish('Mish')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = SeparableConv1D(filters,kernel_size,dilation_rate=dialation_rate,padding='same',
                        strides=stride,kernel_constraint=Spectral_Norm())(x)
    x = Add()([inputs,x])
    
    return x

def Conv_Block_Video_disc_SpectNorm(inputs,dialation_rate=1,stride=1,filters=512,kernel_size=3):
    
    
    #x = Mish('Mish')(inputs)
    x = LeakyReLU(alpha=0.3)(inputs)
    x = SeparableConv1D(filters,kernel_size,dilation_rate=dialation_rate,padding='same',
                        strides=stride,kernel_constraint=Spectral_Norm())(x)
    x = Add()([inputs,x])
    
    return x

def Conv_Block_disc_SpectNorm(inputs,dialation_rate=1,stride=1,filters=512,kernel_size=3):
    
        x = Conv1D(filters,kernel_size,dilation_rate=dialation_rate,padding='same',
                    strides=stride,kernel_constraint=Spectral_Norm())(inputs)
        #x = Mish('Mish')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Conv1D(int(inputs.shape[-1]),1,kernel_constraint=Spectral_Norm())(x)
        x = Add()([inputs,x])
        x = LeakyReLU(alpha=0.3)(x)
        return x


def vec_l2norm(x):

    nr = K.sqrt(tf.math.reduce_sum(K.square(x), axis=1))
    nr = tf.reshape(nr, (-1, 1, 1))
    #nr = tf.broadcast_to(nr, (int(x.shape[1]), int(x.shape[0])))
    return nr


# Generator
def Generator(time_dimensions=500,frequency_bins=257,n_frames=125, 
                lstm = False,lipnet_pretrained=None, train_lipnet=None):
            
    t=time_dimensions
    f=frequency_bins
    frames=n_frames
    lipnet_pretrained = lipnet_pretrained
    train_lipnet = train_lipnet
    lstm = lstm

    audio_input_data=Input(shape=(t*160,1))#audio_shape=(80000,1)

    #audio_encoding
    audio=Conv1D(256,40,padding='same',strides=20, activation='relu')(audio_input_data)
    #audio=Conv1D(256,16,padding='same',strides=2, activation='relu')(audio)
    audio1 = ChannelwiseLayerNorm()(audio)

    #video_processing

    # Lip embeddings input 
    video_data = Input(shape=(50, 512))

    #video_processing

    outv=Conv_Block_Video(video_data)
    outv=Conv_Block_Video(outv)
    outv=Conv_Block_Video(outv)
    outv=Conv_Block_Video(outv)
    outv=Conv_Block_Video(outv)
    outv=Conv1D(256,1)(outv)

    #audio_processing
    outa=Conv_Block_Audio(audio1,dialation_rate=1,filters=256)
    outa=Conv_Block_Audio(outa,dialation_rate=2,filters=256)
    outa=Conv_Block_Audio(outa,dialation_rate=4,filters=256)
    outa=Conv_Block_Audio(outa,dialation_rate=8,filters=256)
    outa=Conv_Block_Audio(outa,dialation_rate=16,filters=256)
    outa=Conv_Block_Audio(outa,dialation_rate=32,filters=256)
    outa=Conv_Block_Audio(outa,dialation_rate=64,filters=256)
    outa=Conv_Block_Audio(outa,dialation_rate=128,filters=256)

    #fusion_process

    outv = UpSampling1D(size=4*8)(outv)

    print('outv:', outv.shape)
    print('outa:', outa.shape)

    #latent_vector = tf.random.normal(shape=tf.shape(outa), name='latent_vector')
    #latent_vector = Lambda(lambda x: tf.random.normal(shape=tf.shape(x)), name='latent_vector')(outa)
    fusion = concatenate([outv, outa], axis=-1)


    print('fusion:', fusion.shape)

    fusion=Conv1D(256,1)(fusion)
    fusion=Conv_Block_Audio(fusion,dialation_rate=1,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=2,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=4,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=8,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=16,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=32,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=64,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=128,filters=256)

    fusion=Conv_Block_Audio(fusion,dialation_rate=1,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=2,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=4,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=8,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=16,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=32,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=64,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=128,filters=256)

    fusion=Conv_Block_Audio(fusion,dialation_rate=1,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=2,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=4,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=8,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=16,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=32,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=64,filters=256)
    fusion=Conv_Block_Audio(fusion,dialation_rate=128,filters=256)
    
    #Decoding
    #mask=Conv1D(256,1,activation='relu', name='mask')(fusion)
    mask = Activation('relu', name='mask')(fusion)
    fusion=Multiply()([audio,mask])
    decode = Lambda(lambda x: K.expand_dims(x, axis=2))(fusion)

    #decode=Conv2DTranspose(256,(16,1),strides=(2,1),padding='same',data_format='channels_last')(decode)
    decode=Conv2DTranspose(1,(40,1),strides=(20,1),padding='same',data_format='channels_last')(decode)
    out = Lambda(lambda x: K.squeeze(x, axis=2), name='out')(decode)
    print('Out:', out.shape)

    model=Model(inputs=[video_data,audio_input_data],outputs=[out])

    return model

# Phase shuffle operation
def apply_phaseshuffle(x, rad, pad_type='reflect'):
  b, x_len, nch = x.get_shape().as_list()

  phase = tf.random.uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
  pad_l = tf.math.maximum(phase, 0)
  pad_r = tf.math.maximum(-phase, 0)
  phase_start = pad_r
  x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

  x = x[:, phase_start:phase_start+x_len]
  x.set_shape([b, x_len, nch])

  return x


# Discriminator
def Discriminator(time_dimensions=500,frequency_bins=257,n_frames=125, phaseshuffle_rad=5, 
                  lstm = False,lipnet_pretrained=None, train_lipnet=None):
            
    t=time_dimensions
    f=frequency_bins
    frames=n_frames
    lipnet_pretrained = lipnet_pretrained
    train_lipnet = train_lipnet
    lstm = lstm

    # Phase Shuffle layer
    if phaseshuffle_rad > 0:
        phaseshuffle = Lambda(lambda x: apply_phaseshuffle(x, phaseshuffle_rad))
        print('Using Phase Shuffle')
    else:
        phaseshuffle = Lambda(lambda x: x)

    audio_input_data1=Input(shape=(t*160,1))#audio_shape=(80000,1)

    # Clean/GT speech
    audio_input_data2=Input(shape=(t*160,1))

    # Additive noise to the clean/GT speech
    noise_dev_inp = Input(shape=(t*160,1))

    norm2 = vec_l2norm(audio_input_data2)
    norm_noise = vec_l2norm(noise_dev_inp)
    audio_input_data2_noisy = audio_input_data2 + (0.1*norm2/(180))*noise_dev_inp
    print('audio_input_data2_noisy:', audio_input_data2_noisy.shape)

    audio_normalized2=audio_input_data2_noisy

    
    audio_normalized = concatenate([audio_input_data1, audio_normalized2],axis=-1)

    print('audio_normalized', audio_normalized.shape)

    #audio_encoding
    audio=Conv1D(256,40,padding='same',strides=20, activation='relu')(audio_normalized)
    #audio=Conv1D(256,16,padding='same',strides=2, activation='relu')(audio)
    audio = ChannelwiseLayerNorm()(audio)
    audio = phaseshuffle(audio)

    print('Audio encode', audio.shape)

    #video_processing

    # Lip embeddings input 
    video_data = Input(shape=(50, 512))

    #video_processing    
    outv=Conv_Block_Video_disc(video_data)
    outv=Conv_Block_Video_disc(outv)
    outv=Conv_Block_Video_disc(outv)
    outv=Conv_Block_Video_disc(outv)
    outv=Conv_Block_Video_disc(outv)
    outv=Conv1D(256,1)(outv)

    # Audio Processing WITH Phase Shuffle
    outa=Conv_Block_Audio_disc(audio,dialation_rate=1,filters=256)
    outa = phaseshuffle(outa)
    outa=Conv_Block_Audio_disc(outa,dialation_rate=2,filters=256)
    outa = phaseshuffle(outa)
    outa=Conv_Block_Audio_disc(outa,dialation_rate=4,filters=256)
    outa = phaseshuffle(outa)
    outa=Conv_Block_Audio_disc(outa,dialation_rate=8,filters=256)
    outa = phaseshuffle(outa)
    outa=Conv_Block_Audio_disc(outa,dialation_rate=16,filters=256)
    outa = phaseshuffle(outa)
    outa=Conv_Block_Audio_disc(outa,dialation_rate=32,filters=256)
    outa = phaseshuffle(outa)
    outa=Conv_Block_Audio_disc(outa,dialation_rate=64,filters=256)
    outa = phaseshuffle(outa)
    outa=Conv_Block_Audio_disc(outa,dialation_rate=128,filters=256)
    outa = phaseshuffle(outa)

    # Upsample Video frames to Audio frames rate
    outv = UpSampling1D(size=4*8)(outv)

    print('outv:', outv.shape)
    print('outa:', outa.shape)

    # Concatenate Audio and Video features
    fusion=concatenate([outv,outa],axis=-1)

    print('fusion:', fusion.shape)

    fusion=Conv1D(256,1)(fusion)

    conv1 = Conv1D(256, 3, padding='same', strides=1)(fusion)
    conv1 = LeakyReLU(alpha=0.3)(conv1)
    conv1 = GlobalLayerNorm()(conv1)
    pool1 = AveragePooling1D(pool_size=2)(conv1)
    #pool1 = Dropout(0.2)(pool1)

    conv2 = Conv1D(512, 3, padding='same', strides=1)(pool1)
    conv2 = LeakyReLU(alpha=0.3)(conv2)
    conv2 = GlobalLayerNorm()(conv2)
    pool2 = AveragePooling1D(pool_size=2)(conv2)
    #pool2 = Dropout(0.2)(pool2)

    conv3 = Conv1D(1024, 3, padding='same', strides=1)(pool2)
    conv3 = LeakyReLU(alpha=0.3)(conv3)
    conv3 = GlobalLayerNorm()(conv3)
    pool3 = AveragePooling1D(pool_size=2)(conv3)
    #pool3 = Dropout(0.2)(pool3)

    out = Flatten()(pool3)

    out_class = Dense(1)(out)

    model=Model(inputs=[video_data, audio_input_data1, audio_input_data2, noise_dev_inp],outputs=[out_class])

    return model