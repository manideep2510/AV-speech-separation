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

# Lip Embedder
def gru(x, input_size, hidden_size, num_layers, num_classes, every_frame=True):

    out = Bidirectional(tf.keras.layers.GRU(hidden_size, return_sequences=True, kernel_initializer='Orthogonal', reset_after=False, name='gru1'), merge_mode='concat')(x)
    out = Bidirectional(tf.keras.layers.GRU(hidden_size, return_sequences=True, kernel_initializer='Orthogonal', reset_after=False, name='gru2'), merge_mode='concat')(out)
    if every_frame:
        out = Dense(num_classes)(out)  # predictions based on every time step
    else:
        out = Dense(num_classes)(out[:, -1, :])  # predictions based on last time-step
    return out


def Lipreading(mode, inputDim=256, hiddenDim=512, nClasses=500, frameLen=29, absolute_max_string_len=128, every_frame=True, pretrain=None):

    frontend3D = Sequential([
                ZeroPadding3D(padding=(2, 3, 3)),
                Conv3D(64, kernel_size=(5, 7, 7), strides=(1, 2, 2), padding='valid', use_bias=False),
                BatchNormalization(),
                #Mish('Mish'),
                Activation('relu'),
                ZeroPadding3D(padding=((0, 4, 8))),
                MaxPooling3D(pool_size=(1, 2, 3), strides=(1, 1, 2))
                ])

    backend_conv1 = Sequential([
                Conv1D(2*inputDim, 5, strides=2, use_bias=False),
                BatchNormalization(),
                #Mish('Mish'),
                Activation('relu'),
                MaxPooling1D(2, 2),
                Conv1D(4*inputDim, 5, strides=2, use_bias=False),
                BatchNormalization(),
                #Mish('Mish'),
                Activation('relu'),
                ])

    backend_conv2 = Sequential([
                Dense(inputDim),
                BatchNormalization(),
                #Mish('Mish'),
                Activation('relu'),
                Dense(nClasses)
                ])

    nLayers=2

    # Forward pass

    input_frames = Input(shape=(frameLen,50,100,1), name='frames_input')
    x = frontend3D(input_frames)
    print('3D Conv Out:', x.shape)
    #x = Lambda(lambda x : tf.transpose(x, [0, 2, 1, 3, 4]), name='lambda1')(x)  #x.transpose(1, 2) tf.tens
    #print('3D Conv Out Transp:', x.shape)
    x = Lambda(lambda x : tf.reshape(x, [-1, int(x.shape[2]), int(x.shape[3]), int(x.shape[4])]), name='lambda2')(x)   #x.view(-1, 64, x.size(3), x.size(4))
    print('3D Conv Out Reshape:', x.shape)

    channels = int(x.shape[-1])
    #resnet18 = ResNet18((None, None, channels), weights=None, include_top=False)

    ResNet18, preprocess_input = Separable_Classifiers.get('resnet18')
    resnet18 = ResNet18((None, None, channels), weights=None, include_top=False)

    x = resnet18(x)
    print('Resnet18 Out:', x.shape)

    x = GlobalAveragePooling2D(name='global_avgpool_resnet')(x)
    x = Dense(inputDim, name='dense_resnet')(x)
    x = BatchNormalization(name='bn_resnet')(x)
    print('Resnet18 Linear Out:', x.shape)

    if mode == 'temporalConv':
        x = Lambda(lambda x : tf.reshape(x, [-1, frameLen, inputDim]), name='lambda3')(x)   #x.view(-1, frameLen, inputDim)

        x = Lambda(lambda x : tf.transpose(x, [0, 2, 1]), name='lambda4')(x)   #x.transpose(1, 2)
        x = backend_conv1(x)
        x = Lambda(lambda x : tf.reduce_mean(x, 2), name='lambda5')(x)
        x = backend_conv2(x)
        #print(x.shape)
    elif mode == 'backendGRU' or mode == 'finetuneGRU':
        x = Lambda(lambda x : tf.reshape(x, [-1, frameLen, inputDim]), name='lambda6')(x)    #x.view(-1, frameLen, inputDim)
        print('Input to GRU:', x.shape)
        x = gru(x, inputDim, hiddenDim, nLayers, nClasses, every_frame)
        print('GRU Out:', x.shape)

    else:
        raise Exception('No model is selected')

    model = Model(inputs=input_frames, outputs=x)

    if pretrain == True:
        model.load_weights(
            '/data/lrs2/sepconv_lipreading_weights.hdf5')
        print('Separable Conv ResNet LSTM Pretrain weights loaded')

    return model

def lipreading(mode, inputDim=256, hiddenDim=512, nClasses=29, frameLen=125, AbsoluteMaxStringLen=128, every_frame=True, pretrain=True):
    model = Lipreading(mode, inputDim=inputDim, hiddenDim=hiddenDim, nClasses=nClasses, frameLen=frameLen, absolute_max_string_len=AbsoluteMaxStringLen, every_frame=every_frame, pretrain=pretrain)
    return model

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
    #video_data = Input(shape=(50, 512))

    lipnet_model = lipreading(mode='backendGRU', inputDim=256, hiddenDim=512, nClasses=29, frameLen=frames, AbsoluteMaxStringLen=128, every_frame=True, pretrain=lipnet_pretrained)

    if train_lipnet == False:
        for layer in lipnet_model.layers:
            layer.trainable = False

    if lstm == True:
        outv = lipnet_model.output
        outv = Dense(128, kernel_initializer='he_normal', name='dense2')(outv)
        outv = Dense(256, kernel_initializer='he_normal', name='dense3')(outv)
        #outv = GlobalAveragePooling2D(outv)
    else:
        outv = lipnet_model.layers[-4].output

    #video_processing    
    video_data=Conv1D(512,1)(outv)
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

    model=Model(inputs=[lipnet_model.input, audio_input_data1, audio_input_data2, noise_dev_inp],outputs=[out_class])

    return model