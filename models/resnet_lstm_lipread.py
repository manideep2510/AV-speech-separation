import sys
sys.path.append('/data/AV-speech-separation/LipNet')
sys.path.append('/data/AV-speech-separation/models')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import keras
from keras.layers import *
from keras import Model, Sequential
import keras.backend as K
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers.core import Lambda
from classification_models.classification_models.resnet import ResNet18, ResNet34, preprocess_input

def GRU(x, input_size, hidden_size, num_layers, num_classes, every_frame=True):

    out = Bidirectional(keras.layers.GRU(hidden_size, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat')(x)
    out = Bidirectional(keras.layers.GRU(hidden_size, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'), merge_mode='concat')(out)
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
                ReLU(),
                ZeroPadding3D(padding=((0, 4, 8))),
                MaxPooling3D(pool_size=(1, 2, 3), strides=(1, 1, 2))
                ])

    backend_conv1 = Sequential([
                Conv1D(2*inputDim, 5, strides=2, use_bias=False),
                BatchNormalization(),
                ReLU(),
                MaxPooling1D(2, 2),
                Conv1D(4*inputDim, 5, strides=2, use_bias=False),
                BatchNormalization(),
                ReLU(),
                ])

    backend_conv2 = Sequential([
                Dense(inputDim),
                BatchNormalization(),
                ReLU(),
                Dense(nClasses)
                ])

    nLayers=2

    # Forward pass

    input_frames = Input(shape=(frameLen,50,100,1), name='frames_input')
    x = frontend3D(input_frames)
    print('3D Conv Out:', x.shape)
    #x = Lambda(lambda x : tf.transpose(x, [0, 2, 1, 3, 4]), name='lambda1')(x)  #x.transpose(1, 2) tf.tens
    #print('3D Conv Out Transp:', x.shape)
    x = Lambda(lambda x : tf.reshape(x, [-1, x.shape[2], x.shape[3], x.shape[4]]), name='lambda2')(x)   #x.view(-1, 64, x.size(3), x.size(4))
    print('3D Conv Out Reshape:', x.shape)

    channels = int(x.shape[-1])
    resnet18 = ResNet18((None, None, channels), weights=None, include_top=False)

    x = resnet18(x)
    print('Resnet18 Out:', x.shape)

    x = GlobalAveragePooling2D(name='global_avgpool_resnet')(x)
    x = Dense(inputDim)(x)
    x = BatchNormalization()(x)
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
        x = GRU(x, inputDim, hiddenDim, nLayers, nClasses, every_frame)
        print('GRU Out:', x.shape)

    else:
        raise Exception('No model is selected')

    model = Model(inputs=input_frames, outputs=x)

    if pretrain == True:
        model.load_weights('/data/models/combResnetLSTM_CTCloss_236k-train_1to3ratio_valWER_epochs9to20_lr1e-5_0.1decay9epochs/weights-04-109.0513.hdf5')
        print('ResNet LSTM Pretrain weights loaded')

    return model

def lipreading(mode, inputDim=256, hiddenDim=512, nClasses=29, frameLen=125, AbsoluteMaxStringLen=128, every_frame=True, pretrain=True):
    model = Lipreading(mode, inputDim=inputDim, hiddenDim=hiddenDim, nClasses=nClasses, frameLen=frameLen, absolute_max_string_len=AbsoluteMaxStringLen, every_frame=every_frame, pretrain=pretrain)
    return model
