# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from keras.layers import *
from keras import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, ReduceLROnPlateau, EarlyStopping, ReduceLROnPlateau
from callbacks import Metrics, learningratescheduler, earlystopping, reducelronplateau
from plotting import plot_loss_and_acc
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from lipnet import LipNet

'''
Usage: 

model = VideoModel(256,96,(298,257,2),(500,50,100,3)).FullModel(lipnet_pretrained = None)

'''


# Model

class VideoModel():

    def __init__(self, filters,filters_audio, audio_ip_shape, video_ip_shape):
        
        self.filters = filters
        self.filters_audio=filters_audio       
        self.audio_ip_shape = audio_ip_shape      #(257,500,2)
        self.video_ip_shape = video_ip_shape      #(125,50,100,3)
    

        self.conv7 = Lambda(lambda x : tf.expand_dims(x, axis = -1))

        self.conv8 = Lambda(lambda x: tf.image.resize_nearest_neighbor(x, size = (298, x.shape[-2])))
        
        

    def FullModel(self, lipnet_pretrained):

        
        ip = Input(shape = (self.audio_ip_shape[0], self.audio_ip_shape[1],self.audio_ip_shape[2]))#shape(batch,257,500,2)
        
        ip_embeddings_1 = Input(shape = (int(self.video_ip_shape[0]), int(self.video_ip_shape[1]),int(self.video_ip_shape[2]), int(self.video_ip_shape[3])))
        ip_embeddings_2 = Input(shape = (int(self.video_ip_shape[0]), int(self.video_ip_shape[1]),int(self.video_ip_shape[2]),int(self.video_ip_shape[3]))); 
      
        #audio_stream = self.AudioModel(ip)
        conv = Conv2D(filters = self.filters_audio, kernel_size = (3,3), strides = (1,1), padding = "same", dilation_rate = (1,1),
                      activation = "relu")(ip) ; print("conv ", conv.shape)
        conv = BatchNormalization(axis=-1)(conv)
        #conv = SpatialDropout2D(rate = dropout)(conv)
        
        conv = Conv2D(filters = self.filters_audio, kernel_size = (3,3), strides = (1,1), padding = "same", dilation_rate = (1,1),
                      activation = "relu")(conv) ; print("conv ", conv.shape)
        conv = BatchNormalization(axis=-1)(conv)
        #conv = SpatialDropout2D(rate = dropout)(conv)
        
        conv = Conv2D(filters = self.filters_audio, kernel_size = (3,3), strides = (1,1), padding = "same", dilation_rate = (1,1),
                      activation = "relu")(conv) ; print("conv ", conv.shape)
        conv = BatchNormalization(axis=-1)(conv)
        #conv = SpatialDropout2D(rate = dropout)(conv)
        
        conv = Conv2D(filters = self.filters_audio* 2, kernel_size = (3,3), strides = (1,1), padding = "same", dilation_rate = (1,1),
                      activation = "relu")(conv) ; print("conv ", conv.shape)
        conv = BatchNormalization(axis=-1)(conv)
        #conv = SpatialDropout2D(rate = dropout)(conv)
        
        conv = Conv2D(filters = self.filters_audio* 2, kernel_size = (3,3), strides = (1,1), padding = "same", dilation_rate = (1,1),
                      activation = "relu")(conv) ; print("conv ", conv.shape)
        conv = BatchNormalization(axis=-1)(conv)
        #conv = SpatialDropout2D(rate = dropout)(conv)
        
        conv = Conv2D(filters = self.filters_audio* 3, kernel_size = (3,3), strides = (1,1), padding = "same", dilation_rate = (1,1),
                      activation = "relu")(conv) ; print("conv ", conv.shape)
        conv = BatchNormalization(axis=-1)(conv)
        #conv = SpatialDropout2D(rate = dropout)(conv)
        
        conv = Conv2D(filters = self.filters_audio* 3, kernel_size = (5,5), strides = (1,1), padding = "same", dilation_rate = (1,1),
                      activation = "relu")(conv) ; print("conv ", conv.shape)
        conv = BatchNormalization(axis=-1)(conv)
        #conv = SpatialDropout2D(rate = dropout)(conv)
        
        conv = Conv2D(filters = self.filters_audio* 3, kernel_size = (5,5), strides = (1,1), padding = "same", dilation_rate = (1,1),
                      activation = "relu")(conv) ; print("conv ", conv.shape)
        conv = BatchNormalization(axis=-1)(conv)
        #conv = SpatialDropout2D(rate = dropout)(conv)
        
        conv = Conv2D(filters = self.filters_audio//12, kernel_size = (5,5), strides = (1,1), padding = "same", dilation_rate = (1,1),
                      activation = "relu")(conv) ; print("conv ", conv.shape)
        audio_stream = BatchNormalization(axis=-1)(conv)
        
        
        #lipnet_model = LipNet(input_shape = (125,50,100,3), pretrained=lipnet_pretrained)
        lipnet_model = LipNet(input_shape = self.video_ip_shape, pretrained=lipnet_pretrained)
        
        x1=lipnet_model(ip_embeddings_1)
        #x1=x1.output
        
        x2=lipnet_model(ip_embeddings_2)
        #x2=x2.output
        
        
        x1 = Dense(128, kernel_initializer='he_normal', name='dense2')(x1)
        x1 = Dense(256, kernel_initializer='he_normal', name='dense3')(x1)
        x1 = self.conv7(x1)
        video_stream_1 = self.conv8(x1)
        
        x2 = Dense(128, kernel_initializer='he_normal', name='dense2_1')(x2)
        x2 = Dense(256, kernel_initializer='he_normal', name='dense3_1')(x2)
        x2 = self.conv7(x2)
        video_stream_2 = self.conv8(x2)



        audio_flatten = TimeDistributed(Flatten())(audio_stream)
        video_flatten_1 = TimeDistributed(Flatten())(video_stream_1)
        video_flatten_2 = TimeDistributed(Flatten())(video_stream_2)
    
    
        

        concated = concatenate([audio_flatten, video_flatten_1,video_flatten_2], axis = 2) 
        lstm = Bidirectional(LSTM(units = 64, return_sequences = True, activation = "tanh"))(concated)   
        flatten = Flatten()(lstm) 
        dense = Dense(100, activation = "relu")(flatten)
        dense = Dense(2 * self.audio_ip_shape[0] * self.audio_ip_shape[1], activation = "sigmoid")(dense) 
        combo_mask = Reshape([2 , self.audio_ip_shape[0], self.audio_ip_shape[1]])(dense) 
        
        mask_1 = Lambda(lambda x : x[:,0])(combo_mask) 
        mask_2 = Lambda(lambda x : x[:,1])(combo_mask) 


        
        model=Model([ip, ip_embeddings_1, ip_embeddings_2], [mask_1, mask_2])

        return model
    
    
