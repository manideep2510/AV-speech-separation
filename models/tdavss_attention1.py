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
from .attention_layers import AttentionLayer
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
    
    def __init__(self,time_dimensions=2,frequency_bins=257,n_frames=50):
        
        self.t=time_dimensions
        self.f=frequency_bins
        self.frames=n_frames
        self.build()
    
    def energy_(self,inputs, states):
        """ Step function for computing energy for a single decoder state """

        outa=inputs
        print('outvshape',outv.shape)
        print('hiddenlenght',self.hidden[-1].shape)
        
        context_vector, attn_states = attention_layer([self.outv,self.hidden[-1],len(self.hidden)-1])

        current_word=Lambda(lambda x:K.expand_dims(x[:,len(self.hidden)-1,:],axis=1))(self.outa)
                #current_word=Lambda(lambda x:x[:,timestep,:])(self.outa)
        print('currentwor',current_word.shape)
        print('context',context_vector.shape)
        decoder_input = Concatenate(axis=2)([K.expand_dims(context_vector,axis=1), current_word])
        print('decoder_inp', decoder_input.shape)
                #decoder_input = Concatenate(axis=1)([context_vector[0], current_word])
        output, hidden_state, cell_state= decoder_recurrent_layer(decoder_input,initial_state=[self.hidden[-1], self.cell[-1]])
        self.hidden.append(hidden_state)
        self.cell.append(cell_state)
        self.outputs.append(context_vector)
        self.weights.append(attn_states)

        return context_vector,[context_vector]
    
    def build(self):
        
        
        self.video_input_data=Input(shape=(self.frames,256))#video_shape=(125,256)
        self.audio_input_data=Input(shape=(32000,1),dtype='float32')#audio_shape=(80000,1)
        
        initial_cell_state = Input(shape=(256,), name='cell_state')
        initial_hidden_state = Input(shape=(256,), name='hidden_state')
        hidden_state, cell_state = initial_hidden_state, initial_cell_state
        print(hidden_state.shape)
        #audio_encoding
        self.audio=Conv1D(256,40,padding='same',strides=20)(self.audio_input_data)
        self.audio=Conv1D(256,16,padding='same',strides=8)(self.audio)
        
      
        #video_processing
        self.video_data=Conv1D(512,1)(self.video_input_data)
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
        print(self.outa.shape)
        #fusion_process
        
        self.outv=UpSampling1D(size=4)(self.outv)
        
        #decoder_recurrent_layer = SimpleRNN(units=256,return_state=True)
        
        
        #decoder_recurrent_layer = LSTM(units=256,return_state=True)
        
        #attention_layer=Attention(context='many-to-many', alignment_type='global',score_function='general')
        self.outputs = []
        self.weights = []
        self.hidden=[]
        self.cell=[]
        self.cell.append(cell_state)
        self.hidden.append(hidden_state)
        print(self.outv.shape)

        last_out, e_outputs, _ = K.rnn(
        self.energy_, self.outa, [hidden_state])
        #attention_layer = Luong(name='attention_layer')
#         attention_layer=Attention(context='many-to-many', alignment_type='global',score_function='general')
#         outputs = []
#         for timestep in range(self.t):
#         # Get current input in from embedded target sequences
#         #current_word = Lambda(lambda x: x[:, timestep: timestep+1, :])(embedded_target)
#     # Apply optional attention mechanism
    
#             context_vector, attention_weights = attention_layer([self.outv,hidden_state,timestep])
#     # Combine information
#             print(context_vector.shape)
#             print(attention_weights.shape)
#             current_word=Lambda(lambda x:K.expand_dims(x[:,timestep,:],axis=1))(self.outa)
#             #urrent_word=Lambda(lambda x:x[:,timestep,:])(self.outa)
#             #print(current_word.shape)
#             decoder_input = Concatenate(axis=2)([K.expand_dims(context_vector,axis=1), current_word])
#             #ecoder_input = Concatenate(axis=1)([context_vector[0], current_word])
#             #print(decoder_input.shape)
#         # Decode target word hidden representation at t = timestep
#             output, hidden_state= decoder_recurrent_layer(decoder_input,initial_state=hidden_state)
#     # Predict next word & append to outputs
#             outputs.append(context_vector)
        
        #print(outputs[0].shape)
       # self.attn_layer = Lambda(lambda x:np.asarray(x))(outputs)
        self.attn_layer=tf.stack(outputs,axis=1)
#         self.attn_layer = Luong_AttentionLayer(name='attention_layer')
#         self.attn_out, self.attn_states = self.attn_layer([self.outv, self.outa], verbose=True)
#         print('attn_out:', self.attn_out.shape)
#         print('attn_states:', self.attn_states.shape)
#         print(self.outv.shape)
#         print(self.outa.shape)

        self.fusion=concatenate([self.attn_layer, self.outv, self.outa],axis=-1)
        print(self.fusion.shape)
        self.fusion=Conv1D(512,1)(self.fusion)

        #self.fusion=concatenate([self.outv,self.outa],axis=-1)
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
        self.out = Lambda(lambda x: K.squeeze(x, axis=2))(self.decode)
  
        
        
        self.model=Model(inputs=[self.video_input_data,self.audio_input_data,initial_hidden_state, initial_cell_state],outputs=[self.out])
        
            
