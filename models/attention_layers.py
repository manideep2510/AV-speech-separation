import tensorflow as tf
import os
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K


class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state """

            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch_size*en_seq_len, latent_dim
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
            # <= batch_size*en_seq_len, latent_dim
            W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
            if verbose:
                print('wa.s>',W_a_dot_s.shape)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>',U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
            if verbose:
                print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """
            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        def create_inital_state(inputs, hidden_size):
            # We are not using initial states, but need to pass something to K.rnn funciton
            fake_state = K.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            fake_state = K.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim
            return fake_state

        fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])
        fake_state_e = create_inital_state(encoder_out_seq, encoder_out_seq.shape[1])  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]


class LuongAttention(tf.keras.Model):
    def __init__(self, rnn_size):
        super(LuongAttention, self).__init__()
        self.wa = tf.keras.layers.Dense(rnn_size)

    def call(self, decoder_output, encoder_output):
        # Dot score: h_t (dot) Wa (dot) h_s
        # encoder_output shape: (batch_size, max_len, rnn_size)
        # decoder_output shape: (batch_size, 1, rnn_size)
        # score will have shape: (batch_size, 1, max_len)
        score = tf.matmul(decoder_output, self.wa(encoder_output), transpose_b=True)

        # alignment vector a_t
        alignment = tf.nn.softmax(score, axis=2)

        # context vector c_t is the average sum of encoder output
        context = tf.matmul(alignment, encoder_output)

        return context, alignment

class Luong(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(Luong, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[1][1])),
                                   initializer='uniform',
                                   trainable=True)
        # self.U_a = self.add_weight(name='U_a',
        #                            shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
        #                            initializer='uniform',
        #                            trainable=True)
        # self.V_a = self.add_weight(name='V_a',
        #                            shape=tf.TensorShape((input_shape[0][2], 1)),
        #                            initializer='uniform',
        #                            trainable=True)

        super(Luong, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):

        encoder_out_seq, decoder_out_seq = inputs
        U_a_dot_h = K.dot(encoder_out_seq, self.W_a)  # <= batch_size, en_seq_len, latent_dim * latent_dim,dec_dim
        if verbose:print('U_a_dot_h', U_a_dot_h.shape)
        """ batch,1,dec_dim"""
        expanded_target=K.expand_dims(decoder_out_seq, 1)
        """batch,en_seq_len,1"""
        if verbose:print('expanded_target', expanded_target.shape)

        e_i=Dot(axes=[2, 2])([U_a_dot_h, expanded_target])
        if verbose:print('e_i', e_i.shape)

        e_i=K.squeeze(e_i,axis=-1)
        if verbose:print('e_i', e_i.shape)
        # <= batch_size, en_seq_len
        e_i = K.softmax(e_i)

        if verbose:print('ei>', e_i.shape)

        c_i = K.sum(encoder_out_seq * K.expand_dims(e_i, -1), axis=1)
        if verbose:print('ci>', c_i.shape)
        return c_i, e_i

class MinimalRNN(Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        
        super(MinimalRNN, self).build(input_shape)
        #self.built = True

    def call(self, inputs, states):
        prev_output,temp = states
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output

class Luong_exp(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(Luong_exp, self).__init__(**kwargs)
        self.decoder_recurrent_layer = LSTM(units=256,return_state=True)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[1][2])),
                                   initializer='uniform',
                                   trainable=True)
        # self.U_a = self.add_weight(name='U_a',
        #                            shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
        #                            initializer='uniform',
        #                            trainable=True)
        # self.V_a = self.add_weight(name='V_a',
        #                            shape=tf.TensorShape((input_shape[0][2], 1)),
        #                            initializer='uniform',
        #                            trainable=True)

        super(Luong_exp, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):

        encoder_out_seq, decoder_out_seq, hidden_state, cell_state = inputs

        self.t=encoder_out_seq.shape[1]
        self.context_vecs=[]
        self.attention_weights=[]


        for timestep in range(self.t):

            U_a_dot_h = K.dot(encoder_out_seq, self.W_a)  # <= batch_size, en_seq_len, latent_dim * latent_dim,dec_dim
            if verbose:print('U_a_dot_h', U_a_dot_h.shape)
            """ batch,1,dec_dim"""
            expanded_target=K.expand_dims(hidden_state, 1)
            """batch,en_seq_len,1"""
            if verbose:print('expanded_target', expanded_target.shape)

            e_i=Dot(axes=[2, 2])([U_a_dot_h, expanded_target])
            if verbose:print('e_i', e_i.shape)

            e_i=K.squeeze(e_i,axis=-1)

            if verbose:print('e_i', e_i.shape)
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:print('ei>', e_i.shape)

            c_i = K.sum(encoder_out_seq * K.expand_dims(e_i, -1), axis=1)
            if verbose:print('ci>', c_i.shape)

            #context_vector, attn_states = attention_layer([self.outv,hidden_state,timestep])
            current_word=Lambda(lambda x:K.expand_dims(x[:,timestep,:],axis=1))(decoder_out_seq)
            #current_word=Lambda(lambda x:x[:,timestep,:])(decoder_out_seq)
            decoder_input = Concatenate(axis=2)([K.expand_dims(c_i,axis=1), current_word])
            #decoder_input = Concatenate(axis=1)(c_i, current_word])
            output, hidden_state, cell_state= self.decoder_recurrent_layer(decoder_input,initial_state=[hidden_state, cell_state])
            self.context_vecs.append(c_i)
            self.attention_weights.append(e_i)

        self.attn_out=tf.stack(self.context_vecs,axis=1)
        self.attn_states=tf.stack(self.attention_weights,axis=2)

        return self.attn_out, self.attn_states

class Luong_exp2(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(Luong_exp2, self).__init__(**kwargs)
        self.decoder_recurrent_layer = LSTM(units=256,return_state=True)



    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[1][2])),
                                   initializer='uniform',
                                   trainable=True)

        super(Luong_exp2, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):

        encoder_out_seq, decoder_out_seq, hidden_state = inputs

        def energy(inputs, states):

	        #encoder_out_seq=inputs[:,:,:self.latent_dim]

	        U_a_dot_h = K.dot(encoder_out_seq, self.W_a)  # <= batch_size, en_seq_len, latent_dim * latent_dim,dec_dim
	        if verbose:print('U_a_dot_h', U_a_dot_h.shape)
	        """ batch,1,dec_dim"""
	        expanded_target=K.expand_dims(states[0], 1)
	        """batch,en_seq_len,1"""
	        if verbose:print('expanded_target', expanded_target.shape)

	        e_i=Dot(axes=[2, 2])([U_a_dot_h, expanded_target])
	        if verbose:print('e_i', e_i.shape)

	        e_i=K.squeeze(e_i,axis=-1)

	        if verbose:print('e_i', e_i.shape)
	        # <= batch_size, en_seq_len
	        e_i = K.softmax(e_i)

	        if verbose:print('ei>', e_i.shape)

	        c_i = K.sum(encoder_out_seq * K.expand_dims(e_i, -1), axis=1)
	        if verbose:print('ci>', c_i.shape)
	        
	        #context_vector, attn_states = attention_layer([self.outv,hidden_state,timestep])
	        #current_word=Lambda(lambda x:K.expand_dims(x[:,timestep,:],axis=1))(inputs[:,self.time,self.latent_dim:])
	        current_word=Lambda(lambda x:K.expand_dims(x,axis=1))(inputs)
	        #current_word=Lambda(lambda x:x[:,timestep,:])(decoder_out_seq)
	        decoder_input = Concatenate(axis=2)([K.expand_dims(c_i,axis=1), current_word])
	        #decoder_input = Concatenate(axis=1)(c_i, current_word])
	        output, hidden_state, cell_state= self.decoder_recurrent_layer(decoder_input,initial_state=[states[0], states[1]])

	        return [c_i,e_i],[hidden_state,cell_state]

        
        fake_out, outputs, _ = K.rnn(energy, decoder_out_seq , [hidden_state,cell_state])
            
        
        #self.attn_out=tf.stack(self.context_vecs,axis=1)
        #self.attn_states=tf.stack(self.attention_weights,axis=2)

        return outputs


class Bahdanau(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(Bahdanau, self).__init__(**kwargs)
        self.decoder_recurrent_layer = GRU(units=256, return_state=True)
        self.out_dense1 = Dense(512)
        self.out_dense2 = Dense(512)
        self.act_leaky_relu = LeakyReLU(alpha=0.1)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(Bahdanau, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        
        encoder_out_seq, decoder_out_seq, hidden_state, cell_state, out_state = inputs

        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy(inputs, states):
            """ Step function for computing energy for a single decoder state """

            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch_size*en_seq_len, latent_dim
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
            # <= batch_size*en_seq_len, latent_dim
            W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
            if verbose:
                print('wa.s>',W_a_dot_s.shape)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(states[0], self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>',U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
            if verbose:
                print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            c_i = K.sum(encoder_out_seq * K.expand_dims(e_i, -1), axis=1)

            #context_vector, attn_states = attention_layer([self.outv,hidden_state,timestep])
            #current_word=Lambda(lambda x:K.expand_dims(x[:,timestep,:],axis=1))(inputs[:,self.time,self.latent_dim:]
            #current_word=Lambda(lambda x:K.expand_dims(x,axis=1))(inputs)
            #current_word=Lambda(lambda x:x[:,timestep,:])(decoder_out_seq)
            decoder_input = Concatenate(axis=2)([K.expand_dims(c_i,axis=1), K.expand_dims(states[1], axis=1)])
            #decoder_input = Concatenate(axis=1)(c_i, current_word])
            output, hidden_state = self.decoder_recurrent_layer(decoder_input,initial_state=[states[0]])

            out = self.out_dense1(Concatenate(axis=-1)([inputs, hidden_state, c_i]))
            out = self.act_leaky_relu(out)
            out = self.out_dense2(out)
            #print('states[2]:', states[2])
            #out_state = out

            return [out, e_i], [hidden_state, out]

        fake_out, outputs, _ = K.rnn(energy, decoder_out_seq , [hidden_state, out_state])
            
        
        #self.attn_out=tf.stack(self.context_vecs,axis=1)
        #self.attn_states=tf.stack(self.attention_weights,axis=2)

        return outputs

class multi_head_self_attention(Layer):

    def __init__(self, n_head, d_model, **kwargs):
        super(multi_head_self_attention, self).__init__(**kwargs)
        self.n_head = n_head
        self.d_q = self.d_v =self.d_k = d_model // n_head
        self.normalize = tf.sqrt(tf.cast(self.d_k, dtype='float32'))

        self.dense1=TimeDistributed(Dense(d_model))
        self.activation=TimeDistributed(Activation('relu'))
        self.dense2=TimeDistributed(Dense(d_model))
    
    def build(self, input_shape):
        
        #assert isinstance(input_shape, list)
        
        self.Q_a={}
        self.K_a={}
        self.V_a={} 

        for i in range(self.n_head):
            self.Q_a[str(i)] = self.add_weight(name='Q_a'+str(i),
                                   shape=tf.TensorShape((input_shape[2], self.d_q)),
                                   initializer='uniform',
                                   trainable=True)
            self.K_a[str(i)] = self.add_weight(name='K_a'+str(i),
                                   shape=tf.TensorShape((input_shape[2], self.d_k)),
                                   initializer='uniform',
                                   trainable=True)
            self.V_a[str(i)] = self.add_weight(name='V_a'+str(i),
                                   shape=tf.TensorShape((input_shape[2], self.d_v)),
                                   initializer='uniform',
                                   trainable=True)

        self.W=self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[2], input_shape[2])),
                                   initializer='uniform',
                                   trainable=True)

        super(multi_head_self_attention, self).build(input_shape)



    def call(self, inputs, verbose=False):

        
        self.features=inputs
        self.concat_features=[]
        #self.batch_size=inputs.shape[0]
        # self.dummy_inputs=tf.zeros((self.batch_size,self.n_head,1))
        # self.dummy_states=tf.zeors((self.batch_size,1))
        # self.head_time=0

        for j in range(self.n_head):
            

            new_q=K.dot(self.features,self.Q_a[str(j)])
            new_k=K.dot(self.features,self.K_a[str(j)])
            new_v=K.dot(self.features,self.V_a[str(j)])
            #print('new_v',new_v.shape)
            weights=K.batch_dot(new_q,new_k,axes=[2,2])/self.normalize
            weights=K.softmax(weights)
            #print('weights',weights.shape)
            attn=K.batch_dot(weights,new_v,axes=[2,1])

            self.concat_features.append(attn)

        self.out=K.dot(Concatenate()(self.concat_features),self.W)
        self.out1=Add()([self.features,self.out])
        self.out1=BatchNormalization()(self.out1)

        #self.out=K.dot(tf.convert_to_tensor(self.concat_features),self.W)

        self.out2=self.dense1(self.out1)
        self.out2=self.activation(self.out2)
        self.out2=self.dense2(self.out2)
        self.out3=Add()([self.out1,self.out2])
        self.final_out=BatchNormalization()(self.out3)

        return self.final_out
