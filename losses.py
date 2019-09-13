import tensorflow as tf
from  keras import backend as K
from keras.losses import sparse_categorical_crossentropy

def l2_loss(spect1, spect2):
    loss = tf.sqrt(tf.nn.l2_loss(spect1[:,:,:,0] - spect2[:,:,:,0]))
    #loss = tf.sqrt(tf.nn.l2_loss(spect1 - spect2))
    return loss

def sparse_categorical_crossentropy_loss(mask, pred):
    shape = K.int_shape(pred)
    print(mask.shape)
    print(pred.shape)
    pred = K.reshape(pred, [-1])
    mask = K.reshape(mask, [-1])

    return sparse_categorical_crossentropy(mask, pred)


def cross_entropy_loss(target_mask, predicted_mask):

    tbm_cross_entropy = (tf.nn.sigmoid_cross_entropy_with_logits(labels=target_mask, logits=predicted_mask))
    shape = K.int_shape(tbm_cross_entropy)
    total = 4*shape[1]*shape[2]
    return tf.reduce_sum(tbm_cross_entropy, name='binary_mask_loss')/total
