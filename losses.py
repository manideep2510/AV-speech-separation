import tensorflow as tf
import keras
from  keras import backend as K
from keras.losses import sparse_categorical_crossentropy

def prepare_label(input_batch, new_size, num_classes, one_hot=True):
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)

    return input_batch

def l2_loss(spect1, spect2):
    #spect2 = spect2[:,:,:,0]
    loss = tf.sqrt(tf.nn.l2_loss(spect1[:,:,:,:2] - spect2[:,:,:,:2]))
    #loss = tf.sqrt(tf.nn.l2_loss(spect1 - spect2))
    return loss

def mse(target_mask, predicted_mask):
    #predicted_mask = predicted_mask[:,:,:,0]
    #print(predicted_mask.shape)
    loss = tf.keras.losses.MSE(target_mask[:,:,:,0], predicted_mask[:,:,:,0])
    loss = K.abs(loss)
    return loss


def sparse_categorical_crossentropy_loss(mask, pred):
    shape = K.int_shape(pred)
    print(mask.shape)
    print(pred.shape)
    pred = K.reshape(pred, [-1])
    mask = K.reshape(mask, [-1])

    return sparse_categorical_crossentropy(mask, pred)


def cross_entropy_loss(target_mask, predicted_mask):
#    raw_masks = predicted_mask
#    predicted_mask = tf.Tensor.getitem(predicted_mask, [:,:,:,:2])
    #predicted_mask = predicted_mask[:8,:,:,:]
    predicted_mask = predicted_mask[:,:,:,:2]
    predicted_mask = tf.reshape(predicted_mask, [-1, 2])
    #target_mask = tf.one_hot(tf.cast(target_mask, dtype=tf.int32), depth=2)
    target_mask = tf.reshape(target_mask, [-1,2])

    tbm_cross_entropy = keras.losses.categorical_crossentropy(target_mask, predicted_mask)
    #shape = K.int_shape(tbm_cross_entropy)
    #total = 4*shape[1]*shape[2]
    return tf.reduce_mean(tbm_cross_entropy, name='binary_mask_loss')
    #return tbm_cross_entropy
#def cross_entropy_loss(target_mask, predicted_mask):

#    tbm_cross_entropy = (tf.nn.sigmoid_cross_entropy_with_logits(labels=target_mask, logits=predicted_mask))
#    return tf.reduce_sum(tbm_cross_entropy, name='binary_mask_loss')


def categorical_crossentropy(y_true, y_pred):

    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)

    return K.mean(tf.keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1))
