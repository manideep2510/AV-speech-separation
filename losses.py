import tensorflow as tf

def l2_loss(spect1, spect2):
    loss = tf.sqrt(tf.nn.l2_loss(spect1[:,:,:,0] - spect2[:,:,:,0]))
    #loss = tf.sqrt(tf.nn.l2_loss(spect1 - spect2))
    return loss

def cross_entropy_loss(target_mask, predicted_mask):

	tbm_cross_entropy = (tf.nn.sigmoid_cross_entropy_with_logits(labels=target_mask, logits=predicted_mask)) 
        return tf.reduce_sum(tbm_cross_entropy, name='binary_mask_loss')
