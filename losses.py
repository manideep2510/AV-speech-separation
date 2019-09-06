import tensorflow as tf

def l2_loss(spect1, spect2):
    loss = tf.sqrt(tf.nn.l2_loss(spect1[:,:,:,0] - spect2[:,:,:,0]))
    #loss = tf.sqrt(tf.nn.l2_loss(spect1 - spect2))
    return loss