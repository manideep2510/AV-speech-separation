import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K

def vec_l2norm(x):

    nr = K.sqrt(tf.math.reduce_sum(K.square(x), axis=1))
    nr = tf.reshape(nr, (-1, 1))
    #nr = tf.broadcast_to(nr, (int(x.shape[1]), int(x.shape[0])))
    return nr

def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator


def snr_loss(s, x):
    """
    Compute SI-SNR
    Arguments:
        x: vector, enhanced/separated signal
        s: vector, reference signal(ground truth)
    """

    x = x[:,:,0]

    #print(x.shape[0])
    #s = x[:,:,1]

    x = tf.reshape(x, (-1, 32000))
    s = tf.reshape(s, (-1, 32000))

    '''print('Pred:', x.shape)
    print('GT:', s.shape)'''

    # zero mean, seems do not hurt results
    x_zm = x - tf.reshape(tf.math.reduce_mean(x, axis=1), (-1, 1))
    s_zm = s - tf.reshape(tf.math.reduce_mean(s, axis=1), (-1, 1))
    t = tf.reshape(tf.math.reduce_sum(x_zm*s_zm, axis=1), (-1, 1)) * s_zm / vec_l2norm(s_zm)**2
    n = x_zm - t

    snr_loss_batch = 20 * log10(vec_l2norm(t) / vec_l2norm(n))

    return -tf.reduce_mean(snr_loss_batch)


def snr_loss_new(s, x):
    """
    Compute SI-SNR
    Arguments:
        x: vector, enhanced/separated signal
        s: vector, reference signal(ground truth)
    """

    x = x[:,:,0]

    #print(x.shape[0])
    #s = x[:,:,1]

    x = tf.reshape(x, (-1, 32000))
    s = tf.reshape(s, (-1, 32000))

    '''print('Pred:', x.shape)
    print('GT:', s.shape)'''

    # zero mean, seems do not hurt results
    x_zm = x - tf.reshape(tf.math.reduce_mean(x, axis=1), (-1, 1))
    s_zm = s - tf.reshape(tf.math.reduce_mean(s, axis=1), (-1, 1))
    t = tf.reshape(tf.math.reduce_sum(x_zm*s_zm, axis=1), (-1, 1)) * s_zm / vec_l2norm(s_zm)**2
    n = x_zm - t

    snr_loss_batch = 20 * log10((vec_l2norm(n) / vec_l2norm(t))+1)

    return tf.reduce_mean(snr_loss_batch)


def snr_acc(s, x):
    """
    Compute SI-SNR
    Arguments:
        x: vector, enhanced/separated signal
        s: vector, reference signal(ground truth)
    """

    x = x[:,:,0]
    #s = x[:,:,1]

    x = tf.reshape(x, (-1, 32000))
    s = tf.reshape(s, (-1, 32000))

    '''print('Pred:', x.shape)
    print('GT:', s.shape)'''

    # zero mean, seems do not hurt results
    x_zm = x - tf.reshape(tf.math.reduce_mean(x, axis=1), (-1, 1))
    s_zm = s - tf.reshape(tf.math.reduce_mean(s, axis=1), (-1, 1))
    t = tf.reshape(tf.math.reduce_sum(x_zm*s_zm, axis=1), (-1, 1)) * s_zm / vec_l2norm(s_zm)**2
    n = x_zm - t

    snr_loss_batch = 20 * log10(vec_l2norm(t) / vec_l2norm(n))

    return tf.reduce_mean(snr_loss_batch)


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#LAMBDA = 100
def generator_loss_LSGAN(disc_generated_output, gen_output, target, LAMBDA):
    #gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    gan_loss = tf.math.reduce_mean(tf.math.square(disc_generated_output - tf.ones_like(disc_generated_output)))/2
    #gan_loss = tf.math.reduce_mean(tf.math.square(disc_generated_output))/2

    # SNR Loss
    loss_snr = snr_loss(target, gen_output)
    #loss_snr = tf.math.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + LAMBDA*(loss_snr)

    return total_gen_loss, gan_loss, loss_snr

def discriminator_loss_LSGAN(disc_real_output, disc_generated_output):
    #real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    real_loss = tf.math.square(disc_real_output - tf.random.uniform(tf.shape(disc_real_output), minval=0.9, maxval=1.0))/2
    #real_loss = tf.math.square(disc_real_output - tf.ones_like(disc_real_output))/2
    '''test_accuracy = tf.keras.metrics.Accuracy()
    real_acc = test_accuracy(tf.ones_like(disc_real_output).numpy().tolist(), sigmoids_to_10(disc_real_output))'''

    #generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    generated_loss = tf.math.square(disc_generated_output - tf.random.uniform(tf.shape(disc_generated_output), minval=0.0, maxval=0.1))/2
    #generated_loss = tf.math.square(disc_generated_output + tf.random.uniform(tf.shape(disc_generated_output), minval=0.9, maxval=1.0))/2
    #generated_loss = tf.math.square(disc_generated_output)/2

    '''fake_acc = test_accuracy(tf.ones_like(disc_generated_output).numpy().tolist(), sigmoids_to_10(disc_generated_output))'''
    total_disc_loss = real_loss + generated_loss

    total_disc_loss = tf.math.reduce_mean(total_disc_loss)

    return total_disc_loss, tf.math.reduce_mean(real_loss), tf.math.reduce_mean(generated_loss)

def discriminator_loss_LSGAN_GP(disc_real_output, disc_generated_output, gradient_penalty, gradient_penalty_weight):
    #real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    #real_loss = tf.math.square(disc_real_output)/2
    real_loss = tf.math.square(disc_real_output - tf.random.uniform(tf.shape(disc_real_output), minval=0.9, maxval=1.0))/2
    #real_loss = tf.math.square(disc_real_output - tf.ones_like(disc_real_output))/2
    '''test_accuracy = tf.keras.metrics.Accuracy()
    real_acc = test_accuracy(tf.ones_like(disc_real_output).numpy().tolist(), sigmoids_to_10(disc_real_output))'''

    #generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    generated_loss = tf.math.square(disc_generated_output - tf.random.uniform(tf.shape(disc_generated_output), minval=0.0, maxval=0.1))/2

    #generated_loss = tf.math.square(disc_generated_output)/2

    '''fake_acc = test_accuracy(tf.ones_like(disc_generated_output).numpy().tolist(), sigmoids_to_10(disc_generated_output))'''
    total_disc_loss = real_loss + generated_loss

    total_disc_loss = tf.math.reduce_mean(total_disc_loss)

    total_disc_loss += gradient_penalty_weight*gradient_penalty

    return total_disc_loss, tf.math.reduce_mean(real_loss), tf.math.reduce_mean(generated_loss), gradient_penalty

def generator_loss_WGAN_GP(disc_generated_output, gen_output, target, LAMBDA):
    #gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    gan_loss = -tf.math.reduce_mean(disc_generated_output)

    # SNR Loss
    loss_snr = snr_loss(target, gen_output)
    #loss_snr = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + LAMBDA*(loss_snr)

    return total_gen_loss, gan_loss, loss_snr

def discriminator_loss_WGAN_GP(disc_real_output, disc_generated_output, gradient_penalty, gradient_penalty_weight=10):

    total_disc_loss = tf.math.reduce_mean(disc_generated_output) - tf.reduce_mean(disc_real_output)

    #gradient_pen = gradient_penalty(inputs, target, gen_output, discriminator)
    total_disc_loss += gradient_penalty_weight*gradient_penalty

    return total_disc_loss, tf.reduce_mean(disc_real_output), tf.math.reduce_mean(disc_generated_output), gradient_penalty

def generator_loss_WGAN(disc_generated_output, gen_output, target, LAMBDA):
    #gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    gan_loss = -tf.math.reduce_mean(disc_generated_output)

    # SNR Loss
    loss_snr = snr_loss(target, gen_output)
    #loss_snr = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + LAMBDA*(loss_snr)

    return total_gen_loss, gan_loss, loss_snr

def discriminator_loss_WGAN(disc_real_output, disc_generated_output):

    total_disc_loss = tf.math.reduce_mean(disc_generated_output) - tf.reduce_mean(disc_real_output)

    return total_disc_loss, tf.reduce_mean(disc_real_output), tf.math.reduce_mean(disc_generated_output)
