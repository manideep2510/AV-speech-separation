import tensorflow as tf
import tensorflow.keras as keras
from  tensorflow.keras import backend as K
from tensorflow.keras.losses import sparse_categorical_crossentropy
 
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

def hinge_loss(y_true, y_pred):

    l = tf.keras.losses.Huber(delta=1)
    loss = l(y_true[:,:,:,:2], y_pred[:,:,:,:2])
    return loss

def l1_loss(spect1, spect2):
    #spect2 = spect2[:,:,:,0] 
    #batch = spect1.shape[0]
    #loss = tf.reduce_mean(tf.math.abs(spect1[:,:,:,:2] - spect2[:,:,:,:2]))
    mae = tf.keras.losses.MeanAbsoluteError()
    loss = mae(spect1[:,:,:,:2], spect2[:,:,:,:2])
    #loss = tf.sqrt(tf.nn.l2_loss(spect1 - spect2))
    return loss

def crm_to_mag_phase(crm, mixed_spect, mixed_phase):
    real = crm[:,:,:,0]
    imag = crm[:,:,:,1]

    real_phase = tf.math.cos(mixed_phase)
    imag_phase = tf.math.sin(mixed_phase)

    mixed_stft_real = tf.math.multiply(real_phase,(tf.math.pow(mixed_spect, 10/3)))
    mixed_stft_imag = tf.math.multiply(imag_phase,(tf.math.pow(mixed_spect, 10/3)))

    stft_real = tf.math.multiply(real, mixed_stft_real)
    stft_imag = tf.math.multiply(imag, mixed_stft_imag)

    mag = tf.math.sqrt(tf.math.abs(tf.math.add(tf.math.square(stft_real), tf.math.square(stft_imag)))+ 1e-10)
    phase = tf.math.atan(tf.math.divide(stft_imag, stft_real))
    #phase = tf.math.angle(tf.dtypes.complex(stft_real, stft_imag))
    phase = tf.where(tf.is_nan(phase), tf.ones_like(phase) * 1.5707964, phase)

    return mag, phase


def mag_phase_loss(crms_gt, y_pred):
    #crms_gt = y_true
    mixed_spect = y_pred[:,:,:,2]
    mixed_phase = y_pred[:,:,:,3]
    crms = y_pred[:,:,:,:2]

    inverse_crm = tf.math.atanh(crms)

    '''Cx = 0.9999999*tf.dtypes.cast((crms_gt>0.9999999), dtype=tf.float32)+crms_gt*tf.dtypes.cast((crms_gt<=0.9999999), dtype=tf.float32)
    crms_gt = -0.9999999*tf.dtypes.cast((Cx<-0.9999999), dtype=tf.float32)+Cx*tf.dtypes.cast((Cx>=-0.9999999), dtype=tf.float32)
'''
    inverse_crm_gt = tf.math.atanh(crms_gt)

    mag_pred, phase_pred = crm_to_mag_phase(inverse_crm, mixed_spect, mixed_phase)
    mag_true, phase_true = crm_to_mag_phase(inverse_crm_gt, mixed_spect, mixed_phase)

    # Mag and Phase loss calculation
    #mae = tf.keras.losses.MeanAbsoluteError()
    #mag_loss = mae(mag_pred, mag_true)
    mag_loss = tf.reduce_mean(tf.math.abs(mag_pred - mag_true))

    phase_loss = tf.reduce_mean(tf.math.multiply(tf.math.cos(tf.math.subtract(phase_pred, phase_true)), mag_true))

    loss = tf.math.subtract(mag_loss, 0.4*phase_loss)
    
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

def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, _, T = source.size()
    mask = source.new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return mask

def cal_loss(source, estimate_source, source_lengths):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    max_snr, perms, max_snr_idx = cal_si_snr_with_pit(source,
                                                      estimate_source,
                                                      source_lengths)
    loss = 0 - tf.reduce_mean(max_snr)
    #reorder_estimate_source = reorder_source(estimate_source, perms, max_snr_idx)
    return loss


def cal_si_snr_with_pit(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    assert K.int_shape(source) == K.int_shape(estimate_source)
    B, C, T = K.int_shape(source)
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]

    # Get max_snr of each utterance
    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= C
    return max_snr, perms, max_snr_idx

def snr_loss(s, x, remove_dc=True):
    """
    Compute SI-SNR
    Arguments:
        x: vector, enhanced/separated signal
        s: vector, reference signal(ground truth)
    """

    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    # zero mean, seems do not hurt results
    if remove_dc:
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)
        t = np.inner(x_zm, s_zm) * s_zm / vec_l2norm(s_zm)**2
        n = x_zm - t
    else:
        t = np.inner(x, s) * s / vec_l2norm(s)**2
        n = x - t
    return 20 * np.log10(vec_l2norm(t) / vec_l2norm(n))
