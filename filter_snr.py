import numpy as np
from scipy.io import wavfile
from scipy import signal
import math
import scipy
import matplotlib.pyplot as plt
from mir_eval.separation import bss_eval_sources
import glob

def si_snr(x, s, remove_dc=True):
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

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

folders = np.loadtxt('/data/AV-speech-separation1/lrs2_3k_val_split.txt', dtype='object').tolist()
true_samples=[]
for item in folders:
    fss = sorted(glob.glob(item+'/*_samples.npy'), key=numericalSort)
    true_samples = true_samples + fss

print('Done reading true samples')

mix_samples = []
for item in true_samples:
    mixed_wav = '/data/mixed_audio_files/' + item.split('/')[-2] + '.wav'
    mix_samples.append(mixed_wav)

print('Done reading mixed samples')

snrs = []
for i, item in enumerate(true_samples):
    true_samp = np.load(item)[:32000]
    mix_samp = wavfile.read(mix_samples[i])[1][:32000]
    snrs.append(si_snr(mix_samp, true_samp))
    if i%1000 == 0:
        print(i, 'Done')

np.savetxt('/data/snrs_2comb_val.txt', snrs)

#snrs = np.loadtxt('/data/snrs_3comb.txt')

c = 0
for i in range(len(snrs)):
    if snrs[i]>= -5 and snrs[i]<=5:
        c = c+1

print('SNRs b/w -5 and 5:', c)

c = 0
for i in range(len(snrs)):
    if snrs[i] > 5:
        c = c+1

print('SNRs greater 5:', c)

c = 0
for i in range(len(snrs)):
    if snrs[i]< -5:
        c = c+1

print('SNRs less than -5:', c)

print('Mean SNR:', np.mean(snrs))
