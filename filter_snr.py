import numpy as np
from scipy.io import wavfile
from scipy import signal
import math
import scipy
import matplotlib.pyplot as plt
#from mir_eval.separation import bss_eval_sources
import glob
import random

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

print('Ok')

#folders = np.loadtxt('/data/AV-speech-separation1/lrs2_comb3_100k_train.txt', dtype='object').tolist()
#folders = random.sample(folders, 10000)

folders = np.loadtxt(
    '/data/AV-speech-separation1/lrs2_comb3_val_snr_filter.txt', dtype='object').tolist()
random.seed(1234)
folders = random.sample(folders, 5000)

true_samples = []
for item in folders:
    true_samples.append(item[:-9]+'_samples.npy')

'''true_samples=[]
for i, item in enumerate(folders):
    fss = sorted(glob.glob(item+'/*_samples.npy'), key=numericalSort)
    true_samples = true_samples + fss
    if i % 100 == 0:
        print(i, 'Done')'''

#print(true_samples[1])

print('Done reading true samples')

mix_samples = []
for item in true_samples:
    mixed_wav = '/data/mixed_audio_files/' + item.split('/')[-2] + '.wav'
    mix_samples.append(mixed_wav)

#print(mix_samples[1])

print('Done reading mixed samples')

snrs = []
snrs_files = []
for i, item in enumerate(true_samples):
    true_samp = np.load(item)[:32000]
    mix_samp = wavfile.read(mix_samples[i])[1][:32000]
    snr_ =  si_snr(mix_samp, true_samp)
    if snr_ >= -5 and snr_ <= 5:
        snrs_files.append(item[:-12]+'_lips.mp4')
    snrs.append(snr_)
    if i%1000 == 0:
        print(i, 'Done')

print('SNR mean:', np.mean(snrs))
#np.savetxt('/data/snrs_2comb_val.txt', snrs)
#np.savetxt('/data/lrs2_comb3_train_snr_filter.txt', snrs_files, fmt='%s')
#snrs = np.loadtxt('/data/snrs_3comb.txt')

plt.hist(snrs, 10, facecolor='g')

plt.xlabel('SNR')
plt.ylabel('Count')
plt.title('SNR Histogram 3 speakers Val')
plt.grid(True)
plt.savefig('/data/AV-speech-separation1/snr_hist_3speak_val.png')

'''c = 0
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

print('Mean SNR:', np.mean(snrs))'''
