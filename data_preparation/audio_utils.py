import os
from os.path import join
from glob import glob 
import random
import shutil
import numpy as np
from pydub import AudioSegment
import tensorflow as tf
import scipy
from scipy.io import wavfile
from scipy import signal
import math
from PIL import Image
import dlib
import skvideo.io
import time
import glob
import subprocess
import random
from PIL import Image
from scipy import signal
from scipy.io import wavfile
import math
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from numba import jit
home = str(Path.home())
# Avoid printing TF log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# For mixing two audios

def two_files_audio_sum(file_1_path, file_2_path,file_sum_name, volume_reduction=0):
    s1 = AudioSegment.from_file(file_1_path)
    s2 = AudioSegment.from_file(file_2_path) - volume_reduction # volume_reduction in dB

    s2_shift = (len(s1)-len(s2)) / 2 if len(s1) > len(s2) else 0
    
    audio_sum = s1.overlay(s2, position=s2_shift)
    audio_sum.export(file_sum_name, format='wav')

    return np.array(audio_sum.get_array_of_samples())

# Function to mix any number of audios

def audios_sum(audio_filenames_list,file_sum_name, volume_reduction=0):
    
    s = []
    for i, audio_file in enumerate(audio_filenames_list):
        
        if i == 0:
            s1 = AudioSegment.from_file(audio_file)
            s.append(s1)
        else:
            s1 = AudioSegment.from_file(audio_file) - volume_reduction # volume_reduction in dB
            s.append(s1)
    
    # Lenghts of audios
    s_len = []
    for i in audio_filenames_list:
        ps = AudioSegment.from_file(i)
        s_len.append(len(ps))
    
    max_index = s_len.index(max(s_len))

    s_max = s[max_index]
    del s[max_index]
    s.insert(0, s_max)

    # Sort elements in s according to their length
    #s = [x for _,x in sorted(zip(s_len,s), reverse=True)]
    '''s_shift = []
    for i, item in enumerate(s):
        
        if i == 0:
            s_shift1 = item
            s_shift.append(s_shift1)
            
        else:
            s_shift1 = (len(s[0])-len(item)) / 2 if len(s[0]) > len(item) else 0
            s_shift.append(s_shift1)'''
            
    for i in range(len(s)):
        
        if i == 0:
            audio_sum = s[0]
            
        elif i > 0:
            audio_sum = audio_sum.overlay(s[i], position=0)

    audio_sum.export(file_sum_name, format='wav')

    return np.array(audio_sum.get_array_of_samples())

def downsampling(samples, sample_rate, downsample_rate):
    secs = len(samples) / float(sample_rate)
    num_samples = int(downsample_rate * secs)

    return signal.resample(samples, num_samples)


'''def compute_spectrograms(audio_file, max_audio_length=500000, sample_rate=16e3, n_fft=512, window_size=25, step_size=10):
    
    
    
    window_frame_size = int(round(window_size / 1e3 * sample_rate))
    step_frame_size = int(round(step_size  / 1e3 * sample_rate))
    
    audio_samples = np.zeros((1, max_audio_length + n_fft//2))
    
    
    rate, samples = wavfile.read(audio_file)
    samples = downsampling(samples, rate, sample_rate)
    audio_samples[0, n_fft//2: len(samples) + n_fft//2] = samples
    num_frames = math.ceil(float(len(samples) + n_fft//2) / step_frame_size)
    
    # Create Graph
    with tf.Graph().as_default():
        samples_tensor = tf.constant(audio_samples, dtype=tf.float32)
        # Compute STFT
        specs_tensor = tf.contrib.signal.stft(samples_tensor, frame_length=window_frame_size, frame_step=step_frame_size,
                                              fft_length=n_fft, pad_end=True)
        # Apply power-law compression
        specs_tensor = tf.abs(specs_tensor) ** 0.3
    
        # Start session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False)) as sess:
            specs = sess.run(specs_tensor)

    return specs, num_frames
'''

# TF constants definition
TF_INTRA_OP_PT = int(os.getenv('TF_INTRA_OP', 0))
TF_INTER_OP_PT = int(os.getenv('TF_INTER_OP', 0))

# For single audio file

'''
Note: The training objective for VL2M is a Target Binary Mask (TBM), 
computed using the spectrogram of the tar- get speaker only. This 
is motivated by our goal of extracting the speech of a target speaker 
as much as possible indepen- dently of the concurrent speakers, so that, 
e.g., we do not need to estimate their number. An additional motivations 
is that the model takes as only input the visual features of the target 
speaker, and a target TBM that only depends on the target speaker allows 
VL2M to learn a function (rather than approximating an ill-posed one-to-many mapping).
'''

def ltass_speaker(audio_file, sample_rate=16e3, max_audio_length=1000000, window_size=25, step_size=10, n_samples=1000):
    """
    Compute the speaker Long-Term Average Speech Spectrum of a speaker.
    """
    
    audio_sample = np.zeros((max_audio_length))
    
    window_frame_size = int(round(window_size / 1e3 * sample_rate))
    step_frame_size = int(round(step_size  / 1e3 * sample_rate))
    
    rate, samples = wavfile.read(audio_file)
    samples = downsampling(samples, rate, sample_rate)
    audio_sample[:len(samples)] = samples
    num_frames = len(samples) // step_frame_size
    
    # Create Graph
    with tf.Graph().as_default():
        samples_tensor = tf.constant(audio_sample, dtype=tf.float32)
        # Compute STFT
        spec_tensor = tf.contrib.signal.stft(samples_tensor, frame_length=window_frame_size, frame_step=step_frame_size, pad_end=False)
        # Apply power-law compression
        spec_tensor = tf.abs(spec_tensor) ** 0.3
        
        # Start session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False,
                                              intra_op_parallelism_threads=TF_INTRA_OP_PT,
                                              inter_op_parallelism_threads=TF_INTER_OP_PT)) as sess:
            spec = sess.run(spec_tensor)

    nf = num_frames
    spectrogram = spec[:nf]
    
    mean_spec = spectrogram.mean()
    stdev_spec = spectrogram.std()
    
    return mean_spec, stdev_spec, spec


def compute_tbm(audio_file, mask_threshold, sample_rate=16e3, max_audio_length=500000, n_fft=512, window_size=25, step_size=10):
    """
    Compute TBMs using LTASS.
    """
    #audio_filenames = sorted(glob(join(audio_folder, '*.wav')))
    #num_frames = np.zeros(len(audio_filenames), dtype=np.int32)
    
    window_frame_size = int(round(window_size / 1e3 * sample_rate))
    step_frame_size = int(round(step_size  / 1e3 * sample_rate))
    
    audio_sample = np.zeros((max_audio_length + n_fft//2))
    
    #for i, wav_file in enumerate(audio_filenames):
    rate, samples = wavfile.read(audio_file)
    samples = downsampling(samples, rate, sample_rate)
    audio_sample[n_fft//2: len(samples) + n_fft//2] = samples
    num_frames = math.ceil(float(len(samples) + n_fft//2) / step_frame_size)
    
    # Create Graph
    with tf.Graph().as_default():
        samples_tensor = tf.constant(audio_sample, dtype=tf.float32)
        # Compute STFT
        spec_tensor = tf.contrib.signal.stft(samples_tensor, frame_length=window_frame_size, frame_step=step_frame_size, pad_end=True)
        spec_tensor = tf.abs(spec_tensor)

        # Start session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False,
                                              intra_op_parallelism_threads=TF_INTRA_OP_PT,
                                              inter_op_parallelism_threads=TF_INTER_OP_PT)) as sess:
            spec = sess.run(spec_tensor)
            
    mask = spec > mask_threshold
    
    return audio_file, mask, num_frames

def save_target_binary_mask_speaker(audio_file, mask_file, mask_factor=0.5, sample_rate=16e3, max_audio_length=500000, ltass_samples=1000):
    
    # Compute thresholds and spectrograms
    #print('Computing LTASS threshold...')
    
    rate, samples = wavfile.read(audio_file)
    samples = downsampling(samples, rate, sample_rate)
    shape_ = samples.shape
    
    if len(shape_)==1:
        
        threshold_mean, threshold_std, _ = ltass_speaker(audio_file, sample_rate, max_audio_length=max_audio_length, n_samples=ltass_samples)
        # Denormalize
        threshold_freq = (threshold_mean + threshold_std * mask_factor) ** (1 / 0.3)

        #print('done.')
        #print('Threshold shape:', threshold_freq.shape)

        # Compute binary masks
        audio_filename, mask, num_frames = compute_tbm(audio_file, threshold_freq, sample_rate, max_audio_length=max_audio_length)

        #for a_file, mask, nf in zip(audio_filenames, masks, num_frames):
        #s_file = os.path.join(audio_file[:-3]+'.npy')
        nf = num_frames
        np.save(mask_file, mask[:nf])

        #print('Done. Target Binary Masks generated:', len(audio_filenames))
        
    elif len(shape_)==2:
        
        print(audio_file)
        garb = np.asarray([0])
        np.save('/data/AV-speech-separation/data_preparation/garbage.npy', garb)
        
        
# Extract audio and save as .wav files 

def get_wav(mp4_file):
    
    out_path = mp4_file[:-3]+"wav"
    #print(out_path)
    command = "ffmpeg -i " + mp4_file + ' -codec:a' + ' pcm_s16le' + ' -ac' + ' 1 ' +'-y '+ out_path
    subprocess.call(command, shell=True)
    

def compare_lengths(file_1_path, file_2_path, max_duration_diff=2000):
    # max_duration_diff in milliseconds
    #print(file_1_path, 'and', file_1_path)
    file_1_path = file_1_path[:-9]+'.wav'
    file_2_path = file_2_path[:-9]+'.wav'
    s1 = AudioSegment.from_file(file_1_path)
    s2 = AudioSegment.from_file(file_2_path)
   
    return abs(len(s1) - len(s2)) < max_duration_diff


#Returns spectrogram, number of useful frames, complex_stft
def compute_spectrograms(audio_file, max_audio_length=500000, sample_rate=16e3, n_fft=512, window_size=25, step_size=10):



    window_frame_size = int(round(window_size / 1e3 * sample_rate))
    step_frame_size = int(round(step_size  / 1e3 * sample_rate))
    overlap_samples=window_frame_size-step_frame_size

    audio_samples = np.zeros(( max_audio_length + n_fft//2))


    rate, samples = wavfile.read(audio_file)
    if(sample_rate!=rate):samples = downsampling(samples, rate, sample_rate)
    audio_samples[ n_fft//2: len(samples) + n_fft//2] = samples
    num_frames = math.ceil(float(len(samples) + n_fft//2) / step_frame_size)


    spec_1=scipy.signal.stft(audio_samples, fs=sample_rate, window='hann', nperseg=window_frame_size, noverlap=overlap_samples, nfft=n_fft, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=-1)
    #power_law_compression
    specs=((abs(spec_1[2]))**0.3)


    return specs, num_frames, spec_1[2]



def split_spectrogram(spec_signal,rate=16e3,step_size=10,time=10):

    step_frame_size = int(round(step_size  / 1e3 * rate))
    n=(rate*time)/step_frame_size

    return spec_signal[:,0:n]



def ibm(spec_mix,spec_signal,threshold=1):

    signal_power=np.square(spec_signal)
    mixed_power=np.square(spec_mix)

    snr=np.divide(signal_power,mixed_power)


    mask2 = np.around(snr, 0)
    mask2[np.isnan(mask2)] = 1
    mask2[mask2 > threshold] = 1
    mask2

    return mask2



def irm(spec_mix,spec_signal):

    signal_power=np.square(spec_signal)
    mixed_power=np.square(spec_mix)
    noise_power=mixed_power-signal_power

    noise_power=noise_power*(noise_power>0)+0.0
    power_sum=signal_power+noise_power
    snr=np.divide(signal_power,power_sum)
    soft_mask=np.sqrt(np.nan_to_num(snr))

    return soft_mask




def tbm(spec_signal,mask_factor=0.5):

    mean_spec = spec_signal.mean()
    stdev_spec = spec_signal.std()
    threshold_freq = (mean_spec + stdev_spec * mask_factor) ** (1 / 0.3)

    spec=spec_signal**(1/0.3)
    mask=spec>threshold_freq

    return mask


def compress_crm(mixed_mag,mixed_phase,signal_mag,signal_phase,K=10,C=0.1):
    
    
    p=np.cos(mixed_phase)+1.j*np.sin(mixed_phase)
    Y=p*(mixed_mag**(10/3))
    
    p=np.cos(signal_phase)+1.j*np.sin(signal_phase)
    S=p*(signal_mag**(10/3))
    
    Yr=Y.real 
    Yi=Y.imag
    Sr=S.real
    Si=S.imag

    Mr=np.divide(np.add((Yr*Sr),(Yi*Si)),np.add(np.square(Yr),np.square(Yi)))
    Mi=np.divide(np.subtract((Yr*Si),(Yi*Sr)),np.add(np.square(Yr),np.square(Yi)))

    Cx=K*np.divide(1-np.exp(-1*C*Mr),1+np.exp(-1*C*Mr))
    Cy=K*np.divide(1-np.exp(-1*C*Mi),1+np.exp(-1*C*Mi))
    
    Cx=np.nan_to_num(Cx)
    Cy=np.nan_to_num(Cy)

    return Cx,Cy

def inverse_crm(real_part,imaginary_part,K=10,C=0.1):


    Mr=(1/C)*np.log(np.divide((K+real_part),(K-real_part)))
    Mi=(1/C)*np.log(np.divide((K+imaginary_part),(K-imaginary_part)))

    return Mr+1.j*Mi


def return_samples_complex(mixed_mag,mixed_phase,mask,sample_rate=16e3, n_fft=512, window_size=25, step_size=10):

    window_frame_size = int(round(window_size / 1e3 * sample_rate))
    step_frame_size = int(round(step_size  / 1e3 * sample_rate))
    overlap_samples=window_frame_size-step_frame_size

    
    p=np.cos(mixed_phase)+1.j*np.sin(mixed_phase)
    mixed_speech=p*(mixed_mag**(10/3))
    
    stft=mixed_speech*mask
    
    predicted_samples=scipy.signal.istft(stft, fs=sample_rate, window='hann', nperseg=window_frame_size, noverlap=overlap_samples, nfft=n_fft, input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2)
    samples=np.asarray(list(map(int, predicted_samples[1])),dtype='int16')
    
    return samples

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


def retrieve_samples(spec_signal,complex_stft,mask,sample_rate=16e3, n_fft=512, window_size=25, step_size=10):

    window_frame_size = int(round(window_size / 1e3 * sample_rate))
    step_frame_size = int(round(step_size  / 1e3 * sample_rate))
    overlap_samples=window_frame_size-step_frame_size


    spec_predicted=spec_signal*mask

    phase=np.angle(complex_stft)
    p = np.cos(phase) + 1.j * np.sin(phase)

    stft=p*(spec_predicted**(10/3))
    predicted_samples=scipy.signal.istft(stft, fs=sample_rate, window='hann', nperseg=window_frame_size, noverlap=overlap_samples, nfft=n_fft, input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2)
    samples=np.asarray(list(map(int, predicted_samples[1])),dtype='int16')

    return samples



def visualize_overlap(predicted_samples,groundtruth_samples):

    fig = plt.figure()
    plt.plot( groundtruth_samples, color="red", alpha = 0.6)
    plt.plot( predicted_samples, color="blue", alpha = 0.4)
    plt.title("Recovery for speaker 1")
    plt.xlabel('Time [sec]')
    plt.ylabel('Signal')
    plt.legend(['Original', 'Recovered via STFT'])


    plt.show()
    plt.close(fig)


def save_audio(samples,rate,file):
    wavfile.write(file,rate,samples)
