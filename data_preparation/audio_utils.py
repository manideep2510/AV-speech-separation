import os
from os.path import join
from glob import glob 
import random
import shutil
import numpy as np
from pydub import AudioSegment
import tensorflow as tf
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
    for i in s:
        l = len(i)
        s_len.append(l)
        
    # Sort elements in s according to their length
    s = [x for _,x in sorted(zip(s_len,s), reverse=True)]
    
    s_shift = []
    for i, item in enumerate(s):
        
        if i == 0:
            s_shift1 = item
            s_shift.append(s_shift1)
            
        else:
            s_shift1 = (len(s[0])-len(item)) / 2 if len(s[0]) > len(item) else 0
            s_shift.append(s_shift1)
            
    for i in range(len(s)):
        
        if i == 0:
            audio_sum = s[0]
            
        elif i > 0:
            audio_sum = audio_sum.overlay(s[i], position=s_shift[i])

    audio_sum.export(file_sum_name, format='wav')

    return np.array(audio_sum.get_array_of_samples())

def downsampling(samples, sample_rate, downsample_rate):
    secs = len(samples) / float(sample_rate)
    num_samples = int(downsample_rate * secs)

    return signal.resample(samples, num_samples)


def compute_spectrograms(audio_file, max_audio_length=500000, sample_rate=16e3, n_fft=512, window_size=25, step_size=10):
    
    
    
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
        np.save('garbage.npy', garb)
        
        
# Extract audio and save as .wav files 

def get_wav(mp4_file):
    
    out_path = mp4_file[:-3]+"wav"
    #print(out_path)
    command = "ffmpeg -i " + mp4_file + ' -codec:a' + ' pcm_s16le' + ' -ac' + ' 1 ' + out_path
    subprocess.call(command, shell=True)
    

def compare_lengths(file_1_path, file_2_path, max_duration_diff=2000):
    # max_duration_diff in milliseconds
    s1 = AudioSegment.from_file(file_1_path)
    s2 = AudioSegment.from_file(file_2_path)
   
    return abs(len(s1) - len(s2)) < max_duration_diff

