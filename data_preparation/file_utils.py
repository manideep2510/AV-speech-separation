import os
from os.path import join
from glob import glob 
import random
import shutil
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
from scipy import signal
import math
from PIL import Image
import skvideo.io
import time
import glob
import subprocess
import random
from pathlib import Path
import shutil
home = '/data'
from numba import jit

from audio_utils import compare_lengths, compute_spectrograms, audios_sum, ibm, irm, compress_crm, inverse_crm, return_samples_complex
import cv2

# From all the existing files, pick certain number files randomly and pair certain number of them together. (Just the file names)
def pair_files(files, combination_no=1, count=10):
    
    if combination_no <= 0 or count <= 0:
        print('Inputs not valid')
        return 0
    
    combined_list = []
    
    picks_no = combination_no*count
    
    #a = time.time()
    for n, item in enumerate(files):
        #print(n)
        
        #global files_copy
        
        files_copy = files[:]
        files_copy.remove(item)
        
        if len(combined_list)>0:
            if combination_no == 1:
                for done in combined_list:
                    if item in done:
                        index = done.index(item)
                        del_index = int(not index)
                        del_file = done[del_index]
                        files_copy.remove(del_file)
                        
        '''# Get the audio files with difference of > 2 seconds
        files_copy_not_2secs = []
        files_copy1 = files_copy[:]
        for i in files_copy1:
            if not compare_lengths(item, i):
                files_copy.remove(i)'''
                
        count_ = 0 
        picks = []
        while count_<=picks_no:
            pick = random.sample(files_copy, 1)[0]
            #print(pick)
            files_copy.remove(pick)
            if compare_lengths(item, pick):
                count_ += 1
                picks.append(pick)
            
        # Sample the audios from the filtered filelist
        random.seed(1)
        picks = random.sample(files_copy, picks_no)
        #print(len(files_copy))
        
        for i in range(count):
            
            if combination_no == 1:
                #random.seed(2)
                picked = picks[i]
                
                combined = [0]*2
                combined[0] = item
                combined[1] = picked

                #picks = [e for e in picks if e is not picked]
                
            else:
                
                random.seed(3)
                picked = random.sample(picks, combination_no)
                
                #picks = [e for e in picks if e is not picked]
                
                picked.append(item)
                
                combined = picked

                for it in picked:
                    picks.remove(it)
                
            combined_list.append(combined)
        #b = time.time()
        #print(n, 'Seconds', b-a)
            
    return combined_list


'''# Generate training folders for one combination (Modified)

def gen_comb_folders(combined_pairs, dest_folder):
    
    try:
        os.mkdir(dest_folder)
    except OSError:
        pass

    videos = combined_pairs
    audios = []
    masks = []
    
    for item in videos:
        audio = item[:-9] + '.wav'
        audios.append(audio)
        
    for item in videos:
        mask = item[:-9] + '_mask.npy'
        masks.append(mask)
    
    folder_name = videos[-1].split('/')[-2] + '_' + videos[-1][:-4].split('/')[-1]
    folder_path = dest_folder + '/' + folder_name
        
    spectrograms = []
    frames = []
    for i in range(len(videos)):
        video = videos[i]
        audio = audios[i]
        
        # Get frames
        frame=get_video_frames(video)
        
        frames.append(frame)
        
    ## Now mix the audios and compute the spectrogram of mixed audio
        
    mix_audio_folder = home + '/mixed_audio_files'
        
    # Filename to save the mixed audio  file
    mixed_audio_filename = folder_name + '.wav'             
        
    mixed_audio_filepath = mix_audio_folder + '/' + mixed_audio_filename
        
    try:
        os.mkdir(mix_audio_folder)
    except OSError:
        pass
        
    mixed_samples = audios_sum(audios, mixed_audio_filepath)
        
    # Compute the spectrogram of mised audio
    s,n=compute_spectrograms(mixed_audio_filepath)
    mixed_spectogram =s[0][:n,:]                     # Useful frames
        

    for p in range(len(videos)):
            
        frame = frames[p]
        #mask = masks[p]
        audio_file = audios[p]
        lips_file = videos[p]
        mask_file = masks[p]
        
        save_path = folder_path + '_' + str(p)
        
        try:
            os.mkdir(save_path)
        except OSError:
            pass
            
        # Save lips.mp4
            
        shutil.copy2(lips_file, save_path)
            
        # Save the mask
                 
        shutil.copy2(mask_file, save_path)
            
        # Save the mixed spectrogram
        np.save(save_path + '/' + 'mixed_spectrogram.png',mixed_spectogram)
'''

# Generate training folders for one combination (Modified)

def gen_comb_folders(combined_pairs, dest_folder):
    
    try:
        os.mkdir(dest_folder)
    except OSError:
        pass

    videos = combined_pairs
    audios = []
    
    for item in videos:
        audio = item[:-9] + '.wav'
        audios.append(audio)
    
    
    folder_name_list = []
    for path in audios:
        split_ = path.split('/')
        fold = split_[-2] + '_' + split_[-1][:-4]
        folder_name_list.append(fold)
        
    folder_name = '_'.join(folder_name_list) + '_' + str(len(videos))
    folder_path = dest_folder + '/' + folder_name
    
    try:
        os.mkdir(folder_path)
    except OSError:
        pass

    ## Now mix the audios and compute the spectrogram of mixed audio
    
    mix_audio_folder = home + '/mixed_audio_files'
        
    # Filename to save the mixed audio  file
    mixed_audio_filename = folder_name + '.wav'             
        
    mixed_audio_filepath = mix_audio_folder + '/' + mixed_audio_filename
        
    try:
        os.mkdir(mix_audio_folder)
    except OSError:
        pass
    
    mixed_samples = audios_sum(audios, mixed_audio_filepath)
        
    # Compute the spectrogram of mised audio
    s, n, c = compute_spectrograms(mixed_audio_filepath)
    mixed_spectogram =s[:,:500]  # Useful frames

    phase=np.angle(c)
    phase=phase[:,:500]

    mixed_spectogram = np.asarray(mixed_spectogram, dtype='float16')
    
    phase_spectogram = np.asarray(phase, dtype='float16')

    # Save the mixed spectrogram
    np.save(folder_path + '/' + 'mixed_spectrogram.npy',mixed_spectogram)
    np.save(folder_path + '/' + 'phase_spectrogram.npy',phase_spectogram)

    for p in range(len(videos)):

        audio_file = audios[p]
        lips_file = videos[p]
       # mask_file = masks[p]
        
        save_path = folder_path
        
        try:
            os.mkdir(save_path)
        except OSError:
            pass
        
        audio_file_split = audio_file.split('/')

        # Save lips.mp4
        
        file_name = audio_file_split[-2] + '_' + audio_file_split[-1][:-4] + '_lips.mp4'
            
        shutil.copy(lips_file, save_path + '/' + file_name)
            
        # Save the mask
        
        s, n, c_ = compute_spectrograms(audio_file)
        s_use=s[:, :500]
        
        mask = ibm(spec_mix = mixed_spectogram,spec_signal = s_use,threshold=1)
        
#        audio_file_split = audio_file.split('/')
        
        file_name = audio_file_split[-2] + '_' + audio_file_split[-1][:-4] + '_mask.png'
        
        cv2.imwrite(save_path + '/' + file_name, mask)

        mask1 = irm(spec_mix = mixed_spectogram,spec_signal = s_use)
        mask1=np.asarray(mask1, dtype='float16')
        file_name = audio_file_split[-2] + '_' + audio_file_split[-1][:-4] + '_softmask.npy'

        np.save(save_path + '/' + file_name, mask1)

        '''# Save phase spect

        phase_=np.angle(c_)
        phase_=phase_[:,:500]
        phase_ = np.asarray(phase_, dtype='float16')

        file_name = audio_file_split[-2] + '_' + audio_file_split[-1][:-4] + '_phase.npy'
        
        np.save(save_path + '/' + file_name, phase_)'''

#        shutil.copy(mask_file, save_path)

def gen_comb_folders_audios(combined_pairs, dest_folder):
    
    try:
        os.mkdir(dest_folder)
    except OSError:
        pass

    videos = combined_pairs
    audios = []
    
    for item in videos:
        audio = item[:-9] + '.wav'
        audios.append(audio)
    
    
    folder_name_list = []
    for path in audios:
        split_ = path.split('/')
        fold = split_[-2] + '_' + split_[-1][:-4]
        folder_name_list.append(fold)
        
    folder_name = '_'.join(folder_name_list) + '_' + str(len(videos))
    folder_path = dest_folder + '/' + folder_name
    
    try:
        os.mkdir(folder_path)
    except OSError:
        pass

    ## Now mix the audios and compute the spectrogram of mixed audio
    
    mix_audio_folder = home + '/mixed_audio_files'
        
    # Filename to save the mixed audio  file
    mixed_audio_filename = folder_name + '.wav'             
        
    mixed_audio_filepath = mix_audio_folder + '/' + mixed_audio_filename
        
    try:
        os.mkdir(mix_audio_folder)
    except OSError:
        pass
    
    #mixed_samples = audios_sum(audios, mixed_audio_filepath)
        
    # Compute the spectrogram of mised audio
    #s, n, c = compute_spectrograms(mixed_audio_filepath)
    #mixed_spectogram =s[:,:500]  # Useful frames

    #phase=np.angle(c)
    #phase=phase[:,:500]

    #mixed_spectogram = np.asarray(mixed_spectogram, dtype='float16')
    
    #phase_spectogram = np.asarray(phase, dtype='float16')

    # Save the mixed spectrogram
    #np.save(folder_path + '/' + 'mixed_spectrogram.npy',mixed_spectogram)
    #np.save(folder_path + '/' + 'phase_spectrogram.npy',phase_spectogram)

    for p in range(len(videos)):

        audio_file = audios[p]
        lips_file = videos[p]
       # mask_file = masks[p]
        
        save_path = folder_path
        
        try:
            os.mkdir(save_path)
        except OSError:
            pass
        
        audio_file_split = audio_file.split('/')

        # Save lips.mp4
        
        file_name = audio_file_split[-2] + '_' + audio_file_split[-1][:-4] + '_samples.npy'
        rate, audio_samples = wavfile.read(audio_file)
        np.save(save_path + '/' + file_name, audio_samples)
        
            
def gen_comb_folders_text(combined_pairs, dest_folder):
    
    try:
        os.mkdir(dest_folder)
    except OSError:
        pass

    videos = combined_pairs
    audios = []
    texts = []
    
    for item in videos:
        audio = item[:-9] + '.wav'
        audios.append(audio)
    
    for item in videos:
        txt = item[:-9] + '.txt'
        texts.append(txt)
    
    folder_name_list = []
    for path in audios:
        split_ = path.split('/')
        fold = split_[-2] + '_' + split_[-1][:-4]
        folder_name_list.append(fold)
        
    folder_name = '_'.join(folder_name_list) + '_' + str(len(videos))
    folder_path = dest_folder + '/' + folder_name
    
    try:
        os.mkdir(folder_path)
    except OSError:
        pass

    ## Now mix the audios and compute the spectrogram of mixed audio
    
    mix_audio_folder = home + '/mixed_audio_files'
        
    # Filename to save the mixed audio  file
    mixed_audio_filename = folder_name + '.wav'             
        
    mixed_audio_filepath = mix_audio_folder + '/' + mixed_audio_filename
        
    try:
        os.mkdir(mix_audio_folder)
    except OSError:
        pass
    
    #mixed_samples = audios_sum(audios, mixed_audio_filepath)
        
    # Compute the spectrogram of mised audio
    #s, n, c = compute_spectrograms(mixed_audio_filepath)
    #mixed_spectogram =s[:,:500]  # Useful frames

    #phase=np.angle(c)
    #phase=phase[:,:500]

    #mixed_spectogram = np.asarray(mixed_spectogram, dtype='float16')
    
    #phase_spectogram = np.asarray(phase, dtype='float16')

    # Save the mixed spectrogram
    #np.save(folder_path + '/' + 'mixed_spectrogram.npy',mixed_spectogram)
    #np.save(folder_path + '/' + 'phase_spectrogram.npy',phase_spectogram)

    for p in range(len(videos)):

        audio_file = audios[p]
        text = texts[p]
       # mask_file = masks[p]
        
        save_path = folder_path
        
        try:
            os.mkdir(save_path)
        except OSError:
            pass
        
        audio_file_split = audio_file.split('/')

        # Save lips.mp4
        
        file_name = audio_file_split[-2] + '_' + audio_file_split[-1][:-4] + '.txt'
        shutil.copy(text, save_path + '/' + file_name)
        
        
def gen_comb_folders_crm_nonumba(combined_pairs, dest_folder):
    
    try:
        os.mkdir(dest_folder)
    except OSError:
        pass

    videos = combined_pairs
    audios = []
    
    for item in videos:
        audio = item[:-9] + '.wav'
        audios.append(audio)
    
    
    folder_name_list = []
    for path in audios:
        split_ = path.split('/')
        fold = split_[-2] + '_' + split_[-1][:-4]
        folder_name_list.append(fold)
        
    folder_name = '_'.join(folder_name_list) + '_' + str(len(videos))
    folder_path = dest_folder + '/' + folder_name
    
    try:
        os.mkdir(folder_path)
    except OSError:
        pass

    ## Now mix the audios and compute the spectrogram of mixed audio
    
    mix_audio_folder = home + '/mixed_audio_files'
        
    # Filename to save the mixed audio  file
    mixed_audio_filename = folder_name + '.wav'             
        
    mixed_audio_filepath = mix_audio_folder + '/' + mixed_audio_filename
        
    try:
        os.mkdir(mix_audio_folder)
    except OSError:
        pass
    
    mixed_samples = audios_sum(audios, mixed_audio_filepath)
        
    # Compute the spectrogram of mised audio
    s, n, c = compute_spectrograms(mixed_audio_filepath)
    mixed_spectogram =s[:,:500]  # Useful frames

    phase=np.angle(c)
    phase=phase[:,:500]

    mixed_spectogram = np.asarray(mixed_spectogram, dtype='float16')
    
    phase_spectogram = np.asarray(phase, dtype='float16')

    # Save the mixed spectrogram
    np.save(folder_path + '/' + 'mixed_spectrogram.npy',mixed_spectogram)
    np.save(folder_path + '/' + 'phase_spectrogram.npy',phase_spectogram)

    for p in range(len(videos)):

        audio_file = audios[p]
        lips_file = videos[p]
       # mask_file = masks[p]
        
        save_path = folder_path
        
        try:
            os.mkdir(save_path)
        except OSError:
            pass
        
        audio_file_split = audio_file.split('/')
            
        # Save the mask
        
        s, n, c_ = compute_spectrograms(audio_file)
        s_use=s[:, :500]
        s_phase = np.angle(c)
        s_phase=s_phase[:,:500]

        Cx,Cy = compress_crm(mixed_mag = mixed_spectogram,mixed_phase = phase,signal_mag = s_use,signal_phase = s_phase, K=1,C=2)
        mask1 = np.stack([Cx,Cy], axis=-1)
        mask1=np.asarray(mask1, dtype='float16')
        file_name = audio_file_split[-2] + '_' + audio_file_split[-1][:-4] + '_crm.npy'

        np.save(save_path + '/' + file_name, mask1)

@jit            
def gen_comb_folders_crm(combined_pairs, dest_folder):
    
    videos = combined_pairs
    audios = []
    
    for item in videos:
        audio = item[:-9] + '.wav'
        audios.append(audio)
    
    
    folder_name_list = []
    for path in audios:
        split_ = path.split('/')
        fold = split_[-2] + '_' + split_[-1][:-4]
        folder_name_list.append(fold)
        
    folder_name = '_'.join(folder_name_list) + '_' + str(len(videos))
    folder_path = dest_folder + '/' + folder_name

    ## Now mix the audios and compute the spectrogram of mixed audio
    
    mix_audio_folder = home + '/mixed_audio_files'
        
    # Filename to save the mixed audio  file
    mixed_audio_filename = folder_name + '.wav'             
        
    mixed_audio_filepath = mix_audio_folder + '/' + mixed_audio_filename
    
    mixed_samples = audios_sum(audios, mixed_audio_filepath)
        
    # Compute the spectrogram of mised audio
    s, n, c = compute_spectrograms(mixed_audio_filepath)
    mixed_spectogram =s[:,:500]  # Useful frames

    phase=np.angle(c)
    phase=phase[:,:500]

    mixed_spectogram = np.asarray(mixed_spectogram, dtype='float16')
    
    phase_spectogram = np.asarray(phase, dtype='float16')

    # Save the mixed spectrogram
    np.save(folder_path + '/' + 'mixed_spectrogram.npy',mixed_spectogram)
    np.save(folder_path + '/' + 'phase_spectrogram.npy',phase_spectogram)

    for p in range(len(videos)):

        audio_file = audios[p]
        lips_file = videos[p]
       # mask_file = masks[p]
        
        save_path = folder_path

        audio_file_split = audio_file.split('/')
            
        # Save the mask
        
        s, n, c_ = compute_spectrograms(audio_file)
        s_use=s[:, :500]
        s_phase = np.angle(c)
        s_phase=s_phase[:,:500]

        Cx,Cy = compress_crm(mixed_mag = mixed_spectogram,mixed_phase = phase,signal_mag = s_use,signal_phase = s_phase, K=1,C=2)
        mask1 = np.stack([Cx,Cy], axis=-1)
        mask1=np.asarray(mask1, dtype='float16')
        file_name = audio_file_split[-2] + '_' + audio_file_split[-1][:-4] + '_crm.npy'

        np.save(save_path + '/' + file_name, mask1)        
