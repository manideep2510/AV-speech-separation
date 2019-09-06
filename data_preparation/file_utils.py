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
home = str(Path.home())

from audio_utils import compare_lengths, compute_spectrograms, audios_sum


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
                random.seed(2)
                picked = random.sample(picks, combination_no)
                
                combined = [0]*2
                combined[0] = item
                combined[1] = picked

                picks = [e for e in picks if e is not picked]
                
            else:
                
                random.seed(3)
                picked = random.sample(picks, combination_no)
                
                picks = [e for e in picks if e is not picked]
                
                picked.append(item)
                
                combined = picked
                
            combined_list.append(combined)
        #b = time.time()
        #print(n, 'Seconds', b-a)
            
    return combined_list


# Generate training folders for one combination (Modified)

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
            
        '''mask_file = save_path + '/' + 'mask.npy'
        # Computes the Target binary mask and save it in destination.
        save_target_binary_mask_speaker(audio_file, mask_file = mask_file)'''
        
        shutil.copy2(mask_file, save_path)
            
        # Save the mixed spectrogram
        np.save(save_path + '/' + 'mixed_spectrogram.png',mixed_spectogram)