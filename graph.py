import numpy as np
from scipy.io import wavfile
from scipy import signal
import math
import scipy
import matplotlib.pyplot as plt
from mir_eval.separation import bss_eval_sources
import glob

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

list_samples1=sorted(glob.glob('./pred_tasnet_6000/*/*original.wav'), key=numericalSort)
list_samples2=sorted(glob.glob('./pred_tasnet_6000/*/*pred.wav'), key=numericalSort)
list_samples3=sorted(glob.glob('./pred_tasnet_6000/*/mixed.wav'), key=numericalSort)

mix_samples=[]
for i in range(len(list_samples3)):
    mix_samples.append(list_samples3[i])
    mix_samples.append(list_samples3[i])

temp=[]
temp1=[]
for i in range(4000):
    if i%100==0:
        print(i, 'Done')
    samples_1=wavfile.read(list_samples1[i])
    samples=[0]*31584
    if(len(samples_1[1])<31584):
        samples[0:len(samples_1[1])]=samples_1[1]
    else: samples=samples_1[1][0:31584]
    samples_2=wavfile.read(list_samples2[i])
    samples_3=wavfile.read(mix_samples[i])
    
#     sam_1=[]
#     sam_2=[]
#     if(len(samples_1[1])<31584): 
#         sam_1=samples_1[1]
#         sam_2=samples_2[1][0:len(sam_1)]
#     else: 
#         sam_2=samples_2[1]
#         sam_1=samples_1[1][0:len(sam_2)]
    
    
    
    samples_mix=[0]*31584
    if(len(samples_3[1])<31584):
        samples_mix[0:len(samples_3[1])]=samples_3[1]
    else: samples_mix=samples_3[1][0:31584]
    #print(len(samples))
    sdr1, sir1, sar1, _ = bss_eval_sources(np.asarray(samples),samples_2[1] , compute_permutation=False)
    sdr, sir, sar, _ = bss_eval_sources(np.asarray(samples), np.asarray(samples_mix) , compute_permutation=False)
    #print(len(samples))
    temp.append(sdr)
    temp1.append(sdr1)

sdr_imp=list(np.array(temp1) - np.array(temp))
sdr_mixed=temp

sdr_mixed1=[]
sdr_imp1=[]
for i in range(len(temp1)):
    if temp[i] >= -5 and temp[i] <= 5:
        sdr_mixed1.append(temp[i])
        sdr_imp1.append(temp1[i])

print('SDR:', np.mean(np.asarray(sdr_imp1)))


'''area = np.pi*1
# Plot
plt.scatter(sdr_mixed, sdr_imp, s=area, c='green', alpha=0.5)
plt.title('SDR Improvement Vs Input SDR')
plt.xlabel('Input SDR (dB)')
plt.ylabel('SDR Improvement (dB)')
#plt.xticks(np.arange(min(sdr_mixed), max(sdr_mixed)+1, 0.5))
#plt.xticks(np.arange(min(sdr_imp), max(sdr_imp)+1, 0.5))
plt.savefig('sdr_improv_4.png')'''
