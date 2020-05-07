import matplotlib.pyplot as plt
import numpy as np

path = 'tdavss_baseline_256dimSepNet_2speakers'

pesq1 = np.loadtxt('/home/ubuntu/metric_results/' + path + '/pesq_val.txt')

snrs1 = np.loadtxt('/home/ubuntu/metric_results/' + path + '/snr_val.txt')

sdrs1 = np.loadtxt('/home/ubuntu/metric_results/' + path + '/sdr_val.txt')

pesq_mix = np.loadtxt('/home/ubuntu/metric_results/' + path + '/pesq_mix.txt')

snrs_mix = np.loadtxt('/home/ubuntu/metric_results/' + path + '/snr_mix.txt')

sdrs_mix = np.loadtxt('/home/ubuntu/metric_results/' + path + '/sdr_mix.txt')


path = 'tdavss_LSGAN_PS2_1350Norm_Lambda100'

pesq2 = np.loadtxt('/home/ubuntu/metric_results/' + path + '/pesq_val.txt')

snrs2 = np.loadtxt('/home/ubuntu/metric_results/' + path + '/snr_val.txt')

sdrs2 = np.loadtxt('/home/ubuntu/metric_results/' + path + '/sdr_val.txt')

'''# Plot
area = np.pi*1
plt.figure()
plt.scatter(snrs_mix, snrs1-snrs_mix, s=area, c='green', alpha=0.5)
plt.scatter(snrs_mix, snrs2-snrs_mix, s=area, c='red', alpha=0.5)
plt.title('SNR Improvement Vs Input SNR')
plt.xlabel('Input SNR (dB)')
plt.ylabel('SNR Improvement (dB)')
plt.savefig('snr_improv.png')


# Plot
plt.figure()
plt.scatter(sdrs_mix, sdrs1-sdrs_mix, s=area, c='green', alpha=0.5)
plt.scatter(sdrs_mix, sdrs2-sdrs_mix, s=area, c='red', alpha=0.5)
plt.title('SDR Improvement Vs Input SDR')
plt.xlabel('Input SDR (dB)')
plt.ylabel('SDR Improvement (dB)')
plt.savefig('sdr_improv.png')


# Plot
plt.figure()
plt.scatter(pesq_mix, pesq1-pesq_mix, s=area, c='green', alpha=0.5)
plt.scatter(pesq_mix, pesq2-pesq_mix, s=area, c='red', alpha=0.5)
plt.title('PESQ Improvement Vs Input PESQ')
plt.xlabel('Input PESQ')
plt.ylabel('PESQ Improvement')
plt.savefig('pesq_improv.png')'''

temp0 = []
temp1 = []
temp2 = []
temp3 = []
temp4 = []
temp5 = []
temp6 = []
temp7 = []
temp8 = []
temp9 = []

for i in range(len(snrs_mix)):
    if snrs_mix[i]>=-5 and snrs_mix[i] <-4:
        temp0.append(snrs1[i])
    elif snrs_mix[i]>=-4 and snrs_mix[i] <-3:
        temp1.append(snrs1[i])
    elif snrs_mix[i]>=-3 and snrs_mix[i] <-2:
        temp2.append(snrs1[i])
    elif snrs_mix[i]>=-2 and snrs_mix[i] <-1:
        temp3.append(snrs1[i])
    elif snrs_mix[i]>=-1 and snrs_mix[i] <0:
        temp4.append(snrs1[i])
    elif snrs_mix[i]>=0 and snrs_mix[i] <1:
        temp5.append(snrs1[i])
    elif snrs_mix[i]>=1 and snrs_mix[i] <2:
        temp6.append(snrs1[i])
    elif snrs_mix[i]>=2 and snrs_mix[i] <3:
        temp7.append(snrs1[i])
    elif snrs_mix[i]>=3 and snrs_mix[i] <4:
        temp8.append(snrs1[i])
    elif snrs_mix[i]>=4 and snrs_mix[i] <=5:
        temp9.append(snrs1[i])

print('----------SNRs----------')

print('AVSS Baseline')
print(np.round(np.mean(temp0), 2), '|', np.round(np.mean(temp1), 2), '|', np.round(np.mean(temp2), 2), '|', np.round(np.mean(temp3), 2), '|', np.round(np.mean(temp4), 2), '|', np.round(np.mean(temp5), 2), '|', np.round(np.mean(temp6), 2), '|', np.round(np.mean(temp7), 2), '|', np.round(np.mean(temp8), 2), '|', np.round(np.mean(temp9), 2))


temp0 = []
temp1 = []
temp2 = []
temp3 = []
temp4 = []
temp5 = []
temp6 = []
temp7 = []
temp8 = []
temp9 = []

for i in range(len(snrs_mix)):
    if snrs_mix[i]>=-5 and snrs_mix[i] <-4:
        temp0.append(snrs2[i])
    elif snrs_mix[i]>=-4 and snrs_mix[i] <-3:
        temp1.append(snrs2[i])
    elif snrs_mix[i]>=-3 and snrs_mix[i] <-2:
        temp2.append(snrs2[i])
    elif snrs_mix[i]>=-2 and snrs_mix[i] <-1:
        temp3.append(snrs2[i])
    elif snrs_mix[i]>=-1 and snrs_mix[i] <0:
        temp4.append(snrs2[i])
    elif snrs_mix[i]>=0 and snrs_mix[i] <1:
        temp5.append(snrs2[i])
    elif snrs_mix[i]>=1 and snrs_mix[i] <2:
        temp6.append(snrs2[i])
    elif snrs_mix[i]>=2 and snrs_mix[i] <3:
        temp7.append(snrs2[i])
    elif snrs_mix[i]>=3 and snrs_mix[i] <4:
        temp8.append(snrs2[i])
    elif snrs_mix[i]>=4 and snrs_mix[i] <=5:
        temp9.append(snrs2[i])

print('AVSS-GAN')
print(np.round(np.mean(temp0), 2), '|', np.round(np.mean(temp1), 2), '|', np.round(np.mean(temp2), 2), '|', np.round(np.mean(temp3), 2), '|', np.round(np.mean(temp4), 2), '|', np.round(np.mean(temp5), 2), '|', np.round(np.mean(temp6), 2), '|', np.round(np.mean(temp7), 2), '|', np.round(np.mean(temp8), 2), '|', np.round(np.mean(temp9), 2))


snrs_mix = pesq_mix
snrs1 = pesq1
snrs2 = pesq2

temp0 = []
temp1 = []
temp2 = []
temp3 = []
temp4 = []
temp5 = []

for i in range(len(snrs_mix)):
    if snrs_mix[i]>=-1 and snrs_mix[i] <0:
        temp0.append(snrs1[i])
    elif snrs_mix[i]>=0 and snrs_mix[i] <1:
        temp1.append(snrs1[i])
    elif snrs_mix[i]>=1 and snrs_mix[i] <2:
        temp2.append(snrs1[i])
    elif snrs_mix[i]>=2 and snrs_mix[i] <3:
        temp3.append(snrs1[i])
    '''elif snrs_mix[i]>=3 and snrs_mix[i] <4:
        temp4.append(snrs1[i])
    elif snrs_mix[i]>=4 and snrs_mix[i] <5:
        temp5.append(snrs1[i])'''

print('----------PESQ----------')

print('AVSS Baseline')
print(np.round(np.mean(temp0), 2), '|', np.round(np.mean(temp1), 2), '|', np.round(np.mean(temp2), 2), '|', np.round(np.mean(temp3), 2))


temp0 = []
temp1 = []
temp2 = []
temp3 = []
temp4 = []
temp5 = []

for i in range(len(snrs_mix)):
    if snrs_mix[i]>=-1 and snrs_mix[i] <0:
        temp0.append(snrs2[i])
    elif snrs_mix[i]>=0 and snrs_mix[i] <1:
        temp1.append(snrs2[i])
    elif snrs_mix[i]>=1 and snrs_mix[i] <2:
        temp2.append(snrs2[i])
    elif snrs_mix[i]>=2 and snrs_mix[i] <3:
        temp3.append(snrs2[i])
    '''elif snrs_mix[i]>=3 and snrs_mix[i] <4:
        temp4.append(snrs2[i])
    elif snrs_mix[i]>=4 and snrs_mix[i] <5:
        temp5.append(snrs2[i])'''

print('AVSS-GAN')
print(np.round(np.mean(temp0), 2), '|', np.round(np.mean(temp1), 2), '|', np.round(np.mean(temp2), 2), '|', np.round(np.mean(temp3), 2))


