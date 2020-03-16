import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar
import numpy as np
import os
import glob
from scipy import signal
from scipy.io import wavfile
import cv2
import random
import time
import wandb
import matplotlib.pyplot as plt
from dataloaders_gan import DataGenerator_train, DataGenerator_val
from losses_gan import snr_loss, snr_acc, generator_loss, discriminator_loss

def media_logging(val_files, generator, batch_size):
    lips = []
    samples = []
    samples_mix = []
    for folder in val_files:
        lips_ = folder
        samples_ = folder[:-9] + '_samples.npy'
        samples_mix_ = '/data/mixed_audio_files/' + folder.split('/')[-2] + '.wav'

        lips.append(lips_)
        samples.append(samples_)
        samples_mix.append(samples_mix_)

    pred_audios = []
    for inputs, target in DataGenerator_val(val_files, batch_size=batch_size, norm=1):
            pred = generator(inputs, training=True)
            pred = pred.numpy()
            pred = pred.tolist()
            pred_audios = pred_audios + pred

    pred_audios = np.asarray(pred_audios).astype('int16')

    save_name_base = []
    for i, item in enumerate(lips):

        save_name_base_ = str(i) + '_' + item.split('/')[-1][:-9]
        save_name_base.append(save_name_base_)

    wandb.log({"Predicted Audio": [wandb.Audio(
        aud, caption=save_name_base[i], sample_rate=16000) for i, aud in enumerate(pred_audios)]}, commit=False)
    wandb.log({"True Audio": [wandb.Audio(
        np.pad(np.load(item), (0, 32000), mode='constant')[:32000], 
        caption=save_name_base[i], sample_rate=16000) for i, item in enumerate(samples)]}, commit=False)
    wandb.log({"Mixed Audio": [wandb.Audio(
        np.pad(wavfile.read(item)[1], (0, 32000), mode='constant')[:32000], 
        caption=save_name_base[i], sample_rate=16000) for i, item in enumerate(samples_mix)]}, commit=True)

@tf.function
def train_step(inputs, target, generator, discriminator, generator_optimizer, discriminator_optimizer, 
               epoch, LAMBDA):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(inputs, training=True)
        
        # std for the instance norm
        noise_inp = tf.random.normal(shape=tf.shape(inputs[1]), stddev=tf.nn.relu(1-((epoch)/10)))

        disc_real_output = discriminator([inputs[0], inputs[1], target, noise_inp], training=True)
        disc_generated_output = discriminator([inputs[0], inputs[1], gen_output, noise_inp], training=True)
        gen_total_loss, gen_gan_loss, gen_snr_loss = generator_loss(disc_generated_output, gen_output, target, LAMBDA=LAMBDA)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

    return gen_total_loss, gen_gan_loss, gen_snr_loss, disc_loss, disc_real_output, disc_generated_output


def fit(folders_list_train, folders_list_val, generator, discriminator, generator_optimizer, 
        discriminator_optimizer, batch_size, epochs, save_path):

    metrics_names = ['gen_total_loss','gen_gan_loss', 'gen_SNR_loss', 'disc_loss']
    gen_total_loss_all = []
    gen_gan_loss_all = []
    loss_snr_all = []
    disc_loss_all = []
    for epoch in range(epochs):
        start = time.time()

        print("\nEpoch {}/{}".format(epoch+1,epochs))

        pb_i = Progbar(len(folders_list_train), stateful_metrics=metrics_names)

        # Train
        gen_total_loss_list = []
        gen_gan_loss_list = []
        loss_snr_list = []
        disc_loss_list = []
        for n, (inputs, target) in enumerate(DataGenerator_train(folders_list_train, batch_size=batch_size, norm=1)):

            # Train on the current batch
            gen_total_loss, gen_gan_loss, loss_snr, disc_loss, disc_real_output, disc_generated_output = train_step(inputs, target, generator, discriminator,
                                                                        generator_optimizer, discriminator_optimizer, 
                                                                        epoch, 10)
            
            '''print('disc_real_output:', disc_real_output.numpy())
            print('disc_generated_output:', disc_generated_output.numpy())'''

            # Progress Bar
            values=[('gen_total_loss',gen_total_loss), ('gen_gan_loss',gen_gan_loss), ('gen_SNR_loss',loss_snr), ('disc_loss',disc_loss)]
            pb_i.add(batch_size, values=values)

            gen_total_loss_list.append(gen_total_loss)
            gen_gan_loss_list.append(gen_gan_loss)
            loss_snr_list.append(loss_snr)
            disc_loss_list.append(disc_loss)

        gen_total_loss_all += gen_total_loss_list
        gen_gan_loss_all += gen_gan_loss_list
        loss_snr_all += loss_snr_list
        disc_loss_all += disc_loss_list

        np.savetxt('/data/results/' + save_path + '/gen_total_loss.txt', gen_total_loss_all)
        np.savetxt('/data/results/' + save_path + '/gen_gan_loss.txt', gen_gan_loss_all)
        np.savetxt('/data/results/' + save_path + '/loss_snr.txt', loss_snr_all)
        np.savetxt('/data/results/' + save_path + '/disc_loss.txt', disc_loss_all)

        #plt.plot(gen_total_loss_all)
        plt.plot(gen_gan_loss_all)
        #plt.plot(loss_snr_all)
        plt.plot(disc_loss_all)
        plt.title('Losses')
        plt.ylabel('loss')
        plt.xlabel('#iterations')
        plt.legend(['Gen loss', 'Disc loss'])
        #plt.show()
        plt.savefig('/data/results/' + save_path + '/loss_gan.png')
        plt.close()

        #plt.plot(gen_total_loss_all)
        #plt.plot(gen_gan_loss_all)
        plt.plot(loss_snr_all)
        #plt.plot(disc_loss_all)
        plt.title('Losses')
        plt.ylabel('loss')
        plt.xlabel('#iterations')
        plt.legend(['SNR loss'])
        #plt.show()
        plt.savefig('/data/results/' + save_path + '/loss_snr.png')
        plt.close()

        snrs = []
        for inputs, target in DataGenerator_val(folders_list_val, batch_size=batch_size, norm=1):
            snrs.append(snr_acc(target, generator(inputs, training=True)))
        val_acc = np.mean(snrs)
        print('Val SNR:', val_acc)

        # Save Weights
        generator.save_weights('/data/models/' + save_path + '/generator-' + str(int(epoch)) + '-' + str(np.round(val_acc, 4)) + '.tf')
        discriminator.save_weights('/data/models/' + save_path + '/discriminator-' + str(int(epoch)) + '-' + str(np.round(val_acc, 4)) + '.tf')

        # Wandb Metrics logging
        wandb.log({"gen_total_loss": np.mean(gen_total_loss_list), 'gen_gan_loss':np.mean(gen_gan_loss_list), 
                          'loss_snr':np.mean(loss_snr_list), 'disc_loss':np.mean(disc_loss_list), 'Val_SNR':np.mean(snrs)}, commit=False)
        
        with open('/data/results/' + save_path + '/logs.txt', "a") as myfile:
            myfile.write("gen_total_loss: " + str(np.mean(gen_total_loss_list)) + ' - ' + 'gen_gan_loss: ' + 
                        str(np.mean(gen_gan_loss_list)) + ' - ' + 'loss_snr: ' + str(np.mean(loss_snr_list)) + 
                        ' - ' + 'disc_loss: ' + str(np.mean(disc_loss_list)) + ' - ' + 'Val_SNR: ' + str(np.mean(snrs)) + '\n')

        val_files = [folders_list_val[10], folders_list_val[30], folders_list_val[50], 
                    folders_list_val[100], folders_list_val[1600], folders_list_val[1620], 
                    folders_list_val[1640], folders_list_val[1660], folders_list_val[70], 
                    folders_list_val[1750]]
        #val_files = folders_list_val[:8]
        media_logging(val_files, generator, batch_size)
