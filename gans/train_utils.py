import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.layers.merge import _Merge
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
from losses_gan import *

def media_logging(val_files, generator, batch_size):
    lips = []
    samples = []
    samples_mix = []
    for folder in val_files:
        lips_ = folder
        samples_ = folder[:-9] + '_samples.npy'
        samples_mix_ = '/home/ubuntu/lrs2/mixed_audios/' + folder.split('/')[-2] + '.wav'

        lips.append(lips_)
        samples.append(samples_)
        samples_mix.append(samples_mix_)

    pred_audios = []
    for inputs, target in DataGenerator_val(val_files, batch_size=batch_size, norm=1350.0):
            pred = generator(inputs, training=True)
            pred = pred.numpy()
            pred = pred.tolist()
            pred_audios = pred_audios + pred

    pred_audios = np.asarray(pred_audios)
    pred_audios = pred_audios/np.mean(np.std(pred_audios, axis=1))
    pred_audios = pred_audios*1350
    pred_audios = pred_audios.astype('int16')

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
               epoch, epoch_start, LAMBDA, gan_type):

    # Calculate Gradient penalty
    # Zero Cenreted GP (https://arxiv.org/pdf/1902.03984)
    def gradient_penalty(inputs, target, noise_inp, gen_output, discriminator, center=0):

        alpha = tf.random.uniform(shape=[tf.shape(inputs[1])[0],1, 1], minval=0., maxval=1.)
        #differences = gen_output - target
        interpolates = alpha*target + ((1-alpha)*gen_output)

        with tf.GradientTape() as tape:

            tape.watch([interpolates, inputs])
            d_hat = discriminator([inputs[0], inputs[1], interpolates, noise_inp], training=True)

        gradients = tape.gradient(d_hat, interpolates)
        slopes = tf.sqrt(tf.math.reduce_sum(tf.square(gradients), axis=[1, 2]))
        gradient_pen = tf.math.reduce_mean((slopes - center)**2)

        return gradient_pen

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(inputs, training=True)
        
        # std for the instance noise
        noise_inp1 = tf.random.normal(shape=tf.shape(inputs[1]), stddev=tf.nn.relu(1-((epoch+epoch_start)/10))) #tf.nn.relu(1-(epoch/10))
        noise_inp2 = tf.random.normal(shape=tf.shape(inputs[1]), stddev=tf.nn.relu(1-((epoch+epoch_start)/10)))
        noise_inp3 = tf.random.normal(shape=tf.shape(inputs[1]), stddev=tf.nn.relu(1-((epoch+epoch_start)/10)))

        disc_real_output = discriminator([inputs[0], inputs[1], target, noise_inp1], training=True)
        disc_generated_output = discriminator([inputs[0], inputs[1], gen_output, noise_inp2], training=True)

        if gan_type == 'LSGAN':
            gen_total_loss, gen_gan_loss, gen_snr_loss = generator_loss_LSGAN(disc_generated_output, gen_output, target, LAMBDA=LAMBDA)
            disc_loss, disc_real_loss, disc_fake_loss = discriminator_loss_LSGAN(disc_real_output, disc_generated_output)
            gradient_penalty = 0

        elif gan_type == 'WGAN_GP':
            gen_total_loss, gen_gan_loss, gen_snr_loss = generator_loss_WGAN_GP(disc_generated_output, gen_output, target, LAMBDA=LAMBDA)
            
            gradient_pen = gradient_penalty(inputs, target, noise_inp1, gen_output, discriminator, center=1)
            disc_loss, disc_real_loss, disc_fake_loss, gradient_penalty = discriminator_loss_WGAN_GP(disc_real_output, disc_generated_output, gradient_pen, 10)

        elif gan_type == 'WGAN':
            gen_total_loss, gen_gan_loss, gen_snr_loss = generator_loss_WGAN(disc_generated_output, gen_output, target, LAMBDA=LAMBDA)
            disc_loss, disc_real_loss, disc_fake_loss = discriminator_loss_WGAN(disc_real_output, disc_generated_output)

        elif gan_type == 'ZERO_GP':
            gen_total_loss, gen_gan_loss, gen_snr_loss = generator_loss_LSGAN(disc_generated_output, gen_output, target, LAMBDA=LAMBDA)
            
            gradient_pen = gradient_penalty(inputs, target, noise_inp3, gen_output, discriminator, center=0)
            disc_loss, disc_real_loss, disc_fake_loss, gradient_penalty = discriminator_loss_LSGAN_GP(disc_real_output, disc_generated_output, gradient_pen, 150)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)

    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    #print(discriminator_gradients.numpy())
    # Gradient Clipping for WGAN                             
    if gan_type == 'WGAN':
        clip_bounds = [-0.01, 0.01]
        discriminator_gradients = [(tf.clip_by_value(grad, clip_bounds[0], clip_bounds[1]))
                                  for grad in discriminator_gradients]

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

    return gen_total_loss, gen_gan_loss, gen_snr_loss, disc_loss, disc_real_loss, disc_fake_loss, disc_real_output, disc_generated_output, gradient_penalty


def fit(folders_list_train, folders_list_val, generator, discriminator, generator_optimizer, 
        discriminator_optimizer, batch_size, epochs, save_path, lr, epoch_start):

    metrics_names = ['gen_total_loss', 'gen_gan_loss',
                     'gen_SNR_loss', 'disc_loss', 'grad_pen'] 
    gen_total_loss_all = []
    gen_gan_loss_all = []
    loss_snr_all = []
    disc_loss_all = []
    disc_real_loss_all = []
    disc_fake_loss_all = []
    grad_pen_all = []
    #snrs_val = []
    for epoch in range(epochs):

        start = time.time()

        print("\nEpoch {}/{}".format(epoch+epoch_start+1,epochs+epoch_start))
        #print('Generator LR:', generator_optimizer._decayed_lr(tf.float32).numpy(), ' - Discriminator LR:', discriminator_optimizer._decayed_lr(tf.float32).numpy())

        pb_i = Progbar(len(folders_list_train), stateful_metrics=metrics_names)

        if epoch+epoch_start < 5:
            lambda_value = 100
        elif epoch+epoch_start >= 5:
            lambda_value = 0.1

        # Train
        gen_total_loss_list = []
        gen_gan_loss_list = []
        loss_snr_list = []
        disc_loss_list = []
        disc_real_loss_list = []
        disc_fake_loss_list = []
        grad_pen_list = []

        for n, (inputs, target) in enumerate(DataGenerator_train(folders_list_train, batch_size=batch_size, norm=1350.0, epoch=epoch+epoch_start)):

            # Train on the current batch
            gen_total_loss, gen_gan_loss, loss_snr, disc_loss, disc_real_loss, disc_fake_loss, disc_real_output, disc_generated_output, gradient_penalty = train_step(inputs, target, generator, 
                                                                            discriminator,generator_optimizer, 
                                                                            discriminator_optimizer, epoch, epoch_start, lambda_value, 'ZERO_GP')
            
            '''print('disc_real_output:', disc_real_output.numpy())
            print('disc_generated_output:', disc_generated_output.numpy())'''
            '''disc_real_loss = disc_real_loss.numpy().tolist()
            disc_fake_loss = disc_fake_loss.numpy().tolist()
            print(disc_real_loss)
            print(disc_fake_loss)
            print('disc_loss:', disc_loss)'''

            '''print('gen opt', generator_optimizer.iterations.numpy())
            print('disc opt', discriminator_optimizer.iterations.numpy())'''

            #print('gradient_penalty:', gradient_penalty.numpy())

            # Progress Bar
            values=[('gen_total_loss',gen_total_loss), ('gen_gan_loss',gen_gan_loss), ('gen_SNR_loss',loss_snr), 
                    ('disc_loss',disc_loss), ('grad_pen',gradient_penalty)]
            pb_i.add(batch_size, values=values)

            gen_total_loss_list.append(gen_total_loss)
            gen_gan_loss_list.append(gen_gan_loss)
            loss_snr_list.append(loss_snr)
            disc_loss_list.append(disc_loss)
            disc_real_loss_list.append(disc_real_loss)
            disc_fake_loss_list.append(disc_fake_loss)
            grad_pen_list.append(gradient_penalty)

        gen_total_loss_all += gen_total_loss_list
        gen_gan_loss_all += gen_gan_loss_list
        loss_snr_all += loss_snr_list
        disc_loss_all += disc_loss_list
        disc_real_loss_all += disc_real_loss_list
        disc_fake_loss_all += disc_fake_loss_list
        grad_pen_all += grad_pen_list

        np.savetxt('/home/ubuntu/results/' + save_path + '/gen_total_loss.txt', gen_total_loss_all)
        np.savetxt('/home/ubuntu/results/' + save_path + '/gen_gan_loss.txt', gen_gan_loss_all)
        np.savetxt('/home/ubuntu/results/' + save_path + '/loss_snr.txt', loss_snr_all)
        np.savetxt('/home/ubuntu/results/' + save_path + '/disc_loss.txt', disc_loss_all)
        np.savetxt('/home/ubuntu/results/' + save_path + '/disc_real_loss.txt', disc_real_loss_all)
        np.savetxt('/home/ubuntu/results/' + save_path + '/disc_fake_loss.txt', disc_fake_loss_all)
        np.savetxt('/home/ubuntu/results/' + save_path + '/grad_pen.txt', grad_pen_all)

        #plt.plot(gen_total_loss_all)
        plt.plot(gen_gan_loss_all)
        #plt.plot(loss_snr_all)
        plt.plot(disc_loss_all)
        plt.title('Losses')
        plt.ylabel('loss')
        plt.xlabel('#iterations')
        plt.legend(['Gen loss', 'Disc loss'])
        #plt.show()
        plt.savefig('/home/ubuntu/results/' + save_path + '/loss_gan.png')
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
        plt.savefig('/home/ubuntu/results/' + save_path + '/loss_snr.png')
        plt.close()

        plt.plot(disc_real_loss_all)
        plt.plot(disc_fake_loss_all)
        plt.title('Losses')
        plt.ylabel('loss')
        plt.xlabel('#iterations')
        plt.legend(['Disc real loss', 'Disc fake loss'])
        plt.savefig('/home/ubuntu/results/' + save_path + '/disc_losses.png')
        plt.close()

        snrs = []
        for inputs, target in DataGenerator_val(folders_list_val, batch_size=batch_size, norm=1350.0):
            snrs.append(snr_acc(target, generator(inputs, training=True)))
        val_acc = np.mean(snrs)
        
        print('Val SNR:', val_acc)

        '''with open('/home/ubuntu/results/' + save_path + '/val_snr.txt', "a") as myfile:
            myfile.write(str(val_acc)+'\n')'''

        # Save Weights
        generator.save_weights('/home/ubuntu/models/' + save_path + '/generator-' + str(int(epoch+epoch_start)) + '-' + str(np.round(val_acc, 4)) + '.tf')
        discriminator.save_weights('/home/ubuntu/models/' + save_path + '/discriminator-' + str(int(epoch+epoch_start)) + '-' + str(np.round(val_acc, 4)) + '.tf')

        # Wandb Metrics logging
        wandb.log({"gen_total_loss": np.mean(gen_total_loss_list), 'gen_gan_loss':np.mean(gen_gan_loss_list), 
                    'SNR':-np.mean(loss_snr_list), 'disc_loss':np.mean(disc_loss_list), 'Val_SNR':np.mean(snrs),
                    'disc_fake_loss':np.mean(disc_fake_loss_list), 'disc_real_loss':np.mean(disc_real_loss_list), 'lr':lr}, commit=False)
        
        with open('/home/ubuntu/results/' + save_path + '/logs.txt', "a") as myfile:
            myfile.write("gen_total_loss: " + str(np.mean(gen_total_loss_list)) + ' - ' + 'gen_gan_loss: ' + 
                        str(np.mean(gen_gan_loss_list)) + ' - ' + 'loss_snr: ' + str(np.mean(loss_snr_list)) + 
                        ' - ' + 'disc_loss: ' + str(np.mean(disc_loss_list)) + ' - ' + 'Val_SNR: ' + str(np.mean(snrs)) + 
                        ' - ' + 'disc_real_loss: ' + str(np.mean(disc_real_loss_list)) + ' - ' + 'disc_fake_loss: ' + str(np.mean(disc_fake_loss_list)) + '\n')
        
        # Reduce LR on Plateau
        '''if epoch+epoch_start >= 4 and epoch+epoch_start <= 35:
            #snrs_val = np.loadtxt('/home/ubuntu/results/' + save_path + '/val_snr.txt')
            if val_acc <= snrs_val[-4] and snrs_val[-1] <= snrs_val[-4] and snrs_val[-2] <= snrs_val[-4] and snrs_val[-3] <= snrs_val[-4]:
                lr = lr*0.5
                tf.keras.backend.set_value(generator_optimizer.lr, lr)
                tf.keras.backend.set_value(discriminator_optimizer.lr, lr)
                print('Generator lr:', generator_optimizer.lr.numpy(), '- Discriminator lr:', discriminator_optimizer.lr.numpy())

        elif epoch+epoch_start > 35:
            #snrs_val = np.loadtxt('/home/ubuntu/results/' + save_path + '/val_snr.txt')
            if val_acc <= snrs_val[-3] and snrs_val[-1] <= snrs_val[-3] and snrs_val[-2] <= snrs_val[-3]:
                lr = lr*0.5
                tf.keras.backend.set_value(generator_optimizer.lr, lr)
                tf.keras.backend.set_value(discriminator_optimizer.lr, lr)
                print('Generator lr:', generator_optimizer.lr.numpy(), '- Discriminator lr:', discriminator_optimizer.lr.numpy())
        
        snrs_val.append(val_acc)'''

        val_files = [folders_list_val[10], folders_list_val[30], folders_list_val[50], 
                    folders_list_val[100], folders_list_val[1600], folders_list_val[1620], 
                    folders_list_val[1640], folders_list_val[1660], folders_list_val[70], 
                    folders_list_val[1750]]
        #val_files = folders_list_val[:8]
        media_logging(val_files, generator, batch_size)
