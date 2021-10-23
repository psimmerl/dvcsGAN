import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LeakyReLU, Dense, Input, BatchNormalization, Activation, Dropout, ReLU
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np
import matplotlib.pyplot as plt

from time import time
from datetime import timedelta

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class GAN():
    ''' Custom GAN to simulate EIC TOPEG events'''

    def __init__(self, **kwargs) -> None:
        self.noise_shape = (64,)
        self.batch_size = 800
        self.epochs = 10000
        self.pflip = 0.05
        self.output_shape = (19,)
        self.input_file = "data/processed/X_train.npy"
        self.save_epoch_rate = 25
        self.features = ['Q2', 'W', 'Gamnu', 'Xbj', 'y', 't', 'phih', 
                        'electron px', 'photon px', 'He(or proton?) px', 
                        'electron py', 'photon py', 'He(or proton?) py', 
                        'electron pz', 'photon pz', 'He(or proton?) pz', 
                        'electron E',  'photon E',  'He(or proton?) E',]
        # drop learning rate so I don't overfit?
        gen_opt = Adam(learning_rate=0.0001)#, beta_1=0.5)#, epsilon=1e-2) #"adam"
        dis_opt = SGD(learning_rate=0.01) #Adam(learning_rate=0.0001, beta_1=0.5) #

        self.dis = self.build_discriminator()
        self.dis.compile(loss="binary_crossentropy", optimizer=dis_opt, metrics=["accuracy"])
        
        self.gen = self.build_generator()
        self.gen.compile(loss="binary_crossentropy", optimizer=gen_opt)
        
        z = Input(self.noise_shape)
        generated = self.gen(z)
        valid = self.dis(generated)
        self.combined = Model(z, valid)
        self.combined.compile(loss="binary_crossentropy", optimizer=gen_opt)


    def build_generator(self) -> Sequential:
        model = Sequential(name="generator")
        alpha=0.2
        
        model.add(Input(self.noise_shape))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha))
        # model.add(BatchNormalization())

        for units in [128, 256, 512, 256, 128]:
            model.add(Dense(units, input_shape=self.noise_shape))
            model.add(LeakyReLU(alpha))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

        model.add(Dense(self.output_shape[0], "tanh"))

        plot_model(model, 'models/generator.png', show_shapes=True)
        model.summary()

        return model

    def build_discriminator(self) -> Sequential:
        model = Sequential(name="discriminator")
        alpha=0.2

        model.add(Input(self.output_shape))
        model.add(Dense(128, input_shape=self.output_shape))
        model.add(LeakyReLU(alpha))

        for units in [128,256,256,128,64,32,15,8]:
            model.add(Dense(units))
            model.add(LeakyReLU(alpha))
            model.add(BatchNormalization())

        model.add(Dense(1, activation="sigmoid"))
        model.summary()
        plot_model(model, 'models/discriminator.png', show_shapes=True)


        return model
    
    def train(self):
        X_train = np.load(self.input_file)
        xlen = X_train.shape[0]
        history = [[],[],[]]
        
        nbatches = np.math.ceil(xlen/self.batch_size)
        btch_pre = len(str(nbatches))

        self.dis.save(f"models/epoch0/discriminator")
        self.dis.save(f"models/epoch0/generator")

        #Generate random vector now so the figures are ~consistent~
        rand_vec = np.random.normal(0, 1, (xlen, self.noise_shape[0]))

        f, axs = plt.subplots(4, 5, figsize=(20,16))
        axs = axs.flatten()
        self.plot_and_save(X_train[:int(xlen/10)], self.gen.predict(rand_vec[:int(xlen/10)]), history, 0, f, axs)

        tStart = time()
        for epoch in range(1,self.epochs+1):
            print(f"\nEpoch {epoch}/{self.epochs} - Time Elapsed {timedelta(seconds=int(time()-tStart))} - Time Left {timedelta(seconds=int((time()-tStart)/epoch*(self.epochs-epoch)))}")
            
            IDXS = np.random.permutation(xlen)
            self.dis.trainable = True
            for ibatch, i in enumerate(np.arange(0, xlen, self.batch_size)):
                idxs = IDXS[i:i+self.batch_size]
                bsz = len(idxs)

                real = X_train[idxs, :]
                noise = np.random.normal(0, 1, (bsz, self.noise_shape[0]))
                fake = self.gen.predict(noise)

                y_real = np.random.uniform(.7,1.2,(bsz,1))#np.ones((bsz,1))#
                y_fake = np.random.uniform(0,.3,(bsz,1))#np.zeros((bsz,1))#

                idx_flip = np.random.randint(0,bsz,int(self.pflip * bsz))
                y_real[idx_flip], y_fake[idx_flip] = y_fake[idx_flip], y_real[idx_flip]


                # y_real = np.ones( (bsz,1)) - np.abs(np.random.normal(0,0.1,(bsz,1)))#+ 0.05 * np.random.normal(0,1,(bsz,1))#
                # y_fake = np.zeros((bsz,1)) + np.abs(np.random.normal(0,0.1,(bsz,1)))#+ 0.05 * np.random.normal(0,1,(bsz,1))#

                # shuffle_idxs = np.random.permutation(2*bsz)[:bsz]
                # d_loss = self.dis.train_on_batch(np.r_[real, fake][shuffle_idxs], 
                #                         np.r_[y_real, y_fake][shuffle_idxs])
                d_loss_r = self.dis.train_on_batch(real, y_real)
                d_loss_f = self.dis.train_on_batch(fake, y_fake)
                d_loss = np.add(d_loss_r, d_loss_f) / 2

                self.dis.trainable = False
                y_valid = np.ones((2*bsz,1))
                noise = np.random.normal(0, 1, (2*bsz, self.noise_shape[0])) # not sure if I can reuse previous noise
                g_loss = self.combined.train_on_batch(noise, y_valid)
                
                ibatch += 1
                arrow = '>' if ibatch != nbatches else '='
                pb = '='*(np.math.floor(30*ibatch/nbatches)-1)+arrow + '.'*(30-np.math.floor(30*ibatch/nbatches))
                print(f"\r{str(ibatch).rjust(btch_pre)}/{nbatches} [{pb}] - [D loss: {d_loss[0]:.4f}, G loss: {g_loss:.4f}]", end="")

            x = np.r_[X_train[:int(xlen/10)], self.gen.predict(rand_vec[:int(xlen/10)])]
            y = np.r_[np.ones((int(xlen/10),1)), np.zeros((int(xlen/10),1))]
            d_loss = self.dis.evaluate(x, y, batch_size=2*self.batch_size)
            g_loss = self.combined.evaluate(rand_vec[:int(xlen/10)], np.ones(int(xlen/10)), batch_size=self.batch_size)
            history[0].append(d_loss[0])
            history[1].append(d_loss[1])
            history[2].append(g_loss)

            if epoch % self.save_epoch_rate == 0:
                self.plot_and_save(X_train[:int(xlen/10)], self.gen.predict(rand_vec[:int(xlen/10)]), history, epoch, f, axs)

    def plot_and_save(self, real, fake, history, epoch, f, axs):
        self.dis.save(f"models/epoch{epoch}/discriminator")
        self.gen.save(f"models/epoch{epoch}/generator")

        for i in range(real.shape[1]):
            axs[i].clear()
            axs[i].hist(real[:,i], histtype='step', bins=200, label='real')
            axs[i].hist(fake[:,i], histtype='step', bins=200, label='fake')
            axs[i].legend()
            axs[i].grid()
            axs[i].set_xlim([-1,1])
            axs[i].set_yscale('log')
            axs[i].set_title(self.features[i])

        axs[-1].clear()
        axs[-1].plot(history[0], label='D loss')
        axs[-1].plot(history[1], label='D acc')
        axs[-1].plot(history[2], label='G loss')
        axs[-1].legend()
        axs[-1].grid()
        axs[-1].set_xlim([0,epoch if epoch else 1])
        axs[-1].set_title(f"Loss History (Epoch {epoch})")
        f.savefig(f"imgs/plots{epoch}", bbox_inches='tight', dpi=200)

if __name__ == "__main__":
    gan = GAN()
    gan.train()
