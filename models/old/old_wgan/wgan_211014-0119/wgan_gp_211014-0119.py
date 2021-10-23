import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt
from joblib import load
import os
from datetime import date, datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class WGAN(keras.models.Model):
    def __init__(self, discriminator=None, generator=None, d_steps=5, 
                    gp_weight=10.0, latent_dim=128, gen_dim=19) -> None:
        super(WGAN, self).__init__()
        self.latent_dim = latent_dim
        self.gen_dim = gen_dim
        self.d_steps = d_steps
        self.gp_weight = gp_weight
        self.dis = discriminator if discriminator else self.build_discriminator()
        self.gen = generator if generator else self.build_generator()

    def build_discriminator(self) -> Sequential:
        model = Sequential(name="discriminator")
        model.add(Input((self.gen_dim,)))
        for units in [512, 512, 512, 512, 512]:
        # for units in [128, 256, 256, 128, 64, 32, 16, 8]:
            model.add(Dense(units))
            model.add(LeakyReLU(0.2))
            model.add(Dropout(0.1))
        model.add(Dense(1))
        # plot_model(model, 'models/discriminator.png', show_shapes=True)
        # model.summary()
        return model

    def build_generator(self) -> Sequential:
        model = Sequential(name="generator")
        model.add(Input(self.latent_dim))
        for units in [512, 512, 512, 512, 512]:
        # for units in [128, 256, 512, 256, 128]:
            model.add(Dense(units))
            model.add(LeakyReLU(0.2))
            # model.add(BatchNormalization())
            # model.add(Dropout(0.5))
        model.add(Dense(self.gen_dim))#, "tanh"))
        # plot_model(model, 'models/generator.png', show_shapes=True)
        # model.summary()
        return model

    def compile(self, d_opt, g_opt):#, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        def discriminator_loss(real, fake):
            return tf.reduce_mean(tf.random.normal((tf.shape(fake)[0], 1), 1, 0.2) * fake) -\
                   tf.reduce_mean(tf.random.normal((tf.shape(real)[0], 1), 1, 0.2) * real)
        def generator_loss(fake):
            return -tf.reduce_mean(fake)#tf.random.normal((tf.shape(fake)[0], 1), 1, 0.05) *

        self.d_opt = d_opt
        self.g_opt = g_opt
        self.d_loss_fn = discriminator_loss
        self.g_loss_fn = generator_loss

    def gradient_penalty(self, batch_size, real, fake):
        # alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        # diff = fake - real
        # interpolated = real + alpha * diff

        epsilon = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        interpolated = epsilon*real + ( 1 - epsilon ) * fake
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.dis(interpolated, training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real):
        if isinstance(real, tuple):
            real = real[0]

        batch_size = tf.shape(real)[0]

        for i in range(self.d_steps):
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                fake = self.gen(random_latent_vectors, training=True)
                fake_logits = self.dis(fake, training=True)
                real_logits = self.dis(real, training=True)

                d_cost = self.d_loss_fn(real=real_logits, fake=fake_logits)
                gp = self.gradient_penalty(batch_size, real, fake)
                d_loss = d_cost + gp * self.gp_weight

            d_gradient = tape.gradient(d_loss, self.dis.trainable_variables)
            self.d_opt.apply_gradients(zip(d_gradient, self.dis.trainable_variables))

        random_latent_vectors = tf.random.normal(shape=(2*batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated = self.gen(random_latent_vectors, training=True)
            gen_img_logits = self.dis(generated, training=True)
            g_loss = self.g_loss_fn(gen_img_logits)

        gen_gradient = tape.gradient(g_loss, self.gen.trainable_variables)
        self.g_opt.apply_gradients(zip(gen_gradient, self.gen.trainable_variables))
        return {"d_loss": d_loss, "g_loss": g_loss}


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, real, feature_names=None, log_dir=None, frequency = 10, sclr=None) -> None:
        self.real = real
        self.freq = frequency
        self.num_samples = real.shape[0]
        # self.sclr = sclr
        if feature_names:
            self.feature_names = feature_names 
        else:
            self.feature_names = np.arange(real.shape[1]).astype(str) 

        if log_dir:
            self.log_dir = log_dir.rstrip('/')
        else:
            self.log_dir = f"models/wgan_{datetime.now().strftime('%y%m%d-%H%M')}"
        os.system(f"mkdir -p {self.log_dir}")
        
        self.f, self.axs = plt.subplots(4,5, figsize=(30,15))
        self.f.tight_layout()
        self.axs = self.axs.flatten()
        self.bins = 250
        for i in range(real.shape[1]):
                ax = self.axs[i]
                a, b, c = ax.hist(self.real[:,i],histtype='step',bins=self.bins,label='real')
                ax.grid(which='both')
                # ax.set_xlim([-1,1])
                ax.set_ylim([1,1.25*max(a)])
                # ax.set_yscale('log')
                ax.set_title(self.feature_names[i])

    def on_epoch_end(self, epoch, logs=None):
        epoch+=1
        if epoch % self.freq == 0:
            os.system(f"mkdir -p {self.log_dir}/epoch{epoch}")
            self.model.dis.save_weights(f"{self.log_dir}/epoch{epoch}/discriminator_weights.hdf5")
            self.model.gen.save_weights(f"{self.log_dir}/epoch{epoch}/generator_weights.hdf5")
            random_latent_vectors = tf.random.normal(shape=(self.num_samples, self.model.latent_dim))
            fake = self.model.gen(random_latent_vectors).numpy()
            
            for yscale in ['linear', 'log']:
                bars = []
                for i in range(self.model.gen_dim):
                    ax = self.axs[i]
                    _,_,b = ax.hist(fake[:,i],histtype='step',bins=self.bins,color='red',label='fake') 
                    ax.set_yscale(yscale)
                    bars.append(b)


                self.axs[0].legend()

                if len(self.model.history.epoch):
                    self.axs[-1].clear()
                    xx = np.array(self.model.history.epoch) + 1
                    self.axs[-1].plot(xx, self.model.history.history['d_loss'], label='D loss')
                    self.axs[-1].plot(xx, self.model.history.history['g_loss'], label='G loss')
                    # self.axs[-1].set_yscale('log')
                    self.axs[-1].legend()
                    self.axs[-1].grid()
                    self.axs[-1].set_title(f"Loss History (Epoch {epoch})")

                self.f.savefig(f"{self.log_dir}/epoch{epoch}/plots_{yscale}.png", bbox_inches='tight', dpi=100)
                for bb in bars:
                    for b in bb:
                        b.remove()

if __name__ == "__main__":
    epochs, batch_size, freq = 100000, 4000, 1000
    dirty_version_preserve = True
    print(f"+-------------------------+")
    print(f"| Starting TOPEG WGAN-GP: |")
    print(f"| epochs     - {epochs:<10} |")
    print(f"| batch size - {batch_size:<10} |")
    print(f"| num GPUs   - {len(tf.config.list_physical_devices('GPU')):<10} |")
    print(f"+-------------------------+")

    train = np.load("data/processed/X_train_minmax.npy")
    test = np.load("data/processed/X_test_minmax.npy")
    sclr = load("data/processed/minmax_sclr")
    train = sclr.inverse_transform(train)
    #!: Remove when not trying to replicate FAT-GAN
    # train = np.load("data/processed/X_train_quantile.npy")
    # test = np.load("data/processed/X_test_quantile.npy")
    # sclr = load("data/processed/quantile_sclr")
    feature_names = ['Q2', 'W', 'Gamnu', 'Xbj', 'y', 't', 'phih', 
                    'electron px', 'photon px', 'proton px', 
                    'electron py', 'photon py', 'proton py', 
                    'electron pz', 'photon pz', 'proton pz', 
                    'electron E',  'photon E',  'proton E']
    
    # train = np.load("data/processed/X_train_custom.npy")
    # feature_names = ['log( Q2 )', 'log( MAX - W )', 'log( MAX - Gamnu )', 'log( Xbj )', 'log( MAX - y )', 'log( t )', 'log( phih / ( MAX - phih ) )', 
    #                 'SIGN * sqrt( |electron px| )', 'SIGN * sqrt( |photon px| )', 'SIGN * sqrt( |pr px| )', 
    #                 'SIGN * sqrt( |electron py| )', 'SIGN * sqrt( |photon py| )', 'SIGN * sqrt( |pr py| )', 
    #                 'log( MAX - electron pz )', 'log( photon pz - MIN )', 'log( MAX - pr pz )', 
    #                 'log( electron E )', 'log( MAX - photon E )', 'log( MAX - pr E )']
    
    monitor = GANMonitor(train, feature_names, frequency=freq)
    callbacks= [ monitor ]

    if dirty_version_preserve:
        os.system(f"cp ./wgan_gp.py {monitor.log_dir}/wgan_gp_{monitor.log_dir.split('_')[-1]}.py")

    wgan = WGAN()
    wgan.compile(d_opt=Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9),
                 g_opt=Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9) )

    m = wgan.fit(train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    
    
    # f, ax = plt.subplots(1, 1, figsize=(8,8))
    # ax.set_title("Loss History")
    # ax.grid()
    # xx = np.arange(1, epochs+1)
    # ax.plot(xx, m.history['d_loss'], label="D Loss")
    # ax.plot(xx, m.history['g_loss'], label="G Loss")
    # ax.set_xlim([xx[0],xx[-1]])
    # ax.legend()
    # f.savefig(f"{monitor.log_dir}/losses.png", bbox_inches='tight', dpi=200)
    # ax.set_yscale('log')
    # f.savefig(f"{monitor.log_dir}/log_losses.png", bbox_inches='tight', dpi=200)

