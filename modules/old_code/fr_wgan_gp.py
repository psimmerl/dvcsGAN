import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Dropout, Lambda
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from modules import KinTool

''' Feature Reduced Wasserstein GAN with Gradient Penalty

Tries to fix the problems with sharp peaks in some of the distribution
since we know all of the kinematic variables can be calculated using 
just the 3 momenta for the final state particles... we just generate those
momenta + phih!

Generate Momenta then calculate kinematic TOPEG variables after 
TOPEG: 19 features
    - Q2, W, gamnu, xbj, y, t, phih
    - electron:
        - px, py, pz, E
    - photon:
        - px, py, pz, E
    - proton:
        - px, py, pz, E

GAN Features:
    - ele_px, ele_py, ele_pz
    - pro_px, pro_py, pro_pz
    - pho_px, pho_py, pho_pz
    - phih
    + Lambda layer at end to add in the kinematic features for the D
    + Prior knowledge about eBeam, nBeam, and rest masses

'''

class FRWGAN(keras.models.Model):
    def __init__(self, discriminator=None, generator=None, d_steps=5, 
                    gp_weight=10.0, latent_dim=128) -> None:
        super(FRWGAN, self).__init__()
        self.latent_dim = latent_dim
        self.gen_dim = 10
        self.aug_dim = 9
        self.d_steps = d_steps
        self.gp_weight = gp_weight
        self.dis = discriminator if discriminator else self.build_discriminator()
        self.gen = generator if generator else self.build_generator()

    def build_discriminator(self) -> Sequential:
        model = Sequential(name="discriminator")
        model.add(Input((self.gen_dim+self.aug_dim,)))
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
        # q2 = Lambda(KinTool.KinTool().Q2())

        # plot_model(model, 'models/generator.png', show_shapes=True)
        # model.summary()
        return model

    def MoreFeat(x):

        # 'Q2', 'W', 'Gamnu', 'Xbj', 'y', 't', 'phih', 
        # 'electron E',  'photon E',  'proton E'

        return x
    
    def compile(self, d_opt, g_opt):#, d_loss_fn, g_loss_fn):
        super(FRWGAN, self).compile()
        def discriminator_loss(real, fake):
            return tf.reduce_mean(tf.random.normal((tf.shape(fake)[0], 1), 1, 0.1) * fake) -\
                   tf.reduce_mean(tf.random.normal((tf.shape(real)[0], 1), 1, 0.1) * real)
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


if __name__ == "__main__":
    epochs, batch_size, freq = 100, 16, 1
    dirty_version_preserve = True
    print(f"+-------------------------+")
    print(f"| Starting TOPEG FRWGAN-GP: |")
    print(f"| epochs     - {epochs:<10} |")
    print(f"| batch size - {batch_size:<10} |")
    print(f"| num GPUs   - {len(tf.config.list_physical_devices('GPU')):<10} |")
    print(f"+-------------------------+")

    train = np.load("data/processed/X_stdscl.npy")
    train = np.load("data/processed/X_stdscl.npy")

    feature_names = ['Q2', 'W', 'Gamnu', 'Xbj', 'y', 't', 'phih', 
                    'electron px', 'photon px', 'proton px', 
                    'electron py', 'photon py', 'proton py', 
                    'electron pz', 'photon pz', 'proton pz', 
                    'electron E',  'photon E',  'proton E']

    monitor = GANMonitor(train, feature_names, frequency=freq, sclr=None)#sclr)
    callbacks= [ monitor ]

    if dirty_version_preserve:
        os.system(f"cp ./fr_wgan_gp.py {monitor.log_dir}/fr_wgan_gp_{monitor.log_dir.split('_')[-1]}.py")

    wgan = FRWGAN()
    wgan.compile(d_opt=Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9),
                 g_opt=Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9) )

    m = wgan.fit(train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    