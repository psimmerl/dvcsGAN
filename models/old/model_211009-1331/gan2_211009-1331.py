import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt
import os
from datetime import date, datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class GAN(keras.models.Model):
    def __init__(self, discriminator=None, generator=None, latent_dim=128, gen_dim=19) -> None:
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        self.gen_dim = gen_dim
        self.dis = discriminator if discriminator else self.build_discriminator()
        self.gen = generator if generator else self.build_generator()

    def build_discriminator(self) -> Sequential:
        model = Sequential(name="discriminator")
        alpha = 0.2
        model.add(Input((self.gen_dim,)))
        # for units in [128, 256, 512, 256, 128, 64, 32, 15, 8]:
        for units in [128, 256, 256, 128, 64, 32, 15, 8]:
            model.add(Dense(units))
            model.add(LeakyReLU(alpha))
            model.add(BatchNormalization())
        model.add(Dense(1, activation="sigmoid"))
        # plot_model(model, 'models/discriminator.png', show_shapes=True)
        # model.summary()
        return model

    def build_generator(self) -> Sequential:
        model = Sequential(name="generator")
        alpha = 0.2

        model.add(Input(self.latent_dim))
        # for units in [512, 1024, 1024, 512, 256]:
        for units in [256, 512, 1024, 512, 256]:
        # for units in [128, 256, 512, 256, 128]:
            model.add(Dense(units))
            model.add(LeakyReLU(alpha))
            # model.add(BatchNormalization())
            model.add(Dropout(0.5))
        model.add(Dense(self.gen_dim, "tanh"))
        # plot_model(model, 'models/generator.png', show_shapes=True)
        # model.summary()
        return model

    def compile(self, d_opt, g_opt, loss_fn):
        super(GAN, self).compile()
        self.d_opt = d_opt
        self.g_opt = g_opt
        self.loss_fn = loss_fn

        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        # self.loss_history = [[],[]]

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real):
        batch_size = tf.shape(real)[0]
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim))

        generated = self.gen(random_latent_vectors)

        # combined = tf.concat([generated, real], axis=0)
        # labels = tf.concat(
        #     [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        # labels += 0.05 * tf.random.uniform(tf.shape(labels))
        # with tf.GradientTape() as tape:
        #     predictions = self.dis(combined)
        #     d_loss = self.loss_fn(labels, predictions)
        # grads = tape.gradient(d_loss, self.dis.trainable_weights)
        # self.d_opt.apply_gradients(zip(grads, self.dis.trainable_weights))

        # idx_flip = np.random.randint(0,batch_size,(batch_size//20,1)) # can't do with tensors need to find different way
        # generated[idx_flip], real[idx_flip] = real[idx_flip], generated[idx_flip]
        y_generated = tf.ones((batch_size, 1)) +  0.1 * tf.random.uniform((batch_size, 1))
        y_real = tf.zeros((batch_size, 1)) +  0.1 * tf.random.uniform((batch_size, 1))

        with tf.GradientTape() as tape:
            predictions = self.dis(generated)
            d_loss = self.loss_fn(y_generated, predictions)
        grads = tape.gradient(d_loss, self.dis.trainable_weights)
        self.d_opt.apply_gradients(zip(grads, self.dis.trainable_weights))

        with tf.GradientTape() as tape:
            predictions = self.dis(real)
            d_loss = self.loss_fn(y_real, predictions)
        grads = tape.gradient(d_loss, self.dis.trainable_weights)
        self.d_opt.apply_gradients(zip(grads, self.dis.trainable_weights))
        
        ##########################################################################################

        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim))
        misleading_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            predictions = self.dis(self.gen(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.gen.trainable_weights)
        self.g_opt.apply_gradients(zip(grads, self.gen.trainable_weights))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        # self.loss_history[0].append(self.d_loss_metric.result().eval(session=sess))
        # self.loss_history[1].append(self.g_loss_metric.result().eval(session=sess))

        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, real, feature_names=None, log_dir=None, frequency = 10) -> None:
        self.real = real
        self.num_samples = real.shape[0]
        if feature_names:
            self.feature_names = feature_names 
        else:
            self.feature_names = np.arange(real.shape[1]).astype(str) 

        if log_dir:
            self.log_dir = log_dir.rstrip('/')
        else:
            self.log_dir = f"models/model_{datetime.now().strftime('%y%m%d-%H%M')}"
        os.system(f"mkdir -p {self.log_dir}")
        
        self.freq = frequency

        self.f, self.axs = plt.subplots(4,5, figsize=(20,15))
        self.axs = self.axs.flatten()
        for i in range(real.shape[1]):
                ax = self.axs[i]
                ax.hist(self.real[:,i],histtype='step',bins=200,label='real')
                ax.grid()
                ax.set_xlim([-1,1])
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
            
            bars = []
            for i in range(self.model.gen_dim):
                ax = self.axs[i]
                _,_,b = ax.hist(fake[:,i],histtype='step',bins=200,color='red',label='fake') 
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
                self.axs[-1].set_xlim([xx[0], xx[-1]+1])

            self.f.savefig(f"{self.log_dir}/epoch{epoch}/plots.png", bbox_inches='tight', dpi=100)
            for bb in bars:
                for b in bb:
                    b.remove()

if __name__ == "__main__":
    epochs, batch_size, freq = 5000, 64, 10
    dirty_version_preserve = True # As andrey would say "this kills kittens"
    print(f"+---------------------+")
    print(f"| Starting TOPEG GAN: |")
    print(f"| epochs     - {epochs:<6} |")
    print(f"| batch size - {batch_size:<6} |")
    print(f"| num GPUs   - {len(tf.config.list_physical_devices('GPU')):<6} |")
    print(f"+---------------------+")

    train = np.load("data/processed/X_train.npy")
    test = np.load("data/processed/X_test.npy")
    feature_names = ['Q2', 'W', 'Gamnu', 'Xbj', 'y', 't', 'phih', 
                    'electron px', 'photon px', 'He(or proton?) px', 
                    'electron py', 'photon py', 'He(or proton?) py', 
                    'electron pz', 'photon pz', 'He(or proton?) pz', 
                    'electron E',  'photon E',  'He(or proton?) E']
    monitor = GANMonitor(train, feature_names, frequency=freq)
    callbacks= [ monitor ]

    if dirty_version_preserve:
        os.system(f"cp ./gan2.py {monitor.log_dir}/gan2_{monitor.log_dir.split('_')[-1]}.py")

    gan = GAN()
    gan.compile(d_opt=Adam(learning_rate=0.0002),#2, beta_1=0.5, beta_2=0.9),
                g_opt=Adam(learning_rate=0.0001),#2, beta_1=0.5, beta_2=0.9),
                loss_fn=BinaryCrossentropy())

    m = gan.fit(train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    f, ax = plt.subplots(1, 1, figsize=(8,8))
    ax.set_title("Loss History")
    ax.grid()
    xx = np.arange(1, epochs+1)
    ax.plot(xx, m.history['d_loss'], label="D Loss")
    ax.plot(xx, m.history['g_loss'], label="G Loss")
    ax.set_xlim([xx[0],xx[-1]])
    ax.legend()
    f.savefig(f"{monitor.log_dir}/losses.png", bbox_inches='tight', dpi=200)
    ax.set_yscale('log')
    f.savefig(f"{monitor.log_dir}/log_losses.png", bbox_inches='tight', dpi=200)

    # gan.dis.load_weights('models/epoch10/discriminator.hdf5')
    # gan.gen.load_weights('models/epoch10/generator.hdf5')
    # random_latent_vectors = tf.random.normal(shape=(train.shape[0], 64))
    # pred = gan.gen.predict(random_latent_vectors)

    # f, axs = plt.subplots(4,5, figsize=(20,15))
    # axs = axs.flatten()
    # for i in range(pred.shape[1]):
    #     ax = axs[i]
    #     ax.hist(train[:,i],histtype='step',bins=200,label='real')
    #     ax.hist(pred[:,i],histtype='step',bins=200,label='real')
    #     ax.grid()
    #     ax.set_xlim([-1,1])
    #     ax.set_yscale('log')
    #     ax.set_title(feature_names[i])

    # f.savefig(f"imgs/loaded.png", bbox_inches='tight', dpi=100)
