import os
import numpy as np
from joblib import load
import tensorflow as tf
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
from json import dump
from data.custom_scaler import custom_scaler
# from data.custom_scalerTF import custom_scalerTF
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, samples=10000, bins=250, log_dir="models/", frequency=10, fshape=(4,5), fsize=(30,15)):
        self.samples = samples
        self.freq = frequency
        self.real = None # load later
        self.sclr = None # load later

        self.log_dir = log_dir.rstrip('/')
        os.system(f"mkdir -p {self.log_dir}")
        
        self.f, self.axs = plt.subplots(*fshape, figsize=fsize)
        self.f.tight_layout()
        self.axs = self.axs.flatten()
        self.bins = bins

        self.smallest_chi2 = (0, 1e10)

    def on_epoch_end(self, epoch, logs=None):
        pd = self.model.pd
        epoch+=1

        if epoch == 1:
            print("\nInitializing the GANMonitor Callback")
            with open(self.log_dir+"/model_params.json", "w") as ff:
                dump(pd, ff)
            with open(self.log_dir+"/epoch_chi2s.txt", "w") as ff:
                ff.write(self.log_dir.split('/')[-1]+"\n")
            # tf.keras.utils.plot_model(self.model.gen, self.log_dir+"/generator.png", show_shapes=True)
            # tf.keras.utils.plot_model(self.model.dis, self.log_dir+"/discriminator.png", show_shapes=True)

            # if pd["CM"]:
            #     real = real[:, [0, 3, 5, 6]]
            # elif pd["scaler"]:
            #     self.sclr = load(pd["scaler"])
            #     real = self.sclr.inverse_transform(real)
        
        # real = self.real   
        if epoch % self.freq == 0:
            os.system(f"mkdir -p {self.log_dir}/epoch{epoch}")
            self.model.dis.save_weights(f"{self.log_dir}/epoch{epoch}/discriminator_weights.h5")
            self.model.gen.save_weights(f"{self.log_dir}/epoch{epoch}/generator_weights.h5")
            
            real = np.load(pd["train_set"])[:self.samples]
            fake = self.model.generate(n_samples=self.samples, batch_size=2500, verbose=False, invtrans=False, pandas_df=False, gpu=True)
            if isinstance(fake, tf.Tensor):
                fake = fake.numpy()
            # RLV = tf.random.normal(shape=(self.samples, pd["latent_dim"]))
            # fake = self.model.gen(RLV, training=False).numpy()
            
            for yscale in ['log', 'linear']:
                chi2 = 0
                if yscale == "linear": 
                    fake = custom_scaler().inverse_transform(fake)
                    real = custom_scaler().inverse_transform(real)
                stacked = np.r_[real, fake]
                for i in range(pd["MCEG_feats"]):
                    ax = self.axs[i]
                    ax.clear()
                    bins = np.linspace(np.min(stacked[:,i]),np.max(stacked[:,i]),self.bins+1)
                    hr, _, _ = ax.hist(real[:,i],histtype='step',bins=bins,label='MCEG')#,linewidth=2)
                    hf, _, _ = ax.hist(fake[:,i],histtype='step',bins=bins,label='WGAN')#,linewidth=2)
                    ax.set_title(pd["feat_names"][i])
                    ax.set_yscale('log')#yscale)
                    ax.grid(which='both')
                    if pd["CM"] and yscale == "log":
                        ax.set_xlim([-1, 1])

                    if 'phih' not in pd["feat_names"][i]:
                        chi2 += np.sum((hr-hf)**2/np.max([np.ones_like(hr),hf], axis=0))
                chi2 = chi2 / ( self.bins * pd["MCEG_feats"] - 1 )
                self.axs[0].legend()

                if pd["CM"]:
                    for xx, yy, ax in zip([4, 0, 1, 1], [0, 3, 0, 2], self.axs[-5:-1]):
                        ax.clear()
                        ax.hist2d(fake[:,xx], fake[:,yy], bins=self.bins, cmap=mpl.cm.jet, norm=mpl.colors.LogNorm())
                        ax.set_title("x: "+pd["feat_names"][xx] + ", y: " + pd["feat_names"][yy])
                        ax.grid(which='both')

                if len(self.model.history.epoch):
                    self.axs[-1].clear()
                    xx = np.array(self.model.history.epoch) + 1
                    self.axs[-1].plot(xx, self.model.history.history['g_loss'], label='G loss')
                    self.axs[-1].plot(xx, self.model.history.history['d_loss'], label='D loss')
                    self.axs[-1].legend()
                    self.axs[-1].grid()
                    self.axs[-1].set_title(f"Loss History (Epoch {epoch}, "+r"$\chi^2$="+f"{chi2:.2f})")

                self.f.savefig(f"{self.log_dir}/epoch{epoch}/plots_{yscale}.png", bbox_inches='tight', dpi=100)
            with open(self.log_dir+"/epoch_chi2s.txt", "a") as ff:
                ff.write(f"{epoch} {chi2}\n")

            if chi2 < self.smallest_chi2[1]:
                self.smallest_chi2 = (epoch, chi2)
            elif epoch >= self.smallest_chi2[0] + 50:
                self.model.STOP_TRAINING = True
                print(f"\033[91;1m\N{GREEK CAPITAL LETTER CHI}\N{SUPERSCRIPT TWO} has not dropped below {self.smallest_chi2[1]:.4f} in 50 epochs. Stopping training!\033[0m\n")

