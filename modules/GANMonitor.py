import os
import numpy as np
from joblib import load
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from json import dump

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

    def on_epoch_end(self, epoch, logs=None):
        pd = self.model.pd
        epoch+=1

        if epoch == 1:
            print("Initializing the GANMonitor Callback")
            with open(self.log_dir+"/model_params.json", "w") as ff:
                dump(pd, ff)

            self.RLV = tf.random.normal(shape=(self.samples, pd["latent_dim"]))
            real = np.load(pd["train_set"])[:self.samples]
            if pd["scaler"]:
                self.sclr = load(pd["scaler"])
                self.real = self.sclr.inverse_transform(real)
        
        real = self.real   
        if epoch % self.freq == 0:
            os.system(f"mkdir -p {self.log_dir}/epoch{epoch}")
            self.model.dis.save_weights(f"{self.log_dir}/epoch{epoch}/discriminator_weights.h5")
            self.model.gen.save_weights(f"{self.log_dir}/epoch{epoch}/generator_weights.h5")

            fake = self.model.gen(self.RLV).numpy()
            if self.sclr:
                fake = self.sclr.inverse_transform(fake)
            
            chi2 = 0
            stacked = np.r_[real, fake]
            for yscale in ['linear', 'log']:
                for i in range(pd["MCEG_feats"]):
                    ax = self.axs[i]
                    ax.clear()
                    bins = np.linspace(np.min(stacked[:,i]),np.max(stacked[:,i]),self.bins+1)
                    vr, _, _ = ax.hist(real[:,i],histtype='step',bins=bins,label='MCEG')#,linewidth=2)
                    vf, _, _ = ax.hist(fake[:,i],histtype='step',bins=bins,label='WGAN')#,linewidth=2)
                    ax.set_title(pd["feat_names"][i])
                    ax.set_yscale(yscale)
                    ax.grid(which='both')

                    if yscale == "linear": chi2 += np.sum(((vr-vf)/(np.max([np.ones_like(vr),np.sqrt(vr)], axis=0)))**2)

                self.axs[0].legend()

                if len(self.model.history.epoch):
                    self.axs[-1].clear()
                    xx = np.array(self.model.history.epoch) + 1
                    self.axs[-1].plot(xx, self.model.history.history['g_loss'], label='G loss')
                    self.axs[-1].plot(xx, self.model.history.history['d_loss'], label='D loss')
                    self.axs[-1].legend()
                    self.axs[-1].grid()
                    self.axs[-1].set_title(f"Loss History (Epoch {epoch}, "+r"$\chi^2$="+f"{chi2:.3e})")

                self.f.savefig(f"{self.log_dir}/epoch{epoch}/plots_{yscale}.png", bbox_inches='tight', dpi=100)