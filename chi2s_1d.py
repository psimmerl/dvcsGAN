import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
try:
    # Disable all GPUSs
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    print("failed to disable gpu")
    # Invalid device or cannot modify virtual devices once initialized.
    exit()
    pass
print(tf.config.get_visible_devices())
tf.random.set_seed(42)

from modules.param_wgan_gp import WGAN, build_param_dict
from json import load as jload
from data.custom_scalerTF import custom_scalerTF


train = np.load("data/raw/X.npy")#[:1_000_000]#np.load(param_dict["train_set"])#
sclr = custom_scalerTF()
print(train.shape)
train[:,-1] = train[:,-1] - 360 * (train[:,-1]>180)

n_samples, batch_size = train.shape[0], 5_000_000
print(f"Nb samples: {n_samples}\nbatch size: {batch_size}")

chi2s = []
# above_min = 0
bins = 250
for mname, epochs in [("CM_small_dLNdKR", 395)]:
    print('-----------------------------')
    print(mname)
    for epoch in range(1, epochs+1):
        with open(f"models/{mname}/model_params.json") as ff:
            param_dict = build_param_dict(jload(ff))
        wgan = WGAN(**param_dict)
        wgan.gen.load_weights(f"models/{mname}/epoch{epoch}/generator_weights.h5")
        wgan.dis.load_weights(f"models/{mname}/epoch{epoch}/discriminator_weights.h5")

        pred = np.zeros((n_samples,param_dict["MCEG_feats"]))

        for i, ss in enumerate(np.arange(batch_size, n_samples+batch_size, batch_size)):
            if ss > n_samples:# not the most elegant but works
                ss = n_samples
            pred[i*batch_size:ss,:] = sclr.inverse_transform(wgan.gen(tf.random.normal(shape=(ss - i*batch_size, param_dict['latent_dim'])), training=False))
            print(f"{ss/n_samples*100:0.0f}%", end="\r")

        chi2, bcs = 0, 0
        for i in range(pred.shape[1]):
            xbins = np.linspace(np.min(np.r_[train[:,i], pred[:,i]]), np.max(np.r_[train[:,i],pred[:,i]]), bins+1)
            for j in range(i, pred.shape[1]):
                if i == j:
                    vr, _ = np.histogram(train[:,i], xbins)
                    vp, _ = np.histogram(pred[:,i], xbins)
                    # bcs += bins
                else:
                    ybins = np.linspace(np.min(np.r_[train[:,j], pred[:,j]]), np.max(np.r_[train[:,j],pred[:,j]]), bins+1)
                    vr, _, _ = np.histogram2d(train[:,i], train[:,j], [xbins,ybins])
                    vp, _, _ = np.histogram2d(pred[:,i], pred[:,j], [xbins,ybins])
                    # bcs += bins**2

                chi2 += np.mean((vp-vr)**2/np.fmax(1,vr))

        chi2s.append((mname, epoch, chi2))#, chi2 * bins / bcs))
        print(*chi2s[-1])
        # if chi2 > chi2s[0]:
        #     above_min += 1
        #     if above_min > 25:
        #         break
        # else:
        #     above_min = 0

        chi2s.sort(key=lambda x:x[2])
        with open("chi2s.txt", 'w') as f:
            f.write("pos model epoch chi2\n")# mean_chi2\n")
            for i, (n, e, c_s) in enumerate(chi2s):#, c_m) in enumerate(chi2s):
                f.write(f"{i} {n} {e} {c_s}\n")# {c_m}\n")

print(chi2s[:10])
