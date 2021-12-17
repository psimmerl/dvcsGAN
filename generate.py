import os

from tensorflow.python import training
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from joblib import load
import numpy as np
import tensorflow as tf
# try:
#     # Disable all GPUS
#     tf.config.set_visible_devices([], 'GPU')
#     visible_devices = tf.config.get_visible_devices()
#     for device in visible_devices:
#         assert device.device_type != 'GPU'
# except:
#     print("failed to disable gpu")
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass
print(tf.config.get_visible_devices())
tf.random.set_seed(42)

from modules.param_wgan_gp import WGAN, build_param_dict
from json import load as jload
# from modules.KinTool import KinTool
# from data.custom_scaler import custom_scaler
from data.custom_scalerTF import custom_scalerTF

mname, epoch = "CM_small_dLN", 1
print(mname, epoch)
# with open(f"models/{mname}/model_params.json") as ff:
with open(f"model_params_{mname}.json") as ff:
    param_dict = build_param_dict(jload(ff))

# print(param_dict)
if 'cs_sclr' in param_dict['scaler']:
    # sclr = custom_scaler()
    sclr = custom_scalerTF()
else:
    sclr = load(param_dict['scaler'])
wgan = WGAN(**param_dict)
# wgan.gen.load_weights(f"models/{mname}/epoch{epoch}/generator_weights.h5")
# wgan.dis.load_weights(f"models/{mname}/epoch{epoch}/discriminator_weights.h5")
wgan.gen.load_weights(f"generator_weights_{mname}.h5")
wgan.dis.load_weights(f"discriminator_weights_{mname}.h5")

# feature_names = param_dict['feat_names']

n_samples, batch_size = 50000000, 1000000
print(f"Nb samples: {n_samples}\nbatch size: {batch_size}")
pred = np.zeros((n_samples,param_dict["MCEG_feats"]))


# nevs = 0
for i, ss in enumerate(np.arange(batch_size, n_samples+batch_size, batch_size)):
    if ss > n_samples:# not the most elegant but works
        ss = n_samples
    ns = ss - i*batch_size
    pred[i*batch_size:ss,:] = sclr.inverse_transform(wgan.gen(tf.random.normal(shape=(ns, param_dict['latent_dim'])), training=False))
    # pred[i*batch_size:ss,:] = sclr.inverse_transform(wgan.gen.predict_on_batch(tf.random.normal(shape=(ns, param_dict['latent_dim']))))
    # pred[i*batch_size:ss,:] = sclr.inverse_transform(wgan.gen.predict(tf.random.normal(shape=(ns, param_dict['latent_dim']))))
    # nevs += ns
    # print(nevs)
