import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np
from datetime import datetime
import tensorflow as tf

from modules.GANMonitor import GANMonitor
from modules.param_wgan_gp import WGAN, build_param_dict

epochs, batch_size = 50000, 8000
freq, verbose = 100, 2

dt = "_bfy"#f"_{datetime.now().strftime('%y%m%d-%H%M')}"
mn = sys.argv[1]
print(mn)
param_dict = {
    "model_name"    : f"reg"+dt,
    "d_units"       : (8, 512),
    "d_dropout"     : False,
    "d_BN"          : False,
    "g_units"       : (8, 512),
    "g_dropout"     : False,
    "g_BN"          : False,
}
if mn == "dDO":
    param_dict["model_name"] = f"dDO"+dt
    param_dict["d_dropout"]  = (1, 0.5)
elif mn == "gBN":
    param_dict["model_name"] = "gBN"+dt
    param_dict["g_BN"]       = 1
elif mn == "dDOgBN":
    param_dict["model_name"] = "dDOgBN"+dt
    param_dict["d_dropout"]  = (1, 0.5)
    param_dict["g_BN"]       = 1
elif mn == "gDO":
    param_dict["model_name"] = "gDO"+dt
    param_dict["g_dropout"]  = (1,0.5)
elif mn == "dDOgDO":
    param_dict["model_name"] = "dDOgDO"+dt
    param_dict["d_dropout"]  = (1, 0.5)
    param_dict["g_dropout"]  = (1, 0.5)
elif mn == "fat":
    param_dict["model_name"] = "fat"+dt
    param_dict["d_dropout"]  = (1, 0.1)
elif mn == "fatBN":
    param_dict["model_name"] = "fatBN"+dt
    param_dict["d_dropout"]  = (1, 0.1)
    param_dict["g_BN"]       = 1
else:
    Exception("Unexpected model number")


param_dict = build_param_dict(param_dict)
print(f"+-------------------------+")
print(f"| Starting MCEG WGAN-GP:  |")
print(f"| model name - {param_dict['model_name']:<10} |")
print(f"| epochs     - {epochs:<10} |")
print(f"| batch size - {batch_size:<10} |")
print(f"| num GPUs   - {len(tf.config.list_physical_devices('GPU')):<10} |")
print(f"| verbose    - {'silent' if verbose == 0 else ('prog bar' if verbose == 1 else 'minimal'):<10} |")
print(f"+-------------------------+")

monitor = GANMonitor(log_dir=f"models/{param_dict['model_name']}/", frequency=freq,
                        samples=100000, bins=250, fshape=(4,5), fsize=(30,15))

wgan = WGAN(**param_dict)
wgan.compile()

train = np.load(param_dict["train_set"])
m = wgan.fit(train, epochs=epochs, batch_size=batch_size, callbacks=[ monitor ], verbose=verbose)