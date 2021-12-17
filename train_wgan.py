import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import sys
import numpy as np
from datetime import datetime
import tensorflow as tf
tf.config.optimizer.set_jit(True)

from modules.GANMonitor import GANMonitor
from modules.param_wgan_gp import WGAN, build_param_dict

for batch_size in [5000, 2500, 10000, 1000, 500, 50, 10, 25000]:
    max_iters, epochs = 5_000_000, 500
    # max_iters, epochs, batch_size = 5_000_000, 500, 5000
    samples_per_epoch = 50_000_000
    epochs = int(max_iters/(samples_per_epoch//batch_size))
    # max_iters = int(epochs*(50_000_000//batch_size))
    freq, verbose = 1, 2

    # Compare pT, |p|, theta, phi
    # Test cartesian, spherical, and cylindrical
    # enforce momentum conservation

    # dt = f"_{datetime.now().strftime('%y%m%d-%H%M')}"
    # mn = sys.argv[1]
    # nns =[ mn ]
    # print(nns, dt)

    nns =[]
    for a in ["kiHU_", "kiHN_"]:
        for b in ["_", "_gLN", "_gLNgKR", "_gLDO", "_gSLDO"]:
            for c in ["", "dLLE", "dNLE"]:
                for d in ["_small", "_tiny", ""]:
                    for e in ["dLN", "dLNdKR", "", "dLDO", "dSLDO"]:
                        nns.append("CM"+d+b+e+c+a+str(batch_size))

    # nns = ["CM_small_dLNdLLEdKR"]#, "CM_dLNdKR", "CM_small_gLDOdLNdKR"]#"CM_gLNdLN", "CM_dLDO", "CM_gLNdLDO", "CM_gLN"]

    for mn in nns:
        param_dict = {
            "model_name"    : mn, #f"reg_"+dt,
            "batch_size"    : batch_size,
            "max_epochs"    : epochs,
            "CM"            : True,
            "train_set"     : "data/processed/X_cs.npy",
            "scaler"        : "data/processed/cs_sclr",
            "feat_names"    : ["Q2", "Xbj", "y", "t", "phih"],
            # "feat_names"    : ["Q2", "W", "Xbj", "y", "t", "phih"],
            # "feat_names"    : [part + mom for part in ["ele ", "pho ", "pro "] for mom in ["px","py","pz","E"]],
            # "feat_names"    : [r"log[$Q^2$-4]", r"log[0.9-$y$]", r"log[$-t$-0.001]", r"$\phi_{trento}$"],
            # "feat_names"    : [r"$Q^2$", r"$y$", r"$-t$", r"$\phi_{trento}$"],
            # "MCEG_feats"    : 4,
            # "kernel_init"   : "he_normal",
            "kernel_init"   : "he_uniform",
            "MMD_loss"      : True,
            "latent_dim"    : 256,

            "d_units"       : (5, 512),
            "d_dropout"     : False,
            "d_BN"          : False,
            "d_LN"          : False,
            "d_opt"         : ("adam", {"lrate" : 0.0001, "beta1" : 0.5, "beta2" : 0.9, "epsilon" : 1e-07}),

            "g_units"       : (5, 512),
            "g_dropout"     : False,
            "g_BN"          : False,
            "g_LN"          : False,
            "g_opt"         : ("adam", {"lrate" : 0.0001, "beta1" : 0.5, "beta2" : 0.9, "epsilon" : 1e-07}), 
            "g_final_act"   : ("tanh", ),
        }

        param_dict["MCEG_feats"] = len(param_dict["feat_names"])
        
        if "kiHU" in mn:
            param_dict["kernel_init"] = "he_uniform"
        if "kiHN" in mn:
            param_dict["kernel_init"] = "he_normal"


        if "gDO" in mn:
            param_dict["g_dropout"] = (1, .5)
        if "gDO2" in mn:
            param_dict["g_dropout"] = (2, .5)
        if "gLDO" in mn:
            param_dict["g_dropout"] = (1, .2)
        if "gSLDO" in mn:
            param_dict["g_dropout"] = (1, .1)
        if "gBN" in mn:
            param_dict["g_BN"] = 1
        if "gLN" in mn:
            param_dict["g_LN"] = 1
        if "gKR" in mn:
            param_dict["g_kernel_reg"] = 0.001

        if "dDO" in mn:
            param_dict["d_dropout"] = (1, .5)
        if "dLDO" in mn:
            param_dict["d_dropout"] = (1, .2)
        if "dSLDO" in mn:
            param_dict["d_dropout"] = (1, .1)
        if "dBN" in mn:
            param_dict["d_BN"] = 1
        if "dLN" in mn:
            param_dict["d_LN"] = 1
        if "dKR" in mn:
            param_dict["d_kernel_reg"] = 0.001
        if "dLLE" in mn:
            param_dict["d_loss_err"] = 0.1
        if "dNLE" in mn:
            param_dict["d_loss_err"] = 0.0

        if "tiny" in mn:
            param_dict["d_units"] = (5, 128)
            param_dict["g_units"] = (5, 128)
            param_dict["latent_dim"] = 64

        if "small" in mn:
            param_dict["d_units"] = (5, 256)
            param_dict["g_units"] = (5, 256)
            param_dict["latent_dim"] = 128

        # if "cust" in mn:
        #     param_dict["d_units"] = [32, 64, 128, 256, 128, 64, 32, 16]
        #     param_dict["g_units"] = [256, 512, 512, 256, 128, 64, 32]
        #     param_dict["latent_dim"] = 128



        param_dict = build_param_dict(param_dict)

        lj = max(9, len(param_dict['model_name']))
        print(f"+--------------{'-'.ljust(lj,'-')}-+")
        print(f"| Starting MCEG{' WGAN-GP:'.ljust(lj)} |")
        print(f"| model name - {param_dict['model_name'].ljust(lj)} |")
        print(f"| iterations - {str(max_iters).ljust(lj)} |")
        print(f"| epochs     - {str(epochs).ljust(lj)} |")
        print(f"| batch size - {str(batch_size).ljust(lj)} |")
        print(f"| num GPUs   - {str(len(tf.config.list_physical_devices('GPU'))).ljust(lj)} |")
        print(f"| verbose    - {('silent' if verbose == 0 else ('prog bar' if verbose == 1 else 'minimal')).ljust(lj)} |")
        print(f"+--------------{'-'.ljust(lj,'-')}-+")


        # train_dataset = tf.data.Dataset.from_tensor_slices((np.load(param_dict["train_set"])[:50000000], param_dict['feat_names']))
        # train_dataset = train_dataset.batch(batch_size)

        train = np.load(param_dict["train_set"])[:50_000_000]
        # if int(max_iters/epochs) > train.shape[0]//batch_size:
        #     print("WARNING: Extra steps per epoch!")

        fshape = (4,5)
        fsize=(30,15)
        if param_dict["CM"]:
            # train = train[:, [0, 3, 5, 6]]
            fshape = (3,4)
            # fshape = (2,3)
            fsize=(20,8)
        
        monitor = GANMonitor(log_dir=f"models/{param_dict['model_name']}/", frequency=freq,
                                samples=250000, bins=200, fshape=fshape, fsize=fsize)

        wgan = WGAN(**param_dict)
        wgan.compile()

        try:
            m = wgan.fit(train, epochs=epochs, batch_size=batch_size, callbacks=[ monitor ], verbose=verbose, steps_per_epoch=samples_per_epoch//batch_size)
        except:
            pass