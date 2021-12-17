import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
mpl.rc('axes', labelsize=20)
mpl.rc('xtick', labelsize=15)
mpl.rc('ytick', labelsize=15)
dpi=320
# mpl.rc('title', fontsize=30)
from joblib import load
import numpy as np
import tensorflow as tf
print(tf.config.get_visible_devices())
tf.random.set_seed(42)

from modules.param_wgan_gp import WGAN, build_param_dict
from json import load as jload
from modules.KinTool import KinTool
# from data.custom_scaler import custom_scaler
from data.custom_scalerTF import custom_scalerTF

mname, epoch = "CM_small_dLNdKR", 316


def mpl_labels(ax, x, y, lab, title, fontsize=20):
    ax.set_xlabel(x, fontsize=fontsize)
    if lab == "MCEG": ax.set_ylabel(y, fontsize=fontsize)
    ax.set_title(lab+" "+title, fontsize=fontsize)
    ax.grid()

train = np.load("data/raw/X.npy")[:1000000]#np.load(param_dict["train_set"])#
sclr = custom_scalerTF()
print(train.shape)

n_samples, batch_size = train.shape[0], 500000
print(f"Nb samples: {n_samples}\nbatch size: {batch_size}")

for epoch in range(395, 0, -1):
    print(mname, epoch)
    with open(f"models/{mname}/model_params.json") as ff:
    # with open(f"model_params_{mname}.json") as ff:
        param_dict = build_param_dict(jload(ff))
    wgan = WGAN(**param_dict)
    wgan.gen.load_weights(f"models/{mname}/epoch{epoch}/generator_weights.h5")
    wgan.dis.load_weights(f"models/{mname}/epoch{epoch}/discriminator_weights.h5")
    # wgan.gen.load_weights(f"generator_weights_{mname}.h5")
    # wgan.dis.load_weights(f"discriminator_weights_{mname}.h5")

    pred = np.zeros((n_samples,param_dict["MCEG_feats"]))
    feature_names = param_dict['feat_names']


    for i, ss in enumerate(np.arange(batch_size, n_samples+batch_size, batch_size)):
        if ss > n_samples:# not the most elegant but works
            ss = n_samples
        pred[i*batch_size:ss,:] = sclr.inverse_transform(wgan.gen(tf.random.normal(shape=(ss - i*batch_size, param_dict['latent_dim'])), training=False))

    print("finished pred")
    if param_dict['CM']:
        train[:,-1] = train[:,-1] - 360 * (train[:,-1]>180)
        feature_names = [r"$Q^2$ [GeV$^2$]", r"$W$", r"$x_{Bj}$", r"$y$", r"$-t$ [GeV$^2$]", r"$\phi_{HADRON}$"]

    f, axs = plt.subplots(2,4, figsize=(30,10)); #f.tight_layout()
    axs = axs.flatten()
    chi2=0
    for i in range(pred.shape[1]):
        ax = axs[i]
        alpha = 1
        ax.grid(which='both')
        ax.set_title(feature_names[i], fontsize=20)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        for k in range(2):
            if k == 1:
                ax = ax.twinx()
                ax.set_yscale('log')
                alpha = 0.5
                ax.set_yticklabels([])
            bins = np.linspace(np.min(np.r_[train[:,i], pred[:,i]]), np.max(np.r_[train[:,i],pred[:,i]]), 1000+1)
            vr, _, _ = ax.hist(train[:,i],histtype='step',bins=bins, label="MCEG", linewidth=3, alpha=alpha)#, weights=np.ones_like(train[:,i])/len(train))#, density=train)
            vf, _, _ = ax.hist(pred[:,i],histtype='step',bins=bins, label="GAN", linewidth=2, alpha=alpha)#, weights=np.ones_like(pred[:,i])/len(pred))#, density=train)
        chi2 += np.sum(((vr-vf)/(np.max([np.ones_like(vr),np.sqrt(vr)], axis=0)))**2)#np.sqrt(vr)
            # ax.set_xlim([-1.1,1.1])
        # ax.legend()
    axs[-1].set_axis_off()
    axs[-1].text(0.1, 0.6, "MCEG", fontsize=100, color='C0')
    axs[-1].text(0.1, 0.25, "GAN", fontsize=100, color='C1')
    # axs[-1].text(0.1, 0.1, "Epoch", fontsize=100, color='C1')

    print("chi2 = ",chi2)
    f.savefig(f"imgs/{epoch}_train_train_all_features.png", bbox_inches='tight')#, dpi=dpi)
    print("finished _train_train_all_features")

    f, axs = plt.subplots(1,2, figsize=(20,10))#; f.tight_layout()
    vmin, vmax = 1, 2500*50
    xyrange, bins = [[0.0007, 0.025],[3.5, 60]], 200#(500,500)
    for ax, hh, lab in zip(axs.flatten(), [train, pred], ["MCEG", "GAN"]):
        c, xb, yb, im = ax.hist2d(hh[:,3-1], hh[:,0], bins=bins, range=xyrange, cmap="cividis", norm=mpl.colors.LogNorm(vmin=vmin,vmax=vmax))
        # ax.contour(c.T,extent=[xb.min(),xb.max(),yb.min(),yb.max()],linewidths=3, cmap=plt.cm.Greys_r, levels=[10, 33, 100, 666, 5000, 25000])
        ax.set_xlabel(r"$x_{B}$ [GeV$^2$]", fontsize=20)
        if lab == "MCEG": ax.set_ylabel(r"$Q^{2}$ [GeV$^2$]", fontsize=20)
        ax.set_title(lab, fontsize=25)
        ax.grid()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.plot([.1, 0.000805, 0.002015, 0.00293, 0.004522, 0.006, 0.00775, 0.01, 0.0121, 0.015, 0.018, 0.025], [4, 4, 10, 14, 20, 25, 30, 35.5, 40, 45.25, 50, 58.25], 'k', linewidth=3)

    f.savefig(f"imgs/{epoch}_q2xb_vbig_lines.png", bbox_inches='tight')#, dpi=500), facecolor="white"
    print("finished _q2xb_vbig_lines")

    f, axs = plt.subplots(1,2, figsize=(20,10))#; f.tight_layout()
    for ax, hh, lab in zip(axs.flatten(), [train, pred], ["MCEG", "GAN"]):
        c, xb, yb, im = ax.hist2d(hh[:,3-1], hh[:,0], bins=bins, range=xyrange, cmap="cividis", norm=mpl.colors.LogNorm(vmin=vmin,vmax=vmax))
        ax.contour(c.T,extent=[xb.min(),xb.max(),yb.min(),yb.max()],linewidths=3, cmap=plt.cm.Greys_r, levels=[10, 33, 100, 666, 5000, 25000])
        ax.set_xlabel(r"$x_{B}$", fontsize=20)
        if lab == "MCEG": ax.set_ylabel(r"$Q^{2}$ [GeV$^2$]", fontsize=20)
        ax.set_title(lab, fontsize=25)
        ax.grid()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        # ax.plot([0.005, 0.000805, 0.002015, 0.00293, 0.004522], [4, 4, 10, 14, 20], 'k', linewidth=3)

    f.savefig(f"imgs/{epoch}_q2xb_big_lines.png", bbox_inches='tight')#, dpi=500), facecolor="white"
    print("finished _q2xb_big_lines")

    f, axs = plt.subplots(1,2, figsize=(20,10))#; f.tight_layout()
    vmin, vmax = 1, 2500*50
    xyrange, bins = [[3.5, 60],[-0.05, 1.05]], 150#(500,500)
    for ax, hh, lab in zip(axs.flatten(), [train, pred], ["MCEG", "GAN"]):
        c, xb, yb, im = ax.hist2d(hh[:,0], hh[:,5-1], bins=bins, range=xyrange, cmap="cividis", norm=mpl.colors.LogNorm(vmin=vmin,vmax=vmax))
        ax.contour(c.T,extent=[xb.min(),xb.max(),yb.min(),yb.max()],linewidths=3, cmap=plt.cm.Greys_r, levels=[10, 33, 100, 666, 5000, 25000])#[10, 33, 100, 333, 1000])
        ax.set_xlabel(r"$Q^{2}$ [GeV$^2$]", fontsize=16)
        if lab == "MCEG": ax.set_ylabel(r"$-t$ [GeV$^2$]", fontsize=16)
        ax.set_title(lab, fontsize=20)
        ax.grid()
    f.savefig(f"imgs/{epoch}_q2t_vbig_lines.png", bbox_inches='tight', dpi=dpi)#, facecolor="white"
    print("finished _q2t_vbig_lines")

    f, axs = plt.subplots(1,2, figsize=(20,10))#; f.tight_layout()
    vmin, vmax = 1, 2500*50
    xyrange, bins = [[3.5, 20],[-0.05, 1.05]], 150#(500,500)
    for ax, hh, lab in zip(axs.flatten(), [train, pred], ["MCEG", "GAN"]):
        c, xb, yb, im = ax.hist2d(hh[:,0], hh[:,5-1], bins=bins, range=xyrange, cmap="cividis", norm=mpl.colors.LogNorm(vmin=vmin,vmax=vmax))
        ax.contour(c.T,extent=[xb.min(),xb.max(),yb.min(),yb.max()],linewidths=3, cmap=plt.cm.Greys_r, levels=[10, 33, 100, 666, 5000, 25000])#[10, 33, 100, 333, 1000])
        ax.set_xlabel(r"$Q^{2}$ [GeV$^2$]", fontsize=20)
        if lab == "MCEG": ax.set_ylabel(r"$-t$ [GeV$^2$]", fontsize=20)
        ax.set_title(lab, fontsize=25)
        ax.grid()
    f.savefig(f"imgs/{epoch}_q2t_lines.png", bbox_inches='tight', dpi=dpi)#, facecolor="white"
    print("finished _q2t_lines")

    bins_xbq2 = [np.linspace(0.0008,0.005, 150+1), np.linspace(4,20, 150+1)]
    info_xbq2 = ("$x_B$", "$Q^2$ [GeV$^2$]", "$Q^2$ vs $x_B$: GAN with MCEG Contours", "xbq2")

    bins_q2t = [np.linspace(4,20, 150+1), np.linspace(0.001,1.001, 150+1)]
    info_q2t = ("$Q^2$ [GeV$^2$]", "$-t$ [GeV$^2$]", "$-t$ vs $Q^2$: GAN with MCEG Contours", "q2t")

    for bins, xi, yi, info in zip([bins_xbq2, bins_q2t], [3-1, 0], [0, 5-1], [info_xbq2, info_q2t]):
        # f, ax = plt.subplots(1,1, figsize=(8,8))#; f.tight_layout()
        f, ax = plt.subplots(1,1, figsize=(24,24))#; f.tight_layout()

        ct, xbt, ybt, imt = ax.hist2d(train[:,xi], train[:,yi], bins=bins)
        cp, xbp, ybp, imp = ax.hist2d(pred[:,xi], pred[:,yi], bins=bins)
        ax.clear()

        levels = np.logspace(0, np.log10(cp.max()), 15+1)
        norm = mpl.colors.LogNorm(vmin=levels[0],vmax=levels[-2])

        csp = ax.contourf(cp.T, extent=[xbp.min(), xbp.max(), ybp.min(), ybp.max()], cmap='tab20', norm=norm, levels=levels)
        cst = ax.contour( ct.T, extent=[xbt.min(), xbt.max(), ybt.min(), ybt.max()], linewidths=2, colors='k',  levels=levels[3:-2])
        cbar = f.colorbar(csp, ax=ax, fraction=0.05, pad=0.01)
        cbar.set_label('count')
        cbar.set_ticks(np.logspace(0, 10, 10+1))

        ax.set_xlabel(info[0], fontsize=20)
        ax.set_ylabel(info[1], fontsize=20)
        ax.set_title(info[2], fontsize=25)
        ax.grid()

        if info[0] == "$x_B$":
            ax.ticklabel_format(axis='x',style='sci', scilimits=(0,0))

        f.savefig(f"imgs/{epoch}_{info[3]}_1plt_contours.png", bbox_inches='tight', dpi=dpi)#, facecolor="white"
        print(f"finished {info[3]}_1plt_contours.png")



