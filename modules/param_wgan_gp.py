import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Dropout

# from modules.KinTool import KinToolTF
from modules.GANMonitor import GANMonitor

class WGAN(keras.models.Model):
    def __init__(self, **kwargs):
        super(WGAN, self).__init__()
        self.pd = kwargs
        self.dis = self.build_network(network="Discriminator")
        self.gen = self.build_network(network="Generator")

    def build_network(self, network):
        pd = self.pd

        model = Sequential(name=network)
        N = network[0].lower()

        in_dim = pd["latent_dim"] if N == "g" else pd["MCEG_feats"]
        model.add(Input(shape=in_dim))
        
        uarr = pd[N+"_units"] if isinstance(pd[N+"_units"],list) else [pd[N+"_units"][1]]*pd[N+"_units"][0]
        for ilay, units in enumerate(uarr):
            model.add(Dense(units))
            
            lact = pd[N+"_act"] if isinstance(pd[N+"_act"],tuple) else pd[N+"_act"][ilay]
            if lact[0] == "leakyrelu":
                model.add(LeakyReLU(alpha=lact[1]["alpha"]))

            if isinstance(pd[N+"_dropout"],tuple) and ilay % pd[N+"_dropout"][0] == 0:
                model.add(Dropout(pd[N+"_dropout"][1]))
            elif isinstance(pd[N+"_dropout"],list) and pd[N+"_dropout"][ilay] > 0.:
                model.add(Dropout(pd[N+"_dropout"][ilay]))

            if pd[N+"_BN"] is not False:
                if isinstance(pd[N+"_BN"],int) and ilay % pd[N+"_BN"] == 0:
                    model.add(BatchNormalization())
                elif isinstance(pd[N+"_BN"],list) and pd[N+"_BN"][ilay]:
                    model.add(BatchNormalization())

        f_act = "linear"
        if  pd[N+"_final_act"][0] == "tanh": 
            f_act = "tanh"

        # needs to change if I do CGAN or if I do a split activation on phih
        # also won't work for advanced activations
        model.add(Dense(1 if N == "d" else pd["MCEG_feats"], activation=f_act)) 

        # plot_model(model, "models/discriminator.png", show_shapes=True)
        # model.summary()
        return model

    # def LambdaKinTool(self, x):
    #     kt = KinToolTF(x, self.sclr,eBeamE=5,nBeamP=275)
    #     kins= [kt.Q2(), kt.W(), kt.Nu(), kt.xBj(), kt.Y(), kt.MinusT()]
    #     E = [kt.ele.E(), kt.pho.E(), kt.pro.E()]
    #     # "Q2", "W", "Gamnu", "Xbj", "y", "t", "phih", 
    #     #             "electron px", "photon px", "proton px", 
    #     #             "electron py", "photon py", "proton py", 
    #     #             "electron pz", "photon pz", "proton pz", 
    #     #             "electron E",  "photon E",  "proton E"
    #     return tf.concat([kins, x, E])

    def compile(self):
        super(WGAN, self).compile()
        pd = self.pd

        d_r_err = pd["d_loss_err"] if isinstance(pd["d_loss_err"],float) else pd["d_loss_err"][0]
        d_f_err = pd["d_loss_err"] if isinstance(pd["d_loss_err"],float) else pd["d_loss_err"][1]
        g_err = pd["g_loss_err"]

        def discriminator_loss(real, fake):
            r_noise = tf.random.normal((tf.shape(real)[0], 1), 1, d_r_err) if d_r_err else 1
            f_noise = tf.random.normal((tf.shape(fake)[0], 1), 1, d_f_err) if d_f_err else 1
            return tf.reduce_mean( f_noise * fake) - tf.reduce_mean( r_noise * real)
        def generator_loss(fake):
            noise = tf.random.normal((tf.shape(fake)[0], 1), 1, g_err) if g_err else 1
            return -tf.reduce_mean( noise * fake)
       
        self.d_loss_fn = discriminator_loss
        self.g_loss_fn = generator_loss

        if pd["d_opt"][0] == "adam":
            pars = pd["d_opt"][1]
            self.d_opt=Adam(learning_rate=pars["lrate"], beta_1=pars["beta1"], beta_2=pars["beta2"])
        if pd["g_opt"][0] == "adam":
            pars = pd["g_opt"][1]
            self.g_opt=Adam(learning_rate=pars["lrate"], beta_1=pars["beta1"], beta_2=pars["beta2"])

    def gradient_penalty(self, batch_size, real, fake):
        pd = self.pd

        if pd["gp_type"] == "normal":
            alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
            interpolated = real + alpha * ( fake - real )
        elif pd["gp_type"] == "uniform":
            epsilon = tf.random.uniform([batch_size, 1], 0.0, 1.0)
            interpolated = epsilon * real + ( 1 - epsilon ) * fake
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.dis(interpolated, training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real):
        pd = self.pd
        
        if isinstance(real, tuple):
            real = real[0]

        batch_size = tf.shape(real)[0]

        for i in range(pd["d_steps"]):
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, pd["latent_dim"])
            )
            with tf.GradientTape() as tape:
                fake = self.gen(random_latent_vectors, training=True)
                fake_logits = self.dis(fake, training=True)
                real_logits = self.dis(real, training=True)

                d_cost = self.d_loss_fn(real=real_logits, fake=fake_logits)
                gp = self.gradient_penalty(batch_size, real, fake)
                d_loss = d_cost + gp * pd["gp_weight"]

            d_gradient = tape.gradient(d_loss, self.dis.trainable_variables)
            self.d_opt.apply_gradients(zip(d_gradient, self.dis.trainable_variables))

        random_latent_vectors = tf.random.normal(shape=(2*batch_size, pd["latent_dim"]))
        with tf.GradientTape() as tape:
            generated = self.gen(random_latent_vectors, training=True)
            gen_img_logits = self.dis(generated, training=True)
            g_loss = self.g_loss_fn(gen_img_logits)

        gen_gradient = tape.gradient(g_loss, self.gen.trainable_variables)
        self.g_opt.apply_gradients(zip(gen_gradient, self.gen.trainable_variables))
        return {"d_loss": d_loss, "g_loss": g_loss}

def build_param_dict(dd=None):
    default_dict = {
        # Model Params
        "model_name"    : f"WGAN_{datetime.now().strftime('%y%m%d-%H%M')}", # if you couldn't guess... it's the models name
        "FAR"           : False,        # Feature Augment and Reduced (fewer features, calc kin feats using physics)
        "latent_dim"    : 128,          # Noise dim for G
        "d_steps"       : 5,            # number of extra training steps for D
        "gp_weight"     : 10.0,         # weight for the gradient penalty
        "gp_type"       : "uniform",    # normal or uniform

        # Data Params
        "scaler"    : "data/processed/standard_sclr",   # loc of scaler
        "train_set" : "data/processed/X_standard.npy",  # loc of scaled train data
        "MCEG_feats"    : 19,                           # Number of features from the MCEG (final G dim and input D dim)
        "feat_names"    : ["Q2", "W", "Gamnu", "Xbj", "y", "t", "phih", #MCEG feature names
                            *[p+v for v in [" px", "py", " pz", " E"] for p in ["electron", "photon", "proton"]]],

        # Discriminator Params
        "d_opt"         : ("adam", {"lrate" : 0.0001, "beta1" : 0.5, "beta2" : 0.9}), # (opt name, params)
        "d_loss_err"    : 0.2,                              # error in D wasserstein loss, float or (real error, fake error)
        "d_units"       : (8, 512),                         # (layers, units/lay) or [units_lay1, units_lay2, ..., units_layN]
        "d_dropout"     : (1, 0.1),                         # False or (n_layers, drop_rate) or [drop_rate1, drop_rate2, ..., drop_rateN]
        "d_BN"          : False,                            # False or int or [bn_bool1, bn_bool2, ..., bn_boolN]
        "d_act"         : ("leakyrelu", {"alpha" : 0.2}),   # hidden layers (activation, params) or list of tuples
        "d_final_act"   : ("linear", ),                     # visible layer (activation, params)

        # Generator Params
        "g_opt"         : ("adam", {"lrate" : 0.0001, "beta1" : 0.5, "beta2" : 0.9}), # (opt name, params)
        "g_loss_err"    : 0.2,                              # error in G wasserstein loss
        "g_units"       : (8, 512),                         # (layers, units/lay) or [units_lay1, units_lay2, ..., units_layN]
        "g_dropout"     : False,                            # False or (n_layers, drop_rate) or [drop_rate1, drop_rate2, ..., drop_rateN]
        "g_BN"          : 1,                                # False or int or [bn_bool1, bn_bool2, ..., bn_boolN]
        "g_act"         : ("leakyrelu", {"alpha" : 0.2}),   # hidden layers (activation, params) or list of tuples
        "g_final_act"   : ("linear", )                      # visible layer (activation, params)
    } 
    if dd:
        for k in dd:
            default_dict[k] = dd[k]

    return default_dict