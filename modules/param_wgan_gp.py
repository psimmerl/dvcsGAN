import numpy as np
from datetime import datetime
import joblib, pandas, json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Input, Dense, Activation, LeakyReLU, BatchNormalization, Dropout, LayerNormalization, Lambda
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
# from modules.KinTool import KinToolTF
# from modules.GANMonitor import GANMonitor
from data.custom_scalerTF import custom_scalerTF
from data.custom_scaler import custom_scaler

class WGAN(keras.models.Model):
    def __init__(self, **kwargs):
        super(WGAN, self).__init__()
        self.pd = build_param_dict(kwargs)
        self.dis = self.build_network(network="Discriminator")
        self.gen = self.build_network(network="Generator")
        # self.sclr = custom_scalerTF()

    def build_network(self, network):
        pd = self.pd

        init = pd["kernel_init"]
        if pd["kernel_init"] == "normal":
            init = RandomNormal(mean=0., stddev=1.)
        if pd["kernel_init"] == "normal_mod":
            init = RandomNormal(mean=0., stddev=0.02)

        model = Sequential(name=network)
        N = network[0].lower()

        in_dim = pd["latent_dim"] if N == "g" else pd["MCEG_feats"]
        model.add(Input(shape=in_dim))
        
        # uarr = pd[N+"_units"] if isinstance(pd[N+"_units"],list) else [pd[N+"_units"][1]]*pd[N+"_units"][0]
        uarr = [pd[N+"_units"][1]]*pd[N+"_units"][0]
        for ilay, units in enumerate(uarr):
            if pd[N+"_kernel_reg"]:
                model.add(Dense(units, kernel_initializer=init, kernel_regularizer=l2(pd[N+"_kernel_reg"])))#, bias_regularizer=l2(0.001)))
            else:
                model.add(Dense(units, kernel_initializer=init))

            if pd[N+"_BN"] is not False:
                if isinstance(pd[N+"_BN"],int) and ilay % pd[N+"_BN"] == 0:
                    model.add(BatchNormalization())
                elif isinstance(pd[N+"_BN"],list) and pd[N+"_BN"][ilay]:
                    model.add(BatchNormalization())
            
            if pd[N+"_LN"] is not False:
                if isinstance(pd[N+"_LN"],int) and ilay % pd[N+"_LN"] == 0:
                    model.add(LayerNormalization())
                elif isinstance(pd[N+"_LN"],list) and pd[N+"_LN"][ilay]:
                    model.add(LayerNormalization())

            act = pd[N+"_act"] #if isinstance(pd[N+"_act"],tuple) else pd[N+"_act"][ilay]
            if act[0] == "leakyrelu":
                model.add(LeakyReLU(alpha=act[1]["alpha"]))

            # if isinstance(pd[N+"_dropout"],tuple) and ilay % pd[N+"_dropout"][0] == 0:
            if pd[N+"_dropout"] and ilay % pd[N+"_dropout"][0] == 0:
                model.add(Dropout(pd[N+"_dropout"][1]))
            # elif isinstance(pd[N+"_dropout"],list) and pd[N+"_dropout"][ilay] > 0.:
                # model.add(Dropout(pd[N+"_dropout"][ilay]))

        f_act = pd[N+"_final_act"][0]
        # if  pd[N+"_final_act"][0] == "tanh": 
        #     f_act = "tanh"

        # needs to change if I do cGAN or if I do a split activation on phih
        # also won't work for advanced activations
        if N == "g" and pd["CM"]:
            model.add(Dense(4, activation=f_act))
            # model.add(Activation('linear', dtype='float32' ))
            model.add(Lambda(self.LambdaKinTool, output_shape=pd["MCEG_feats"]))
        else:
            model.add(Dense(1 if N == "d" else pd["MCEG_feats"], activation=f_act)) 

        # model.summary()
        return model

    def LambdaKinTool(self, x):
        sclr = custom_scalerTF()
        x = sclr.inverse_transform(x)
        
        q2, y, t, phih  = x[:,0], x[:,1], x[:,2], x[:,3]

        xb = q2 / (2 * y * sclr.nbeb)
        # nu = q2 / (2 * sclr.mpro * xb)
        # nu = y * sclr.nbeb / sclr.mpro
        # w  = tf.sqrt(sclr.mpro2 - q2 + q2/xb)
        # w  = tf.sqrt(sclr.mpro2 - q2 + 2 * y * sclr.nbeb)
        # w  = tf.sqrt(tf.math.maximum(sclr.mpro2 - q2 + 2 * y * sclr.nbeb, sclr.w_range[0]**2))

        x = tf.stack([q2,xb,y,t,phih],axis=1)
        # x = tf.stack([q2,w,xb,y,t,phih],axis=1)
        # x = tf.stack([q2,w,nu,xb,y,t,phih],axis=1)
        return sclr.transform(x)

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
            self.d_opt=Adam(learning_rate=pars["lrate"], beta_1=pars["beta1"], beta_2=pars["beta2"], epsilon=pars["epsilon"])
        if pd["g_opt"][0] == "adam":
            pars = pd["g_opt"][1]
            self.g_opt=Adam(learning_rate=pars["lrate"], beta_1=pars["beta1"], beta_2=pars["beta2"], epsilon=pars["epsilon"])

    ''' Regularization functions for the losses '''
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

    def MMD_loss_gaus(self, real, fake):
        batch_size = tf.shape(real)[0]
        # batch_size_f = tf.shape(fake)[0]
        # assert batch_size_f//2 == batch_size_r # overcautious should be fine without this

        sig = 1 #!TODO: include in param dict
        weight = 1000 #!TODO: include in param dict

        wts, sigmas = None, [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
        if wts is None:
            wts = [1] * len(sigmas)

        XX = tf.matmul(real, real, transpose_b=True)
        XY = tf.matmul(real, fake, transpose_b=True)
        YY = tf.matmul(fake, fake, transpose_b=True)
            
        X_sqnorms = tf.linalg.diag_part(XX)
        Y_sqnorms = tf.linalg.diag_part(YY)

        r = lambda x: tf.expand_dims(x, 0)
        c = lambda x: tf.expand_dims(x, 1)

        K_XX, K_XY, K_YY = 0, 0, 0
        
        XYsqnorm = -2 * XY + c(X_sqnorms) + r(Y_sqnorms)
        for sigma, wt in zip(sigmas, wts):
            gamma = 1 / (2 * sigma**2)
            K_XY += wt * tf.exp(-gamma * XYsqnorm)
            
        # if K_XY_only:
        #     return K_XY
        
        XXsqnorm = -2 * XX + c(X_sqnorms) + r(X_sqnorms)
        YYsqnorm = -2 * YY + c(Y_sqnorms) + r(Y_sqnorms)
        for sigma, wt in zip(sigmas, wts):
            gamma = 1 / (2 * sigma**2)
            K_XX += wt * tf.exp(-gamma * XXsqnorm)
            K_YY += wt * tf.exp(-gamma * YYsqnorm)
            
        # return K_XX, K_XY, K_YY, tf.reduce_sum(wts)
        const_diagonal = tf.reduce_sum(wts)

        m = tf.cast(tf.shape(K_XX)[0], tf.float32)
        n = tf.cast(tf.shape(K_YY)[0], tf.float32)

        # if biased:
        #     mmd2 = (tf.reduce_sum(K_XX) / (m * m)
        #         + tf.reduce_sum(K_YY) / (n * n)
        #         - 2 * tf.reduce_sum(K_XY) / (m * n))
        # else:
        if const_diagonal is not False:
            const_diagonal = tf.cast(const_diagonal, tf.float32)
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = tf.linalg.trace(K_XX)
            trace_Y = tf.linalg.trace(K_YY)

        mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
            + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
            - 2 * tf.reduce_sum(K_XY) / (m * n))

        return mmd2 

        # r1, r2, f1, f2 = real[:batch_size//2], real[batch_size//2:], fake[:batch_size//2], fake[batch_size//2:]

        # r1r2 = tf.reduce_mean(tf.exp(-tf.reduce_sum((r1-r2)**2,axis=1) / sig))
        # f1f2 = tf.reduce_mean(tf.exp(-tf.reduce_sum((f1-f2)**2,axis=1) / sig))
        # rf  = tf.reduce_mean(tf.exp(-tf.reduce_sum((real-fake)**2,axis=1) / sig))

        # return weight * (r1r2 + f1f2 - 2 * rf)**2

    def momentum_regularization(self, fake):
        eta = 100
        # eta1, eta2, eta3, eta4 = 100, 100, 100, 100
        px = tf.square(tf.reduce_sum(fake[7:10]), axis=1)
        py = tf.square(tf.reduce_sum(fake[10:13]), axis=1)
        pz = tf.square(tf.reduce_sum(tf.concat([fake[13:16], tf.constant(-270)])), axis=1)
        # E = tf.square(tf.reduce_sum(tf.concat([fake[13:16], tf.constant(-280)]), axis=1)

        return eta * ( tf.reduce_mean(tf.concat([px, py, pz])) ) #+ eta4*E
        
    ''' --------------------------------------- '''
    def train_step(self, real):
        pd = self.pd
        if isinstance(real, tuple):
            real = real[0]
        batch_size = tf.shape(real)[0]

        for i in range(pd["d_steps"]):
            # random_latent_vectors = tf.random.normal(
            #     shape=(batch_size, pd["latent_dim"])
            # )
            with tf.GradientTape() as tape:
                # fake = self.gen(random_latent_vectors, training=False)
                fake = self.generate(pd["batch_size"], batch_size=2500, invtrans=False, training=False, pandas_df=False)
                fake_logits = self.dis(fake, training=True)
                real_logits = self.dis(real, training=True)
                # fake_logits[:batch_size//20,:],real_logits[:batch_size//20,:]=real_logits[:batch_size//20,:],fake_logits[:batch_size//20,:]
                # if pd["MMD_loss"]:
                #     d_cost = -self.MMD_loss_gaus(real_logits, fake_logits)
                # else:
                d_cost = self.d_loss_fn(real=real_logits, fake=fake_logits)

                gp = self.gradient_penalty(batch_size, real, fake)
                d_loss = d_cost + gp * pd["gp_weight"]

            d_gradient = tape.gradient(d_loss, self.dis.trainable_variables)
            self.d_opt.apply_gradients(zip(d_gradient, self.dis.trainable_variables))

        #!WARNING: 2 * batch_size might break MMD loss?
        # random_latent_vectors = tf.random.normal(shape=(batch_size, pd["latent_dim"]))
        with tf.GradientTape() as tape:
            # gen = self.gen(random_latent_vectors, training=True)
            gen = self.generate(pd["batch_size"], batch_size=2500, invtrans=False, training=True, pandas_df=False)
            gen_lgts = self.dis(gen, training=False)
            g_loss = self.g_loss_fn(gen_lgts) 

            if pd["MMD_loss"]:
                g_loss += 100 * self.MMD_loss_gaus(real, gen)

        gen_gradient = tape.gradient(g_loss, self.gen.trainable_variables)
        self.g_opt.apply_gradients(zip(gen_gradient, self.gen.trainable_variables))
        return {"d_loss": d_loss, "g_loss": g_loss}

    def generate(self, n_samples=100, batch_size=10000, gpu=True, verbose=False, invtrans=True, pandas_df=True, training=False):
        pd = self.pd

        if gpu:
            pred = tf.zeros((1,pd["MCEG_feats"]))
            if invtrans and 'cs_sclr' in pd['scaler']:
                sclr = custom_scalerTF()
            elif invtrans:
                sclr = joblib.load(pd['scaler'])
            for i, ss in enumerate(range(batch_size, n_samples+batch_size, batch_size)):
                if ss > n_samples:# not the most elegant but works
                    ss = n_samples
                # pred[i*batch_size:ss,:] = sclr.inverse_transform(wgan.gen.predict_on_batch(tf.random.normal(shape=(ns, param_dict['latent_dim']))))
                # pred[i*batch_size:ss,:] = sclr.inverse_transform(wgan.gen.predict(tf.random.normal(shape=(ns, param_dict['latent_dim']))))
                if invtrans:
                    pred[i*batch_size:ss,:] = sclr.inverse_transform(pred[i*batch_size:ss,:])
                else:
                    pred = tf.concat([pred,self.gen(tf.random.normal(shape=(ss - i*batch_size, pd['latent_dim'])), training=training)], axis=0)

                if verbose:
                    print(f"{ss}/{n_samples} ({ss/n_samples*100:0.0f}%)  ", end= "\n" if ss >= n_samples else "\r")

            pred = pred[1:]

        else:
            pred = np.zeros((n_samples,pd["MCEG_feats"]))
            if invtrans and 'cs_sclr' in pd['scaler']:
                sclr = custom_scaler()
            elif invtrans:
                sclr = joblib.load(pd['scaler'])
            with tf.device("/cpu:0"):
                for i, ss in enumerate(range(batch_size, n_samples+batch_size, batch_size)):
                    if ss > n_samples:
                        ss = n_samples
                    pred[i*batch_size:ss,:] = self.gen(tf.random.normal(shape=(ss - i*batch_size, pd['latent_dim'])), training=False)#
                    if invtrans:
                        pred[i*batch_size:ss,:] = sclr.inverse_transform(pred[i*batch_size:ss,:])
                    if verbose:
                        print(f"{ss}/{n_samples} ({ss/n_samples*100:0.0f}%)  ", end= "\n" if ss >= n_samples else "\r")
        if pandas_df:
            return pandas.DataFrame(dict(zip(pd["feat_names"],pred.T)))
        else:
            return pred
            # feature_names = [r"$Q^2$ [GeV$^2$]", r"$W$", r"$x_{Bj}$", r"$y$", r"$-t$ [GeV$^2$]", r"$\phi_{HADRON}$"]
            # feature_names = [r"$Q^2$ [GeV$^2$]", r"$W$", r"$\nu$", r"$x_{Bj}$", r"$y$", r"$-t$ [GeV$^2$]", r"$\phi_{HADRON}$"]

def load_model(param_dict=None, gen_weights=None, dis_weights=None):
    if param_dict and gen_weights and dis_weights:
        with open(param_dict) as ff:
            param_dict = build_param_dict(json.load(ff))
        wgan = WGAN(**param_dict)
        wgan.gen.load_weights(gen_weights)
        wgan.dis.load_weights(dis_weights)
        return wgan
    else:
        raise Exception("Something is none")


def build_param_dict(dd=None):
    default_dict = {
        # Model Params
        "model_name"    : f"WGAN_{datetime.now().strftime('%y%m%d-%H%M')}", # if you couldn't guess... it's the models name
        "CM"           : False,        # Feature Augment and Reduced (fewer features, calc kin feats using physics)
        "latent_dim"    : 128,          # Noise dim for G
        "d_steps"       : 5,            # number of extra training steps for D
        "gp_weight"     : 10.0,         # weight for the gradient penalty
        "gp_type"       : "uniform",    # normal or uniform
        "MMD_loss"      : False,        # bool
        "mom_con"       : False,
        # Data Params
        "scaler"    : "data/processed/standard_sclr",   # loc of scaler
        "train_set" : "data/processed/X_standard.npy",  # loc of scaled train data
        "MCEG_feats"    : 19,                           # Number of features from the MCEG (final G dim and input D dim)
        "feat_names"    : ["Q2", "W", "Gamnu", "Xbj", "y", "t", "phih", #MCEG feature names
                            *[p+v for v in ["px", "py", "pz", "E"] for p in ["electron ", "photon ", "proton "]]],

        #!TODO g and d separate
        "kernel_init" : "glorot_uniform",

        # Discriminator Params
        "d_opt"         : ("adam", {"lrate" : 0.0001, "beta1" : 0.5, "beta2" : 0.9}), # (opt name, params)
        "d_loss_err"    : 0.2,                              # error in D wasserstein loss, float or (real error, fake error)
        "d_units"       : (8, 512),                         # (layers, units/lay) or [units_lay1, units_lay2, ..., units_layN]
        "d_dropout"     : (1, 0.1),                         # False or (n_layers, drop_rate) or [drop_rate1, drop_rate2, ..., drop_rateN]
        "d_kernel_reg"  : None,
        "d_BN"          : False,                            # False or int or [bn_bool1, bn_bool2, ..., bn_boolN]
        "d_LN"          : False,                            # False or int or [bn_bool1, bn_bool2, ..., bn_boolN]
        "d_act"         : ("leakyrelu", {"alpha" : 0.2}),   # hidden layers (activation, params) or list of tuples
        "d_final_act"   : ("linear", ),                     # visible layer (activation, params)

        # Generator Params
        "g_opt"         : ("adam", {"lrate" : 0.0001, "beta1" : 0.5, "beta2" : 0.9}), # (opt name, params)
        "g_loss_err"    : 0.2,                              # error in G wasserstein loss
        "g_units"       : (8, 512),                         # (layers, units/lay) or [units_lay1, units_lay2, ..., units_layN]
        "g_dropout"     : False,                            # False or (n_layers, drop_rate) or [drop_rate1, drop_rate2, ..., drop_rateN]
        "g_kernel_reg"  : None,
        "g_BN"          : 1,                                # False or int or [bn_bool1, bn_bool2, ..., bn_boolN]
        "g_LN"          : False,                            # False or int or [bn_bool1, bn_bool2, ..., bn_boolN]
        "g_act"         : ("leakyrelu", {"alpha" : 0.2}),   # hidden layers (activation, params) or list of tuples
        "g_final_act"   : ("linear", )                      # visible layer (activation, params)
    } 

    if dd:
        for k in dd:
            default_dict[k] = dd[k]

    return default_dict