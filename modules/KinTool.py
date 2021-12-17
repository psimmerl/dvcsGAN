import math
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt

class LorentzVector():
    _metric = np.array([[1,  0,  0,  0],
                        [0, -1,  0,  0],
                        [0,  0, -1,  0],
                        [0,  0,  0, -1]])

    def __init__(self, m, px, py, pz, kind="mass"):
        if kind=="mass":
            e = np.sqrt(m**2 + px**2 + py**2 + pz**2)
        else:
            e=m
        self.lv = np.array([e, px, py, pz]).T

    def __str__(self):
        return str(self.lv)

    def __add__(self, other):
        return LorentzVector(*(self.lv + other.lv), kind="e")

    def __sub__(self, other):
        return LorentzVector(*(self.lv - other.lv), kind="e")

    def __matmul__(self, other):
        return self.lv.T @ self._metric @ other.lv

    def __pow__(self, pow):
        if pow != 2:
            raise ValueError("Not implemented for powers != 0!")
        return self.lv.T @ self._metric @ self.lv
    
    def copy(self):
        return LorentzVector(*self.lv, kind="e")

    def mass2(self):
        return self.lv.T @ self._metric @ self.lv
    
    def mass(self):
        return np.sign(self.mass2())*np.sqrt(np.abs(self.mass2()))
    
    def E(self):
        return self.lv[0]

    def boost(self, other):
        ''' boosts this lorentz vector into 
            the input lorentz vectors frame '''
        
        b = np.array[0, 0, 0]
        g = 1/np.sqrt(1-(b @ b))
        L = np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                if i == 0 and j == 0:
                    L[i,j] = g
                elif i == 0 or j == 0:
                    L[i,j] = -b[i] * g
                else:
                    L[i,j] = (g-1)*b[i]*b[j]/(b@b)+(i==j)
        
        self.lv = L @ self.lv

        return self

class LorentzVector2():
    _metric = [1,  -1,  -1,  -1]

    def __init__(self, m, px, py, pz, kind="mass"):
        if kind=="mass":
            e = np.sqrt(m**2 + px**2 + py**2 + pz**2)
        else:
            e=m
        self.lv = np.array([e, px, py, pz])

    def __str__(self):
        return str(self.lv)

    def __add__(self, other):
        return LorentzVector2(*(self.lv + other.lv).T, kind="e")

    def __sub__(self, other):
        return LorentzVector2(*(self.lv - other.lv).T, kind="e")

    def __matmul__(self, other): # broken
        return np.sum(self.lv * self._metric * other.lv, axis=1)

    def __pow__(self, pow):
        if pow != 2:
            raise ValueError("Not implemented for powers != 0!")
        return self.lv @ self.lv
    
    def copy(self):
        return LorentzVector2(*self.lv.T, kind="e")

    def mass2(self):
        return np.sum(self.lv * self._metric * self.lv, axis=1)
    
    def mass(self):
        return np.sign(self.mass2())*np.sqrt(np.abs(self.mass2()))
    
    def E(self):
        return self.lv[:, 0]

    def _boost(self, other):
        ''' boosts this lorentz vector into 
            the input lorentz vectors frame '''
        
        b = np.array[0, 0, 0]
        g = 1/np.sqrt(1-(b @ b))
        L = np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                if i == 0 and j == 0:
                    L[i,j] = g
                elif i == 0 or j == 0:
                    L[i,j] = -b[i] * g
                else:
                    L[i,j] = (g-1)*b[i]*b[j]/(b@b)+(i==j)
        
        self.lv = L @ self.lv

        return self
        
class KinTool():
    def Q2(self,  ebeam,        ele): return -(ebeam - ele).mass2()
    def xBj(self, ebeam, nbeam, ele): return self.Q2(ebeam,ele)/(2 * (nbeam @ (ebeam-ele)))
    def Nu(self,  ebeam, nbeam, ele): return self.Q2(ebeam,ele)/(2*nbeam.mass()*self.xBj(ebeam,nbeam,ele))
    def W(self,   ebeam, nbeam, ele): return np.sqrt((nbeam + (ebeam - ele))**2)
    def Y(self,   ebeam, nbeam, ele): return (nbeam @ (ebeam - ele)) / (nbeam @ ebeam)
    def MinusT(self,     nbeam, pro): return -(pro - nbeam)**2
    
# class KinToolTF(tf.keras.layers.Layer, KinTool): 
#     pro_mass = tf.constant(0.9382720813) # GeV/c^2
#     he4_mass = tf.constant(3.7284013254) # GeV/c^2
#     ele_mass = tf.constant(0.0005109989461) # GeV/c^2
#     def __init__(self, sclr, eBeamE=tf.constant(5), nBeamP=tf.constant(275)):
#         super(KinToolTF, self).__init__()
#         self.eBeamE = tf.constant(eBeamE)
#         self.nBeamP = tf.constant(nBeamP)
#         self.means = tf.constant(sclr.means_) # The fastest way to do this would be to manually hardcode in the sclr values, or init the class only once
#         self.stds = tf.sqrt(sclr.vars_)
#         self.ebeam = LorentzVector(self.ele_mass, 0, 0, -self.eBeamE)
#         self.nbeam = LorentzVector(self.pro_mass, 0, 0, self.nBeamP)
    
#     def call(self, x):
#         x = self.stds[:] * x + self.means[:]
#         ele = LorentzVector(self.ele_mass, x[1], x[4], x[7])
#         pho = LorentzVector(0, x[2], x[5], x[8])
#         pro = LorentzVector(self.pro_mass, x[3], x[6], x[9])
#         kins= [self.Q2(self.ebeam, ele), 
#                 self.W(self.ebeam, self.nbeam, ele), 
#                 self.Nu(self.ebeam, self.nbeam, ele), 
#                 self.xBj(self.ebeam, self.nbeam, ele), 
#                 self.Y(self.ebeam, self.nbeam, pro), 
#                 self.MinusT()]
#         e =   [ele.E(), pho.E(), pro.E()]
#         return tf.concat(( [kins, x, e] - self.means ) / tf.keras.sqrt)

#     def Q2(self, x):
#         return ( -(self.ebeam - self.ele).mass2() - self.means[0] ) / tf.keras.sqrt(self.vars[0])
    
#     def xBj(self, x):
#         return ( self.Q2()/(2 * (self.nbeam @ (self.ebeam-self.ele))) - self.means[3] ) / tf.keras.sqrt(self.vars[3])

#     def Nu(self, x):
#         return ( self.Q2()/(2*self.nbeam.mass()*self.self.xBj(self.ebeam,self.nbeam,self.ele)) - self.means[2] ) / tf.keras.sqrt(self.vars[2])
    
#     def W(self, x):
#         return ( tf.keras.sqrt((self.nbeam + (self.ebeam - self.ele))**2) - self.means[1] ) / tf.keras.sqrt(self.vars[1])
    
#     def Y(self, x):
#         return ( (self.nbeam @ (self.ebeam - self.ele)) / (self.nbeam @ self.ebeam) - self.means[4] ) / tf.keras.sqrt(self.vars[4])
    
#     def MinusT(self):
#         return ( -(self.pro - self.nbeam)**2 - self.means[5] ) / tf.keras.sqrt(self.vars[5])
    
#     def E(self):
#         return ( [self.ele.E(), self.pho.E(), self.pro.E()] - self.means[16:] ) / tf.keras.sqrt(self.vars[16:])
    
if __name__ == "__main__":
    data = np.load("data/raw/X.npy")
    feature_names = ['Q2', 'W', 'Gamnu', 'Xbj', 'y', 't', 'phih', 
                    'electron px', 'photon px', 'proton px', 
                    'electron py', 'photon py', 'proton py', 
                    'electron pz', 'photon pz', 'proton pz', 
                    'electron E',  'photon E',  'proton E']
    mpro, mele, mpho = 0.9382720813, 0.0005109989461, 0.0
    ebeam = LorentzVector(mele, 0, 0, -np.sqrt(5**2 - mele**2))
    nbeam = LorentzVector(mpro, 0, 0, 275)
    KT = KinTool()

    # print(ebeam @ nbeam)
    errs = np.zeros((data.shape[0], 12))
    for i, d in enumerate(data):
        if i % 1000 == 0:
            print(i, end="\r")
        # ele = LorentzVector(0, d[7], d[10], d[13])
        ele = LorentzVector(mele, d[7], d[10], d[13])
        pho = LorentzVector(mpho, d[8], d[11], d[14])
        pro = LorentzVector(mpro, d[9], d[12], d[15])
        phoc = ebeam + nbeam - (ele + pro)
        pho = phoc
        
        ele2 = LorentzVector(d[16], d[7], d[10], d[13],kind='e')
        pho2 = LorentzVector(d[17], d[8], d[11], d[14],kind='e')
        pro2 = LorentzVector(d[18], d[9], d[12], d[15],kind='e')

        # print(f"ele E: {ele.E():.8f} {d[16]:.8f}")
        # print(f"pho E: {pho.E():.8f} {d[17]:.8f}")
        # print(f"pro E: {pro.E():.8f} {d[18]:.8f}")
        # print(f"ele M: {ele.mass():.8f} {ele2.mass():.8f}")
        # print(f"pho M: {pho.mass():.8f} {pho2.mass():.8f}")
        # print(f"pro M: {pro.mass():.8f} {pro2.mass():.8f}\n")

        # print(f"ele E: {(ele.E() - d[16])/d[16]*100:.4f}%")
        # print(f"pho E: {(pho.E() - d[17])/d[17]*100:.4f}%")
        # print(f"pro E: {(pro.E() - d[18])/d[18]*100:.4f}%")
        # print(f"ele M: {(ele.mass() - ele2.mass())/ele2.mass()*100:.4f}%")
        # print(f"pho M: {(pho.mass() - pho2.mass())/pho2.mass()*100:.4f}%")
        # print(f"pro M: {(pro.mass() - pro2.mass())/pro2.mass()*100:.4f}%\n")

        # phoc = (ele + pro) - (ebeam + nbeam)
        # print(f"\nphoc E: {phoc.E():.8f} {d[17]:.8f}")
        # print(f"phoc M: {phoc.mass():.8f} {pho2.mass():.8f}")
        # print(f"phoc E: {( phoc.E() - d[17] ) / d[17] * 100:.8f}%")
        # print(f"phoc M: {( phoc.mass() - pho2.mass() ) / pho2.mass() * 100:.8f}%\n")
        
        # # ele = ele2
        # # pho = pho2
        # # pro = pro2

        # print(f"Q2: {KT.Q2(ebeam,ele):.8f} {d[0]:.8f}")
        # print(f"W:  {KT.W(ebeam,nbeam,ele):.8f} {d[1]:.8f}")
        # print(f"Nu: {KT.Nu(ebeam,nbeam,ele):.8f} {d[2]:.8f}")
        # print(f"xB: {KT.xBj(ebeam,nbeam,ele):.8f} {d[3]:.8f}")
        # print(f"y:  {KT.Y(ebeam,nbeam,ele):.8f} {d[4]:.8f}")
        # print(f"t:  {KT.MinusT(nbeam,pro):.8f} {d[5]:.8f}\n")

        # print(f"Q2: {(KT.Q2(ebeam,ele) - d[0])/d[0]*100:.4f}%")
        # print(f"W:  {(KT.W(ebeam,nbeam,ele) - d[1])/d[1]*100:.4f}%")
        # print(f"Nu: {(KT.Nu(ebeam,nbeam,ele) - d[2])/d[2]*100:.4f}%")
        # print(f"xB: {(KT.xBj(ebeam,nbeam,ele) - d[3])/d[3]*100:.4f}%")
        # print(f"y:  {(KT.Y(ebeam,nbeam,ele) - d[4])/d[4]*100:.4f}%")
        # print(f"t:  {(KT.MinusT(nbeam,pro) - d[5])/d[5]*100:.4f}%")

        # print("--------------")

        errs[i, :] = np.array([
        (ele.E() - d[16])/d[16]*100, 
        (pho.E() - d[17])/d[17]*100, 
        (pro.E() - d[18])/d[18]*100, 
        (ele.mass() - ele2.mass())/ele2.mass()*100, 
        (pho.mass() - pho2.mass())/pho2.mass()*100, 
        (pro.mass() - pro2.mass())/pro2.mass()*100, 
        (KT.Q2(ebeam,ele) - d[0])/d[0]*100, 
        (KT.W(ebeam,nbeam,ele) - d[1])/d[1]*100, 
        (KT.Nu(ebeam,nbeam,ele) - d[2])/d[2]*100, 
        (KT.xBj(ebeam,nbeam,ele) - d[3])/d[3]*100, 
        (KT.Y(ebeam,nbeam,ele) - d[4])/d[4]*100, 
        (KT.MinusT(nbeam,pro) - d[5])/d[5]*100 ])

    mins = np.min(errs, axis=0)
    maxs = np.max(errs, axis=0)
    mm = np.mean(errs, axis=0)
    ss = np.std(errs, axis=0)
    nn = ["ele E","pho E","pro E","ele M","pho M","pro M","Q2","W","Nu","xB","y","t"]

    f, axs = plt.subplots(3,4, figsize=(20,10));# f.tight_layout()
    axs = axs.flatten()

    print(f"-: ave% std%, max% min%")
    for ax, err, mn, mx, m, s, n in zip(axs, errs.T, mins, maxs, mm, ss, nn):
        print(f"{n}: {m:.6f}% {s:.6f}%, {mn:.6f}% {mx:.6f}%")
        ax.hist(err, bins=100, weights=np.ones_like(err)/len(err))
        ax.set_title(n+r" (% error)")
        # ax.set_xlabel("% error")
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.grid()

    f.savefig("imgs/perc_error_kintool.png", bbox_inches="tight")

    


