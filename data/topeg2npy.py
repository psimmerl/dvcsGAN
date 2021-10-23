import ROOT
import numpy as np
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler, RobustScaler
from joblib import dump

seed=42
np.random.seed(seed)

if __name__ == "__main__":
    print("Converting TOPEG TNtuple to numpy array")
    #1000000evs
    rdf = ROOT.RDataFrame("TOPEG", "data/CORE_5x275-M4-Nh.root")
    #10000evs
    # # rdf = ROOT.RDataFrame("TOPEG", "/home/psimmerl/Documents/JLab/eic/CORE_5x275-M4-Nh.root")

    dd = rdf.AsNumpy()
    features = ['Q2', 'W', 'Gamnu', 'Xbj', 'y', 't', 'phih', 'part_px', 'part_py', 'part_pz', 'part_e']
    X = np.array(dd.pop(features[0]), dtype=np.float32)
    for feature in features[1:]:
        x = dd.pop(feature)
        if "RVec" in str(type(x[0])):
            xx = np.zeros((X.shape[0], dd['Nb_part'][0]), dtype=np.float32)
            for i in range(len(x)):
                xx[i,:] = [i for i in x[i]]
                # print(xx[i])
                # break
            x = xx
        X = np.c_[X, x]

    #####################################################
    # sclr = QuantileTransformer(output_distribution='normal', random_state=seed)
    # sclr = MinMaxScaler(feature_range=(-1,1)); lab = "minmax"
    # sclr = StandardScaler(); lab = "standard"
    sclr = RobustScaler(); lab = "robust"

    idx = np.random.permutation(X.shape[0])
    X = X[idx]

    np.save("data/raw/X.npy", X)
    X = sclr.fit_transform(X)
    np.save(f"data/processed/X_{lab}.npy", X)

    # print(X_train.shape, X_test.shape)
    # train_split = 0.8
    # X_train = X[idx[:int(train_split*X.shape[0])]]
    # X_test = X[idx[int(train_split*X.shape[0]):]]

    # print(X_train.shape, X_test.shape)

    # np.save("data/raw/X.npy", X_train)
    # np.save("data/raw/X_test.npy", X_test)


    # X_train = sclr.fit_transform(X_train)
    # X_test = sclr.transform(X_test)

    # np.save("data/processed/X_train_minmax.npy", X_train)
    # np.save("data/processed/X_test_minmax.npy", X_test)

    dump(sclr, f"data/processed/{lab}_sclr")
