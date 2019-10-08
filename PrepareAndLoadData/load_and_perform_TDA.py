import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def load_handcrafted_features(loc):
    features = np.load(loc,allow_pickle=True)
    features = np.delete(features, (-1),axis=0)
    for i in range(features.shape[0]):
        features[i] = np.real(np.array(features[i][0]))

    feature_type = ["AR0", "AR1", "AR2", "AR3", "AR4",
                    "CC0", "CC1", "CC2", "CC3", "CC4",
                    "DASDV", "HIST1", "HIST2", "HIST3", "IEMG",
                    "IQR1", "IQR2", "LD", "MAVFD", "MAVFDn",
                    "MAV", "MAVSD", "MAVSDn", "MAVSLP", "MDF",
                    "MMAV1", "MMAV2", "MNF", "MNP", "MPK",
                    "MSR", "MYOP", "RANGE", "RMS", "SKEW",
                    "SM", "SSC", "SSI", "STD", "TM",
                    "TTP", "VAR", "WAMP", "WL", "ZC"]
    feature_names = [];
    for c in range(10):
        for f in range(len(feature_type)):
            feature_names.append(feature_type[f] + "_" + str(c))

    # Remove padding on final feature row
    return features,feature_names


def load_learned_features(layer, var_retained=0.95):
    # Objectives:
    # 1: Load in learned features and prepare for analysis
    # 2: Reduce Dimensionality of learned features through PCA
    # 3: Label learned features with generic names

    #1:
    layer_contents = pd.read_csv("../Dataset/processed_dataset/learned_features_layer_" + str(layer) + ".csv")

    # All the info is in string format -> convert back to floats
    learned_features_string = layer_contents.LearnedFeatures.values
    learned_features_list = [[]]
    for w in range(len(learned_features_string)):
        learned_features_list[w].extend(# Add to inner list (window w)
            [float(i) for i in (list(map(lambda foo: foo.replace('\n', ''), #Remove \n characters
                     (list(filter(None, #Remove empty elements (when multiple spaces are next to eachother
                                  learned_features_string[w][1:-1].split(' ')))))))]) # Split string every space
        # Make region for next window
        learned_features_list.append(list())

    del learned_features_list[-1]
    learned_features_raw = np.array(learned_features_list)

    #2:
    pca = PCA(n_components=len(learned_features_raw[0]))
    pca.fit(learned_features_raw)
    # Determine amount of PCs needed for 95% variance of learned features
    num_pc = 0
    for p in range(pca.explained_variance_ratio_.shape[0]):
        if np.sum(pca.explained_variance_ratio_[:p]) >= var_retained:
            num_pc = p
            break
    pca = PCA(n_components=num_pc)

    learned_features_representation = pca.fit_transform(learned_features_raw)

    # 3:
    learned_features_names = []
    for i in range(num_pc):
        learned_features_names.append("Learned_Feature" +str(i) +"_" + str(layer))

    return learned_features_representation, learned_features_names


def gather_features():

    # SECTION: Load handcrafted heatures
    print("Loading Handcrafted Features")
    handcrafted_features, handcrafted_names = load_handcrafted_features("../Dataset/processed_dataset/FEATURES_train.npy")

    # SECTION: Load learned features
    print("Loading Learned Features")

    var_retained_pca = 0.99
    # Takes about 5 minutes per layer
    learned_features0, learned_names0 = load_learned_features(0, var_retained_pca)
    learned_features1, learned_names1 = load_learned_features(1, var_retained_pca)
    learned_features2, learned_names2 = load_learned_features(2, var_retained_pca)
    learned_features3, learned_names3 = load_learned_features(3, var_retained_pca)
    learned_features4, learned_names4 = load_learned_features(4, var_retained_pca)
    learned_features5, learned_names5 = load_learned_features(5, var_retained_pca)

    features = [[]]
    for w in range(handcrafted_features.shape[0]):
        features[w].append(
            np.concatenate((handcrafted_features[w], learned_features0[w], learned_features1[w], learned_features2[w],
                            learned_features3[w], learned_features4[w], learned_features5[w]), axis=0))
        features.append(list())

    features = np.delete(features, (-1),axis=0)
    features = np.array(features)
    for i in range(features.shape[0]):
        features[i] =np.array(features[i][0])

    names = handcrafted_names + learned_names0 + learned_names1 + \
            learned_names2 + learned_names3 + learned_names3 + learned_names4

    return features, names


if __name__ == "__main__":

    features, names = gather_features()

    # This is really hacky, but all the previous feature vectors are all in different formats.
    # This step goes back to a clean nparray.
    feature_arr = np.zeros((97186, 481))
    for i in range(feature_arr.shape[0]):
        feature_arr[i,:] = features[i].flatten()

    print("Reducing dimensionality of windows")
    #Transpose dataset
    feature_arr = np.transpose(feature_arr)
    #Use PCA
    pca = PCA(n_components=300)
    pca.fit(feature_arr)
    # Determine amount of PCs needed for 95% variance of learned features
    num_pc = 0
    for p in range(pca.explained_variance_ratio_.shape[0]):
        if np.sum(pca.explained_variance_ratio_[:p]) >= 0.99:
            num_pc = p
            break
    pca = PCA(n_components=num_pc)

    feature_arr = pca.fit_transform(feature_arr)
    # 99% of variance is retained using only 10 PCs - we should probably normalize features using zscore (scale among handcrafted features is quite different)

    # SECTION: Perform TDA on prepared dataset
    print("Beginning topological data analysis")
    # TODO

    print("Done!")