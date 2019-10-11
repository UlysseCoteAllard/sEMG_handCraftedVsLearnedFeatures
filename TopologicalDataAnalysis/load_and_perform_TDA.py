import numpy as np
import pandas as pd
import kmapper as km
from sklearn.decomposition import PCA
import sklearn
import scipy
from itertools import compress

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
            feature_names.append("Handcrafted_" + feature_type[f] + "_" + str(c))


    new_arr = np.zeros((features.shape[0], 450))
    for i in range(new_arr.shape[0]):
        new_arr[i,:] = features[i]

    norm_features = scipy.stats.zscore(new_arr, axis=0, ddof=1)

    norm_features_nonan = norm_features[:,~np.isnan(norm_features).any(axis=0)]
    feature_names_nonan = list(compress(feature_names, ~np.isnan(norm_features)[0]))

    return norm_features_nonan, feature_names_nonan


def load_learned_features(layer):
    # Objectives:
    # 1: Load in learned features and prepare for analysis
    # 2: Label learned features with generic names

    #1:
    layer_contents = pd.read_csv("../Dataset/processed_dataset/learned_features_layer_" + str(layer) + ".csv")

    # All the info is in string format -> convert back to floats
    #learned_features_string = layer_contents.LearnedFeatures.values
    learned_features_string = layer_contents.values
    learned_features_list = [[]]
    for w in range(len(learned_features_string)):
        # CODE FOR COMBINED CHANNEL AVERAGE
        learned_features_list[w].extend([float(i) for i in (list(filter(None,list(map(lambda foo: foo.replace(']',''), list(map(lambda foo: foo.replace('[',''),list(map(lambda foo: foo.replace('\n',''), learned_features_string[w][2][2:-2].split(' '))))))))))])
        # CODE FOR ALL CHANNEL AVERAGE
        #learned_features_list[w].extend(# Add to inner list (window w)
        #    [float(i) for i in (list(map(lambda foo: foo.replace('\n', ''), #Remove \n characters
        #             (list(filter(None, #Remove empty elements (when multiple spaces are next to eachother
        #                          learned_features_string[w][1:-1].split(' ')))))))]) # Split string every space
        # Make region for next window
        learned_features_list.append(list())

    del learned_features_list[-1]
    learned_features = np.array(learned_features_list)

    # 2: zscore of deep features

    new_arr = np.zeros((learned_features.shape[0], len(learned_features_list[0])))
    for i in range(new_arr.shape[0]):
        new_arr[i, :] = learned_features[i]

    learned_features = scipy.stats.zscore(new_arr, axis=0, ddof=1)

    learned_features_names = []
    for i in range(int(len(learned_features[0])/10)):
        for j in range(10):
            learned_features_names.append("LeF" + str(layer) + '_c' + str(j)+ '_n' + str(i))

    return learned_features, learned_features_names


def gather_features():

    # SECTION: Load learned features
    print("Loading Learned Features")

    # Takes takes about 18 minutes to load and prepare all data
    learned_features0, learned_names0 = load_learned_features(0)
    print("Loaded0")
    learned_features1, learned_names1 = load_learned_features(1)
    print("Loaded1")
    learned_features2, learned_names2 = load_learned_features(2)
    print("Loaded2")
    learned_features3, learned_names3 = load_learned_features(3)
    print("Loaded3")
    learned_features4, learned_names4 = load_learned_features(4)
    print("Loaded4")
    learned_features5, learned_names5 = load_learned_features(5)
    print("Loaded5")

    # SECTION: Load handcrafted heatures
    print("Loading Handcrafted Features")
    handcrafted_features, handcrafted_names = load_handcrafted_features("../Dataset/processed_dataset/FEATURES_train.npy")

    features = [[]]
    for w in range(learned_features4.shape[0]):
        features[w].append(
            np.concatenate((handcrafted_features[w], learned_features0[w], learned_features1[w], learned_features2[w],
                            learned_features3[w], learned_features4[w], learned_features5[w]), axis=0))
        features.append(list())

    features = np.delete(features, (-1), axis=0)
    all_features = np.array(features)
    for i in range(all_features.shape[0]):
        all_features[i] =np.array(features[i][0])

    feat_names = handcrafted_names + learned_names0 + learned_names1 + \
            learned_names2 + learned_names3 + learned_names3 + learned_names4

    return all_features, feat_names


def perform_TDA(data, names, NR, PO, filt, cf, layer, tooltips,nclusters):
    mapper = km.KeplerMapper(verbose=0)

    projected_data = mapper.project(data, projection=filt)

    graph = mapper.map(projected_data, data,
                       cover=km.Cover(n_cubes=NR, perc_overlap=PO),
                       clusterer=sklearn.cluster.AgglomerativeClustering(n_clusters=nclusters,
                                                                         affinity='euclidean',
                                                                         memory=None,
                                                                         connectivity=None,
                                                                         compute_full_tree='auto',
                                                                         linkage='ward',
                                                                         pooling_func='deprecated')
                       )

    #cf = mapper.project(data, projection="knn_distance_2")
    html = mapper.visualize(graph,
                            X_names=names,
                            custom_tooltips=tooltips,
                            path_html="../Dataset/TopologyResults/All/MAP_" + str(NR) +"_" + str(PO) + "_"+ filt + "_" + str(nclusters) + ".html",
                            title="Handcrafted vs. Learned Features - Layer" + str(layer),
                            color_function=np.asarray(cf))

if __name__ == "__main__":

    features, names = gather_features()

    # This is really hacky, but all the previous feature vectors are all in different formats.
    # This step goes back to a clean nparray.



    feature_arr = np.zeros((97186, 4250))
    for i in range(feature_arr.shape[0]):
        feature_arr[i,:] = features[i]


    print("Reducing dimensionality of windows")
    #Transpose dataset
    feature_arr = np.transpose(feature_arr)
    #Use PCA
    pca = PCA(n_components=800)
    pca.fit_transform(feature_arr)
    # Determine amount of PCs needed for 95% variance of learned features
    num_pc = 0
    for p in range(pca.explained_variance_ratio_.shape[0]):
        if np.sum(pca.explained_variance_ratio_[:p]) >= 0.99:
            num_pc = p
            break
    pca = PCA(n_components=num_pc)

    feature_arr = pca.fit_transform(feature_arr)

    # SECTION: Perform TDA on prepared dataset
    print("Beginning topological data analysis")

    filt = "knn_distance_2"

    for N in range(10):
        NR = N*5+5
        for P in range(15):
            PO = P*0.05+0.05
            for clusters in range(5,20,1):
                #cf = [int(all(x in elem for x in ['LeF0']) ) for elem in names ]
                #perform_TDA(feature_arr, names, NR, PO, filt, cf, 0, np.asarray(names))

                #cf = [int(all(x in elem for x in ['LeF1'])) for elem in names]
                #perform_TDA(feature_arr, names, NR, PO, filt, cf, 1, np.asarray(names))

                #cf = [int(all(x in elem for x in ['LeF2'])) for elem in names]
                #perform_TDA(feature_arr, names, NR, PO, filt, cf, 2, np.asarray(names))

                #cf = [int(all(x in elem for x in ['LeF3'])) for elem in names]
                #perform_TDA(feature_arr, names, NR, PO, filt, cf, 3, np.asarray(names))

                #cf = [int(all(x in elem for x in ['LeF4'])) for elem in names]
                #perform_TDA(feature_arr, names, NR, PO, filt, cf, 4, np.asarray(names))

                #cf = [int(all(x in elem for x in ['LeF5'])) for elem in names]
                #perform_TDA(feature_arr, names, NR, PO, filt, cf, 5, np.asarray(names))

                cf = [int(all(x in elem for x in ['LeF'])) for elem in names]
                perform_TDA(feature_arr, names, NR, PO, filt, cf, 6, np.asarray(names),clusters)

    print("Done!")

