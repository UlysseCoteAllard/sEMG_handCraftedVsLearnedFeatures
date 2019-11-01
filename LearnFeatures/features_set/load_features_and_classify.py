import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def load_handcrafted_features(handcraft_features_to_use,
                              path_training="../features_data/handcrafted_trainfeatures_151.csv",
                              path_test="../features_data/handcrafted_testfeatures_151.csv",
                              number_of_channels=10):
    data_training = pd.read_csv(path_training)
    data_test = pd.read_csv(path_test)

    features_to_use = ['TrueLabel']
    for feature in handcraft_features_to_use:
        for channel_index in range(1, number_of_channels+1):
            features_to_use.append(feature + str(channel_index))
    return data_training[features_to_use], data_test[features_to_use]

def load_learned_features(layer):
    layer_contents = pd.read_csv("../features_data/learned_features_layer_" + layer + ".csv")
    # All the info is in string format -> convert back to floats
    learned_features_string = layer_contents.LearnFeatures.values
    learned_features_list = [[]]

    for w in range(len(learned_features_string)):
        # CODE FOR COMBINED CHANNEL AVERAGE
        learned_features_list[w].extend(
            [float(i) for i in (list(filter(None, list(map(lambda foo: foo.replace(']', ''), list(
                map(lambda foo: foo.replace('[', ''),
                    list(map(lambda foo: foo.replace('\n', ''), learned_features_string[w][2:-2].split(' '))))))))))])
        learned_features_list.append(list())

    del learned_features_list[-1]

    return np.array(learned_features_list)


def performLDA(train_examples, labels_train, test_examples):
    lda = LinearDiscriminantAnalysis()
    lda.fit(train_examples, labels_train)
    print(test_examples)
    predictions = lda.predict(test_examples)

    return predictions


def prepare_learned_features(layers_to_get, number_of_features_per_layer=14):
    data_train = {}
    data_test = {}
    list_features_to_use = []
    for layer in layers_to_get:
        name_features = []
        test_features = load_learned_features(str(layer) + "_TEST_dataset")
        print(np.shape(test_features))
        train_features = load_learned_features(str(layer))

        for feature in range(int(test_features.shape[1] / 10)):
            name_features.append('_feature_' + str(feature) + "_CH_")
            for channel in range(1, 11):
                data_test['layer_' + str(layer)
                          + '_feature_' + str(feature)
                          + '_CH_' + str(channel)] = test_features[:, (10*feature) + (channel-1)]
                data_train['layer_' + str(layer)
                           + '_feature_' + str(feature)
                           + '_CH_' + str(channel)] = train_features[:, (10*feature) + (channel-1)]

        np.random.shuffle(name_features)
        name_features = name_features[:number_of_features_per_layer]
        features_to_use_for_this_layer = []
        for feature_name in name_features:
            for channel in range(1, 11):
                features_to_use_for_this_layer.append('layer_' + str(layer) + feature_name + str(channel))
        list_features_to_use.extend(features_to_use_for_this_layer)
        print(list_features_to_use)
        print(np.shape(test_features))
        # train_features = load_learned_features(str(layer))

    pd_test = pd.DataFrame(data_test)
    pd_test.to_csv("../features_data/learned_features_TEST_pandas.csv")
    print(pd_test)
    test_examples = pd_test[list_features_to_use]
    pd_train = pd.DataFrame(data_train)
    pd_train.to_csv("../features_data/learned_features_TRAIN_pandas.csv")
    train_examples = pd_train[list_features_to_use]
    return train_examples, test_examples

if __name__ == '__main__':

    test_labels = np.genfromtxt('../features_data/labels_for_learned_features_TEST.csv', delimiter=',')
    labels_test = np.array(test_labels[1:, 2]).astype(int)
    train_labels = np.genfromtxt('../features_data/labels_for_learned_features.csv', delimiter=',')
    labels_train = np.asarray(train_labels[1:, 2]).astype(int)
    layers_to_get = [0, 1, 2, 3, 4, 5]
    list_features_to_use = []
    train_examples_learned, test_examples_learned = prepare_learned_features(layers_to_get=layers_to_get,
                                                             number_of_features_per_layer=64)

    predictions = performLDA(train_examples_learned, labels_train, test_examples_learned)
    print(np.shape(test_examples_learned))
    print(accuracy_score(labels_test, predictions))




    handcraft_features_to_use = ["AR_1_CH", "AR_3_CH", "DAR_1_CH", "DAR_3_CH", "CC_3_CH", "DCC_2_CH",
                                 "SSC_1_CH", "FR_1_CH", "MDF_1_CH", "MNF_1_CH", "PKF_1_CH", "ZC_1_CH",
                                 "TDPSD_1_CH", "TDPSD_2_CH",  "TDPSD_3_CH", "TDPSD_4_CH", "TDPSD_5_CH", "TDPSD_6_CH",
                                 #"AFB_1_CH",
                                 "MFL_1_CH", "WAMP_1_CH"]
    dataset_training, dataset_test = load_handcrafted_features(handcraft_features_to_use)
    labels_train = dataset_training['TrueLabel']
    labels_test = dataset_test['TrueLabel']
    train_examples_handcrafted = dataset_training.drop('TrueLabel', axis=1)
    test_examples_handcrafted = dataset_test.drop('TrueLabel', axis=1)
    print(train_examples_handcrafted)

    predictions = performLDA(train_examples_handcrafted, labels_train, test_examples_handcrafted)

    print(accuracy_score(labels_test, predictions))

    train_examples = pd.concat([train_examples_handcrafted, train_examples_learned], axis=1)
    test_examples = pd.concat([test_examples_handcrafted, test_examples_learned], axis=1)

    predictions = performLDA(train_examples, labels_train, test_examples)
    print(accuracy_score(labels_test, predictions))

