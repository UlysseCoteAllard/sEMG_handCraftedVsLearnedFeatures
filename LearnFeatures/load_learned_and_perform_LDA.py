import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import matplotlib.colors as colors


def load_learned_features(layer):
    layer_contents = pd.read_csv("./features_data/learned_features_layer_" + layer + ".csv")
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


def perform_LDA(test_features, training_features, test_labels, training_labels, layer, classes):

    lda = LinearDiscriminantAnalysis()
    lda.fit(training_features, training_labels)
    predictions = lda.predict(test_features)

    return predictions


def perform_LDA_single_feature(test_features, training_features, test_labels, training_labels, layer, classes):
    features_predictions = []
    for feature in range(int(test_features.shape[1] / 10)):
        examples_train = training_features[:, feature * 10:(feature + 1) * 10]
        examples_test = test_features[:, feature * 10:(feature + 1) * 10]
        clf = LinearDiscriminantAnalysis()
        predictions = np.array(clf.fit(examples_train, training_labels).predict(examples_test))
        features_predictions.append(predictions)
    return features_predictions

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_confusion_matrix(ground_truth, predictions, class_names,
                          normalize=False, title=None, fontsize=24):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    class_names = np.array(class_names)
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    confusion_matrix_calculated = confusion_matrix(np.ravel(np.array(ground_truth)),
                                                            np.ravel(np.array(predictions)))
    if normalize:
        confusion_matrix_calculated = confusion_matrix_calculated.astype('float') / \
                                      confusion_matrix_calculated.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        print("Normalized confusion matrix")
    else:
        fmt = 'd'
        print('Confusion matrix, without normalization')
    df_cm = pd.DataFrame(
        confusion_matrix_calculated, index=class_names, columns=class_names,
    )
    print(confusion_matrix_calculated)
    max_confusion_matrix = np.max(confusion_matrix_calculated)
    min_confusion_matrix = np.min(confusion_matrix_calculated)
    cmap = plt.get_cmap("magma")
    new_cmap = truncate_colormap(cmap, min_confusion_matrix, max_confusion_matrix)
    print(max_confusion_matrix, "  ", min_confusion_matrix)

    plt.figure(figsize=(14, 12))  # Sample figsize in inches
    try:
        heatmap = sns.heatmap(df_cm, annot=False, fmt=fmt, cbar=False, annot_kws={"size": 28}, cmap=new_cmap)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    #heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), fontsize=fontsize)
    heatmap.set(
        # ... and label them with the respective list entries
        #title="Accuracy: " + str(accuracy_score(ground_truth, predictions)),
        #ylabel='True label',
        #xlabel='Predicted label'
    )

    #heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=30, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels([], rotation=30, ha='right', fontsize=fontsize)
    heatmap.yaxis.set_ticklabels([], fontsize=fontsize)
    '''
    heatmap.xaxis.label.set_size(fontsize + 4)
    heatmap.yaxis.label.set_size(fontsize + 4)
    heatmap.title.set_size(fontsize + 6)
    '''
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.gcf().subplots_adjust(top=0.90)
    heatmap.figure.tight_layout()
    heatmap.figure.savefig("./confusion_matrix/" + title + ".svg", dpi=1200)
    heatmap.figure.clf()
    plt.clf()
    plt.close()

def handle_handcrafted_features(classes, path="./features_data/handcrafted_predictions.csv"):
    data = pd.read_csv(path)

    features_by_category = [["AFB_", "DAMV_", "DASDV_", "DLD_", "DTM_", "DVARV_", "DV_", "IEMG_", "LD_", "M2_",
                             "MMAV1_", "MMAV2_", "MAV_", "MAX_", "MHW_", "MNP_", "MTW_", "RMS_", "SM_", "SSI_", "TM_",
                             "TTP_", "VAR_", "V_", "WL_"],
                            ["FR_", "MDF_", "MNF_", "SSC_", "ZC_"],
                            ["SAMPEN_", "APEN_", "WAMP_", "BC_", "KATZ_", "MFL_"],
                            ["AR_", "CC_", "DAR_", "DCC_", "DFA_", "PSR_", "SNR_"],
                            ["CE_", "DPR_", "HIST_", "KURT_", "MAVS_", "OHM_", "PKF_", "PSDFD_", "SKEW_", "SMR_",
                             "TSPSD_", "CVF_", "VFD_"],
                            ["UNIQUE"]]

    category = ["SAP", "FI", "NLC", "TSM", "UNI", "UNIQUE"]
    accuracies_handcrafted = {"SAP": [], "FI": [], "NLC": [], "TSM": [], "UNI": [], "UNIQUE": []}
    true_labels = []
    for i, column_name in enumerate(data):
        if i == 0:
            true_labels = data[column_name]
        else:
            plot_confusion_matrix(data[column_name], true_labels, classes, normalize=True,
                                  title="Handcrafted_" + str(column_name))
            category_for_accuracy = 0
            for j, features_category in enumerate(features_by_category):
                found = False
                for feature in features_category:
                    if feature in column_name:
                        category_for_accuracy = j
                        found = True
                        break
                if found:
                    break

            accuracies_handcrafted[category[category_for_accuracy]].append(accuracy_score(true_labels,
                                                                                          data[column_name]))
    print(accuracies_handcrafted)
    for category in accuracies_handcrafted:
        print(len(accuracies_handcrafted[category]))
        print(np.mean(accuracies_handcrafted[category]))
        print(np.std(accuracies_handcrafted[category]))
    #print(np.mean(accuracies_handcrafted))


if __name__ == '__main__':
    classes = ["Neutral", "Radial Deviation", "Wrist Flexion", "Ulnar Deviation", "Wrist Extension", "Supination",
               "Pronation", "Power Grip", "Open Hand", "Chuck Grip", "Pinch Grip"]

    sns.set(font_scale=3.5)
    np.set_printoptions(precision=1)

    #handle_handcrafted_features(classes=classes)

    import os
    print(os.listdir("./"))
    test_labels = np.genfromtxt('./features_data/labels_for_learned_features_TEST.csv', delimiter=',')
    test_labels = np.array(test_labels[1:, 2]).astype(int)
    train_labels = np.genfromtxt('./features_data/labels_for_learned_features.csv', delimiter=',')
    train_labels = np.asarray(train_labels[1:, 2]).astype(int)

    layers_predictions_single_features = []
    layers_predictions_all_features = []

    for layer in range(6):

        test_features = load_learned_features(str(layer) + "_TEST_dataset")
        print(np.shape(test_features))
        train_features = load_learned_features(str(layer))

        #layers_predictions_single_features.append(perform_LDA_single_feature(test_features, train_features, test_labels,
        #                                                                     train_labels, layer, classes))
        layers_predictions_all_features.append(perform_LDA(test_features, train_features, test_labels, train_labels,
                                                           layer, classes))
        for predictions in layers_predictions_all_features:
            print(accuracy_score(test_labels, predictions))
    print(layers_predictions_all_features)
    np.save("./lda_predictions/single_features_predictions.npy", layers_predictions_single_features)
    np.save("./lda_predictions/all_features_predictions.npy", layers_predictions_all_features)
    '''
    layers_predictions_single_features = np.load("./lda_predictions/single_features_predictions.npy")
    layers_predictions_all_features = np.load("./lda_predictions/all_features_predictions.npy")
    
    for block in range(6):
        for feature in range(64):
            predictions = layers_predictions_single_features[block][feature]
            plot_confusion_matrix(test_labels, predictions, classes, normalize=True,
                                  title="Block_" + str(block) + "_Features_" + str(feature))
    for block in range(6):
        predictions = layers_predictions_all_features[block]
        plot_confusion_matrix(test_labels, predictions, classes, normalize=True,
                              title="Block_" + str(block) + "_All_Features")
     
    accuracies_single_feature = []
    for block in range(6):
        for feature in range(64):
            predictions = layers_predictions_single_features[block][feature]
            accuracies_single_feature.append(accuracy_score(test_labels, predictions))
        print("Accuracy single features_data, BLOCK: ", str(block), " is: ", np.mean(accuracies_single_feature), " STD is: ",
              np.std(accuracies_single_feature))

    for block in range(6):
        accuracies_all_features = []
        predictions = layers_predictions_all_features[block]
        print("Accuracy all features_data, BLOCK: ", str(block), " is: ", accuracy_score(test_labels, predictions))
    '''