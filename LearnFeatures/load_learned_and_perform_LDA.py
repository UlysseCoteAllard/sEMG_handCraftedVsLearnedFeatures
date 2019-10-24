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


def load_learned_features(layer):
    layer_contents = pd.read_csv("./features/learned_features_layer_" + layer +".csv")
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


def perform_LDA(test_features, training_features, test_labels, training_labels, layer):
    classes = ["Neutral", "Radial Deviation", "Wrist Flexion", "Ulnar Deviation", "Wrist Extension", "Supination",
               "Pronation", "Power Grip", "Open Hand", "Chuck Grip", "Pinch Grip"]

    lda = LinearDiscriminantAnalysis()
    lda.fit(training_features, training_labels)
    predictions = lda.predict(test_features)

    plot_confusion_matrix(test_labels, predictions, classes, normalize=True,
                          title="Layer_" + str(layer) + "_All_Features")


def perform_LDA_single_feature(test_features, training_features, test_labels, training_labels, layer):
    classes = ["Neutral", "Radial Deviation", "Wrist Flexion", "Ulnar Deviation", "Wrist Extension", "Supination",
               "Pronation", "Power Grip", "Open Hand", "Chuck Grip", "Pinch Grip"]
    '''
    print(np.shape(training_features))
    labels_one_against_all_array = range(11)
    for labels_one_against_all in labels_one_against_all_array:
        test_labels_one_against_all = []
        training_labels_one_against_all = []
        for label in test_labels:
            if label == labels_one_against_all:
                test_labels_one_against_all.append(0)
            else:
                test_labels_one_against_all.append(1)
        for label in training_labels:
            if label == labels_one_against_all:
                training_labels_one_against_all.append(0)
            else:
                training_labels_one_against_all.append(1)

        # Get all the index for class 0
        indexes_class_zero = np.argwhere(np.array(training_labels_one_against_all) == 0).flatten()
        number_of_class_one_to_get = len(indexes_class_zero)
        indexes_class_one = np.random.choice(np.argwhere(np.array(training_labels_one_against_all)).flatten(),
                                             number_of_class_one_to_get)
        indexes_all = []
        indexes_all.extend(np.array(indexes_class_zero).tolist())
        indexes_all.extend(np.array(indexes_class_one).tolist())
        labels_balanced_train = np.array(training_labels_one_against_all)[indexes_all]
        examples_balanced_train = np.array(training_features)[indexes_all]

        #for feature in range(int(test_features.shape[1]/10)):
        for feature in range(1):
            #examples_train = examples_balanced_train[:, feature * 10:(feature+1) * 10]
            examples_train = examples_balanced_train
            examples_train, labels_train = shuffle(examples_train, labels_balanced_train)
            #examples_test = test_features[:, feature * 10:(feature+1) * 10]
            examples_test = test_features
            examples_test, labels_test = shuffle(examples_test, test_labels_one_against_all)
            lda = LinearDiscriminantAnalysis()
            lda.fit(examples_train, labels_train)
            predictions = lda.predict(examples_test)

            plot_confusion_matrix(labels_test, predictions, [classes[labels_one_against_all], "Others"],
                                  normalize=True,
                                  title=classes[labels_one_against_all] + "_Layer_" + str(layer) + "_F_" + str(feature))
            if feature > 0:
                break
        print(training_labels_one_against_all)
    '''
    #accuracy = np.zeros(int(teF.shape[1]/10))
    for feature in range(int(test_features.shape[1] / 10)):
        examples_train = training_features[:, feature * 10:(feature + 1) * 10]
        examples_train_shuffled, labels_train_shuffled = shuffle(examples_train, training_labels)
        examples_test = test_features[:, feature * 10:(feature + 1) * 10]
        examples_test_shuffled, labels_test_shuffled = shuffle(examples_test, test_labels)
        clf = LinearDiscriminantAnalysis()
        predictions = np.array(clf.fit(examples_train_shuffled, labels_train_shuffled).predict(examples_test_shuffled))
        plot_confusion_matrix(labels_test_shuffled, predictions, classes,
                              normalize=True,
                              title="Layer_" + str(layer) + "_F_" + str(feature))

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

    plt.figure(figsize=(14, 12))  # Sample figsize in inches
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt, cbar=False, annot_kws={"size": 28})
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), fontsize=fontsize)
    heatmap.set(
        # ... and label them with the respective list entries
        title="Accuracy: " + str(accuracy_score(ground_truth, predictions)),
        ylabel='True label',
        xlabel='Predicted label')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=30, ha='right', fontsize=fontsize)
    heatmap.xaxis.label.set_size(fontsize + 4)
    heatmap.yaxis.label.set_size(fontsize + 4)
    heatmap.title.set_size(fontsize + 6)


    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.gcf().subplots_adjust(top=0.90)
    heatmap.figure.tight_layout()
    heatmap.figure.savefig("./confusion_matrix/" + title + ".svg", dpi=1200)
    heatmap.figure.clf()
    plt.clf()
    plt.close()


if __name__ == '__main__':
    sns.set()
    np.set_printoptions(precision=1)
    import os
    print(os.listdir("./"))
    test_labels = np.genfromtxt('./features/labels_for_learned_features_TEST.csv', delimiter=',')
    test_labels = np.array(test_labels[1:, 2]).astype(int)
    train_labels = np.genfromtxt('./features/labels_for_learned_features.csv', delimiter=',')
    train_labels = np.asarray(train_labels[1:, 2]).astype(int)

    for layer in range(6):

        test_features = load_learned_features(str(layer) + "_TEST_dataset")
        train_features = load_learned_features(str(layer))

        perform_LDA_single_feature(test_features, train_features, test_labels, train_labels, layer)
        perform_LDA(test_features, train_features, test_labels, train_labels, layer)
        #np.savetxt("learned_accuracy" + str(layer) + "conf.csv",accuracy,delimiter=',')

        #for cm in range(11): #cm:class mask

        #    test_labels_mask = test_labels == cm
        #    test_labels_mask = test_labels_mask.astype(int)
        #    train_labels_mask = train_labels == cm
        #    train_labels_mask = train_labels_mask.astype(int)
        #    accuracy = perform_LDA(test_features,train_features,test_labels_mask, train_labels_mask)
        #    np.savetxt("learned_accuracy" + str(layer) + "_" + str(cm) + '.csv',
        #               accuracy,
        #               delimiter=',')


