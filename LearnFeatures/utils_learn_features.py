import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix as confusion_matrix_function

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def print_barplots(samples_d1, samples_d2, name_samples_d1, name_samples_d2, fontsize=48):
    sns.set_context("paper")
    sns.set_style("whitegrid")
    #sns.set_style("whitegrid", {'axes.grid' : False})
    training_method = []
    participants_index = []
    accuracies = []
    for i in range(len(samples_d2)):
        accuracies.append(samples_d2[i])
        training_method.append(name_samples_d2)
        participants_index.append(i)
    for i in range(len(samples_d1)):
        accuracies.append(samples_d1[i])
        training_method.append(name_samples_d1)
        participants_index.append(i)
    df = pd.DataFrame({"Training Method": training_method, "Accuracy": accuracies,
                       "Participant": participants_index})

    sns.set(font_scale=3.5)
    with sns.axes_style("whitegrid"):
        sns.catplot(x="Participant", y="Accuracy", hue="Training Method", ci="sd", data=df, legend_out=True, kind="bar")
    plt.title("Accuracy per participant in respect to the training method")
    plt.show()


def print_box_plots(samples_d1, samples_d2, name_samples_d1, name_samples_d2, fontsize=48):
    sns.set(style="whitegrid")
    training_method = []
    participants_index = []
    accuracies = []
    for i in range(len(samples_d2)):
        accuracies.append(samples_d2[i])
        training_method.append(name_samples_d2)
        participants_index.append(i)
    for i in range(len(samples_d1)):
        accuracies.append(samples_d1[i])
        training_method.append(name_samples_d1)
        participants_index.append(i)
    df = pd.DataFrame({"Training Method": training_method, "Accuracy": accuracies,
                       "Participant": participants_index})
    boxplot = sns.boxplot(x="Training Method", y="Accuracy", data=df, linewidth=4.)
    boxplot.xaxis.label.set_size(fontsize + 4)
    boxplot.yaxis.label.set_size(fontsize + 4)
    boxplot.title.set_size(fontsize + 6)
    boxplot
    #sns.catplot(x="Training Method", y="Accuracy", data=df, ax=ax, color=".25")


def print_confusion_matrix(ground_truth, predictions, class_names, fontsize=24,
                           normalize=True, fig=None, axs=None, title=None):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """

    print(np.shape(predictions[0]))
    # Calculate the confusion matrix across all participants
    predictions = [x for y in predictions for x in y]
    ground_truth = [x for y in ground_truth for x in y]
    #predictions = [x for y in predictions for x in y]
    #ground_truth = [x for y in ground_truth for x in y]
    print(np.shape(ground_truth))
    confusion_matrix_calculated = confusion_matrix_function(np.ravel(np.array(ground_truth)),
                                                            np.ravel(np.array(predictions)))

    if normalize:
        confusion_matrix_calculated = confusion_matrix_calculated.astype('float') /\
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

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt, cbar=False, annot_kws={"size": fontsize}, cmap=new_cmap)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.set(
        # ... and label them with the respective list entries
        title=title,
        ylabel='True label',
        xlabel='Predicted label')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=30, ha='right', fontsize=fontsize)
    heatmap.xaxis.label.set_size(fontsize + 4)
    heatmap.yaxis.label.set_size(fontsize + 4)
    heatmap.title.set_size(fontsize + 6)
    return fig, axs


def get_p_value(samples_d1, samples_d2):
    return wilcoxon(samples_d1, samples_d2)


def get_cohen_D(samples_d1, samples_d2):
    # Pooled standard deviation
    difference = np.array(samples_d1) - np.array(samples_d2)
    std_difference = np.std(difference)
    cohen_d = (np.mean(samples_d1)-np.mean(samples_d2))/std_difference
    return cohen_d

if __name__ == '__main__':
    accuracy_participants_ADANN = [0.873046875, 0.748779296875, 0.939208984375, 0.825927734375, 0.93505859375,
                                   0.759033203125, 0.85205078125, 0.806396484375, 0.76416015625, 0.832763671875,
                                   0.8583984375, 0.8404947916666666, 0.86083984375, 0.8642578125, 0.8974609375,
                                   0.83203125, 0.85400390625, 0.8540736607142857, 0.8509114583333334,
                                   0.8289930555555556, 0.8231336805555556, 0.8752170138888888]
    accuracy_participants_normal_training = [0.62060546875, 0.569580078125, 0.712646484375, 0.698974609375,
                                             0.697998046875, 0.717529296875, 0.58984375, 0.64794921875, 0.56103515625,
                                             0.658203125, 0.640380859375, 0.7306857638888888, 0.703369140625,
                                             0.640380859375, 0.778076171875, 0.552978515625, 0.553466796875,
                                             0.41573660714285715, 0.7547743055555556, 0.7259114583333334,
                                             0.6692708333333334, 0.6681857638888888]

    print(get_p_value(accuracy_participants_ADANN, accuracy_participants_normal_training))
    print(get_cohen_D(accuracy_participants_ADANN, accuracy_participants_normal_training))

    print_barplots(accuracy_participants_ADANN, accuracy_participants_normal_training, "ADANN", "Standard", fontsize=48)

    #print_box_plots(accuracy_participants_ADANN, accuracy_participants_normal_training, "ADANN", "Standard")
