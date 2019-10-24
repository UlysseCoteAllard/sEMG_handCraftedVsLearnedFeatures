import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix as confusion_matrix_function

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

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt, cbar=False, annot_kws={"size": 28})
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
