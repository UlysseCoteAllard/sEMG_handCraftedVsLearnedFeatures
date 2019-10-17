import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from Models.rawConvNet import Model
from Models.model_training import train_model_no_TL
from LearnFeatures.utils_learn_features import print_confusion_matrix
from PrepareAndLoadData.load_prepared_dataset_in_dataloaders import load_dataloader_for_training_without_TL,\
    load_dataloaders


def pretrain_raw_convNet(path, filter_size=(1, 11)):
    participants_dataloaders_train, participants_dataloaders_validation = load_dataloader_for_training_without_TL(
        path, batch_size=512)

    # Define Model
    model = Model(number_of_class=11, number_of_blocks=6, dropout_rate=0.35, filter_size=filter_size).cuda()

    # Define Loss functions
    cross_entropy_loss_classes = nn.CrossEntropyLoss(reduction='mean').cuda()

    # Define Optimizer
    learning_rate = 0.0404709
    print(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Define Scheduler
    precision = 1e-8
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=15,
                                                     verbose=True, eps=precision)

    best_weights = train_model_no_TL(model=model, criterion=cross_entropy_loss_classes, optimizer=optimizer,
                                     scheduler=scheduler, dataloaders={"train": participants_dataloaders_train,
                                                                       "val": participants_dataloaders_validation},
                                     precision=precision)

    torch.save(best_weights, f="../weights/no_TL_best_weights.pt")


def test_network_raw_convNet(path_dataset='../Dataset/processed_dataset',
                             path_weights='../weights/no_TL_best_weights.pt',
                             filter_size=(1, 26)):
    with open("results/test_accuracy_filter_size_" + str(filter_size[1]) + ".txt", "a") as myfile:
        myfile.write("Test")
    _, _, participants_dataloaders_test = load_dataloaders(path_dataset, batch_size=512, get_test_set=True)

    model = Model(number_of_class=11, number_of_blocks=6, dropout_rate=.35, filter_size=filter_size).cuda()
    best_weights = torch.load(path_weights)
    model.load_state_dict(best_weights)

    predictions = []
    ground_truth = []
    accuracies = []
    for participant_index, dataset_test in enumerate(participants_dataloaders_test):
        predictions_participant = []
        ground_truth_participant = []
        with torch.no_grad():
            model.eval()
            for inputs, labels in dataset_test:
                inputs = inputs.cuda()
                output, _ = model(inputs)
                _, predicted = torch.max(output.data, 1)
                predictions_participant.extend(predicted.cpu().numpy())
                ground_truth_participant.extend(labels.numpy())
        print("Participant: ", participant_index, " Accuracy: ",
              np.mean(np.array(predictions_participant) == np.array(ground_truth_participant)))
        predictions.append(predictions_participant)
        ground_truth.append(ground_truth_participant)
        accuracies.append(np.mean(np.array(predictions_participant) == np.array(ground_truth_participant)))
    print("OVERALL ACCURACY: " + str(np.mean(accuracies)))

    with open("results/test_accuracy_no_TL_filter_size_" + str(filter_size[1]) + ".txt", "a") as myfile:
        myfile.write("Predictions: \n")
        myfile.write(str(predictions) + '\n')
        myfile.write("Ground Truth: \n")
        myfile.write(str(ground_truth) + '\n')
        myfile.write("ACCURACIES: \n")
        myfile.write(str(accuracies) + '\n')
        myfile.write("OVERALL ACCURACY: " + str(np.mean(accuracies)))

    return predictions, ground_truth

if __name__ == "__main__":
    filter_size = (1, 26)
    #pretrain_raw_convNet(path="../Dataset/processed_dataset", filter_size=filter_size)

    predictions, ground_truth = test_network_raw_convNet(filter_size=filter_size)

    classes = ["Neutral", "Radial Deviation", "Wrist Flexion", "Ulnar Deviation", "Wrist Extension", "Supination",
               "Pronation", "Power Grip", "Open Hand", "Chuck Grip", "Pinch Grip"]
    font_size = 24
    sns.set(style='dark')

    fig, axs = print_confusion_matrix(ground_truth=ground_truth, predictions=predictions,
                                      class_names=classes, title="ConvNet standard training", fontsize=font_size)

    # fig.suptitle("ConvNet using AdaDANN training", fontsize=28)
    mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')  # works fine on Windows!
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.13)
    plt.gcf().subplots_adjust(top=0.90)
    plt.show()

