import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from Models.rawConvNet import Model
from Models.model_training import train_regressor
from PrepareAndLoadData.load_prepared_dataset_in_dataloaders import load_dataloaders_for_regression
from PrepareAndLoadData import feature_extraction


def get_regressor_and_test_it(feature, path_dataset='../Dataset/processed_dataset',
                              path_weights='../weights/TL_best_weights.pt',
                              path_bn_statistics="../weights/bn_statistics.pt",
                              number_of_repetitions=1):
    participants_train, participants_validation, participants_test = load_dataloaders_for_regression(path_dataset,
                                                                                                     feature=feature,
                                                                                                     batch_size=512,
                                                                                                     validation_cycle=3,
                                                                                                     get_test_set=True,
                                                                                                     drop_last=True,
                                                                                                     shuffle=True)

    mean_error_layer = []
    for number_of_block in range(6):
        mean_error_participant = []
        for i in range(len(participants_train)):
            mean_error_repetition = []
            for _ in range(number_of_repetitions):
                # Define Model
                model = Model(number_of_class=11, number_of_blocks=6, dropout_rate=0.35)
                best_weights = torch.load(path_weights)
                model.load_state_dict(best_weights)
                list_dictionaries_bn_weights = torch.load(path_bn_statistics)
                BN_weights = list_dictionaries_bn_weights[i]
                model.load_state_dict(BN_weights, strict=False)

                model.transform_to_regressor(layer_to_regress_from=number_of_block, freeze_features_extraction=True,
                                             channel_to_use=0)
                model.cuda()

                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.0404709)

                precision = 1e-8
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                                 verbose=True, eps=precision)

                model = train_regressor(model, criterion, optimizer, scheduler, dataloaders={
                    "train": participants_train[i], "val": participants_validation[i]}, precision=precision)

                average_mean = 0
                loss = nn.MSELoss()
                total = 0
                with torch.no_grad():
                    model.eval()
                    for _, data in enumerate(participants_validation[i], 0):
                        inputs, labels = data
                        inputs, labels = inputs.cuda(), labels.cuda()
                        output = model(inputs, use_regressor_forward=True).view(-1)
                        average_mean += loss(output, labels).item()
                        total += labels.size(0)
                    print("AVERAGE MEAN : ", average_mean/total)
                    mean_error_repetition.append(average_mean / total)
            mean_error_participant.append(np.mean(mean_error_repetition))
        print("BLOCK INDEX: ", number_of_block, "  MEAN ERROR PARTICIPANT : ", mean_error_participant)
        mean_error_layer.append(mean_error_participant)
    #print(mean_error_layer)
    mean_error_layer = np.array(mean_error_layer)
    np.save("ResultsRegression/data_for_feature_" + feature.__name__ + ".npy", mean_error_layer)


def generate_regression_graph(data, feature, colors):
    font = {'family': 'normal',
            'size': 32}

    matplotlib.rc('font', **font)
    sns.set(font_scale=2)
    sns.set_palette(sns.color_palette([colors]))
    sns.set_style("whitegrid")


    print(data)
    print(np.shape(data))
    layers = []
    participants = []
    mseloss = []
    for i, layer in enumerate(data):
        for j, participants_mse in enumerate(layer):
            mseloss.append(participants_mse)
            layers.append(i + 1)
            participants.append(j)

    df = pd.DataFrame({"Block": layers,
                       "Participants": participants,
                       "Mean Square Error": mseloss})
    print(df)

    sns.catplot("Block", "Mean Square Error", data=df, kind="point")
    plt.title(feature.__name__[3:])
    plt.tight_layout()
    plt.rcParams['figure.facecolor'] = 'white'
    plt.savefig("ResultsRegression/" + feature.__name__ + ".svg", dpi=1200)


if __name__ == "__main__":

    features = [feature_extraction.getMAV, feature_extraction.getRMS, feature_extraction.getSTD,
                feature_extraction.getMDF, feature_extraction.getZC, feature_extraction.getSSC,
                feature_extraction.getAR, feature_extraction.getCC, feature_extraction.getDAR,
                feature_extraction.getMFL, feature_extraction.getSampEn, feature_extraction.getBC,
                feature_extraction.getSKEW, feature_extraction.getKURT, feature_extraction.getHIST]

    
    for feature in features:
        get_regressor_and_test_it(feature=feature, path_dataset='../Dataset/processed_dataset',
                                  path_weights='../weights/TL_best_weights.pt',
                                  path_bn_statistics="../weights/bn_statistics.pt",
                                  number_of_repetitions=20)

    colors = ["#B85450", "#82B366", "#6C8EBF", "#647687"]
    colors = np.repeat(colors, 4)
    print(colors)
    for i, feature in enumerate(features):
        data = np.load("ResultsRegression/data_for_feature_" + feature.__name__ + ".npy")
        generate_regression_graph(data, feature, colors[i])
