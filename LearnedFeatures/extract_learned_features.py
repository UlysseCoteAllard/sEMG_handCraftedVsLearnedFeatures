import numpy as np
import pandas as pd

import torch

from Models.rawConvNet import Model
from PrepareAndLoadData.load_prepared_dataset_in_dataloaders import load_dataloaders

def extract_learned_features_average(path_dataset='../Dataset/processed_dataset',
                                     path_weights='../weights/TL_best_weights.pt',
                                     path_bn_statistics="../weights/bn_statistics.pt"):
    participants_dataloaders_train = load_dataloaders(path_dataset, batch_size=512, validation_cycle=None,
                                                      get_test_set=False, drop_last=False, shuffle=False)
    # Define Model
    model = Model(number_of_class=11, number_of_blocks=6, dropout_rate=0.35).cuda()
    best_weights = torch.load(path_weights)
    model.load_state_dict(best_weights)
    list_dictionaries_bn_weights = torch.load(path_bn_statistics)

    'Generate the features from the training dataset and'
    'Format the features in a dictionary in the shape: Layers x Participant x Examples. with Layers being the keys of'
    'the dictionary'

    print(model.get_number_of_blocks())
    features_learned_per_layers = {}
    for i in range(model.get_number_of_blocks()):
        features_learned_per_layers['layer_' + str(i)] = []

    with torch.no_grad():
        for dataset_index, dataset in enumerate(participants_dataloaders_train):
            BN_weights = list_dictionaries_bn_weights[dataset_index]
            model.load_state_dict(BN_weights, strict=False)
            model.eval()
            features_participant = {}
            for i in range(model.get_number_of_blocks()):
                features_participant['layer_' + str(i)] = []
            for i, data in enumerate(dataset):
                inputs, _ = data
                inputs = inputs.cuda()
                _, features_batch = model(inputs)
                for key in features_batch:
                    features_participant[key].extend(features_batch[key])
            for key in features_participant:
                features_learned_per_layers[key].append(features_participant[key])

    for key_layer in features_learned_per_layers:
        for participant in features_learned_per_layers[key_layer]:
                print(np.shape(participant[0]))

    'Make number_of_blocks_dataframe (i.e. number of layers). The dataframe will be of the shape:'
    ' Participant x Examples x Mean Output Feature Maps'
    dataframes_grouped_by_layers = []
    for key_layer in features_learned_per_layers:
        participant_label = []
        feature_maps_mean = []
        participant_index = 0
        for participant in features_learned_per_layers[key_layer]:
            participant_label.extend(np.ones(len(participant))*participant_index)
            feature_maps_mean.extend(participant)
            participant_index += 1
        data = {'Participant': participant_label,
                'LearnedFeatures': feature_maps_mean}
        df = pd.DataFrame(data)
        dataframes_grouped_by_layers.append(df)
        print(df)

    'Save the list of dataframes'
    for i, dataframe in enumerate(dataframes_grouped_by_layers):
        dataframe.to_csv(path_or_buf="../LearnedFeatures/learned_features_layer_" + str(i) + ".csv")


if __name__ == "__main__":
    extract_learned_features_average(path_dataset='../Dataset/processed_dataset',
                                     path_weights='../weights/TL_best_weights.pt',
                                     path_bn_statistics="../weights/bn_statistics.pt")
