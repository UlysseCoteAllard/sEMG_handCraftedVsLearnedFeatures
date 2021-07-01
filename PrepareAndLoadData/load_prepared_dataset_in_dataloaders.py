import numpy as np
from sklearn.decomposition import PCA

import torch
from torch.utils.data import TensorDataset


def scramble(examples, labels):
    random_vec = np.arange(len(labels))
    np.random.shuffle(random_vec)
    new_labels = []
    new_examples = []

    for i in random_vec:
        new_labels.append(labels[i])
        new_examples.append(examples[i])
    return new_examples, new_labels


def get_dataloader_for_regression(examples_datasets, feature_calculator, number_of_cycle,
                                  validations_cycles=None, batch_size=128, drop_last=True, shuffle=True,
                                  electrode_to_use=0, pca=None):
    participants_dataloaders = []
    participants_dataloaders_validation = []

    for participant_examples in examples_datasets:
        X, Y, X_valid, Y_valid = [], [], [], []

        for cycle in range(number_of_cycle):
            for example in participant_examples[cycle]:
                if validations_cycles is None:
                    X.append(example)
                    feature = feature_calculator(example[electrode_to_use])
                    Y.append(feature)
                else:
                    if cycle < validations_cycles:
                        X.append(example)
                        feature = feature_calculator(example[electrode_to_use])
                        Y.append(feature)
                    else:
                        X_valid.append(example)
                        feature = feature_calculator(example[electrode_to_use])
                        Y_valid.append(feature)

        if hasattr(Y[0], "__len__"):
            if pca is None:
                pca = PCA(n_components=1)
                Y = np.squeeze(pca.fit_transform(Y))
            else:
                Y = np.squeeze(pca.transform(Y))

        X = np.expand_dims(X, axis=1)
        train = TensorDataset(torch.from_numpy(np.array(X, dtype=np.float32)),
                              torch.from_numpy(np.array(Y, dtype=np.float32)))
        examplesloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        participants_dataloaders.append(examplesloader)

        if validations_cycles is not None:
            if pca is not None:
                Y_valid = np.squeeze(pca.transform(Y_valid))
            X_valid = np.expand_dims(X_valid, axis=1)
            validation = TensorDataset(torch.from_numpy(np.array(X_valid, dtype=np.float32)),
                                       torch.from_numpy(np.array(Y_valid, dtype=np.float32)))
            validationloader = torch.utils.data.DataLoader(validation, batch_size=len(X_valid), shuffle=shuffle,
                                                           drop_last=drop_last)
            participants_dataloaders_validation.append(validationloader)

    if validations_cycles is None:
        return participants_dataloaders, pca
    else:
        return participants_dataloaders, participants_dataloaders_validation, pca


def load_dataloaders_for_regression(path, feature, number_of_cycle=4, batch_size=128, validation_cycle=3,
                                    get_test_set=True, drop_last=True, shuffle=True, electrode_to_use=0):
    participants_dataloaders_test = []

    'Get training dataset'
    datasets_train = np.load(path + "/RAW_3DC_train.npy")
    examples_datasets_train, _ = datasets_train
    if validation_cycle is None:
        participants_dataloaders_train, pca = get_dataloader_for_regression(examples_datasets_train,
                                                                           feature_calculator=feature,
                                                                           number_of_cycle=number_of_cycle,
                                                                           validations_cycles=validation_cycle,
                                                                           batch_size=batch_size, drop_last=drop_last,
                                                                           shuffle=shuffle,
                                                                           electrode_to_use=electrode_to_use)
        if get_test_set:
            'Get testing dataset'
            if get_test_set:
                datasets_test = np.load(path + "/RAW_3DC_test.npy")
                examples_datasets_test, _ = datasets_test

                participants_dataloaders_test, _ = get_dataloader_for_regression(examples_datasets_test,
                                                                                 feature_calculator=feature,
                                                                                 number_of_cycle=number_of_cycle,
                                                                                 validations_cycles=None,
                                                                                 batch_size=batch_size,
                                                                                 drop_last=drop_last, shuffle=False,
                                                                                 electrode_to_use=electrode_to_use,
                                                                                 pca=pca)

            return participants_dataloaders_train, participants_dataloaders_test
        else:
            return participants_dataloaders_train
    else:
        participants_dataloaders_train, participants_dataloaders_validation, pca = get_dataloader_for_regression(
            examples_datasets_train, feature_calculator=feature, number_of_cycle=number_of_cycle,
            validations_cycles=validation_cycle, batch_size=batch_size, drop_last=drop_last, shuffle=False,
            electrode_to_use=electrode_to_use)
        if get_test_set:
            'Get testing dataset'
            if get_test_set:
                datasets_test = np.load(path + "/RAW_3DC_test.npy")
                examples_datasets_test, _ = datasets_test

                participants_dataloaders_test, _ = get_dataloader_for_regression(examples_datasets_test,
                                                                              feature_calculator=feature,
                                                                              number_of_cycle=number_of_cycle,
                                                                              validations_cycles=None,
                                                                              batch_size=batch_size,
                                                                              drop_last=drop_last, shuffle=False,
                                                                              electrode_to_use=electrode_to_use,
                                                                              pca=pca)

            return participants_dataloaders_train, participants_dataloaders_validation, participants_dataloaders_test
        else:
            return participants_dataloaders_train, participants_dataloaders_validation


def get_dataloader(examples_datasets, labels_datasets, cycle_used, batch_size=128,
                   drop_last=True, shuffle=True, aggre=False):
    # cycle_used is a list for specifying which cycles are used
    # aggre is used to specify if data is saved by participants (=False) or not (=True)
    dataloaders = []
    
    if aggre:
        X, Y = [], []
        for participant_examples, participant_labels in zip(examples_datasets, labels_datasets):
            for cycle in cycle_used:
                X.extend(participant_examples[cycle])
                Y.extend(participant_labels[cycle])
        X = np.expand_dims(X, axis=1)  # For data aggregation
        data = TensorDataset(torch.from_numpy(np.array(X, dtype=np.float32)),
                              torch.from_numpy(np.array(Y, dtype=np.int64)))
        dataloaders = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    else:
        for participant_examples, participant_labels in zip(examples_datasets, labels_datasets):
            X, Y = [], []
            for cycle in cycle_used:
                X.extend(participant_examples[cycle])
                Y.extend(participant_labels[cycle])
            X = np.expand_dims(X, axis=1)  # Save each dataset by participant index
            data = TensorDataset(torch.from_numpy(np.array(X, dtype=np.float32)),
                              torch.from_numpy(np.array(Y, dtype=np.int64)))
            examplesloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
            dataloaders.append(examplesloader)
    return dataloaders


def load_dataloaders(path, number_of_cycle=4, batch_size=128, valid_cycle_num=1, get_test_set=True, drop_last=True, aggre_required=False):
    # valid_cycle_num: to specify how many cylcles are needed for validation
    
    #  These comments can be delected later if the code modification is approved 
    #  aggre_requires = False is the same as the original function 'loader_dataloaders'
    #  aggre_requires = False is the same as the original function 'loader_dataloaders_for_training_without_TL'
    #  valid_cycle_num = 1 is the same as the original variable 'validation_cycle'=3 here
    #  The variable shuffle can be removed since it is usually set to true for training and validation, while false for testing 
    participants_dataloaders_test = []
    if get_test_set:
        'Get testing dataset'
        datasets_test = np.load(path + "/RAW_3DC_test.npy")
        examples_datasets_test, labels_datasets_test = datasets_test
        test_cycles = list(range(number_of_cycle))
        participants_dataloaders_test = get_dataloader(examples_datasets_test, labels_datasets_test, test_cycles,
                                                       batch_size=batch_size, drop_last=drop_last, shuffle=False,
                                                       aggre=aggre_required)
    'Get training dataset'
    datasets_train = np.load(path + "/RAW_3DC_train.npy")
    examples_datasets_train, labels_datasets_train = datasets_train
    train_cycles = list(range(number_of_cycle))
    valid_cycles = []
    for _ in range(validation_cycle):
        valid_cycles.append(train_cycles.pop())
    
    #  for example, if number_of_cycle=4 and valid_cycle_num=1,
    #  train_cycles = [0,1,2] and valid_cycles = [3] here
    
    participants_dataloaders_train = get_dataloader(examples_datasets_train, labels_datasets_train, train_cycles,
                                                    batch_size=batch_size, drop_last=drop_last, shuffle=True,
                                                    aggre=aggre_required)
    participants_dataloaders_valid = []
    if valid_cycle_num !=0:
        'Get validation dataset'
        participants_dataloaders_valid = get_dataloader(examples_datasets_train, labels_datasets_train,
                                                        valid_cycles, batch_size=batch_size, 
                                                        drop_last=drop_last, shuffle=True,
                                                        aggre=aggre_required)
    
    return participants_dataloaders_train, participants_dataloaders_validation, participants_dataloaders_test


if __name__ == "__main__":
    train_dataset, validation_dataset, test_dataset = load_dataloaders("../Dataset/processed_dataset")
