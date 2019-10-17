import numpy as np

import torch
from torch.utils.data import TensorDataset

import numpy as np

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
                                  electrode_to_use=0):
    participants_dataloaders = []
    participants_dataloaders_validation = []

    for participant_examples in examples_datasets:
        X, Y, X_valid, Y_valid = [], [], [], []

        for cycle in range(number_of_cycle):
            for example in participant_examples[cycle]:
                if validations_cycles is None:
                    X.append(example)
                    Y.append(feature_calculator(example[electrode_to_use]))
                else:
                    if cycle < validations_cycles:
                        X.append(example)
                        Y.append(feature_calculator(example[electrode_to_use]))
                    else:
                        X_valid.append(example)
                        Y_valid.append(feature_calculator(example[electrode_to_use]))

        X = np.expand_dims(X, axis=1)
        train = TensorDataset(torch.from_numpy(np.array(X, dtype=np.float32)),
                              torch.from_numpy(np.array(Y, dtype=np.float32)))
        examplesloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        participants_dataloaders.append(examplesloader)

        if validations_cycles is not None:
            X_valid = np.expand_dims(X_valid, axis=1)
            validation = TensorDataset(torch.from_numpy(np.array(X_valid, dtype=np.float32)),
                                       torch.from_numpy(np.array(Y_valid, dtype=np.float32)))
            validationloader = torch.utils.data.DataLoader(validation, batch_size=len(X_valid), shuffle=shuffle,
                                                           drop_last=drop_last)
            participants_dataloaders_validation.append(validationloader)

    if validations_cycles is None:
        return participants_dataloaders
    else:
        return participants_dataloaders, participants_dataloaders_validation


def load_dataloaders_for_regression(path, feature, number_of_cycle=4, batch_size=128, validation_cycle=3,
                                    get_test_set=True, drop_last=True, shuffle=True, electrode_to_use=0):
    participants_dataloaders_test = []
    'Get testing dataset'
    if get_test_set:
        datasets_test = np.load(path + "/RAW_3DC_test.npy")
        examples_datasets_test, _ = datasets_test

        participants_dataloaders_test = get_dataloader_for_regression(examples_datasets_test,
                                                                      feature_calculator=feature,
                                                                      number_of_cycle=number_of_cycle,
                                                                      validations_cycles=None,
                                                                      batch_size=batch_size,
                                                                      drop_last=drop_last, shuffle=False,
                                                                      electrode_to_use=electrode_to_use)
    'Get training dataset'
    datasets_train = np.load(path + "/RAW_3DC_train.npy")
    examples_datasets_train, _ = datasets_train
    if validation_cycle is None:
        participants_dataloaders_train = get_dataloader_for_regression(examples_datasets_train,
                                                                       feature_calculator=feature,
                                                                       number_of_cycle=number_of_cycle,
                                                                       validations_cycles=validation_cycle,
                                                                       batch_size=batch_size, drop_last=drop_last,
                                                                       shuffle=shuffle,
                                                                       electrode_to_use=electrode_to_use)
        if get_test_set:
            return participants_dataloaders_train, participants_dataloaders_test
        else:
            return participants_dataloaders_train
    else:
        participants_dataloaders_train, participants_dataloaders_validation = get_dataloader_for_regression(
            examples_datasets_train, feature_calculator=feature, number_of_cycle=number_of_cycle,
            validations_cycles=validation_cycle, batch_size=batch_size, drop_last=drop_last, shuffle=False,
            electrode_to_use=electrode_to_use)
        if get_test_set:
            return participants_dataloaders_train, participants_dataloaders_validation, participants_dataloaders_test
        else:
            return participants_dataloaders_train, participants_dataloaders_validation

def get_dataloader_for_training_no_TL(examples_datasets, labels_datasets, number_of_cycle, validations_cycles=None,
                                      batch_size=128, drop_last=True, shuffle=True):
    X, Y, X_valid, Y_valid = [], [], [], []
    for participant_examples, participant_labels in zip(examples_datasets, labels_datasets):
        for cycle in range(number_of_cycle):
            if validations_cycles is None:
                X.extend(participant_examples[cycle])
                Y.extend(participant_labels[cycle])
            else:
                if cycle < validations_cycles:
                    X.extend(participant_examples[cycle])
                    Y.extend(participant_labels[cycle])
                else:
                    X_valid.extend(participant_examples[cycle])
                    Y_valid.extend(participant_labels[cycle])
    X = np.expand_dims(X, axis=1)
    train = TensorDataset(torch.from_numpy(np.array(X, dtype=np.float32)),
                          torch.from_numpy(np.array(Y, dtype=np.int64)))
    examplesloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    if validations_cycles is not None:
        X_valid = np.expand_dims(X_valid, axis=1)
        validation = TensorDataset(torch.from_numpy(np.array(X_valid, dtype=np.float32)),
                                   torch.from_numpy(np.array(Y_valid, dtype=np.int64)))
        validationloader = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=shuffle,
                                                       drop_last=drop_last)
        return examplesloader, validationloader
    else:
        return examplesloader


def get_dataloader(examples_datasets, labels_datasets, number_of_cycle, validations_cycles=None, batch_size=128,
                   drop_last=True, shuffle=True):
    participants_dataloaders = []
    participants_dataloaders_validation = []

    for participant_examples, participant_labels in zip(examples_datasets, labels_datasets):
        X, Y, X_valid, Y_valid = [], [], [], []

        for cycle in range(number_of_cycle):
            if validations_cycles is None:
                X.extend(participant_examples[cycle])
                Y.extend(participant_labels[cycle])
            else:
                if cycle < validations_cycles:
                    X.extend(participant_examples[cycle])
                    Y.extend(participant_labels[cycle])
                else:
                    X_valid.extend(participant_examples[cycle])
                    Y_valid.extend(participant_labels[cycle])
        X = np.expand_dims(X, axis=1)
        train = TensorDataset(torch.from_numpy(np.array(X, dtype=np.float32)),
                              torch.from_numpy(np.array(Y, dtype=np.int64)))
        examplesloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        participants_dataloaders.append(examplesloader)

        if validations_cycles is not None:
            X_valid = np.expand_dims(X_valid, axis=1)
            validation = TensorDataset(torch.from_numpy(np.array(X_valid, dtype=np.float32)),
                                       torch.from_numpy(np.array(Y_valid, dtype=np.int64)))
            validationloader = torch.utils.data.DataLoader(validation, batch_size=len(X_valid), shuffle=shuffle,
                                                           drop_last=drop_last)
            participants_dataloaders_validation.append(validationloader)

    if validations_cycles is None:
        return participants_dataloaders
    else:
        return participants_dataloaders, participants_dataloaders_validation

def load_dataloader_for_training_without_TL(path, number_of_cycle=4, batch_size=128, validation_cycle=3, drop_last=True,
                                            shuffle=True):

    datasets_train = np.load(path + "/RAW_3DC_train.npy")
    'Get training dataset'
    examples_datasets_train, labels_datasets_train = datasets_train
    if validation_cycle is None:
        participants_dataloaders_train = get_dataloader_for_training_no_TL(examples_datasets_train,
                                                                           labels_datasets_train, number_of_cycle,
                                                                           validations_cycles=validation_cycle,
                                                                           batch_size=batch_size, drop_last=drop_last,
                                                                           shuffle=shuffle)
        return participants_dataloaders_train
    else:
        participants_dataloaders_train, participants_dataloaders_validation = get_dataloader_for_training_no_TL(
            examples_datasets_train, labels_datasets_train, number_of_cycle, validations_cycles=validation_cycle,
            batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
        return participants_dataloaders_train, participants_dataloaders_validation

def load_dataloaders(path, number_of_cycle=4, batch_size=128, validation_cycle=3, get_test_set=True, drop_last=True,
                     shuffle=True):
    participants_dataloaders_test = []
    'Get testing dataset'
    if get_test_set:
        datasets_test = np.load(path + "/RAW_3DC_test.npy")
        examples_datasets_test, labels_datasets_test = datasets_test

        participants_dataloaders_test = get_dataloader(examples_datasets_test, labels_datasets_test, number_of_cycle,
                                                       validations_cycles=None, batch_size=batch_size,
                                                       drop_last=drop_last, shuffle=False)
    'Get training dataset'
    datasets_train = np.load(path + "/RAW_3DC_train.npy")
    examples_datasets_train, labels_datasets_train = datasets_train
    if validation_cycle is None:
        participants_dataloaders_train = get_dataloader(examples_datasets_train, labels_datasets_train, number_of_cycle,
                                                        validations_cycles=validation_cycle, batch_size=batch_size,
                                                        drop_last=drop_last, shuffle=shuffle)
        if get_test_set:
            return participants_dataloaders_train, participants_dataloaders_test
        else:
            return participants_dataloaders_train
    else:
        participants_dataloaders_train, participants_dataloaders_validation = get_dataloader(examples_datasets_train,
                                                                                             labels_datasets_train,
                                                                                             number_of_cycle,
                                                                                             validations_cycles=
                                                                                             validation_cycle,
                                                                                             batch_size=batch_size,
                                                                                             drop_last=drop_last,
                                                                                             shuffle=shuffle)
        if get_test_set:
            return participants_dataloaders_train, participants_dataloaders_validation, participants_dataloaders_test
        else:
            return participants_dataloaders_train, participants_dataloaders_validation


if __name__ == "__main__":
    train_dataset, validation_dataset, test_dataset = load_dataloaders("../Dataset/processed_dataset")
