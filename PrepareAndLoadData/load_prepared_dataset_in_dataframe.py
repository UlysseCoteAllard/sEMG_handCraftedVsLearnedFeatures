import numpy as np
import pandas as pd


def get_dataframe(examples_datasets_train, labels_datasets_train, number_of_cycle):
    participants_dataframes = []
    for participant_examples, participant_labels in zip(examples_datasets_train, labels_datasets_train):
        X = []
        Y = []

        for cycle in range(number_of_cycle):
            X.extend(participant_examples[cycle])
            Y.extend(participant_labels[cycle])
        data = {'examples': X,
                'labels': Y}
        df = pd.DataFrame(data)
        participants_dataframes.append(df)
    return participants_dataframes


def load_dataframe(number_of_cycle=4):
    datasets_train = np.load("../Dataset/processed_dataset/RAW_3DC_train.npy")
    examples_datasets_train, labels_datasets_train = datasets_train

    participants_dataframes_train = get_dataframe(examples_datasets_train, labels_datasets_train, number_of_cycle)

    datasets_test = np.load("../Dataset/processed_dataset/RAW_3DC_test.npy")
    examples_datasets_test, labels_datasets_test = datasets_test

    participants_dataframes_test = get_dataframe(examples_datasets_test, labels_datasets_test, number_of_cycle)

    return participants_dataframes_train, participants_dataframes_test


if __name__ == "__main__":
    train_dataset, test_dataset = load_dataframe()

