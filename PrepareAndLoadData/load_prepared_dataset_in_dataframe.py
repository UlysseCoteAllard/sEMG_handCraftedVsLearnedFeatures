import numpy as np
import pandas as pd

import feature_extraction as fe

def get_dataframe(examples_datasets, labels_datasets, number_of_cycle):
    participants_dataframes = []
    for participant_examples, participant_labels in zip(examples_datasets, labels_datasets):
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
    datasets_train = np.load("../Dataset/processed_dataset/RAW_3DC_train.npy",allow_pickle=True)
    examples_datasets_train, labels_datasets_train = datasets_train

    participants_dataframes_train = get_dataframe(examples_datasets_train, labels_datasets_train, number_of_cycle)

    datasets_test = np.load("../Dataset/processed_dataset/RAW_3DC_test.npy",allow_pickle=True)
    examples_datasets_test, labels_datasets_test = datasets_test

    participants_dataframes_test = get_dataframe(examples_datasets_test, labels_datasets_test, number_of_cycle)

    return participants_dataframes_train, participants_dataframes_test


if __name__ == "__main__":
    train_dataset, test_dataset = load_dataframe()

    num_subj = 22
    num_window = 4000

    train_features = [[]]
    test_features = [[]]

    train_class = []
    test_class = []

    window = 0

    for s in range(len(train_dataset)):
        subj_data = train_dataset[s]

        for w in range(len(subj_data)):
            win_data = subj_data.values[w]
            # Collect the feature vector for window #
            train_features[window].append(fe.extract_features(win_data[0]))
            # Add list to list of list (make slot for next feature vector)
            train_features.append(list())
            train_class.append(win_data[1])
            window += 1

    train_features = np.array(train_features)
    train_class = np.array(train_class)

    np.save("../Dataset/processed_dataset/FEATURES_train",train_features)
    np.save("../Dataset/processed_dataset/CLASS_train",train_class)


    for s in range(len(test_dataset)):
        subj_data = test_dataset[s]

        for w in range(len(subj_data)):
            win_data = subj_data.values[w]
            # Collect the feature vector for window #
            test_features[window].append(fe.extract_features(win_data[0]))
            # Add list to list of list (make slot for next feature vector)
            test_features.append(list())
            test_class.append(win_data[1])
            window += 1

    test_features = np.array(test_features)
    test_class = np.array(test_class)
    np.save("../Dataset/processed_dataset/FEATURES_test",test_features)
    np.save("../Dataset/processed_dataset/CLASS_test",test_class)

    # Save test features_data
    # Save test class


