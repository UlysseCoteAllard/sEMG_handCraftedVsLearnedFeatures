import numpy as np

import feature_extraction as fe
from PrepareAndLoadData import load_prepared_dataset_in_dataframe

def extract_fetures_from_dataset():
    train_dataset, test_dataset = load_prepared_dataset_in_dataframe.load_dataframe()

    num_subj = 22
    num_window = 4000

    train_features = [[]]
    test_features = [[]]

    train_class = []
    test_class = []

    window = 0

    for s in range(len(train_dataset)):
        subj_data = train_dataset[s]
        print("Current subject: ", s)
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

    np.save("../Dataset/processed_dataset/FEATURES_train", train_features)
    np.save("../Dataset/processed_dataset/CLASS_train", train_class)
    '''
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
    np.save("../Dataset/processed_dataset/FEATURES_test", test_features)
    np.save("../Dataset/processed_dataset/CLASS_test", test_class)

    # Save test features_data
    # Save test class
    '''

if __name__ == "__main__":
    extract_fetures_from_dataset()
