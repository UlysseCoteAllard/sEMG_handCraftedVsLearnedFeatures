import os
import numpy as np

from PrepareAndLoadData.prepare_dataset_utils import butter_bandpass_filter

def format_examples(emg_examples, window_size=150, size_non_overlap=50):
    examples_formatted = []
    example = []
    for emg_vector in emg_examples:
        if len(example) == 0:
            example = emg_vector
        else:
            example = np.row_stack((example, emg_vector))

        if len(example) >= window_size:
            # The example is of the shape TIME x CHANNEL. Make it of the shape CHANNEL x TIME
            example = example.transpose()
            # Go over each channel and bandpass filter it between 20 and 495 Hz.
            example_filtered = []
            for channel in example:
                channel_filtered = butter_bandpass_filter(channel, lowcut=20, highcut=495, fs=1000, order=4)
                example_filtered.append(channel_filtered)
            # Add the filtered example to the list of examples to return and transpose the example array again to go
            # back to TIME x CHANNEL
            examples_formatted.append(example_filtered)
            example = example.transpose()
            # Remove part of the data of the example according to the size_non_overlap variable
            example = example[size_non_overlap:]

    return examples_formatted


def get_data_and_process_it_from_file(get_train_data, path, number_of_gestures=11, number_of_cycles=4, window_size=150,
                                      size_non_overlap=50):
    examples_datasets, labels_datasets = [], []
    train_or_test_str = "train" if get_train_data else "test"

    participants_directories = os.listdir(path)
    for participant_directory in participants_directories:
        print("Preparing data of: " + participant_directory)

        examples_participants, labels_participant = [], []

        for cycle in range(number_of_cycles):
            path_emg = path + "/" + participant_directory + "/" + "%s/EMG/3dc_EMG_gesture_%d_" % (train_or_test_str,
                                                                                                  cycle)
            examples, labels = [], []
            for gesture_index in range(number_of_gestures):
                examples_to_format = []
                for line in open(path_emg + '%d.txt' % gesture_index):
                    # strip() remove the "\n" character, split separate the data in a list. np.float_ transform
                    # each element of the list from a str to a float
                    emg_signal = np.float32(line.strip().split(","))
                    examples_to_format.append(emg_signal)
                examples_formatted = format_examples(examples_to_format, window_size=window_size,
                                                     size_non_overlap=size_non_overlap)
                examples.extend(examples_formatted)
                labels.extend(np.ones(len(examples_formatted)) * gesture_index)

            examples_participants.append(examples)
            labels_participant.append(examples)

        examples_datasets.append(examples_participants)
        labels_datasets.append(labels_participant)

    return examples_datasets, labels_datasets

def read_data(path, number_of_gestures=11, number_of_cycles=4, window_size=150, size_non_overlap=50):
    print("Loading and preparing datasets...")
    'Get and process the train data'
    print("Taking care of the training data...")
    list_dataset_train_emg, list_labels_train_emg = get_data_and_process_it_from_file(get_train_data=True, path=path,
                                                                                      number_of_gestures=
                                                                                      number_of_gestures,
                                                                                      number_of_cycles=number_of_cycles,
                                                                                      window_size=window_size,
                                                                                      size_non_overlap=size_non_overlap)
    np.save("../Dataset/processed_dataset/RAW_3DC_train", (list_dataset_train_emg, list_labels_train_emg))
    print("Finished with the training data...")
    'Get and process the test data'
    print("Starting with the test data...")
    list_dataset_train_emg, list_labels_train_emg = get_data_and_process_it_from_file(get_train_data=False, path=path,
                                                                                      number_of_gestures=
                                                                                      number_of_gestures,
                                                                                      number_of_cycles=number_of_cycles,
                                                                                      window_size=window_size,
                                                                                      size_non_overlap=size_non_overlap)
    np.save("../Dataset/processed_dataset/RAW_3DC_test", (list_dataset_train_emg, list_labels_train_emg))
    print("Finished with the test data")
    

if __name__ == '__main__':
    read_data(path="../Dataset/3DC_dataset")
