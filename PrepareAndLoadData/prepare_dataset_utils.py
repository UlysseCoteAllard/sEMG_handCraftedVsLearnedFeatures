from scipy import signal
import matplotlib.pyplot as plt


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    lowcut_normalized = lowcut / nyq
    highcut_normalized = highcut / nyq
    b, a = signal.butter(N=order, Wn=[lowcut_normalized, highcut_normalized], btype='band', output="ba")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def applies_high_pass_for_dataset(dataset, frequency):
    dataset_to_return = []
    for example in dataset:
        example_formatted = []
        for vector_electrode in example:
            filtered_signal = butter_bandpass_filter(vector_electrode, 20, 495, frequency)
            example_formatted.append(filtered_signal)
        dataset_to_return.append(example_formatted)
    return dataset_to_return


def show_filtered_signal(noisy_signal, filtered_signal, fs=1000):
    plt.plot(noisy_signal, label='Noisy signal')
    plt.plot(filtered_signal, label='Filtered signal (%g Hz)' % fs)
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()
