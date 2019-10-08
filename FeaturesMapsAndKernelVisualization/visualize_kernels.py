import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision.utils import make_grid

import torch

from Models.rawConvNet import Model


def visualizeKernels(path_weights='../weights/TL_best_weights.pt', use_trained_weights=True):
    sns.set()
    cmap="viridis"
    example_mat = np.linspace(1, 0, 25).reshape(5, 5)
    plt.imshow(example_mat, cmap=cmap)
    # Define Model
    model = Model(number_of_class=11, number_of_blocks=6, dropout_rate=0.35)
    if use_trained_weights:
        best_weights = torch.load(path_weights)
        model.load_state_dict(best_weights)

    for name, parameter in model.named_parameters():
        if "conv2D" in name and "weight" in name:
            kernels = parameter.detach().clone()
            plt.figure(figsize=(16, 4))
            for idx, filt in enumerate(kernels):
                if idx == 0:
                    plt.title(name)
                plt.subplot(16, 4, idx + 1)
                plt.imshow(filt[0, :, :], interpolation=None, cmap=cmap)
                plt.axis('off')
    #plt.show()

    #plt.imshow(example_mat, cmap=cmap)
    for name, parameter in model.named_parameters():
        if "conv2D" in name and "weight" in name:
            plt.title(name)
            kernels = parameter.detach().clone()
            plt.figure()
            plt.imshow(kernels[:, 0, 0, :], cmap=cmap)
            plt.axis('off')

    #plt.show()

    for name, parameter in model.named_parameters():
        if "conv2D" in name and "weight" in name:
            kernels = parameter.detach().clone()
            plt.figure(figsize=(16, 4))
            for idx, filt in enumerate(kernels):
                if idx == 0:
                    plt.title(name)
                plt.subplot(16, 4, idx + 1)
                plt.plot(filt[0, 0, :])
                plt.axis('off')
    plt.show()


if __name__ == "__main__":
    visualizeKernels(use_trained_weights=False)
    visualizeKernels()
