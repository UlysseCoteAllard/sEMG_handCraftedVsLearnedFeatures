import numpy as np
import matplotlib.pyplot as plt

import torch

from Models.rawConvNet import Model
from PrepareAndLoadData.load_prepared_dataset_in_dataloaders import load_dataloaders

def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def visualizeFeaturesMaps(path_dataset='../Dataset/processed_dataset', path_weights='../weights/TL_best_weights.pt',
                          path_bn_statistics="../weights/bn_statistics.pt"):
    # Define Model
    model = Model(number_of_class=11, number_of_blocks=6, dropout_rate=0.35)
    best_weights = torch.load(path_weights)
    model.load_state_dict(best_weights)
    list_dictionaries_bn_weights = torch.load(path_bn_statistics)
    BN_weights = list_dictionaries_bn_weights[5]
    model.load_state_dict(BN_weights, strict=False)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    for i in range(6):
        model._features_extractor[i][0].register_forward_hook(get_activation('conv2D_' + str(i)))

    path_dataset = '../Dataset/processed_dataset'
    participants_dataloaders_train = load_dataloaders(path_dataset, batch_size=1, validation_cycle=None,
                                                      get_test_set=False, drop_last=False, shuffle=True)
    gestures = ["Neutral", "Radial Deviation", "Wrist Flexion", "Ulnar Deviation", "Wrist Extension", "Supination",
                "Pronation", "Power Grip", "Open Hand", "Chuck Grip", "Pinch Grip"]
    label_found = None
    for image, label in participants_dataloaders_train[5]:
        sample_image = image
        label_found = label
        break
    print(np.shape(sample_image))
    fig = plt.figure(figsize=(10, 1))
    fig.suptitle(gestures[label_found.item()])
    for i in range(10):
        plt.subplot(10, 1, i + 1)
        plt.plot(sample_image[0][0][i])
        plt.axis('off')

    #plt.show()
    output, _ = model(sample_image)
    for i in range(6):
        act = activation['conv2D_' + str(i)].squeeze()
        #fig, axarr = plt.subplots(act.size(0))
        for idx in range(act.size(0)):
            if idx > 4:
                break
            fig = plt.figure(figsize=(10, 1))
            fig.suptitle(gestures[label_found.item()] + " Layer Activation : " + str(i+1))
            if i == 5:
                print(np.shape(act[idx]))
                plt.imshow(act, cmap='gray')
                break
            else:
                for k in range(10):
                    plt.subplot(10, 1, k+1)
                    print(np.shape(act))
                    target = torch.ones(151)
                    target *= float('nan')
                    target[150] = 0
                    target[:len(act[idx][k])] = act[idx][k]
                    plt.plot(range(151), target)
                    plt.axis('off')
            #axarr[idx].imshow(act[idx])
    plt.show()
    print(output.shape)

if __name__ == "__main__":
    visualizeFeaturesMaps()