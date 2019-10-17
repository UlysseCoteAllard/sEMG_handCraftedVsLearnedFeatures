
"""
Created on Thu Oct 26 11:23:47 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as mpl_color_map

import torch
from torch.nn import ReLU, LeakyReLU

from Models.rawConvNet import Model
from FeaturesMapsAndKernelVisualization import grad_cam
from PrepareAndLoadData.load_prepared_dataset_in_dataloaders import load_dataloaders
from FeaturesMapsAndKernelVisualization.misc_functions import (get_example_params, convert_to_grayscale,
                                                               save_gradient_images, get_positive_negative_saliency)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def apply_colormap_to_1D_signal(input_to_network, activation, title, label):
    titles_figure = ["Neutral Gesture", "Radial Deviation", "Wrist Flexion", "Ulnar Deviation", "Wrist Extension",
                     "Supination", "Pronation", "Power Grip", "Open Hand", "Chuck Grip", "Pinch Grip"]
    title_figure = titles_figure[label]
    sns.set()
    color_cmap = truncate_colormap(mpl_color_map.inferno, 0.2, .95)
    if activation is not None and activation.max() > 0.:
        #activation = np.swapaxes(activation, 1, 0)
        print("Activation: ", np.shape(activation))
        print(activation.tolist())
        activation = np.log(activation+1e-16)
        activation = activation.max() + activation
        activation[activation < -3] = activation.min()
        activation = (activation-activation.min())/(activation.max()-activation.min())
        print(activation.tolist())

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 40}

    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(10, 1, sharey=True, sharex=True, figsize=(20, 14))
    fig.text(.5, .965, title_figure, ha='center')
    fig.text(0.5, 0.01, 'Time (ms)', ha='center')
    fig.text(0.01, 0.5, 'Channel', va='center', rotation='vertical')
    print("INPUT TO NETWORK: ", np.shape(input_to_network))
    for i in range(10):
        x_for_linear_interpolation = np.linspace(0, len(input_to_network[0][0][i]), 2000)


        print(np.shape(input_to_network[0][0][i]))
        channel_linearly_interpolated = np.interp(x_for_linear_interpolation,
                                                  np.linspace(0, len(input_to_network[0][0][i]),
                                                              len(input_to_network[0][0][i])),
                                                  input_to_network[0][0][i])
        if activation is not None:
            activation_channel = np.array(activation[i])
            activation_channel[activation_channel == float('nan')] = 0.
            if np.mean(activation_channel) > 0.:
                activation_interpolated = np.interp(x_for_linear_interpolation,
                                                    np.linspace(0, len(input_to_network[0][0][i]),
                                                                len(input_to_network[0][0][i])),
                                                    activation_channel)
                #activation_interpolated[activation_interpolated == float('nan')] = 0.0
                #activation_interpolated[(0 < activation_interpolated) & (activation_interpolated <= 0.2)] = 0.2
                print(activation_channel)
                channel_linearly_interpolated[activation_interpolated <= 0.001] = float('nan')
                ax[i].scatter(x_for_linear_interpolation, channel_linearly_interpolated,
                              c=color_cmap(np.abs(activation_interpolated)), edgecolors='none', s=12**2)
        ax[i].plot(np.linspace(0, 151, 151), input_to_network[0][0][i], linewidth=4, color='forestgreen')
        ax[i].tick_params(axis='both', which='major', labelsize=32)
        ax[i].set_facecolor('white')
        plt.setp(ax[i].get_yticklabels(), visible=False)
    plt.setp(ax[0].get_yticklabels(), visible=True)
    plt.tight_layout(pad=1.12)
    plt.subplots_adjust(wspace=0, hspace=0)
    print(title)
    #plt.show()
    plt.savefig("results/results_guided_gradcam/" + title + ".svg", dpi=1200)


def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    """
        Guided grad cam is just pointwise multiplication of cam mask and
        guided backprop mask
    Args:
        grad_cam_mask (np_arr): Class activation map mask
        guided_backprop_mask (np_arr):Guided backprop mask
    """
    print(grad_cam_mask.tolist())
    print("BLASDASD")
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    print(cam_gb.tolist())
    print("wertfdswrewre")
    return cam_gb


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model, input_x):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers(input_x)

    def hook_layers(self, input_x):
        def hook_function(grad):
            self.gradients = grad[0]
        # Register hook to the first layer
        first_layer = list(self.model.children())[0][0][0]
        #first_layer.register_backward_hook(hook_function)
        input_x.register_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model._features_extractor._modules.items():
            for layer in module:
                if isinstance(layer, LeakyReLU):
                    layer.register_backward_hook(relu_backward_hook_function)
                    layer.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output, _ = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


if __name__ == '__main__':
    'Everything related to the dataset'
    participant_id = 21
    path_dataset = '../Dataset/processed_dataset'
    participants_dataloaders_train = load_dataloaders(path_dataset, batch_size=1, validation_cycle=None,
                                                      get_test_set=False, drop_last=False, shuffle=False)
    gestures = ["Neutral", "Radial_Deviation", "Wrist_Flexion", "Ulnar_Deviation", "Wrist_Extension", "Supination",
                "Pronation", "Power_Grip", "Open_Hand", "Chuck_Grip", "Pinch_Grip"]


    x = torch.tensor(np.random.normal(scale=450, size=(1, 1, 10, 151)), dtype=torch.float32)
    x, label_found = None, None
    k = 0
    for image, label in participants_dataloaders_train[participant_id]:
        label_found = label
        if label.item() == 8:
            k += 1
        if k > 20:
            # x = image
            if x is None:
                x = image
            break

    #print(x)
    for gesture_label in range(11):
        k = 0
        print(gesture_label)
        for image, label in participants_dataloaders_train[participant_id]:
            label_found = label
            if label.item() == gesture_label:
                k += 1
            if k > 50:
                #x = image
                if x is None:
                    x = image
                break

        'Everything related to the models'
        path_weights = '../weights/TL_best_weights.pt'
        path_bn_statistics = "../weights/bn_statistics.pt"
        model = Model(number_of_class=11, number_of_blocks=6, dropout_rate=0.35, filter_size=(1, 26))
        best_weights = torch.load(path_weights)
        model.load_state_dict(best_weights)
        list_dictionaries_bn_weights = torch.load(path_bn_statistics)
        BN_weights = list_dictionaries_bn_weights[participant_id]
        model.load_state_dict(BN_weights, strict=False)

        # Get Grad-CAM
        grad_cam_object = grad_cam.GradCam(model, target_layer=5)
        # Generate cam mask
        cam_mask = grad_cam_object.generate_cam(x, label_found)

        # Guided backprop
        x = torch.autograd.Variable(x, requires_grad=True)
        output, _ = model(x)
        print("OUTPUT: ", output)
        GBP = GuidedBackprop(model, x)
        # Get gradients
        guided_grads_mask = GBP.generate_gradients(x, label_found)

        guided_grad_cam_result = guided_grad_cam(cam_mask, guided_grads_mask.T)

        # Normalize
        guided_grad_cam_result = np.swapaxes(guided_grad_cam_result, 1, 0)

        guided_grad_cam_result = np.array(guided_grad_cam_result)
        guided_grad_cam_result = guided_grad_cam_result.clip(min=0)
        print(guided_grad_cam_result.tolist())
        guided_grad_cam_result = (guided_grad_cam_result - guided_grad_cam_result.min()) / guided_grad_cam_result.max()
        print("SHAPE GUIDED: ", np.shape(guided_grad_cam_result))
        print(guided_grad_cam_result.tolist())
        print('')
        '''
        im_max = np.percentile(guided_grads_mask, 99)
        print("max: ", np.shape(im_max))
        im_min = np.min(guided_grads_mask)
        print("min: ", np.shape(im_min))
        grayscale_guided_grads = (np.clip((guided_grads_mask - im_min) / (im_max - im_min), 0, 1))
        '''
        apply_colormap_to_1D_signal(x.data.numpy(), guided_grad_cam_result, gestures[label_found.item()] +
                                    "_realInput_OpenHand_" + str(5), label=label_found.item())

        '''
        # Save grayscale gradients
        save_gradient_images(guided_grads_mask, 'results_guided_gradcam/_GGrad_Cam')
        grayscale_cam_gb = convert_to_grayscale(guided_grad_cam_result)
        save_gradient_images(grayscale_cam_gb, 'results_guided_gradcam/_GGrad_Cam_gray')
        '''
        print('Guided grad cam completed')
