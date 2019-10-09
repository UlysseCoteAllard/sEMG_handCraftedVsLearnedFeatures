"""
Created on Thu Oct 26 11:06:51 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F

from Models.rawConvNet import Model
from PrepareAndLoadData.load_prepared_dataset_in_dataloaders import load_dataloaders
from FeaturesMapsAndKernelVisualization.misc_functions import get_example_params, save_class_activation_images, \
    apply_colormap_to_1D_signal



class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model._features_extractor._modules.items():
            print(module)
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model._output(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model._features_extractor.zero_grad()
        self.model._output.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        print("Guided Gradient Size: ", np.shape(guided_gradients))
        #weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        weights = np.mean(guided_gradients, axis=(1, 2))
        print("Weight : ", np.shape(weights))
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        print("TARGET: ", np.shape(target))
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        print(np.shape(cam))
        print("BEFORE RESIZING: ", cam)
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(np.swapaxes(cam, 1, 0)).resize((input_image.shape[2],
                                                                       input_image.shape[3]), Image.ANTIALIAS)) / 255
        print("CAM READY: ", cam)
        print(np.shape(cam))
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.
        return cam


if __name__ == '__main__':
    path_dataset = '../Dataset/processed_dataset'
    path_weights = '../weights/TL_best_weights.pt'
    path_bn_statistics = "../weights/bn_statistics.pt"
    model = Model(number_of_class=11, number_of_blocks=6, dropout_rate=0.35)
    best_weights = torch.load(path_weights)
    model.load_state_dict(best_weights)
    list_dictionaries_bn_weights = torch.load(path_bn_statistics)
    BN_weights = list_dictionaries_bn_weights[5]
    model.load_state_dict(BN_weights, strict=False)

    participants_dataloaders_train = load_dataloaders(path_dataset, batch_size=1, validation_cycle=None,
                                                      get_test_set=False, drop_last=False, shuffle=True)
    gestures = ["Neutral", "Radial Deviation", "Wrist Flexion", "Ulnar Deviation", "Wrist Extension", "Supination",
                "Pronation", "Power Grip", "Open Hand", "Chuck Grip", "Pinch Grip"]
    x, label_found = None, None
    for image, label in participants_dataloaders_train[5]:
        x = image
        label_found = label
        if label == 9:
            break

    print(np.shape(x))
    sample_image = F.to_pil_image(x[0]).transpose(Image.ROTATE_90)
    #sample_image.transpose(Image.ROTATE_90).show()
    # Get params
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)
    pretrained_model = model
    apply_colormap_to_1D_signal(x.numpy(), None, gestures[label_found.item()] + " Input data")
    for i in range(6):
        # Grad cam
        grad_cam = GradCam(pretrained_model, target_layer=i)

        # Generate cam mask
        cam = grad_cam.generate_cam(x, label_found)

        # Save mask

        apply_colormap_to_1D_signal(x.numpy(), cam, gestures[label_found.item()] + " Layer_" + str(i))
    import matplotlib.pyplot as plt
    save_class_activation_images(sample_image, cam, gestures[label_found.item()])
    print('Grad cam completed')

