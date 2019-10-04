import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from Models.utils import ReversalGradientLayerF


class Model(nn.Module):
    def __init__(self, number_of_class, number_of_blocks, number_of_channels=10, number_of_features_output=64,
                 filter_size=(1, 25)):
        super(Model, self).__init__()
        self._number_of_channel_input = number_of_channels
        self._number_of_features_output = number_of_features_output

        list_blocks = []
        for i in range(number_of_blocks):
            if i == 0:
                list_blocks.append(self.generate_bloc(number_features_input=self._number_of_channel_input,
                                                      number_of_features_output=self._number_of_features_output,
                                                      filter_size=filter_size))
            else:
                list_blocks.append(self.generate_bloc(number_features_input=self._number_of_features_output,
                                                      number_of_features_output=self._number_of_features_output,
                                                      filter_size=filter_size))

        self._features_extractor = nn.ModuleList(list_blocks)

        self._output = nn.Linear(self._number_of_channel_input*self._number_of_features_output, number_of_class)
        self._output_domain = nn.Linear(self._number_of_channel_input*self._number_of_features_output, 1)

        print(self)
        print("Number Parameters: ", self.get_n_params())


    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def forward(self, x, lambda_value=None):
        for i, block in enumerate(self._features_extractor):
            x = block(x)
        # Perform the average pooling channel wise (i.e. for each channel of the armband), take the average output of
        # the features
        x = F.adaptive_avg_pool2d(x, (self._number_of_channel_input, 1)).view(-1, self._number_of_channel_input *
                                                                              self._number_of_features_output)
        output = self._output(x)
        if lambda_value is None:
            return output
        else:
            reversed_layer = ReversalGradientLayerF.grad_reverse(x, lambda_value)
            output_domain = self._output_domain(reversed_layer)
            return output, output_domain


    def generate_bloc(self, number_features_input=64, number_of_features_output=64, filter_size=(1, 25)):
        block = nn.Sequential(
            nn.Conv2d(in_channels=number_features_input, out_channels=number_of_features_output,
                      kernel_size=filter_size, stride=1),
            nn.BatchNorm2d(num_features=number_of_features_output, momentum=0.99, eps=1e-3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        return block
