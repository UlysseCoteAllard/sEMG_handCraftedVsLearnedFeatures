import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.utils import ReversalGradientLayerF


class Model(nn.Module):
    def __init__(self, number_of_class, number_of_blocks, number_of_channels=10, number_of_features_output=64,
                 filter_size=(1, 26), dropout_rate=0.5):
        super(Model, self).__init__()
        self._number_of_channel_input = number_of_channels
        self._number_of_features_output = number_of_features_output

        list_blocks = []
        for i in range(number_of_blocks):
            if i == 0:
                list_blocks.append(self.generate_bloc(block_id=i, number_features_input=1,
                                                      number_of_features_output=self._number_of_features_output,
                                                      filter_size=filter_size, dropout_rate=dropout_rate))
            else:
                list_blocks.append(self.generate_bloc(block_id=i, number_features_input=self._number_of_features_output,
                                                      number_of_features_output=self._number_of_features_output,
                                                      filter_size=filter_size, dropout_rate=dropout_rate))

        self._features_extractor = nn.ModuleList(list_blocks)

        self._output = nn.Linear(self._number_of_channel_input*self._number_of_features_output, number_of_class)
        self._output_domain = nn.Linear(self._number_of_channel_input*self._number_of_features_output, 2)

        self._number_of_blocks = number_of_blocks

        'Regressor related variables'
        self._output_regressor = None
        self._layers_to_regress_from = None
        self._size_feature_maps = [126, 101, 76, 51, 26, 1]
        self._channel_to_use = None

        print(self)
        print("Number Parameters: ", self.get_n_params())

    def get_number_of_blocks(self):
        return self._number_of_blocks

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def forward(self, x, lambda_value=None, use_regressor_forward=False, use_forward_visualization=False):
        if use_regressor_forward:
            return self.regressor_forward(x)

        if use_forward_visualization:
            return self.forward_visualization(x)
        features_calculated = {}
        for i, block in enumerate(self._features_extractor):
            for _, layer in enumerate(block):
                x = layer(x)
                if isinstance(layer, nn.LeakyReLU):
                    features_calculated['layer_' + str(i)] = torch.mean(x, dim=(3)).detach().cpu().numpy()
        # Perform the average pooling channel wise (i.e. for each channel of the armband), take the average output of
        # the features_data
        #features_extracted = F.adaptive_avg_pool2d(x, (self._number_of_channel_input, 1)).view(
        #    -1, self._number_of_channel_input * self._number_of_features_output)
        features_extracted = x.view(-1, self._number_of_channel_input * self._number_of_features_output)
        output = self._output(features_extracted)
        if lambda_value is None:
            return output, features_calculated
        else:
            reversed_layer = ReversalGradientLayerF.grad_reverse(features_extracted, lambda_value)
            output_domain = self._output_domain(reversed_layer)
            return output, output_domain, features_calculated

    def generate_bloc(self, block_id, number_features_input=64, number_of_features_output=64, filter_size=(1, 26),
                      dropout_rate=0.5):
        block = nn.Sequential(OrderedDict([
            ("conv2D_" + str(block_id), nn.Conv2d(in_channels=number_features_input, out_channels=
            number_of_features_output, kernel_size=filter_size, stride=1)),
            ("batchNorm_" + str(block_id), nn.BatchNorm2d(num_features=number_of_features_output, momentum=0.99,
                                                          eps=1e-3)),
            ("leakyRelu_" + str(block_id), nn.LeakyReLU(negative_slope=0.1, inplace=True)),
            ("dropout2D_" + str(block_id), nn.Dropout2d(p=dropout_rate))
        ]))

        return block

    def regressor_feature_extraction_forward(self, x):
        for i, block in enumerate(self._features_extractor):
            for j, layer in enumerate(block):
                x = layer(x)
                if isinstance(layer, nn.LeakyReLU) and i == self._layers_to_regress_from:
                    return x
        return x

    def regressor_forward(self, x):
        # Only keep the electrode that is being regressed on
        x = x.narrow(2, self._channel_to_use, self._channel_to_use + 1)

        x = self.regressor_feature_extraction_forward(x)
        flatten_x = x.reshape(-1, self._number_of_features_output *
                              self._size_feature_maps[self._layers_to_regress_from])
        output_regressor = self._output_regressor(flatten_x)
        return output_regressor

    def transform_to_regressor(self, layer_to_regress_from, freeze_features_extraction=True, channel_to_use=0):
        assert 0 <= layer_to_regress_from <= self._number_of_blocks

        self._layers_to_regress_from = layer_to_regress_from
        self._size_feature_maps = [126, 101, 76, 51, 26, 1]
        self._channel_to_use = channel_to_use
        if freeze_features_extraction:
            # Remove the possibility of changing the weights of the network
            for param in self.parameters():
                param.requires_grad = False

        self._output_regressor = nn.Linear(self._number_of_features_output *
                                           self._size_feature_maps[self._layers_to_regress_from], 1)
