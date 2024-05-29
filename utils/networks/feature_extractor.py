# Adapted from https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#multiple-inputs-and-dictionary-observations

import gymnasium as gym
import torch as th
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch


class OneHotEncoder(nn.Module):
    # Written with the Help of ChatGPT
    def __init__(self, num_classes, shape):
        super(OneHotEncoder, self).__init__()
        self.num_classes = num_classes
        self.shape = shape

    def forward(self, x):
        with torch.no_grad():
            x = x.to(torch.int64)
            x = x.flatten()
            x = F.one_hot(x, num_classes=self.num_classes)
            x = x.to(torch.float32)
            x = x.view(-1, self.shape[0], self.shape[1], self.num_classes)
            x = x.permute(0, 3, 1, 2)
        return x


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_intermediate_dimension: int,
        n_feature_extractor_layers: int,
        feature_extractor_layer_width: int,
        feature_extractor_output_dimension: int,
    ):
        """
        Custom feature extractor for the MultiInputPolicy.

        First the input is processed by a onehot encoder, followed by a CNN layer, and one linear layer used for downscaling.
        The output is then processed by a number of linear layers of dimension `feature_extractor_layer_width`.
        The last layer has `feature_extractor_output_dimension` units.

        """
        super().__init__(observation_space, features_dim=1)
        self.shape = observation_space["chars"].shape
        self.cnn_intermediate_dimension = cnn_intermediate_dimension
        self.n_feature_extractor_layers = n_feature_extractor_layers
        self.feature_extractor_layer_width = feature_extractor_layer_width
        self.feature_extractor_output_dimension = feature_extractor_output_dimension

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "chars":
                # Assume the chars are 2D grid of integers
                # Transform them to one-hot encoding - resulting in
                extractors[key] = nn.Sequential(
                    OneHotEncoder(int(subspace.high_repr), self.shape),
                    nn.Conv2d(
                        int(subspace.high_repr),
                        self.cnn_intermediate_dimension,
                        kernel_size=(1, 1),
                        stride=1,
                        padding=0,
                    ),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                total_concat_size += observation_space["chars"].shape[0] * observation_space["chars"].shape[1]

            else:
                raise NotImplementedError("Image observation not supported")

        self.extractors = nn.ModuleDict(extractors)
        self.downscaling = nn.Sequential(
            nn.Linear(total_concat_size, self.feature_extractor_layer_width),
            nn.ReLU(),
        )

        self.linear_layers = nn.Sequential()
        for layer_number in range(n_feature_extractor_layers):
            input_size = self.feature_extractor_layer_width
            if layer_number == n_feature_extractor_layers - 1:
                output_size = feature_extractor_output_dimension
            else:
                output_size = self.feature_extractor_layer_width
            self.linear_layers.add_module(f"linear_{layer_number}", nn.Linear(input_size, output_size))
            self.linear_layers.add_module(f"ReLu_{layer_number}", nn.ReLU())

        ##### KEEEEEP THIS AT ALL COST, BECAUSE STABLE BASELIENS USES IT FOR SOME REASON
        self._features_dim = feature_extractor_output_dimension
        ##### KEEEEEP THIS AT ALL COST, BECAUSE STABLE BASELIENS USES IT FOR SOME REASON

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        # Concatenate the extracted features
        encoded_tensor = th.cat(encoded_tensor_list, dim=1)
        encoded_tensor = self.downscaling(encoded_tensor)
        encoded_tensor = self.linear_layers(encoded_tensor)

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return encoded_tensor
