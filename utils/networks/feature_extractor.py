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
    def __init__(self, observation_space: gym.spaces.Dict,  n_layers:int, layer_width:int, cnn_intermediate_dimension: int = 1):
        super().__init__(observation_space, features_dim=1)
        self.cnn_intermediate_dimension = cnn_intermediate_dimension
        self.shape = observation_space["chars"].shape
        self.n_layers = n_layers
        self.layer_width = layer_width

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
                    nn.Conv2d(int(subspace.high_repr), self.cnn_intermediate_dimension, kernel_size=(1, 1), stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                total_concat_size += observation_space["chars"].shape[0] * observation_space["chars"].shape[1]

            else:
                raise NotImplementedError("Image observation not supported")

        self.extractors = nn.ModuleDict(extractors)

        self.downscaling = nn.Sequential()
        for layer_number in range(n_layers):
            self.downscaling.add_module(f"linear_{layer_number}", nn.Linear(total_concat_size, self.layer_width))
            self.downscaling.add_module(f"ReLu_{layer_number}", nn.ReLU())

        # Update the features dim manually
        self._features_dim = 256

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        # Concatenate the extracted features
        encoded_tensor = th.cat(encoded_tensor_list, dim=1)
        encoded_tensor = self.downscaling(encoded_tensor)

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return encoded_tensor
