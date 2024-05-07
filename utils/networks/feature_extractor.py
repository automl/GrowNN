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
        x = x.to(torch.int32)
        x_flat = x.flatten()
        x_one_hot = F.one_hot(x_flat, num_classes=self.num_classes)
        x_one_hot = x_one_hot.view(-1, self.shape[0], self.shape[1], self.num_classes)
        # Change the shape to [batch_size, num_classes, 21, 79] for CNN
        return x_one_hot.permute(0, 3, 1, 2).to(torch.float32)

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, cnn_intermediate_dimension: int = 1):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)
        self.cnn_intermediate_dimension = cnn_intermediate_dimension
        self.shape = observation_space["glyphs"].shape

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "glyphs":
                # Assume the glyphs are 2D grid of integers
                # Transform them to one-hot encoding - resulting in
                extractors[key] = nn.Sequential(OneHotEncoder(int(subspace.high_repr), self.shape),
                                                nn.Conv2d(int(subspace.high_repr), self.cnn_intermediate_dimension, kernel_size=(1, 1), stride=1, padding=0),
                                                nn.ReLU(),
                                                nn.Flatten())
                total_concat_size += observation_space["glyphs"].shape[0] * observation_space["glyphs"].shape[1]

            else:
                raise NotImplementedError("Image observation not supported")
            
        self.extractors = nn.ModuleDict(extractors)
        self.downscaling = nn.Sequential(
            nn.Linear(total_concat_size, 256),
            nn.ReLU()
        )
        
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