from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces.box import Box
from typing import List
from torch import nn


class FixedArchitectureFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, feature_extractor_architecture: List[int], **kwargs):
        super(FixedArchitectureFeaturesExtractor, self).__init__(observation_space, features_dim=feature_extractor_architecture[-1], ** kwargs)
        self.architecture = feature_extractor_architecture
        self.input_layer = nn.Linear(in_features=observation_space.shape[0], out_features=feature_extractor_architecture[0])
        self.processing = nn.Sequential(
            *[nn.Linear(in_features=feature_extractor_architecture[i], out_features=feature_extractor_architecture[i + 1]) for i in range(len(feature_extractor_architecture) - 1)]
        )

    def forward(self, observations):
        x = self.input_layer(observations)
        x = self.processing(x)
        return x
