from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces.box import Box
from typing import List
from torch import nn
from utils.networks.net2deeper import Net2Deeper


class FixedArchitectureFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, feature_extractor_architecture: List[int], **kwargs):
        super(FixedArchitectureFeaturesExtractor, self).__init__(observation_space, features_dim=feature_extractor_architecture[-1], **kwargs)
        self.architecture = feature_extractor_architecture
        self.input_layer = nn.Linear(in_features=observation_space.shape[0], out_features=feature_extractor_architecture[0])
        self.processing = nn.Sequential(
            *[nn.Linear(in_features=feature_extractor_architecture[i], out_features=feature_extractor_architecture[i + 1]) for i in range(len(feature_extractor_architecture) - 1)]
        )

    def forward(self, observations):
        x = self.input_layer(observations)
        x = self.processing(x)
        return x


class Net2DeeperFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, feature_extractor_width: List[int], **kwargs):
        super(Net2DeeperFeatureExtractor, self).__init__(observation_space, features_dim=feature_extractor_width, **kwargs)
        self.feature_extractor_width = feature_extractor_width
        self.input_layer = nn.Linear(in_features=observation_space.shape[0], out_features=feature_extractor_width)
        self.net2deeper_network = Net2Deeper(feature_extractor_width, feature_extractor_width)

    def forward(self, observations):
        x = self.input_layer(observations)
        x = self.net2deeper_network(x)
        return x
    
    def add_layer(self):
        self.net2deeper_network.add_layer()
        return self

