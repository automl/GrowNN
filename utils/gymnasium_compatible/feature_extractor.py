from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces.box import Box
from typing import List
from torch import nn
from utils.networks.net2deeper import Net2Deeper
from utils.networks.net2wider import Net2Wider


class FixedArchitectureFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor with a fixed architecture. The architecture is defined by the feature_extractor_architecture parameter.
    """

    def __init__(self, observation_space: Box, feature_extractor_architecture: List[int], **kwargs):
        """
        Constructor for the FixedArchitectureFeaturesExtractor class. The architecture is defined by the feature_extractor_architecture parameter.
        :param observation_space: The observation space of the environment
        :type observation_space: Box
        :param feature_extractor_architecture: The architecture of the feature extractor. Each list element is the number of neurons in a layer.
        :type feature_extractor_architecture: List[int]
        """
        super(FixedArchitectureFeaturesExtractor, self).__init__(observation_space, features_dim=feature_extractor_architecture[-1], **kwargs)
        self.architecture = feature_extractor_architecture
        self.input_layer = nn.Linear(in_features=observation_space.shape[0], out_features=feature_extractor_architecture[0])
        self.processing = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(in_features=feature_extractor_architecture[i], out_features=feature_extractor_architecture[i + 1]), nn.ReLU())
                for i in range(len(feature_extractor_architecture) - 1)
            ]
        )

    def forward(self, observations):
        x = self.input_layer(observations)
        x = self.processing(x)
        return x


class Net2DeeperFeatureExtractor(BaseFeaturesExtractor):
    """
    Net2DeeperFeatureExtractor class. This class is a feature extractor that can grow in depth and always starts with depth one.
    """

    def __init__(self, observation_space: Box, feature_extractor_width: List[int], **kwargs):
        """
        Constructor for the Net2DeeperFeatureExtractor class. The architecture is defined by the feature_extractor_width parameter.
        :param observation_space: The observation space of the environment
        :type observation_space: Box
        :param feature_extractor_width: The width of the feature extractor. Each list element is the number of neurons in a layer.
        :type feature_extractor_width: List[int]
        """
        super(Net2DeeperFeatureExtractor, self).__init__(observation_space, features_dim=feature_extractor_width, **kwargs)
        self.feature_extractor_width = feature_extractor_width
        self.input_layer = nn.Linear(in_features=observation_space.shape[0], out_features=feature_extractor_width)
        self.net2deeper_network = Net2Deeper(feature_extractor_width, feature_extractor_width)

    def forward(self, observations):
        x = self.input_layer(observations)
        x = self.net2deeper_network(x)
        return x

    def add_layer(self):
        """
        Adds a layer to the feature extractor.
        """
        self.net2deeper_network.add_layer()
        return self


class Net2WiderFeatureExtractor(BaseFeaturesExtractor):
    """
    Net2WiderFeatureExtractor class.
    """
    def __init__(self, observation_space: Box, input_size: int, output_size:int, n_layers:int, increase_factor:float, noise_level:float, **kwargs):
        """
        Constructor for the Net2WiderFeatureExtractor class.
        :param observation_space: The observation space of the environment
        :type observation_space: Box
        :param input_size: The number neurons which the input layer has
        :type input_size: int
        :param output_size: The number of neurons in the output layer
        :type output_size: int
        :param n_layers: The number of layers of the feature extractor
        :type n_layers: int
        :param increase_factor: The factor by which the width of the network is increased
        :type increase_factor: float
        :param noise_level: The noise level which is used to break the symetry of added neurons
        :type noise_level: float
        """
        super(Net2WiderFeatureExtractor, self).__init__(observation_space, features_dim=output_size, **kwargs)
        self.input_layer = nn.Linear(in_features=observation_space.shape[0], out_features=input_size)
        self.net2wider_network = Net2Wider(input_size=input_size, output_size=output_size, n_layers=n_layers, increase_factor=increase_factor, noise_level=noise_level)

    def forward(self, observations):
        x = self.input_layer(observations)
        x = self.net2wider_network(x)
        return x

    def grow_width(self):
        self.net2wider_network.increase_network_width()
        return self
