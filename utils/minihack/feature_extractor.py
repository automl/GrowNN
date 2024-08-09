# Adapted from https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#multiple-inputs-and-dictionary-observations

import gymnasium as gym
import torch as th
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import abc
from utils.networks.net2deeper import Net2Deeper
from typing import List
from utils.networks.net2wider import Net2Wider


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


class NoCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, n_feature_extractor_layers: int, feature_extractor_layer_width: int, feature_extractor_output_dimension: int):
        super(NoCNNFeatureExtractor, self).__init__(observation_space, features_dim=1)
        self.shape = observation_space["chars"].shape
        self.n_feature_extractor_layers = n_feature_extractor_layers
        self.feature_extractor_layer_width = feature_extractor_layer_width
        self.feature_extractor_output_dimension = feature_extractor_output_dimension
        self.build_feature_extractor(observation_space)

    def build_feature_extractor(self, observation_space: gym.spaces.Dict):
        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "chars":
                # Assume the chars are 2D grid of integers
                # Transform them to one-hot encoding - resulting in
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                )
                total_concat_size += observation_space["chars"].shape[0] * observation_space["chars"].shape[1]

            else:
                raise NotImplementedError("Image observation not supported")

        self.extractors = nn.ModuleDict(extractors)
        self.downscaling = self.downscaling = nn.Sequential(
            nn.Linear(total_concat_size, self.feature_extractor_layer_width),
            nn.ReLU(),
        )
        self.linear_layers = nn.Sequential()
        for layer_number in range(self.n_feature_extractor_layers):
            input_size = self.feature_extractor_layer_width
            if layer_number == self.n_feature_extractor_layers - 1:
                output_size = self.feature_extractor_output_dimension
            else:
                output_size = self.feature_extractor_layer_width
            self.linear_layers.add_module(f"linear_{layer_number}", nn.Linear(input_size, output_size))
            self.linear_layers.add_module(f"ReLu_{layer_number}", nn.ReLU())

        ##### KEEEEEP THIS AT ALL COST, BECAUSE STABLE BASELIENS USES IT FOR SOME REASON
        self._features_dim = self.feature_extractor_output_dimension
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


class AbstractFeatureExtractor(BaseFeaturesExtractor, abc.ABC):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_intermediate_dimension: int,
        n_feature_extractor_layers: int,
        feature_extractor_layer_width: int,
        feature_extractor_output_dimension: int,
    ):
        """
        Abstract Custom feature extractor for the MultiInputPolicy. All subclasses are initialised with the same amount of
        parameters, but the amount of layers may be increased over time.
        """
        super().__init__(observation_space, features_dim=1)
        self.shape = observation_space["chars"].shape
        self.cnn_intermediate_dimension = cnn_intermediate_dimension
        self.n_feature_extractor_layers = n_feature_extractor_layers
        self.feature_extractor_layer_width = feature_extractor_layer_width
        self.feature_extractor_output_dimension = feature_extractor_output_dimension
        self.build_feature_extractor(observation_space)

    @abc.abstractmethod
    def build_feature_extractor(self, observation_space: gym.spaces.Dict):
        """
        Abstract Method to initialise the feature extracot.
        """

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


class FixedArchitectureFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, cnn_intermediate_dimension: int, feature_extractor_architecture: List[int], feature_extractor_output_dimension: int):
        super().__init__(observation_space, features_dim=1)
        self.shape = observation_space["chars"].shape
        self.cnn_intermediate_dimension = cnn_intermediate_dimension
        self.feature_extractor_architecture = feature_extractor_architecture
        self.feature_extractor_output_dimension = feature_extractor_output_dimension
        assert self.feature_extractor_architecture[-1] == self.feature_extractor_output_dimension

        self.build_feature_extractor(observation_space)

    def build_feature_extractor(self, observation_space: gym.spaces.Dict):
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
                total_concat_size += observation_space["chars"].shape[0] * observation_space["chars"].shape[1] * self.cnn_intermediate_dimension

            else:
                raise NotImplementedError("Image observation not supported")

        self.extractors = nn.ModuleDict(extractors)
        self.downscaling = nn.Sequential(
            nn.Linear(total_concat_size, self.feature_extractor_architecture[0]),
            nn.ReLU(),
        )

        self.linear_layers = nn.Sequential()
        for layer_number in range(len(self.feature_extractor_architecture) - 1):
            input_size = self.feature_extractor_architecture[layer_number]
            output_size = self.feature_extractor_architecture[layer_number + 1]
            self.linear_layers.add_module(f"linear_{layer_number}", nn.Linear(input_size, output_size))
            if layer_number != len(self.feature_extractor_architecture) - 2:
                self.linear_layers.add_module(f"ReLu_{layer_number}", nn.ReLU())

        ##### KEEEEEP THIS AT ALL COST, BECAUSE STABLE BASELIENS USES IT FOR SOME REASON
        self._features_dim = self.feature_extractor_output_dimension
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


class CustomCombinedExtractor(AbstractFeatureExtractor):
    def build_feature_extractor(self, observation_space: gym.spaces.Dict):
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
                total_concat_size += observation_space["chars"].shape[0] * observation_space["chars"].shape[1] * self.cnn_intermediate_dimension

            else:
                raise NotImplementedError("Image observation not supported")

        self.extractors = nn.ModuleDict(extractors)
        self.downscaling = nn.Sequential(
            nn.Linear(total_concat_size, self.feature_extractor_layer_width),
            nn.ReLU(),
        )

        self.linear_layers = nn.Sequential()
        for layer_number in range(self.n_feature_extractor_layers):
            input_size = self.feature_extractor_layer_width
            if layer_number == self.n_feature_extractor_layers - 1:
                output_size = self.feature_extractor_output_dimension
            else:
                output_size = self.feature_extractor_layer_width
            self.linear_layers.add_module(f"linear_{layer_number}", nn.Linear(input_size, output_size))
            self.linear_layers.add_module(f"ReLu_{layer_number}", nn.ReLU())

        ##### KEEEEEP THIS AT ALL COST, BECAUSE STABLE BASELIENS USES IT FOR SOME REASON
        self._features_dim = self.feature_extractor_output_dimension
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


class Net2DeeperFeatureExtractor(AbstractFeatureExtractor):
    """
    Custom feature extractor for the MultiInputPolicy.

    First the input is processed by a onehot encoder, followed by a CNN layer, and one linear layer used for downscaling.
    The output is then processed by a number of linear layers of dimension `feature_extractor_layer_width`.
    The last layer has `feature_extractor_output_dimension` units.

    The object is initialised with the given amount of layers, but the amount of layers is increased over time.
    """

    def build_feature_extractor(self, observation_space: gym.spaces.Dict):
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
                total_concat_size += observation_space["chars"].shape[0] * observation_space["chars"].shape[1] * self.cnn_intermediate_dimension

            else:
                raise NotImplementedError("Image observation not supported")

        self.extractors = nn.ModuleDict(extractors)
        self.downscaling = nn.Sequential(
            nn.Linear(total_concat_size, self.feature_extractor_layer_width),
            nn.ReLU(),
        )

        self.linear_layers = Net2Deeper(self.feature_extractor_layer_width, self.feature_extractor_output_dimension)

        ##### KEEEEEP THIS AT ALL COST, BECAUSE STABLE BASELIENS USES IT FOR SOME REASON
        self._features_dim = self.feature_extractor_output_dimension
        ##### KEEEEEP THIS AT ALL COST, BECAUSE STABLE BASELIENS USES IT FOR SOME REASON

    def add_layer(self):
        self.linear_layers.add_layer()


class Net2WiderFeatureExtractor(AbstractFeatureExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_intermediate_dimension: int,
        n_feature_extractor_layers: int,
        feature_extractor_layer_width: int,
        feature_extractor_output_dimension: int,
        increase_factor: float,
        noise_level: float,
    ):
        self.increase_factor = increase_factor
        self.noise_level = noise_level
        super().__init__(
            observation_space=observation_space,
            cnn_intermediate_dimension=cnn_intermediate_dimension,
            n_feature_extractor_layers=n_feature_extractor_layers,
            feature_extractor_layer_width=feature_extractor_layer_width,
            feature_extractor_output_dimension=feature_extractor_output_dimension,
        )

    def build_feature_extractor(self, observation_space: gym.spaces.Dict):
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
                total_concat_size += observation_space["chars"].shape[0] * observation_space["chars"].shape[1] * self.cnn_intermediate_dimension

            else:
                raise NotImplementedError("Image observation not supported")

        self.extractors = nn.ModuleDict(extractors)
        self.downscaling = nn.Sequential(
            nn.Linear(total_concat_size, self.feature_extractor_layer_width),
            nn.ReLU(),
        )

        self.linear_layers = Net2Wider(
            self.feature_extractor_layer_width, self.feature_extractor_output_dimension, self.n_feature_extractor_layers, increase_factor=self.increase_factor, noise_level=self.noise_level
        )

        ##### KEEEEEP THIS AT ALL COST, BECAUSE STABLE BASELIENS USES IT FOR SOME REASON
        self._features_dim = self.feature_extractor_output_dimension
        ##### KEEEEEP THIS AT ALL COST, BECAUSE STABLE BASELIENS USES IT FOR SOME REASON

    def increase_width(self):
        self.linear_layers.increase_network_width()
