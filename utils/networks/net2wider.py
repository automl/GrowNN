from torch import nn
import torch
import copy

class Net2Wider(nn.Module):
    """
    Implemenation of the Net2Wider Algorithm from Net2Net: Accelerating Learning via Knowledge Transfer.
    Instead of serving as a single layer as in Net2Deeper, this network contains the entire linear passthrough.

    """

    def __init__(
        self, input_size: int, output_size: int, n_layers: int, increase_factor: int, noise_level: float
    ):
        super(Net2Wider, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.increase_factor = increase_factor
        self.noise_level = noise_level

        self.sequential_container = nn.Sequential()
        for i in range(self.n_layers):
            self.sequential_container.add_module(
                f"linear_{i}", nn.Linear(input_size, output_size)
            )
            self.sequential_container.add_module(f"relu_{i}", nn.ReLU())
        

    def forward(self, x: torch.Tensor):
        x1 = self.sequential_container[0](x)
        x2 = self.sequential_container[1](x1)
        x3 = self.sequential_container[2](x2)
        x4 = self.sequential_container[3](x3)
        x5 = self.sequential_container[4](x4)
        x6 = self.sequential_container[5](x5)
        return x6

    def increase_network_width(self):
        """
        Increase the width of the entire network, i.e. increase the number of neurons in each layer.
        """
        for transformed_layer_id, layer in enumerate(self.sequential_container):
            if isinstance(layer, nn.Linear):
                self.increase_layer_width(transformed_layer_id=transformed_layer_id)

    def increase_layer_width(self, transformed_layer_id: int):
        """
        Increase the width of a specific layer
        """

        if transformed_layer_id % 2 == 1:
            raise ValueError("Cannot increase the width of Relu layer")

        if transformed_layer_id == 0:
            return

        # In the following we call the first affected layer, input layer and the second affected layer, output layer
        input_layer: nn.Linear = self.sequential_container[transformed_layer_id - 2]
        output_layer: nn.Linear = self.sequential_container[transformed_layer_id]

        ### Select Neurons to Copy
        n_new_neurons = int(output_layer.weight.shape[1] * self.increase_factor)
        copy_indices = torch.sort(torch.randint(0, 2, (n_new_neurons,)))

        ### Transformations
        self.sequential_container[transformed_layer_id - 2] = (
            self.transform_input_layer(
                input_layer=input_layer,
                copy_indices=copy_indices,
            )
        )
        self.sequential_container[transformed_layer_id] = self.transform_output_layer(
            preexisting_layer=output_layer,
            copy_indices=copy_indices,
        )

    def transform_input_layer(
        self,
        input_layer: nn.Linear,
        copy_indices: torch.Tensor,
    ):
        input_weights = input_layer.weight[copy_indices.values]

        # Create new layer

        weigths = torch.cat(
            (input_layer.weight, input_weights), dim=0
        )

        new_input_layer = nn.Linear(
            weigths.shape[1],
            weigths.shape[0],
            bias=input_layer.bias is not None,
        )
        new_input_layer.weight.data = weigths

        return new_input_layer

    def transform_output_layer(
        self,
        preexisting_layer: nn.Linear,
        copy_indices: torch.Tensor,
    ):
        # Remember columns of the matrix get split
        # Remember how often the neurons are split
        # Set the weights of the new output layer to the same value
        # Add noise
        weigths = preexisting_layer.weight.data

        index_counts = torch.bincount(copy_indices.values)
        for index, count in enumerate(index_counts):
            if count > 0:
                weigths[:, index] = weigths[:, index] / (count + 1)
                added_weights = (weigths[:, index]).repeat(count).reshape(2, -1)
                weigths = torch.cat((weigths, added_weights), dim=1)
        weigths = weigths + torch.normal(0, 0.1, weigths.shape)
        preexisting_layer.weight.data = weigths
        transformed_layer = nn.Linear(
            weigths.shape[1],
            weigths.shape[0],
            bias=preexisting_layer.bias is not None,
        )
        return transformed_layer
