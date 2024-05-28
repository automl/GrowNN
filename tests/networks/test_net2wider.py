from utils.networks.net2wider import Net2Wider
import pytest
from torch import nn
import torch


@pytest.fixture
def net2wider():
    torch.manual_seed(1)
    return Net2Wider(input_size=2, output_size=2, n_layers=3, increase_factor=2, noise_level=0.1)


@pytest.fixture
def transformed_layer_id():
    return 2


def test_transform_input_layer(net2wider: Net2Wider, transformed_layer_id: int):
    input_layer: nn.Linear = net2wider.sequential_container[transformed_layer_id - 2]

    ### Select Neurons to Copy
    n_new_neurons = int(input_layer.weight.shape[1] * net2wider.increase_factor)
    copy_indices = torch.sort(torch.randint(0, 2, (n_new_neurons,)))

    new_input_layer = net2wider.transform_input_layer(
        input_layer=input_layer,
        copy_indices=copy_indices,
    )

    # Assert shaoe of the new input layer
    assert new_input_layer.weight.shape[1] == 2
    assert new_input_layer.weight.shape[0] == 6

    # Assert shape resulting of forward pass
    output = new_input_layer(torch.normal(-1, 1, (1, 2)))
    assert output.shape == (
        1,
        6,
    )


def test_tranform_output_layer(net2wider: Net2Wider, transformed_layer_id: int):
    output_layer = net2wider.sequential_container[transformed_layer_id]

    ### Select Neurons to Copy
    n_new_neurons = int(output_layer.weight.shape[0] * net2wider.increase_factor)
    copy_indices = torch.sort(torch.randint(0, 2, (n_new_neurons,)))
    new_output_layer = net2wider.transform_output_layer(
        preexisting_layer=output_layer,
        copy_indices=copy_indices,
    )
    assert new_output_layer.weight.shape[0] == 2
    assert new_output_layer.weight.shape[1] == 6

    # Assert shape resulting of forward pass
    output = new_output_layer(torch.normal(-1, 1, (3, 6)))
    assert output.shape == (
        3,
        2,
    )


def test_increase_layer_width(net2wider: Net2Wider, transformed_layer_id: int):    
    net2wider.increase_layer_width(transformed_layer_id=transformed_layer_id)
    x = torch.normal(-1, 1, (3, 2))
    output = net2wider(x)
    output.shape == (3, 2)

def test_increase_network_width(net2wider: Net2Wider):
    net2wider.increase_network_width()
    x = torch.normal(-1, 1, (3, 2))
    output = net2wider(x)
    output.shape == (3, 2)