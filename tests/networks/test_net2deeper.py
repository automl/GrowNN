from utils.networks.net2deeper import Net2Deeper

from torch import nn
import torch
import pytest

@pytest.fixture
def input_size():
    return 32
@pytest.fixture
def net(input_size):
    return Net2Deeper(input_size, input_size)

def test_net2deeper_initialization(net:Net2Deeper):
    assert isinstance(net, nn.Module)
    assert (
        len(net.sequential_container) == 2
    )  # One Linear layer and one ReLU activation
    assert isinstance(net.sequential_container[0], nn.Linear)
    assert isinstance(net.sequential_container[1], nn.ReLU)


def test_net2deeper_forward_pass(net:Net2Deeper, input_size):
    x = torch.randn(1, input_size)
    y = net(x)
    assert y.shape == (1, input_size)


def test_net2deeper_add_layer(net:Net2Deeper):
    net = net.add_layer()
    assert (
        len(net.sequential_container) == 4
    )  # Two Linear layers and two ReLU activations
    assert isinstance(net.sequential_container[0], nn.Linear)
    assert isinstance(net.sequential_container[1], nn.ReLU)
    assert isinstance(net.sequential_container[2], nn.Linear)
    assert isinstance(net.sequential_container[3], nn.ReLU)


def test_identiy_weights_change(net: Net2Deeper):
    net = net.add_layer()

    # Crreate Input with Batchsize 32
    x = torch.randn(32, net.input_size)
    y = torch.randn(32, 1)

    # Optimize the network
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    for i in range(1000):
        optimizer.zero_grad()
        output = net(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    
    # Check if the weights of the added layer have changed
    assert not torch.equal(net.sequential_container[2].weight.data, torch.eye(net.input_size))
    weights = net.sequential_container[2].weight.data
    # Check if any weight is zero
    assert not torch.equal(weights, torch.zeros_like(weights))