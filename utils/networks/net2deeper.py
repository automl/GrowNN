from torch import nn
from torch.nn import functional as F
import torch

class Net2Deeper(nn.Module):
    """
    Implemenation of the Net2Deeper Algorithm from Net2Net: Accelerating Learning via Knowledge Transfer.
    Essentially this serves as a container of linear layers that can be added to the model to increase the depth of the network.
    """

    def __init__(self, input_size, output_size):
        super(Net2Deeper, self).__init__() 
        self.input_size = input_size
        self.output_size = output_size
        self.sequential_container = nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU())

    def forward(self, x):
        x = self.sequential_container(x)
        return x 
    
    def add_layer(self):
        """
        Add a layer to the network
        """
        new_layer = self.get_identity_layer()
        self.sequential_container.append(new_layer)
        self.sequential_container.append(nn.ReLU())
        return self        

    def get_identity_layer(self):
        """
        Get the identity weights for a given layer
        """
        values = torch.eye(self.input_size)
        layer = nn.Linear(self.input_size, self.output_size)
        layer.weight.data = values
        return layer