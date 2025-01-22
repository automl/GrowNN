from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter
import yaml

# Create a Configuration Space
cs = ConfigurationSpace()

# Add hyperparameters
learning_rate = UniformFloatHyperparameter('lr', lower=1e-5, upper=1e-1, log=True)
num_heads = CategoricalHyperparameter('num_heads', [6, 8, 12])
num_hidden_neurons = CategoricalHyperparameter('num_hidden_neurons', [64, 128, 256, 512])
num_hidden_layers = CategoricalHyperparameter('num_hidden_layers', [1, 2, 3])
activation_function_type = CategoricalHyperparameter('activation_function', ['GELU', 'LeakyReLU'])
scheduler_type = CategoricalHyperparameter('scheduler', ['', 'ReduceLROnPlateau', 'CosineAnnealingLR'])
max_value = CategoricalHyperparameter('max_value', [1, 2, 3])

# Add hyperparameters to the ConfigurationSpace
cs.add_hyperparameters([
    learning_rate,
    num_heads,
    num_hidden_neurons,
    num_hidden_layers,
    activation_function_type,
    scheduler_type,
    max_value
])

cs.to_json("configspace.json")