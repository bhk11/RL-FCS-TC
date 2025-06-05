from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.common.preprocessing import get_flattened_obs_dim


import torch.nn as nn

def create_network(input_dim, hidden_sizes, output_dim, activations):
    """
    Create a neural network with customizable layers, sizes, and activation functions.

    Args:
    - input_dim (int): The size of the input layer.
    - hidden_sizes (list): List of integers representing the sizes of each hidden layer.
    - output_dim (int): The size of the output layer.
    - activations (list): List of tuples where each tuple contains the activation function
                          name as the first element and any parameters as subsequent elements.
                          Pass None for layers without activation.

    Returns:
    - network (nn.Sequential): The created neural network.
    """

    layers = []
  
    # Input layer
    #m = nn.Linear(20, 30) ==> 20 is input, 30 is output
    #input = torch.randn(128, 20) ==> 128 batch, 20 features
    #output = m(input)
    #print(output.size())
    #torch.Size([128, 30]) ==> 128 batch, 30 features

    layers.append(nn.Linear(input_dim, hidden_sizes[0])) # ???
    
    # Activation function for the first hidden layer
    if activations[0] is not None:
        activation, *params = activations[0]
        act_func = getattr(nn, activation)(*params)
        layers.append(act_func)

    # Hidden layers
    for i in range(1, len(hidden_sizes)):
        layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))

        # Activation function
        if activations[i] is not None:
            activation, *params = activations[i]
            act_func = getattr(nn, activation)(*params)
            layers.append(act_func)

    # Output layer
    layers.append(nn.Linear(hidden_sizes[-1], output_dim))

    # Activation function for output layer
    if activations[-1] is not None:
        activation, *params = activations[-1]
        act_func = getattr(nn, activation)(*params)
        layers.append(act_func)

    return nn.Sequential(*layers)


class CustomQNetwork(QNetwork):
    def __init__(self, activation_fn_, net_arch_, *args, **kwargs):
        super(CustomQNetwork, self).__init__(*args, **kwargs)

        # Custom initialization or modifications can be added here
        # For example, you can modify the network architecture or add custom layers
        # self.net_arch = [64, 64]  # Example of modifying the network architecture 

        action_dim = int(self.action_space.n)
        obs_dim = get_flattened_obs_dim(self.observation_space)
        self.q_net = create_network(
            input_dim=obs_dim,
            hidden_sizes=net_arch_,
            output_dim=action_dim,
            activations=activation_fn_
        )



class CustomMlpPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        #self.obs_space = kwargs.pop('observation_space', None)
        #self.act_space = kwargs.pop('action_space', None)
        self.activation_fn_ = kwargs.pop('activation_fn', None)
        self.net_arch_ = kwargs.pop('net_arch', None)

        super(CustomMlpPolicy, self).__init__(*args, **kwargs)

        # Custom initialization or modifications can be added here
        # For example, you can modify the network architecture or add custom layers
        # self.net_arch = [64, 64]  # Example of modifying the network architecture 
        #self.q_net
        #self.q_net_target

    def make_q_net(self) -> CustomQNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return CustomQNetwork(self.activation_fn_,self.net_arch_,**net_args).to(self.device)



