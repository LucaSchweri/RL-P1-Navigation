import torch
from importlib import reload
import networks.dqn
reload(networks.dqn)
from networks.dqn import DeepQNetwork
import networks.dropout_dqn
reload(networks.dropout_dqn)
from networks.dropout_dqn import DropoutDeepQNetwork
import networks.dueling_dqn
reload(networks.dueling_dqn)
from networks.dueling_dqn import DuelingDeepQNetwork


def get_network(name, state_size, action_size, seed):
    """returns the initialized network loaded to the correct device
    
    Params
    ======
        name (str): name of network
        state_size (int or tuple): state space size
        action_size (int): action space size
        seed (int): random seed
    """
    
    if name == "dqn":
        net = DeepQNetwork(state_size, action_size, seed)
    elif name == "dropout-dqn":
        net = DropoutDeepQNetwork(state_size, action_size, seed)
    elif name == "dueling-dqn":
        net = DuelingDeepQNetwork(state_size, action_size, seed)
    else:
        raise NotImplementedError(f"Network ({name}) not found!")
        
    if torch.cuda.is_available():
        net = net.to("cuda")
        
    return net