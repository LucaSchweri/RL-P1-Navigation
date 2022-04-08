import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDeepQNetwork(nn.Module):
    """Simple Dueling Deep Q-Network
    """
    
    def __init__(self, state_size, action_size, seed):
        """initializes network
    
        Params
        ======
            state_size (int or tuple): state space size
            action_size (int): action space size
            seed (int): random seed
        """
        
        super(DuelingDeepQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.linear1 = nn.Linear(state_size, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear31 = nn.Linear(128, 32)
        self.linear32 = nn.Linear(128, 32)
        self.linear41 = nn.Linear(32, action_size)
        self.linear42 = nn.Linear(32, 1)

    def forward(self, state):
        """network forward pass
    
        Params
        ======
            state (array): state (input to network)
        """
        
        hidden_state = F.relu(self.linear1(state))
        hidden_state = F.relu(self.linear2(hidden_state))
        hidden_state1 = F.relu(self.linear31(hidden_state))
        advantage_values = self.linear41(hidden_state1)
        hidden_state2 = F.relu(self.linear32(hidden_state))
        state_value = self.linear42(hidden_state2)
        
        action_values = state_value + advantage_values
        
        return action_values