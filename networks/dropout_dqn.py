import torch
import torch.nn as nn
import torch.nn.functional as F


class DropoutDeepQNetwork(nn.Module):
    """Simple Deep Q-Network using dropout
    """
    
    def __init__(self, state_size, action_size, seed):
        """initializes network
    
        Params
        ======
            state_size (int or tuple): state space size
            action_size (int): action space size
            seed (int): random seed
        """
        
        super(DropoutDeepQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.linear1 = nn.Linear(state_size, 512)
        self.dropout1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(128, 32)
        self.linear4 = nn.Linear(32, action_size)

    def forward(self, state):
        """network forward pass
    
        Params
        ======
            state (array): state (input to network)
        """
        
        hidden_state = F.relu(self.dropout1(self.linear1(state)))
        hidden_state = F.relu(self.dropout2(self.linear2(hidden_state)))
        hidden_state = F.relu(self.linear3(hidden_state))
        action_values = self.linear4(hidden_state)
        
        return action_values