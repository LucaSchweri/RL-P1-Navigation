import numpy as np
import random
from collections import deque
import json

import torch
import torch.nn.functional as F
import torch.optim as optim

from importlib import reload
import networks.networks
reload(networks.networks)
from networks.networks import get_network

device = "cuda" if torch.cuda.is_available() else "cpu"

class DQNAgent():
    """DeepQ-network Agent
    
    Configurations (config.json)
    ======
        network (str): name of network
        lr (float): learning rate
        buffer_size (int): size of replay buffer
        batch_size (int): batch size
        update_net_steps (int): how often the agent should learn from experience replay
        discount_factor (float): discount factor
        epsilon (dict): contains all information about epsilon and its decay strategy
        target_ema (float): exponential moving average parameter for the target network
        double_q_learning (bool): whether or not the agent should use double Q-learning
        prioritized_experience_replay (bool): whether or not the agent should use prioritized experience replay
        state_perturbations (float): perturbation amount used on the states in the experience replay learning (if set 0 -> no state perturbations)
    """

    def __init__(self, config, state_size, action_size, seed):
        """initializes agent
    
        Params
        ======
            config (dict): agent configurations
            state_size (int or tuple): state space size
            action_size (int): action space size
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = np.random.seed(seed)
        
        # Hyperparameters
        self.batch_size = config["batch_size"]
        self.update_net_steps = config["update_net_steps"]
        self.tau = config["target_ema"]
        self.gamma = config["discount_factor"]
        self.double_q_learning = config["double_q_learning"]
        self.state_perturbations = config["state_perturbations"]

        # Networks
        self.net = get_network(config["network"], state_size, action_size, seed)
        self.target_net = get_network(config["network"], state_size, action_size, seed)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config["lr"])

        # Replay Memory
        if config["prioritized_experience_replay"]:
            self.replay_buffer = PrioritizedReplayBuffer(action_size, config["buffer_size"], config["batch_size"], seed)
        else:
            self.replay_buffer = ReplayBuffer(action_size, config["buffer_size"], config["batch_size"], seed)
        
        # epsilon
        self.epsilon = config["epsilon"]["start"]
        self.test_epsilon = config["epsilon"]["test"]
        if config["epsilon"]["decay"] == "constant":
            self.update_epsilon = lambda t: config["epsilon"]["start"]
        elif config["epsilon"]["decay"] == "linear":
            self.update_epsilon = lambda t: max((config["epsilon"]["start"] - config["epsilon"]["end"]) * (config["epsilon"]["to_end"] - t) / config["epsilon"]["to_end"] + config["epsilon"]["end"], config["epsilon"]["end"])
        elif config["epsilon"]["decay"] == "polynomial": 
            self.update_epsilon = lambda t: max(1 / (t + 1), config["epsilon"]["end"])
        elif config["epsilon"]["decay"] == "exponential":
            self.update_epsilon = lambda t: max(config["epsilon"]["factor"] ** t, config["epsilon"]["end"])
        elif config["epsilon"]["decay"] == "stepwise_linear":
            self.update_epsilon = lambda t: (config["epsilon"]["start"] - config["epsilon"]["mid"]) * (config["epsilon"]["to_mid"] - t) / config["epsilon"]["to_mid"] + config["epsilon"]["mid"] if t < config["epsilon"]["to_mid"] else max((config["epsilon"]["mid"] - config["epsilon"]["end"]) * (config["epsilon"]["to_end"] - t + config["epsilon"]["to_mid"]) / config["epsilon"]["to_end"] + config["epsilon"]["end"], config["epsilon"]["end"])
        else:
            raise NotImplementedError(f"Epsilon Decay Function ({config['epsilon']['decay']}) not found")
        self.t_step = 0
        self.episodes = 0
    
    def step(self, state, action, reward, next_state, is_done):
        """Saves SARS tuple in replay buffer and use experience replay when appropiate
    
        Params
        ======
            state (array): state
            action (int): action
            reward (float): reward
            next_state (array): next state
            is_done (bool): whether the episode has ended
        """
        
        # Save Experience
        self.replay_buffer.add(state, action, reward, next_state, is_done)
        
        # Learn
        target_diff = -1
        if (self.t_step + 1) % self.update_net_steps == 0 and len(self.replay_buffer) > self.batch_size:
            target_diff = self.learn()
            
        if is_done:
            self.episodes += 1
            self.epsilon = self.update_epsilon(self.episodes)
            
        self.t_step += 1
        
        return target_diff

    def act(self, state, test=False):
        """Returns action given state
    
        Params
        ======
            state (array): state
            test (bool): whether it is used in training or testing
        """
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.net.eval()
        with torch.no_grad():
            action_values = self.net(state)[0].detach().cpu().numpy()
        self.net.train()

        # Epsilon-greedy action selection
        if test:
            self.epsilon = self.test_epsilon
        probs = np.full((self.action_size,), self.epsilon / self.action_size)
        max_value = np.max(action_values)
        best_actions = np.argwhere(action_values == max_value)
        probs[best_actions] += (1 - self.epsilon) / len(best_actions)
        
        action = np.random.choice(np.arange(self.action_size), p=probs)
        return action, action_values[action]

    def learn(self):
        """Experience replay learning
    
        Params
        ======
        """
        states, actions, rewards, next_states, dones, weights = self.replay_buffer.sample()
        
        if self.state_perturbations > 0:
            perturbations = (torch.rand_like(states) * 2 * self.state_perturbations - self.state_perturbations).to(device)
            states += perturbations
            next_states += perturbations
        
        self.net.eval()
        self.target_net.eval()
        with torch.no_grad():
            if self.double_q_learning:
                best_actions = self.net(next_states).argmax(dim=1).unsqueeze(1)
                target = rewards + self.gamma * self.target_net(next_states).gather(1, best_actions)
            else:
                target = rewards + self.gamma * self.target_net(next_states).max(dim=1)[0].unsqueeze(1)
                    
            target[dones.bool()] = rewards[dones.bool()] 
        self.net.train()
        self.target_net.train()
                    
        prediction = self.net(states).gather(1, actions)
        
        target_diffs = np.abs((prediction - target)[:, 0].detach().cpu().numpy())
        
        if weights is None:
            loss_fcn = torch.nn.MSELoss()
            loss = loss_fcn(prediction, target)
        else:
            self.replay_buffer.update_priorities(target_diffs)
            loss = torch.sum(weights * (prediction[:, 0] - target[:, 0]) ** 2)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            
                
        # ------------------- update target network ------------------- #
        self.update_target()  
        
        return np.mean(target_diffs)
    
    def optimistic_init(self, states):
        """initializes network to always return 10 (optimistic value)
    
        Params
        ======
            states (list): a sample of possible states
        """
        
        optimizer = optim.Adam(self.net.parameters(), lr=0.01)
        
        states = torch.from_numpy(np.vstack(states)).float().to(device)
        
        for _ in range(100):
            prediction = self.net(states)
            target = torch.full_like(prediction, 10).to(device)

            loss_fcn = torch.nn.MSELoss()
            loss = loss_fcn(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(param.data)

    def update_target(self):
        """Exponential moving average of target network
    
        Params
        ======
        """
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(self.tau*param.data + (1.0-self.tau)*target_param.data)
            
    def save(self, name):
        """Saves network parameters
    
        Params
        ======
            name (str): method name
        """
        
        torch.save(self.net.state_dict(), f"./data/{name}/net_checkpoint.pth")
        torch.save(self.target_net.state_dict(), f"./data/{name}/target_checkpoint.pth")
    
    def load(self, name):
        """Loads network parameters
    
        Params
        ======
            name (str): method name
        """
        
        self.net.load_state_dict(torch.load(f"./data/{name}/net_checkpoint.pth", map_location=device))
        self.target_net.load_state_dict(torch.load(f"./data/{name}/target_checkpoint.pth", map_location=device))

class ReplayBuffer:
    """Normal replay buffer
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initalizes replay buffer
    
        Params
        ======
            action_size (int): action size
            buffer_size (int): buffer size
            batch_size (int): batch size
            seed (int): random seed
        """
        
        self.buffer = deque(maxlen=int(buffer_size))  
        self.batch_size = batch_size
        self.seed = np.random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Adds tuple to replay buffer
    
        Params
        ======
            state (array): state
            action (int): action
            reward (int): reward
            next_state (array): next state
            done (bool): whether th episode has ended
        """
        
        elem = [state, action, reward, next_state, done]
        self.buffer.append(elem)
    
    def sample(self):
        """Get a experience sample from the replay buffer
    
        Params
        ======
        """
        
        idx = np.random.choice([i for i in range(len(self.buffer))], size=self.batch_size)
        
        experiences = [self.buffer[int(i)] for i in idx]

        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones, None)

    def __len__(self):
        return len(self.buffer)
    
class PrioritizedReplayBuffer:
    """Prioritized replay buffer
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initalizes replay buffer
    
        Params
        ======
            action_size (int): action size
            buffer_size (int): buffer size
            batch_size (int): batch size
            seed (int): random seed
        """
        
        self.buffer = deque(maxlen=int(buffer_size))  
        self.batch_size = batch_size
        self.seed = np.random.seed(seed)
        self.e = 1e-3
        self.alpha = 0.8
        self.beta = 0.8
        self.max_priority = 1
    
    def add(self, state, action, reward, next_state, done):
        """Adds tuple to replay buffer
    
        Params
        ======
            state (array): state
            action (int): action
            reward (int): reward
            next_state (array): next state
            done (bool): whether th episode has ended
        """
        
        elem = [state, action, reward, next_state, done, self.max_priority]
        self.buffer.append(elem)
    
    def sample(self):
        """Get a experience sample from the replay buffer
    
        Params
        ======
        """
        
        priorities = np.array([e[-1] ** self.alpha for e in self.buffer])
        priorities /= np.sum(priorities)
        
        self.idx = np.random.choice([i for i in range(len(self.buffer))], size=self.batch_size, p=priorities)
        
        experiences = [self.buffer[int(i)] for i in self.idx]

        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(device)
        
        weights = torch.from_numpy(1 / ((len(self) * priorities[self.idx]) ** self.beta)).float().to(device)
        weights /= torch.sum(weights)
  
        return (states, actions, rewards, next_states, dones, weights)

    def update_priorities(self, new_deltas):
        """Update priorities
    
        Params
        ======
            new_deltas (array): differences to target
        """
        
        new_deltas += self.e
        if np.max(new_deltas) > self.max_priority:
            self.max_priority = np.max(new_deltas)
        
        for i, d in zip(self.idx, new_deltas):
            self.buffer[int(i)][-1] = d

    def __len__(self):
        return len(self.buffer)