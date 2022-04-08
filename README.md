# Project 1: Navigation
by Luca Schweri, 2022

## Project Details

### Environment

The environment is a square map and the task is to navigate in this map to collect as many yellow bananas as possible.

### Rewards

- **+1**: collecting yellow banana
- **-1**: collecting blue banana

### Actions

- **0**: move forward
- **1**: move backward
- **2**: turn left
- **3**: turn right

### States

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.

### Solved

The environment is solved as soon as the agent is able to get an average score higher than 13 over 100 consecutive episode.


## Getting Started

To set up your python environment use the following commands:

```
conda create --name p1_udacity python=3.6
conda activate p1_udacity

pip install .
conda install -n p1_udacity pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

```

Download the Unity environment and change the path to the environment in the first code cell of the jupyter notebook [navigation.ipynb](navigation.ipynb):
- Linux: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

## Instructions

To train or test a new approach, first modify the [config.json](config.json) file and then open the jupyter notebook [navigation.ipynb](navigation.ipynb) by using the command
```
jupyter notebook
```

In the notebook you find three sections (which are better described in the jupyter notebook):
- **Trainig**: Can be used to train an agent.
- **Test**: Can be used to test an already trained agent.
- **Comparison**: Can be used to compare different approaches.

Note that you might need to change the location of the Unity environment in the first code cell.