import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(scores, names, ma=100, show=False):
    """Plots the score over the episodes
    
    Params
    ======
        scores (list): list of scores
        names (list): names of the methods used
        ma (int): moving average window
        show (bool): whether the figure should be displayed or saved as image
    """
    
    fig = plt.figure(figsize=(14, 10))
    for score, name in zip(scores, names):
        moving_average = np.convolve(score, np.ones(ma), 'valid') / ma
        if len(names) == 1:
            plt.plot(score, label=name)
            plt.plot([i for i in range(ma-1, len(score))], moving_average, label=f"mean({name})")
        else:
            plt.plot([i for i in range(ma-1, len(score))], moving_average, label=name)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    if len(names) == 1 and not show:
        plt.savefig(f"./data/{names[0]}/learning_curve.png")
    else:
        plt.show()
    plt.close(fig)
                    

def plot_avg_q_values(values, names, ma=100, show=False):
    """Plots the average q-values over the episodes
    
    Params
    ======
        values (list): list of q-values
        names (list): names of the methods used
        ma (int): moving average window
        show (bool): whether the figure should be displayed or saved as image
    """
    
    fig = plt.figure(figsize=(14, 10))
    for value, name in zip(values, names):
        moving_average = np.convolve(value, np.ones(ma), 'valid') / ma
        if len(names) == 1:
            plt.plot(value, label=name)
            plt.plot([i for i in range(ma-1, len(value))], moving_average, label=f"mean({name})")
        else:
            plt.plot([i for i in range(ma-1, len(value))], moving_average, label=name)
    plt.xlabel("Episode")
    plt.ylabel("Average Q-Values")
    plt.legend()
    if len(names) == 1 and not show:
        plt.savefig(f"./data/{names[0]}/avg_q_values.png")
    else:
        plt.show()
    plt.close(fig)
                               
def plot_avg_target_differences(values, names, ma=100, show=False):
    """Plots the average difference to the taget over the episodes
    
    Params
    ======
        values (list): list of differences to the target
        names (list): names of the methods used
        ma (int): moving average window
        show (bool): whether the figure should be displayed or saved as image
    """
    
    fig = plt.figure(figsize=(14, 10))
    for value, name in zip(values, names):
        moving_average = np.convolve(value, np.ones(ma), 'valid') / ma
        if len(names) == 1:
            plt.plot(value, label=name)
            plt.plot([i for i in range(ma-1, len(value))], moving_average, label=f"mean({name})")
        else:
            plt.plot([i for i in range(ma-1, len(value))], moving_average, label=name)
    plt.xlabel("Episode")
    plt.ylabel("Average Target Differences")
    plt.legend()
    if len(names) == 1 and not show:
        plt.savefig(f"./data/{names[0]}/avg_target_differences.png")
    else:
        plt.show()
    plt.close(fig)
    
def plot_test_q_values(values, rewards):
    """Plots q-values and rewards over one episode
    
    Params
    ======
        values (list): list of q-values
        rewards (list): list of rewards received
    """
    
    fig = plt.figure(figsize=(14, 10))
    for i, r in enumerate(rewards):
        if r == 1:
            plt.axvline(x=i, color='green')
        elif r == -1:
            plt.axvline(x=i, color='red')
    plt.plot(values)
    plt.xlabel("Steps")
    plt.ylabel("Q-Values")
    plt.show()
    plt.close(fig)