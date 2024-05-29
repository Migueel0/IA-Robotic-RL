import mdptoolbox.mdp as mdp
import numpy as np
import matplotlib.pyplot as plt


actions = ['esperar','N','NE','E','SE','S','SO','O','NO']

def read_stage(data):
    
    with open(data,'r') as file:
        lines = file.readlines()
    numbers = [float(num) for num in lines[0].split()]
    lines.pop(0)
    lines.reverse()
    matrix = []
    for line in lines:
        row = [int(char) for char in line.strip()]
        matrix.append(row)
    return np.array(matrix),(numbers[0],numbers[1])

def view_stage(stage,goal):
    plt.figure(figsize=(len(stage[0]), len(stage)))
    plt.imshow(1-stage, cmap='gray', interpolation='none')
    plt.xlim(-0.5, len(stage[0]) - 0.5)
    plt.ylim(-0.5, len(stage) - 0.5)
    plt.gca().add_patch(plt.Circle(goal,radius = 0.5,edgecolor = 'red', facecolor = 'red'))
    return plt


def calculate_states(stage):
    states = []
    for i in range(0,stage.shape[1]):
        for j in range(0,stage.shape[0]):
            states.append(tuple([i,j]))
    return states

def collision(state,stage):
    return stage[state[1],state[0]]==1

def apply_action(stage,state,action):
    if collision(state,stage) or action not in actions:
        return state
    x = state[0]
    y = state[1]
    
    if action == 'N':
        y += 1
    elif action == 'S':
        y -= 1
    elif action == 'E':
        x += 1
    elif action == 'O':
        x -= 1
    elif action == 'NE':
        y += 1
        x += 1
    elif action == 'SE':
        y -= 1
        x += 1
    elif action == 'SO':
        y -= 1
        x -= 1
    elif action == 'NO':
        y += 1
        x -= 1
    return x,y


def get_reward(stage,state,goal,K):
    if collision(state,stage):
        value = -K
    else:
        value = - np.sqrt( (state[0]-goal[0])**2 + (state[1]-goal[1])**2)
    return value

def view_rewards(stage,states,goal,K):
    
    plt = view_stage(stage,goal)
    rewards = [get_reward(stage,state,goal,K) for state in states]
    rewards = [np.nan if element == -1000 else element for element in rewards]
    max_reward = np.nanmax(rewards)
    min_reward = np.nanmin(rewards)
    for state in states:
        r = get_reward(stage,state,goal,K)
        if r == -1000:
            continue
        a = (r-min_reward)/(max_reward-min_reward)
        rect = plt.Rectangle((state[0] - 0.5, state[1] - 0.5), 1, 1, alpha = a,linewidth=1, edgecolor='blue', facecolor='blue')
        plt.gca().add_patch(rect)
    return plt







