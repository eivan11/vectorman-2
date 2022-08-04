import time
import retro
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
#from IPython.display import clear_output
import math
from statistics import mean
import argparse
import base64
import imageio
import IPython
import pygame
import imutils
import cv2

import sys
# sys.path.append('../../')
from dqn_agent import DQNAgent
from dqn_cnn import DQNCnn
from stack_frame import preprocess_frame, stack_frame
from csv_utils import data_write_csv







  
# See the environment properties
def viewEnvironment():
    print("The size of frame is: ", env.observation_space.shape)
    print("No. of Default Actions: ", env.action_space.n)

    env.reset()
    plt.figure()
    plt.axis('off')
    plt.imshow(env.reset())
    plt.title('Original Frame')
    plt.show()


def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (1, -1, -1, 1), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames


def test_agent(n_timesteps, epsilon, n_episodes):
    print("Testing Starts")
    # env.viewer = None
    start_epoch = 0
    success_count = 0

    for i_episode in range(start_epoch + 1, n_episodes + 1):
        state = stack_frames(None, env.reset(), True)
        score = 0
        timestamp = 0
        while timestamp < n_timesteps:
            env.render()
            action = agent.act_test(state, eps=epsilon)
            next_state, reward, done, info = env.step(possible_actions[action])

            timestamp += 1
            # print(timestamp)
            score += reward
            state = stack_frames(state, next_state, False)
            if done:
                break

        if score >= 9000:
            success_count += 1
        success_rate = (success_count/i_episode) * 100

        print("\rEpisode: {}\tsuccess rate: {:.2f}%".format(i_episode, success_rate), end="")
    # env.render(close=True)

def epsilon_by_episode(frame_idx, eps_start, eps_end, eps_decay):
    epsilon = eps_end + (eps_start - eps_end) * math.exp(-1. * frame_idx / eps_decay)
    return epsilon

def train(n_episodes, eps_start, eps_end, eps_decay):
    print("Training Starts")
    start_epoch = 0
    scores_window = deque(maxlen=20) # For storing Moving Average of the Score/Reward
    score_history = [[]]


    for i_episode in range(start_epoch + 1, n_episodes + 1):
        state = stack_frames(None, env.reset(), True)
        score = 0
        eps = epsilon_by_episode(i_episode, eps_start, eps_end, eps_decay)

        # Punish the agent for not moving forward
        x_area = deque(maxlen=100)
        area_stuck = 0
        prev_avg = 0

        timestamp = 0

        while timestamp < 10000:
            env.render()
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(possible_actions[action])
            x_area.append(info["score"])
            x_area_avg = mean(x_area)
            timestamp += 1

            # Punish the agent for getting stuck in an area for too long
            if prev_avg - 1 <= x_area_avg <= prev_avg + 1:
                area_stuck += 1
            else:
                area_stuck = 0

            prev_avg = x_area_avg
            if area_stuck > 100:
                reward -= 1

            score += reward
            next_state = stack_frames(state, next_state, False)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        scores_window.append(score)  # save most recent score
        agent.save_networks(i_episode, n_episodes)

        score_data = [i_episode, score, np.mean(scores_window), eps]
        score_history.append(score_data)
        data_write_csv("./csv/reward_history.csv", score_history)

        print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, score, np.mean(scores_window), eps))

    # return scores

def str2bool(value, raise_exc=False):
    _true_set = {'yes', 'true', 't', 'y', '1'}
    _false_set = {'no', 'false', 'f', 'n', '0'}
    if isinstance(value, str):
        value = value.lower()
        if value in _true_set:
            return True
        if value in _false_set:
            return False

    if raise_exc:
        raise ValueError('Expected "%s"' % '", "'.join(_true_set | _false_set))
    return None

if __name__ == "__main__":
    # Create the environment using gym-retro
    env = retro.make('Vectorman2-Genesis', 'Level2')
    env.seed(0)

    # Set GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Modified Possible Actions
    possible_actions = {
        # No Operation
        0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Left
        1: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        # Right
        2: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        # Left, Down
        3: [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        # Right, Down
        4: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        # Down
        5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        # Down, B
        6: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        # B
        7: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }

    # Creating Agent
    INPUT_SHAPE = (4, 84, 84)
    ACTION_SIZE = len(possible_actions)
    SEED = 0
    GAMMA = 0.99  # discount factor
    BUFFER_SIZE = 100000  # replay buffer size
    BATCH_SIZE = 32  # Update batch size
    LR = 0.0001  # learning rate
    TAU = 1e-3  # for soft update of target parameters
    UPDATE_EVERY = 100  # how often to update the network
    UPDATE_TARGET = 10000  # After which threshold replay to be started
    EPS_START = 0.99  # starting value of epsilon
    EPS_END = 0.01  # Ending value of epsilon
    EPS_DECAY = 100  # Rate by which epsilon to be decayed
    SAVE_EVERY = 10 # Save the model every n episode
    ROOT_DIR = "./models/"

    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train_mode", help="Whether to train or test the model", default=False)
    parser.add_argument("-lm", "--load_model", help="Whether to Load Pre-Trained Policy & Target Network or not", default=False)
    parser.add_argument("-mn", "--model_name", help="Name of the policy & target network, separated by comma (,). No spaces",
                        default="modified-reward_policy_net,modified-reward_target_net")

    args = parser.parse_args()
    train_mode = str2bool(args.train_mode)
    LOAD_MODEL = str2bool(args.load_model)

    MODEL_NAME = args.model_name.split(",")

    agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY,
                     UPDATE_TARGET, DQNCnn, LOAD_MODEL, SAVE_EVERY, ROOT_DIR, MODEL_NAME)

    if train_mode:
        train(1000, EPS_START, EPS_END, EPS_DECAY) # Train
    else:
        test_agent(10000, 0.01, 100) # Test
