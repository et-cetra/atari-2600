#
# Code assembled and tweaked by:
#   Joshua Derbe
#   Sean Luo
#   Mariya Shmalko
#   Arkie Owen
#
#
# Base algorithm outline created with heavy reference to
#   Lapan, M. (2018) Deep Reinforcement Learning Hands-On, Packt Publishing.
#
# This version of files is the vanilla DQN with no extensions

import time

import gym
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

import nnmodels
from lib import wrappers
from lib.helper import Agent, ExperienceBuffer, calc_loss, RewardBuffer
from tensorboardX import SummaryWriter
import argparse
from gym import spaces

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
REWARD_BOUND = 18
REWARD_EPISODES = 100

# Hyper parameters
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 100000
LEARNING_RATE = 0.0001
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

if __name__ == "__main__":
    start_time = time.time()
    print(time.strftime("%H:%M:%S", time.localtime()))

    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--outpath", help="output model path")
    parser.add_argument("-m","--mode",help="mode: c for cpu, g for gpu-cuda")
    parser.add_argument("-md","--model",help="choose dqn model(int): 0-default 1-resnet 2-a2c", type=int)
    parser.add_argument("-r","--reward",help="set the reward bound for specific env:", type=int)
    
    args = parser.parse_args()

    # ------------------------------
    # using the args

    # Make our device the GPU
    device = torch.device("cpu" if (args.mode is None) or (args.mode == "c") else "cuda")

    # Create the environment with wrappers
    env = wrappers.make_env(DEFAULT_ENV_NAME, lives=False, fire=True)
    print(env.unwrapped.get_action_meanings())
    if args.reward is not None:
        REWARD_BOUND = args.reward

    # Initialise the neural network, which will try to learn the Q values
    # Initialise the target network, which provides a copy of the network weights
    # from previous training iterations
    if args.model == 0 or args.model is None:
        print("Using vanilla network")
        net = nnmodels.VanillaDQN(env.observation_space.shape, env.action_space.n).to(device)
        tgtNet = nnmodels.VanillaDQN(env.observation_space.shape, env.action_space.n).to(device)
    # -------------------------

    # Create writer for tensorboard to keep track of variables
    writer = SummaryWriter()

    # Create experience buffer to hold past records of (s, a, r, s')
    # That is, (state, action, reward, new state)
    buffer = ExperienceBuffer(REPLAY_SIZE)

    # Create actual Agent that will be interacting with the environment
    agent = Agent(env, buffer)

    # Epsilon will control whether we randomly take an action or try to choose a good action from our neural network
    # Epsilon will gradually decrease, so we choose less and less randomly as time goes on
    epsilon = EPSILON_START

    # Create optimiser to train the neural net
    optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    totalRewards = []
    bestMean = -99999
    rewardbuffer = RewardBuffer(REWARD_EPISODES)
    framesTotal = 0
    lastFrameTotal = 0
    lastTime = time.time()

    # MAIN TRAINING LOOP
    while True:
        framesTotal += 1
        
        # Decay epsilon
        epsilon = max(EPSILON_FINAL, EPSILON_START - (framesTotal)/(EPSILON_DECAY_LAST_FRAME))

        # Choose an action and perform a step, and obtain the reward from that step
        reward = agent.play_step(net, epsilon, device=device)

        # Reward is not None, we have finished the episode/game, reward is total reward for the episode
        if reward is not None:
            totalRewards.append(reward)
            rewardbuffer.append(reward)
            mean = rewardbuffer.mean()

            speed = (framesTotal - lastFrameTotal) / (time.time() - lastTime)
            lastFrameTotal = framesTotal
            lastTime = time.time()

            # Display meta information
            print("{nFrames:d}: finished {nGames:d} games, reward for this game is {reward:f}. Mean: {mean:f}. Speed: {speed:.2f} f/s".format(nFrames = framesTotal, nGames = len(totalRewards), reward = reward, mean = mean, speed = speed))

            # Report variables in tensorboard
            writer.add_scalar("epsilon", epsilon, framesTotal)
            writer.add_scalar("reward", reward, framesTotal)
            writer.add_scalar("mean", mean, framesTotal)
            writer.add_scalar("speed", speed, framesTotal)
            if mean > bestMean:
                print("New best mean from {oldMean:f} {bestMean:f}, model saved".format(oldMean = bestMean, bestMean = mean))
                bestMean = mean
                torch.save(net.state_dict(), (DEFAULT_ENV_NAME + "-best.dat"))
            writer.add_scalar("best_mean", bestMean, framesTotal)

            # Check if we have "solved" the game
            if mean >= REWARD_BOUND:
                print("Solved in {nGames:d} games and {nFrames:d} frames!".format(nGames = len(totalRewards), nFrames = framesTotal))
                break

        # If our reward was None, then we have not finished the episode/game

        if len(buffer) < REPLAY_START_SIZE:
            # If our buffer is not large enough to train on yet, repeat the loop
            continue

        if framesTotal % SYNC_TARGET_FRAMES == 0:
            # Sync our main network with the target network every SYNC_TARGET_FRAMES
            tgtNet.load_state_dict(net.state_dict())

        # Zero gradients
        optimiser.zero_grad()

        # Sample a random set of experiences from the experience buffer
        batch = buffer.sample(BATCH_SIZE)

        # Calculate loss
        loss = calc_loss(batch, net, tgtNet, GAMMA, device)
        writer.add_scalar("loss", loss, framesTotal)

        # Backpropogate and optimise to minimise the loss
        loss.backward()
        optimiser.step()
    
    # POST MAIN TRAINING LOOP

    # Export the trained model parameters to a file.
    SAVE_PATH = "./"+args.env+"saved_model.dat" if (args.env) else "./" + DEFAULT_ENV_NAME + "_saved_model.dat"
    net.eval()
    with torch.no_grad():
        torch.save(net.state_dict(), SAVE_PATH)
    writer.close()
    print("--- %s seconds ---" % (time.time() - start_time))

    