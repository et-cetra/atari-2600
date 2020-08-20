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
# This version of files includes double DQN, priority experience buffer, Canny edge detector wrapper, and dueling DQN

import time
import gym
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

import nnmodels
from lib import wrappers
from lib.helper import Agent, ExperienceBuffer, calc_loss, RewardBuffer, PriorityExperienceBuffer
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

# Hyper parameters for prioritised replay buffer
ALPHA_PRIORITY = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000


if __name__ == "__main__":
    start_time = time.time()
    print(time.strftime("%H:%M:%S", time.localtime()))

    # Additional command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--outpath", help="output model path")
    parser.add_argument("-m","--mode",help="mode: c for cpu, g for gpu-cuda")
    parser.add_argument("-md","--model",help="choose dqn model(int): 0-default(Dueling Network) 1-vanilla", type=int)
    parser.add_argument("-r","--reward",help="set the reward bound for specific env:", type=int)

    args = parser.parse_args()

    # ------------------------------
    # using the args

    env_name = DEFAULT_ENV_NAME

    # Make our device the GPU
    device = torch.device("cpu" if (args.mode is None) or (args.mode == "c") else "cuda")

    # Create the environment with wrappers
    env = wrappers.make_env(env_name, lives=False, fire=True)
    print("{} environment".format(env_name))
    print(env.unwrapped.get_action_meanings())
    if args.reward is not None:
        REWARD_BOUND = args.reward

    # Initialise the neural network, which will try to learn the Q values
    # Initialise the target network, which provides a copy of the network weights
    # from previous training iterations
    if args.model == 0 or args.model is None:
        print("Using Dueling Network")
        net = nnmodels.DuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
        tgtNet = nnmodels.DuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
    elif args.model == 1:
        net = nnmodels.VanillaDQN(env.observation_space.shape, env.action_space.n).to(device)
        tgtNet = nnmodels.VanillaDQN(env.observation_space.shape, env.action_space.n).to(device)
    # -------------------------

    # Create writer for tensorboard to keep track of variables
    writer = SummaryWriter()

    # Initialise experience buffer to hold past records of (s, a, r, s')
    # That is, (state, action, reward, new state)
    buffer = PriorityExperienceBuffer(REPLAY_SIZE, ALPHA_PRIORITY)

    # Create actual Agent that will be interacting with the environment
    agent = Agent(env, buffer)

    # Epsilon controls whether we randomly take an action or try to choose a good action from our neural network
    # Epsilon gradually decreases, so we choose less randomly as time goes on
    epsilon = EPSILON_START

    # Create optimiser to train the neural net
    optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Initialise reward buffer
    totalRewards = []
    bestMean = -99999
    rewardbuffer = RewardBuffer(REWARD_EPISODES)
    framesTotal = 0
    lastFrameTotal = 0
    lastTime = time.time()

    # Beta for prioritised experience buffer
    beta = BETA_START

    # MAIN TRAINING LOOP
    while True:
        framesTotal += 1

        # Decay epsilon
        epsilon = max(EPSILON_FINAL, EPSILON_START - (framesTotal)/(EPSILON_DECAY_LAST_FRAME))

        # Increase beta
        beta = min(1.0, BETA_START + framesTotal * (1.0 - BETA_START) / BETA_FRAMES)

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
            writer.add_scalar("beta", beta, framesTotal)
            if mean > bestMean:
                print("New best mean from {oldMean:f} {bestMean:f}, model saved".format(oldMean = bestMean, bestMean = mean))
                bestMean = mean
                # Save best model so far
                torch.save(net.state_dict(), (env_name + "-best.dat"))
            writer.add_scalar("best_mean", bestMean, framesTotal)

            # Check if we have "solved" the game
            if mean >= REWARD_BOUND:
                print("Solved in {nGames:d} games and {nFrames:d} frames!".format(nGames = len(totalRewards), nFrames = framesTotal))
                break

        # If our reward was None, then we have not finished the episode/game

        # Continue if buffer is not large enough to train on
        if len(buffer) < REPLAY_START_SIZE:
            continue

        # Sync our main network with the target network every SYNC_TARGET_FRAMES
        if framesTotal % SYNC_TARGET_FRAMES == 0:
            tgtNet.load_state_dict(net.state_dict())

        # Zero gradients
        optimiser.zero_grad()

        # Sample a random set of experiences from the experience buffer
        batch, batchIndices, batchWeights = buffer.sample(BATCH_SIZE, beta)

        # Calculate loss
        loss, samplePriorities = calc_loss(batch, batchWeights, net, tgtNet, GAMMA, device)
        writer.add_scalar("loss", loss, framesTotal)

        # Backpropogate and optimise to minimise the loss
        loss.backward()
        optimiser.step()

        # Update priorites in buffer based on the loss
        buffer.update_priorities(batchIndices, samplePriorities.data.cpu().numpy())

    # POST MAIN TRAINING LOOP

    # Export the trained model parameters to a file.
    SAVE_PATH = "./"+args.env+"saved_model.dat" if (args.env) else "./" + DEFAULT_ENV_NAME + "_saved_model.dat"
    net.eval()
    with torch.no_grad():
        torch.save(net.state_dict(), SAVE_PATH)
    writer.close()
    print("--- %s seconds ---" % (time.time() - start_time))

    