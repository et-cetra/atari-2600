import time
import gym
import numpy as np
import collections
import torch
from nnmodels import DuelingDQN
import argparse
from lib import wrappers
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
DEFAULT_MODEL_NAME = "PongNoFrameskip-v4-best.dat"

FPS = 30

# Runs the trained model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--mode",help="mode: c for cpu, g for gpu-cuda")
    parser.add_argument("-md", "--model", required=False, help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    args = parser.parse_args()

    env_name = DEFAULT_ENV_NAME if args.env is None else args.env

    device = torch.device("cpu" if (args.mode is None) or (args.mode == "c") else "cuda")

    env = wrappers.make_env(env_name, lives=True, fire=True)
    print("{} environment".format(env_name))
    print(env.unwrapped.get_action_meanings())

    model = DEFAULT_MODEL_NAME if (args.model is None) else args.model
    net = DuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
    net.load_state_dict(torch.load(model, map_location=torch.device(device)))
    total_reward = 0.0
    total_steps = 0
    obs = env.reset()
    
    net.eval()
    env.step(1)
    with torch.no_grad():
        while True:
            start_ts = time.time()
            env.render()
            _, act_v = torch.max(net(torch.tensor(np.array([obs], copy=False)).to(device)),dim=1)
            action = int(act_v.item())
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            total_steps += 1
            if done:
                env.reset()
                env.step(1)
            delta = 1/FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)

    print("Episode done in %d steps, total reward %.2f" % (total_steps, total_reward))
    env.close()
    env.env.close()