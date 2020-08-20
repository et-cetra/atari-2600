#
# List of wrappers for OpenAI environments, compiled from:
#   https://github.com/openai/baselines
#

import cv2
import gym
import gym.spaces
import numpy as np
import collections


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """
        Do no-op action for a number of steps in [1, noop_max].
        """
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """
        Take action on reset for environments that are fixed until firing.
        """
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """
        Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


# Combines repetition of actions during K frames and pixels from two consecutive frames
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """
        Return only every `skip`-th frame
        """
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """
        Clear past frame buffer and init. to first obs. from inner env.
        """
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """
        Bin reward to {+1, 0, -1} by its sign.
        """
        return np.sign(reward)

DEF_H = 84
DEF_W = 110
RGB = 3
GREYSCALE = 1

# Converts input from (normal) 210 x 160 with RGB channels to a greyscale 84 x 84 image
# Resizes image and crops top and bottom of the result
class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, RGB]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, RGB]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114        # Turning image greyscale
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)

        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [DEF_H, DEF_H, GREYSCALE])                                # Resizing to 84 x 84
        return x_t.astype(np.uint8)


# Creates a stack of subsequent frames along the first dimension and returns them as an observation
# Purpose: give the network an idea about the dynamics of the object - speed and direction of movement
class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0), old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

## calculating the difference between frames: i.e. taking out some functionality from the nn
## base on BufferWrapper
## need further adj
class BufferDiffWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        x = np.zeros_like(obs,dtype=self.dtype)
        x[1,:] = obs[:-1]
        diff = obs - x
        return np.concat((obs,diff[1,:]),axis=0)

# Changes the shape of observations from HWC to CHW format required by PyTorch
# Input shape has a colour channel as the last dimension, PyTorch's convolution layers assume colour channel to be the first dimension
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

# Converts observation data from bytes to floats and scales every pixel's value between 0..1
class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

# Author: Sean Luo
# Image feature extraction: intelligent edge extraction from pixels
class CVWrapperMultFrameCannyOnly(gym.ObservationWrapper):
    def __init__(self, env, dtype=np.float32):
        super(CVWrapperMultFrameCannyOnly, self).__init__(env)
    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())
    def observation(self,obs):
        old_shape = obs.shape
        new_obs = np.zeros((old_shape[0],old_shape[1],old_shape[2]),dtype=np.float32)
        new_obs[0:old_shape[0],:,:] = obs
        for f in range(old_shape[0]):
            new_obs[f,:,:] = self.Canny(obs[f,:,:])
        return new_obs
    def Canny(self,obs):
        obs = obs.astype(np.uint8)
        sigma = 0.9
        v = np.median(obs)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(obs, lower, upper)
        return edged

# Main wrapper creation. Uses relevant wrappers.
def make_env(env_name, lives=False, fire=False):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = gym.wrappers.Monitor(env, "recording", force=True)
    if (fire == True):
        env = FireResetEnv(env)
    else:
        env = NoopResetEnv(env)
    env = ClipRewardEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    if (lives == True):
        env = EpisodicLifeEnv(env)
    env = ScaledFloatFrame(env)
    return env