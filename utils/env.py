from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import gym
import time
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import dmc2gym

from typing import Any, List, Union, Optional, Callable
import numpy as np
import torch
from gym.spaces import Box
import cv2
from torchvision.transforms import ColorJitter
from collections import deque
import dmc2gym
from stable_baselines3.common.preprocessing import is_image_space

def make_dmc_aug_env(args, render_size, source=True):
    if source:
        env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=True,
        height=render_size,
        width=render_size,
        frame_skip=args.action_repeat
        )
    else:
        env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.target_task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=True,
        height=render_size,
        width=render_size,
        frame_skip=args.action_repeat
        )
    env = FrameStack(env, k=args.frame_stack)
    return env


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


# copied from RAD, modified for 3 dims
def random_crop(imgs, out=84):
    """
        args:
        imgs: np.array shape (C,H,W)
        out: output size (e.g. 84)
        returns np.array
    """
    c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max)
    h1 = np.random.randint(0, crop_max)
    cropped = np.empty((c, out, out), dtype=imgs.dtype)
    cropped = imgs[:,h1:h1 + out, w1:w1 + out]
    return cropped

def random_translate(imgs, size, return_random_idxs=False, h1s=None, w1s=None):
    c, h, w = imgs.shape
    assert size >= h and size >= w
    outs = np.zeros((c, size, size), dtype=imgs.dtype)
    h1 = np.random.randint(0, size - h + 1) if h1s is None else h1s
    w1 = np.random.randint(0, size - w + 1) if w1s is None else w1s
    outs[:,h1:h1 + h, w1:w1 + w] = imgs
    if return_random_idxs:  # So can do the same to another set of imgs.
        return outs, dict(h1s=h1s, w1s=w1s)
    return outs


def center_crop_images(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image

def random_crop_vec(imgs, out=84):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        out: output size (e.g. 84)
        returns np.array
    """
    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, out, out), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped


def center_crop_vec(image, output_size):
    h, w = image.shape[2:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, :, top:top + new_h, left:left + new_w]
    return image

def random_translate_vec(imgs, size, return_random_idxs=False, h1s=None, w1s=None):
    n, c, h, w = imgs.shape
    assert size >= h and size >= w
    outs = np.zeros((n, c, size, size), dtype=imgs.dtype)
    h1s = np.random.randint(0, size - h + 1, n) if h1s is None else h1s
    w1s = np.random.randint(0, size - w + 1, n) if w1s is None else w1s
    for out, img, h1, w1 in zip(outs, imgs, h1s, w1s):
        out[:, h1:h1 + h, w1:w1 + w] = img
    if return_random_idxs:  # So can do the same to another set of imgs.
        return outs, dict(h1s=h1s, w1s=w1s)
    return outs

def center_translate_vec(imgs, size):
    n, c, h, w = imgs.shape
    assert size >= h and size >= w
    out = np.zeros((n, c, size, size), dtype=imgs.dtype)
    h1 = (size - h + 1)//2
    w1 = (size - w + 1)//2
    out[:, :, h1:h1 + h, w1:w1 + w] = imgs
    return out


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cartpole')
    parser.add_argument('--task_name', default='swingup')
    parser.add_argument('--pre_transform_image_size', default=100, type=int)
    parser.add_argument('--seed', default=0)
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--encoder_type', default='pixel', type=str)

    args = parser.parse_args()

    env = make_dmc_env(args)
    print(is_image_space(env.observation_space))
    print(env.action_space)
    print(env.observation_space)
    obs = env.reset()
    for _ in range(50):
        obs,_,_,_ = env.step(env.action_space.sample())
    print(obs.shape)
    print(obs.max())
    print(obs.min())
    print(np.mean(obs))
    #cv2.imwrite('test3.png',np.transpose(obs,(1,2,0)))