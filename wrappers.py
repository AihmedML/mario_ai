import gym
import numpy
import cv2
from collections import deque


class SkipFrame:
    def __init__(self, env, skip=4):
        self.env = env
        self.skip = skip

    def step(self, action):
        total_reward = 0
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def __getattr__(self, name):
        return getattr(self.env, name)


class GrayScale:
    def __init__(self, env):
        self.env = env

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84))
        obs = numpy.array(obs, dtype=numpy.float32) / 255.0
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84))
        obs = numpy.array(obs, dtype=numpy.float32) / 255.0
        return obs

    def render(self):
        return self.env.render()

    def __getattr__(self, name):
        return getattr(self.env, name)


class FrameStack:
    def __init__(self, env, stack_size=4):
        self.env = env
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.stack_size):
            self.frames.append(obs)
        return numpy.array(self.frames)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return numpy.array(self.frames), reward, done, info

    def render(self):
        return self.env.render()

    def __getattr__(self, name):
        return getattr(self.env, name)


class RAMMap:
    """Processes game frame into clean 84x84 mini-map with Mario, enemies, ground."""

    def __init__(self, env):
        self.env = env

    def _build_map(self, obs):
        # obs is the raw RGB frame (240, 256, 3)
        grid = numpy.zeros((84, 84), dtype=numpy.float32)

        # shrink frame to 84x84
        small = cv2.resize(obs, (84, 84))

        # detect ground/blocks (brown/orange: high red, medium green, low blue)
        r, g, b = small[:,:,0], small[:,:,1], small[:,:,2]

        # ground and bricks (brownish colors)
        ground = ((r > 150) & (g < 150) & (b < 100)) | ((r > 180) & (g > 100) & (b < 80))
        grid[ground] = 0.3

        # question blocks and coins (yellow/orange)
        yellow = (r > 200) & (g > 150) & (b < 100)
        grid[yellow] = 0.5

        # pipes (green)
        green = (g > 150) & (r < 150) & (b < 150)
        grid[green] = 0.4

        # mario (red tones — his hat and shirt)
        mario = (r > 180) & (g < 80) & (b < 80)
        grid[mario] = 1.0

        # enemies (goombas are brown, koopas are green)
        # enemies are small moving things — detect brownish at specific positions
        enemy_brown = (r > 100) & (r < 180) & (g > 50) & (g < 120) & (b < 60)
        grid[enemy_brown] = numpy.maximum(grid[enemy_brown], 0.7)

        # sky stays 0 (black)
        return grid

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        grid = self._build_map(obs)
        return grid, reward, done, info

    def reset(self):
        obs = self.env.reset()
        grid = self._build_map(obs)
        return grid

    def render(self):
        return self.env.render()

    def __getattr__(self, name):
        return getattr(self.env, name)


class RewardShaper:
    """Adds reward shaping: bonus for moving right, penalty for dying."""

    def __init__(self, env):
        self.env = env
        self.last_x = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        x_pos = info.get("x_pos", 0)
        # bonus for moving right
        reward += (x_pos - self.last_x) * 0.5
        # penalty for dying
        if done and info.get("life", 0) < 2:
            reward -= 50
        self.last_x = x_pos
        return obs, reward, done, info

    def reset(self):
        self.last_x = 0
        return self.env.reset()

    def render(self):
        return self.env.render()

    def __getattr__(self, name):
        return getattr(self.env, name)


class StuckDetector:
    def __init__(self, env, max_no_progress=500):
        self.env = env
        self.max_no_progress = max_no_progress
        self.best_x = 0
        self.no_progress_count = 0

    def reset(self):
        self.best_x = 0
        self.no_progress_count = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        x_pos = info.get("x_pos", 0)
        if x_pos > self.best_x:
            self.best_x = x_pos
            self.no_progress_count = 0
        else:
            self.no_progress_count += 1
        if self.no_progress_count >= self.max_no_progress:
            done = True
        return obs, reward, done, info

    def render(self):
        return self.env.render()

    def __getattr__(self, name):
        return getattr(self.env, name)
