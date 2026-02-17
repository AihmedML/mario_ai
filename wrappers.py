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

    class FrameStack:
        def __init__(self, env, stack_size=4):
            self.env = env
            self.stack_size = stack_size
            self.frames = deque(maxlen=stack_size)

        def reset(self):
            obs = self.env.reset()
            for i in range(self.stack_size):
                self.frames.append(obs)
            return numpy.array(self.frames)

        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            self.frames.append(obs)
            return numpy.array(self.frames), reward, done, info

        def render(self):
            return self.env.render()
