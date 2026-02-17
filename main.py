import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time

env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

env.reset()
for step in range(5000):
    action = env.action_space.sample()
    env.step(action)
    env.render()
    time.sleep(0.02)

env.close()
