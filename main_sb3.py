"""
Stable-Baselines3 PPO for Mario — for comparison with our custom PPO.
Install: pip install stable-baselines3==1.8.0

Run: python main_sb3.py
"""
import warnings
warnings.filterwarnings("ignore")

import gym
from gym import spaces
import numpy as np
import cv2
from collections import deque
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

NUM_ENVS = 32


class MarioWrapper(gym.Wrapper):
    """All-in-one wrapper that SB3 can understand."""

    def __init__(self, env, skip=4, stack_size=4, max_no_progress=500):
        super().__init__(env)
        self.skip = skip
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
        self.max_no_progress = max_no_progress
        self.best_x = 0
        self.no_progress_count = 0
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(stack_size, 84, 84), dtype=np.float32
        )

    def _process_frame(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84))
        return np.array(resized, dtype=np.float32) / 255.0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.best_x = 0
        self.no_progress_count = 0
        frame = self._process_frame(obs)
        for _ in range(self.stack_size):
            self.frames.append(frame)
        return np.array(self.frames)

    def step(self, action):
        total_reward = 0
        done = False
        info = {}
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        # stuck detection
        x_pos = info.get("x_pos", 0)
        if x_pos > self.best_x:
            self.best_x = x_pos
            self.no_progress_count = 0
        else:
            self.no_progress_count += 1
        if self.no_progress_count >= self.max_no_progress:
            done = True

        frame = self._process_frame(obs)
        self.frames.append(frame)
        # scale rewards to [-1, 1] like custom PPO
        scaled_reward = np.clip(total_reward / 15.0, -1.0, 1.0)
        return np.array(self.frames), scaled_reward, done, info


class PrintCallback(BaseCallback):
    """Print episode rewards like our custom training loop."""

    def __init__(self):
        super().__init__()
        self.episode_count = 0
        self.best_reward = float("-inf")

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                reward = info["episode"]["r"]
                if reward > self.best_reward:
                    self.best_reward = reward
                self.episode_count += 1
                print(
                    f"Episode {self.episode_count} | "
                    f"Reward: {reward:.0f} | "
                    f"Best: {self.best_reward:.0f} | "
                    f"Steps: {self.num_timesteps}"
                )
        return True


def make_env():
    def _init():
        env = gym_super_mario_bros.make("SuperMarioBros-v0")
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = MarioWrapper(env)
        env = Monitor(env)
        return env
    return _init


if __name__ == "__main__":
    # DummyVecEnv = sequential (safer on Windows with NES emulator)
    env = DummyVecEnv([make_env() for _ in range(NUM_ENVS)])

    policy_kwargs = dict(normalize_images=False)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=0,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        n_epochs=3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.1,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./mario_sb3_logs/",
    )

    print(f"Training with SB3 PPO | {NUM_ENVS} envs | {model.device}")
    print("Same hyperparams as our custom PPO")
    print("-" * 50)

    model.learn(total_timesteps=2_000_000, callback=PrintCallback())
    model.save("mario_sb3_model")
    env.close()
    print("Done! Model saved to mario_sb3_model.zip")
