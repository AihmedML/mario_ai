import warnings

warnings.filterwarnings("ignore")

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from wrappers import SkipFrame, GrayScale, FrameStack
from agent import MarioAgent

env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = SkipFrame(env)
env = GrayScale(env)
env = FrameStack(env)

agent = MarioAgent(num_actions=7)

episodes = 1000
update_target_every = 10

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.pick_action(state)
        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.learn()
        state = next_state
        total_reward += reward
        env.render()

    if episode % update_target_every == 0:
        agent.update_target()

    print(
        f"Episode {episode} | Reward: {total_reward:.0f} | Epsilon: {agent.epsilon:.3f}"
    )

env.close()
