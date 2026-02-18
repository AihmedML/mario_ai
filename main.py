import warnings

warnings.filterwarnings("ignore")

import os
import numpy
import cv2
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from wrappers import SkipFrame, GrayScale, FrameStack, StuckDetector
from agent import MarioAgent
from visualize import Visualizer

NUM_ENVS = 32
ENABLE_RENDER = False
ENABLE_VIZ = False
VIZ_EVERY_N_FRAMES = 4
USE_RIGHT_ONLY_ACTIONS = True


class VecEnv:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.envs = []
        action_set = RIGHT_ONLY if USE_RIGHT_ONLY_ACTIONS else SIMPLE_MOVEMENT
        for _ in range(num_envs):
            env = gym_super_mario_bros.make("SuperMarioBros-v0")
            env = JoypadSpace(env, action_set)
            env = SkipFrame(env)
            env = GrayScale(env)
            env = FrameStack(env)
            env = StuckDetector(env, max_no_progress=200)
            self.envs.append(env)

    def reset(self):
        return numpy.array([env.reset() for env in self.envs])

    def step(self, actions):
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, done, info = env.step(int(action))
            if done:
                obs = env.reset()
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return (
            numpy.array(obs_list),
            numpy.array(reward_list),
            numpy.array(done_list),
            info_list,
        )

    def render(self, idx=0):
        self.envs[idx].render()

    def close(self):
        for env in self.envs:
            env.close()


if __name__ == "__main__":
    action_set = RIGHT_ONLY if USE_RIGHT_ONLY_ACTIONS else SIMPLE_MOVEMENT
    action_names = [" + ".join(combo) if combo else "NOOP" for combo in action_set]
    viz = Visualizer(action_names) if ENABLE_VIZ else None

    env = VecEnv(NUM_ENVS)
    agent = MarioAgent(num_actions=len(action_names), num_envs=NUM_ENVS)

    # auto-load best checkpoint if it exists
    if os.path.exists("mario_best.pth"):
        agent.load("mario_best.pth")
        print(f"  ** Loaded checkpoint mario_best.pth (step {agent.step_count}, best_avg {getattr(agent, 'best_avg', 0):.0f}) **")
    elif os.path.exists("mario_model.pth"):
        agent.load("mario_model.pth")
        print(f"  ** Loaded checkpoint mario_model.pth (step {agent.step_count}, best_avg {getattr(agent, 'best_avg', 0):.0f}) **")

    states = env.reset()
    episode = 0
    episode_rewards = numpy.zeros(NUM_ENVS)
    frame_count = 0

    # collapse detection + auto-recovery + auto-tuning
    recent_rewards = []
    best_avg = getattr(agent, "best_avg", 0)
    collapse_patience = 50  # check every 50 episodes
    collapse_threshold = 0.5  # if avg drops below 50% of best, reload
    plateau_count = 0  # how many checks without improvement

    while episode < 10000:
        actions = agent.pick_action(states)
        next_states, rewards, dones, infos = env.step(actions)
        agent.remember(rewards, dones, infos)

        states = next_states
        episode_rewards += rewards
        frame_count += 1

        if ENABLE_RENDER:
            env.render(0)
        if ENABLE_VIZ and frame_count % VIZ_EVERY_N_FRAMES == 0:
            viz.show(states[0], agent, episode, episode_rewards[0])

        # handle done episodes
        for i in range(NUM_ENVS):
            if dones[i]:
                if ENABLE_VIZ:
                    viz.end_episode(episode_rewards[i])
                print(
                    f"Episode {episode} | Env {i} | Reward: {episode_rewards[i]:.0f} | Steps: {agent.step_count}"
                )
                recent_rewards.append(episode_rewards[i])
                episode_rewards[i] = 0
                episode += 1

                if episode % 50 == 0:
                    agent.best_avg = best_avg
                    agent.save()

                # collapse detection every 50 episodes (after warmup)
                if (
                    len(recent_rewards) >= collapse_patience
                    and episode % collapse_patience == 0
                ):
                    avg = numpy.mean(recent_rewards[-collapse_patience:])
                    if avg > best_avg:
                        best_avg = avg
                        plateau_count = 0
                        agent.best_avg = best_avg
                        agent.save("mario_best.pth")
                        print(f"  ** New best avg: {best_avg:.0f} - saved checkpoint **")
                    else:
                        plateau_count += 1

                        # auto-tune: if plateaued for 4 checks (200 eps), sharpen
                        if plateau_count % 4 == 0 and plateau_count > 0:
                            old_start = agent.entropy_coeff_start
                            old_end = agent.entropy_coeff_end
                            old_decay = agent.entropy_decay_updates
                            old_lr = agent.max_lr
                            # speed up entropy decay and lower entropy schedule.
                            # This persists because learn() recomputes entropy from start/end/decay.
                            agent.entropy_decay_updates = max(200, int(agent.entropy_decay_updates * 0.7))
                            agent.entropy_coeff_end = max(0.001, agent.entropy_coeff_end * 0.9)
                            agent.entropy_coeff_start = max(
                                agent.entropy_coeff_end + 0.001,
                                agent.entropy_coeff_start * 0.9,
                            )
                            agent.max_lr = max(5e-5, agent.max_lr * 0.85)
                            progress = min(1.0, agent.update_count / max(1, agent.entropy_decay_updates))
                            agent.entropy_coeff = agent.entropy_coeff_start + (
                                agent.entropy_coeff_end - agent.entropy_coeff_start
                            ) * progress
                            print(
                                f"  >> AUTO-TUNE: plateau {plateau_count} checks | "
                                f"ent_start {old_start:.4f}->{agent.entropy_coeff_start:.4f} | "
                                f"ent_end {old_end:.4f}->{agent.entropy_coeff_end:.4f} | "
                                f"decay {old_decay}->{agent.entropy_decay_updates} | "
                                f"ent {agent.entropy_coeff:.4f} | "
                                f"max_lr {old_lr:.6f}->{agent.max_lr:.6f}"
                            )
                            agent.best_avg = best_avg
                            agent.save()

                        if avg < best_avg * collapse_threshold and os.path.exists(
                            "mario_best.pth"
                        ):
                            print(
                                f"  !! COLLAPSE DETECTED - avg {avg:.0f} vs best {best_avg:.0f} - reloading best model !!"
                            )
                            agent.load("mario_best.pth")
                            best_avg = getattr(agent, "best_avg", best_avg)
                            plateau_count = 0
                            recent_rewards.clear()

        # PPO update when rollout is full
        if agent.ready_to_learn():
            stats = agent.learn(states)
            if stats is not None:
                print(
                    "  PPO | "
                    f"Loss {stats['loss']:.4f} | "
                    f"Pi {stats['policy_loss']:.4f} | "
                    f"V {stats['value_loss']:.4f} | "
                    f"Ent {stats['entropy']:.4f} | "
                    f"EntCoef {agent.entropy_coeff:.4f} | "
                    f"KL {stats['approx_kl']:.5f} | "
                    f"Clip {stats['clip_frac']:.2f} | "
                    f"EV {stats['explained_var']:.3f} | "
                    f"LR {stats['lr']:.6f} | "
                    f"Stop {int(stats['early_stopped'])}"
                )

    env.close()
    cv2.destroyAllWindows()
