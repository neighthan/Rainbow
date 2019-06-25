import os
from gzip import GzipFile
from pathlib import Path
from time import sleep
from typing import Optional
import numpy as np
import torch
from .env import Env


def test(
    args,
    timestep: int,
    dqn,
    trainer,
    evaluate: bool = False,
    render: bool = False,
    render_delay: float = 0.05,
    evaluation_episodes: Optional[int] = None,
    env=None,
    max_env_steps: int = -1,
):
    evaluation_episodes = evaluation_episodes or args.evaluation_episodes
    env = env or Env(args)
    env.eval()
    trainer.timesteps.append(timestep)
    timestep_rewards = []
    timestep_q_vals = []

    done = True
    all_obs = []
    for episode_i in range(evaluation_episodes):
        while True:
            if done:
                state = env.reset()
                reward_sum = 0
                done = False
                i = 0
                if episode_i == 0:
                    all_obs.append(state)

            action = dqn.act_e_greedy(state)  # Choose an action ε-greedily
            state, reward, done, *_ = env.step(action)
            reward_sum += reward

            if episode_i == 0:
                all_obs.append(state)

            if render or args.render:
                env.render()
                sleep(render_delay)

            if done or i == max_env_steps:
                timestep_rewards.append(reward_sum)
                break
            i += 1
    env.close()

    # TODO - this is for the test environment; for Atari, we need to preprocess a little differently
    # (e.g. do * 255 before .astype(np.uint8). Check `env` to determine what to do?)
    all_obs = np.stack(
        [
            np.flipud(frame.numpy().astype(np.uint8).squeeze().transpose())
            for frame in all_obs
        ]
    )
    with GzipFile(Path(trainer.results_dir) / f"video_{timestep}.npy.gz", "w") as f:
        np.save(f, arr=all_obs)

    # Test Q-values over validation memory
    for state in trainer.val_mem:
        timestep_q_vals.append(dqn.evaluate_q(state))

    avg_reward = sum(timestep_rewards) / len(timestep_rewards)
    avg_q_val = sum(timestep_q_vals) / len(timestep_q_vals)
    if not evaluate:
        trainer.rewards.append(timestep_rewards)
        trainer.q_vals.append(timestep_q_vals)

        if avg_reward > trainer.best_avg_reward:
            trainer.best_avg_reward = avg_reward
            dqn.save(trainer.results_dir)

    return avg_reward, avg_q_val
