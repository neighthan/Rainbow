import os
import torch
from .env import Env


timesteps = []
rewards = []
q_vals = []
best_avg_reward = -1e10


def test(args, timestep, dqn, val_mem, evaluate=False, render=False):
    global timesteps, rewards, q_vals, best_avg_reward
    env = Env(args)
    env.eval()
    timesteps.append(timestep)
    timestep_rewards, timestep_q_vals = [], []

    # Test performance over several episodes
    done = True
    for _ in range(args.evaluation_episodes):
        while True:
            if done:
                state = env.reset()
                reward_sum = 0
                done = False

            action = dqn.act_e_greedy(state)  # Choose an action ε-greedily
            state, reward, done, *_ = env.step(action)
            reward_sum += reward
            if render or args.render:
                env.render()

            if done:
                timestep_rewards.append(reward_sum)
                break
    env.close()

    # Test Q-values over validation memory
    for state in val_mem:  # Iterate over valid states
        timestep_q_vals.append(dqn.evaluate_q(state))

    avg_reward = sum(timestep_rewards) / len(timestep_rewards)
    avg_q_val = sum(timestep_q_vals) / len(timestep_q_vals)
    if not evaluate:
        # Append to results
        rewards.append(timestep_rewards)
        q_vals.append(timestep_q_vals)

        # Save model parameters if improved
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            dqn.save("results")

    # Return average reward and Q-value
    return avg_reward, avg_q_val
