import os
import torch

from .env import Env


# Globals
timesteps = []
rewards = []
q_vals = []
best_avg_reward = -1e10


# Test DQN
def test(args, timestep, dqn, val_mem, evaluate=False):
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
            state, reward, done, *info = env.step(action)
            reward_sum += reward
            if args.render:
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

        # Plot
        _plot_line(timesteps, rewards, "Reward", path="results")
        _plot_line(timesteps, q_vals, "Q", path="results")

        # Save model parameters if improved
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            dqn.save("results")

    # Return average reward and Q-value
    return avg_reward, avg_q_val


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=""):
    try:
        import plotly
        from plotly.graph_objs import Scatter
        from plotly.graph_objs.scatter import Line
    except ImportError:
        return
    max_colour, mean_colour, std_colour, transparent = (
        "rgb(0, 132, 180)",
        "rgb(0, 172, 237)",
        "rgba(29, 202, 255, 0.2)",
        "rgba(0, 0, 0, 0)",
    )

    ys = torch.tensor(ys_population, dtype=torch.float32)
    ys_min, ys_max, ys_mean, ys_std = (
        ys.min(1)[0].squeeze(),
        ys.max(1)[0].squeeze(),
        ys.mean(1).squeeze(),
        ys.std(1).squeeze(),
    )
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(
        x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash="dash"), name="Max"
    )
    trace_upper = Scatter(
        x=xs,
        y=ys_upper.numpy(),
        line=Line(color=transparent),
        name="+1 Std. Dev.",
        showlegend=False,
    )
    trace_mean = Scatter(
        x=xs,
        y=ys_mean.numpy(),
        fill="tonexty",
        fillcolor=std_colour,
        line=Line(color=mean_colour),
        name="Mean",
    )
    trace_lower = Scatter(
        x=xs,
        y=ys_lower.numpy(),
        fill="tonexty",
        fillcolor=std_colour,
        line=Line(color=transparent),
        name="-1 Std. Dev.",
        showlegend=False,
    )
    trace_min = Scatter(
        x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash="dash"), name="Min"
    )

    plotly.offline.plot(
        {
            "data": [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
            "layout": dict(
                title=title, xaxis={"title": "Step"}, yaxis={"title": title}
            ),
        },
        filename=os.path.join(path, title + ".html"),
        auto_open=False,
    )
