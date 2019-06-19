import argparse
from datetime import datetime
import atari_py
import numpy as np
import torch
from tqdm import tnrange


from .agent import Agent
from .env import Env
from .memory import ReplayMemory
from .test import test
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Rainbow")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--game",
        type=str,
        default="space_invaders",
        choices=atari_py.list_games(),
        help="ATARI game",
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=int(50e6),
        metavar="STEPS",
        help="Number of training steps (4x number of frames)",
    )
    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=int(108e3),
        metavar="LENGTH",
        help="Max episode length in game frames (0 to disable)",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=4,
        metavar="T",
        help="Number of consecutive states processed",
    )
    parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
    parser.add_argument(
        "--hidden-size", type=int, default=512, metavar="SIZE", help="Network hidden size"
    )
    parser.add_argument(
        "--noisy-std",
        type=float,
        default=0.1,
        metavar="σ",
        help="Initial standard deviation of noisy linear layers",
    )
    parser.add_argument(
        "--atoms",
        type=int,
        default=51,
        metavar="C",
        help="Discretised size of value distribution",
    )
    parser.add_argument(
        "--V-min",
        type=float,
        default=-10,
        metavar="V",
        help="Minimum of value distribution support",
    )
    parser.add_argument(
        "--V-max",
        type=float,
        default=10,
        metavar="V",
        help="Maximum of value distribution support",
    )
    parser.add_argument(
        "--model", type=str, metavar="PARAMS", help="Pretrained model (state dict)"
    )
    parser.add_argument(
        "--memory-capacity",
        type=int,
        default=int(1e6),
        metavar="CAPACITY",
        help="Experience replay memory capacity",
    )
    parser.add_argument(
        "--replay-frequency",
        type=int,
        default=4,
        metavar="k",
        help="Frequency of sampling from memory",
    )
    parser.add_argument(
        "--priority-exponent",
        type=float,
        default=0.5,
        metavar="ω",
        help="Prioritised experience replay exponent (originally denoted α)",
    )
    parser.add_argument(
        "--priority-weight",
        type=float,
        default=0.4,
        metavar="β",
        help="Initial prioritised experience replay importance sampling weight",
    )
    parser.add_argument(
        "--multi-step",
        type=int,
        default=3,
        metavar="n",
        help="Number of steps for multi-step return",
    )
    parser.add_argument(
        "--discount", type=float, default=0.99, metavar="γ", help="Discount factor"
    )
    parser.add_argument(
        "--target-update",
        type=int,
        default=int(8e3),
        metavar="τ",
        help="Number of steps after which to update target network",
    )
    parser.add_argument(
        "--reward-clip",
        type=int,
        default=1,
        metavar="VALUE",
        help="Reward clipping (0 to disable)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0000625, metavar="η", help="Learning rate"
    )
    parser.add_argument(
        "--adam-eps", type=float, default=1.5e-4, metavar="ε", help="Adam epsilon"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, metavar="SIZE", help="Batch size"
    )
    parser.add_argument(
        "--learn-start",
        type=int,
        default=int(20e3),
        metavar="STEPS",
        help="Number of steps before starting training",
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate only")
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=100000,
        metavar="STEPS",
        help="Number of training steps between evaluations",
    )
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=10,
        metavar="N",
        help="Number of evaluation episodes to average over",
    )
    parser.add_argument(
        "--evaluation-size",
        type=int,
        default=500,
        metavar="N",
        help="Number of transitions to use for validating Q",
    )
    parser.add_argument(
        "--render", action="store_true", help="Display screen (testing only)"
    )
    parser.add_argument(
        "--enable-cudnn",
        action="store_true",
        help="Enable cuDNN (faster but nondeterministic)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()


def main():
    args = parse_args()
    env = Env(args)
    env.train()
    agent = Agent(args, env)
    trainer = Trainer(args)
    trainer.training_loop(env, agent, args)
    env.close()


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(1, 10000))
    torch.cuda.manual_seed(np.random.randint(1, 10000))


class Trainer:
    def __init__(self, env, args):
        set_random_seed(args.seed)
        if torch.cuda.is_available() and not args.disable_cuda:
            args.device = torch.device("cuda")
            torch.backends.cudnn.enabled = args.enable_cudnn
        else:
            args.device = torch.device("cpu")

        self.mem = ReplayMemory(args, capacity=args.memory_capacity)

        # Construct validation memory
        # self.val_mem = ReplayMemory(args, capacity=args.evaluation_size)
        # done = True
        # for T in range(args.evaluation_size):
        #     if done:
        #         state, done = env.reset(), False

        #     next_state, _, done = env.step(np.random.randint(0, env.action_space()))
        #     self.val_mem.append(state, None, None, done)
        #     state = next_state

    def training_loop(self, env, agent, args, eval_func=None):
        if True:  # what was the env variable again?
            range_ = tnrange
        else:
            range_ = range

        priority_weight_increase = (1 - args.priority_weight) / (args.max_timesteps - args.learn_start)

        # Training loop
        agent.train()
        done = True
        n_actions = env.action_space()
        for T in range_(args.max_timesteps):
            if done:
                state, done = env.reset(), False

            if T % args.replay_frequency == 0:
                agent.reset_noise()  # Draw a new set of noisy weights

            if T > args.learn_start:
                action = agent.act(state)  # Choose an action greedily (with noisy weights)
            else:
                action = np.random.randint(n_actions)
            next_state, reward, done, *info = env.step(action)  # Step
            if info and "experience" in info[0]:
                for action, reward in info[0]["experience"]:
                    if args.reward_clip > 0:
                        reward = max(min(reward, args.reward_clip), -args.reward_clip)
                    self.mem.append(state, action, reward, done)
            else:
                if args.reward_clip > 0:
                    reward = max(min(reward, args.reward_clip), -args.reward_clip)
                self.mem.append(state, action, reward, done)

            # Train and test
            if T >= args.learn_start:
                self.mem.priority_weight = min(
                    self.mem.priority_weight + priority_weight_increase, 1
                )  # Anneal importance sampling weight β to 1

                if T % args.replay_frequency == 0:
                    # Train with n-step distributional double-Q learning
                    agent.learn(self.mem)

                if T % args.evaluation_interval == 0 and eval_func:
                    agent.eval()  # Set DQN (online network) to evaluation mode
                    eval_func()
                    # avg_reward, avg_Q = test(args, T, self, val_mem)  # Test
                    # log('T = ' + str(T) + ' / ' + str(max_timesteps) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
                    agent.train()  # Set DQN (online network) back to training mode

                # Update target network
                if T % args.target_update == 0:
                    agent.update_target_net()

            state = next_state
