import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random


# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:

    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.network_sync_rate = params["network_sync_rate"]
        self.reward_threshold = params["reward_threshold"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    def run(self, is_training=True, render=False):

        env = gym.make(
            "FlappyBird-v0",
            render_mode="human" if render else None
        )

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)

        # TRAINING MODE
        if is_training:

            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)

            target_dqn.load_state_dict(
                policy_dqn.state_dict()
            )

            steps = 0

            self.optimizer = optim.Adam(
                policy_dqn.parameters(),
                lr=self.alpha
            )

            best_reward = float("-inf")

        # TEST MODE
        else:

            if not os.path.exists(self.MODEL_FILE):
                print("Model file not found:", self.MODEL_FILE)
                return

            policy_dqn.load_state_dict(
                torch.load(
                    self.MODEL_FILE,
                    map_location=device
                )
            )

            policy_dqn.eval()

        # Episodes
        episodes = 10000 if is_training else 50

        for episode in range(episodes):

            state, _ = env.reset()

            state = torch.tensor(
                state,
                dtype=torch.float,
                device=device
            )

            episode_reward = 0

            terminated = False

            while (
                not terminated
                and episode_reward < self.reward_threshold
            ):

                # ACTION SELECTION
                if is_training and random.random() < epsilon:

                    action = env.action_space.sample()

                    action = torch.tensor(
                        action,
                        dtype=torch.long,
                        device=device
                    )

                else:

                    with torch.no_grad():

                        action = (
                            policy_dqn(
                                state.unsqueeze(0)
                            )
                            .squeeze()
                            .argmax()
                        )

                next_state, reward, terminated, _, _ = env.step(
                    action.item()
                )

                episode_reward += reward

                next_state = torch.tensor(
                    next_state,
                    dtype=torch.float,
                    device=device
                )

                reward = torch.tensor(
                    reward,
                    dtype=torch.float,
                    device=device
                )

                # STORE EXPERIENCE
                if is_training:

                    memory.append(
                        (
                            state,
                            action,
                            next_state,
                            reward,
                            terminated
                        )
                    )

                    steps += 1

                state = next_state

            # PRINT RESULT
            if is_training:

                print(
                    f"Episode {episode+1} | "
                    f"Reward {episode_reward} | "
                    f"Epsilon {epsilon:.4f}"
                )

            else:

                print(
                    f"Episode {episode+1} | "
                    f"Reward {episode_reward}"
                )

            # EPSILON DECAY
            if is_training:

                epsilon = max(
                    epsilon * self.epsilon_decay,
                    self.epsilon_min
                )

                # SAVE BEST MODEL
                if episode_reward > best_reward:

                    log_msg = (
                        f"Best reward = {episode_reward} "
                        f"Episode = {episode+1}"
                    )

                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")

                    torch.save(
                        policy_dqn.state_dict(),
                        self.MODEL_FILE
                    )

                    best_reward = episode_reward

            # TRAIN NETWORK
            if (
                is_training
                and len(memory) > self.mini_batch_size
            ):

                mini_batch = memory.sample(
                    self.mini_batch_size
                )

                self.optimize(
                    mini_batch,
                    policy_dqn,
                    target_dqn
                )

                if steps > self.network_sync_rate:

                    target_dqn.load_state_dict(
                        policy_dqn.state_dict()
                    )

                    steps = 0

        env.close()

    def optimize(
        self,
        mini_batch,
        policy_dqn,
        target_dqn
    ):

        state, action, next_state, reward, terminations = zip(
            *mini_batch
        )

        state = torch.stack(state)
        action = torch.stack(action)
        next_state = torch.stack(next_state)
        reward = torch.stack(reward)

        terminations = (
            torch.tensor(terminations)
            .float()
            .to(device)
        )

        with torch.no_grad():

            target_q = reward + (
                1 - terminations
            ) * self.gamma * target_dqn(
                next_state
            ).max(dim=1)[0]

        current_q = (
            policy_dqn(state)
            .gather(
                dim=1,
                index=action.unsqueeze(1)
            )
            .squeeze()
        )

        loss = self.loss_fn(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train or test model."
    )

    parser.add_argument(
        "hyperparameters",
        help="Parameter set name"
    )

    parser.add_argument(
        "--train",
        help="Training mode",
        action="store_true"
    )

    args = parser.parse_args()

    dql = Agent(
        param_set=args.hyperparameters
    )

    if args.train:

        dql.run(is_training=True)

    else:

        dql.run(
            is_training=False,
            render=True
        )