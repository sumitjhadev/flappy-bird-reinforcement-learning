import gymnasium as gym
import flappy_bird_gymnasium
import torch
import os

from dqn import DQN

RUNS_DIR = "runs"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def run(param_set="flappybirdv0", render=True):

    model_path = os.path.join(RUNS_DIR, f"{param_set}.pt")

    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}. Train first with:")
        print(f"  python agent.py {param_set} --train")
        return

    env = gym.make(
        "FlappyBird-v0",
        render_mode="human" if render else None
    )

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    policy_dqn = DQN(num_states, num_actions).to(device)
    policy_dqn.load_state_dict(torch.load(model_path, map_location=device))
    policy_dqn.eval()

    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float, device=device)

    terminated = False
    total_reward = 0

    while not terminated:

        with torch.no_grad():
            action = policy_dqn(state.unsqueeze(0)).squeeze().argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if truncated:
            break

        state = torch.tensor(next_state, dtype=torch.float, device=device)

    print(f"Episode finished. Total reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    run()