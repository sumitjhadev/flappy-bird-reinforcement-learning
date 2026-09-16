import re
import os
import matplotlib.pyplot as plt

LOG_FILE = os.path.join("runs", "flappybirdv0.log")
OUT_FILE = os.path.join("assets", "reward_curve.png")

episodes = []
rewards = []

pattern = re.compile(r"Best reward = (-?\d+\.?\d*) Episode = (\d+)")

with open(LOG_FILE, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            rewards.append(float(match.group(1)))
            episodes.append(int(match.group(2)))

os.makedirs("assets", exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(episodes, rewards, marker="o", color="#2E8B57")
plt.title("Best Reward Progression - Flappy Bird DQN")
plt.xlabel("Episode")
plt.ylabel("Best Reward So Far")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_FILE, dpi=150)

print(f"Saved plot to {OUT_FILE}")