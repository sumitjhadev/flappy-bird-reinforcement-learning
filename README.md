# Flappy Bird DQN

A Deep Q-Network (DQN) agent trained to play Flappy Bird, built with PyTorch and [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium).

![Trained agent playing Flappy Bird](assets/demo.gif)

## Overview

This project implements a DQN agent from scratch, including:
- A fully connected Q-network (`dqn.py`)
- Experience replay buffer (`experience_replay.py`)
- Epsilon-greedy exploration with decay
- A separate target network synced periodically for training stability
- YAML-based hyperparameter configuration for easy experimentation

## Training Results

The agent was trained for 10,000 episodes on CPU. Below is the progression of the best reward achieved over the course of training:

![Reward progression](assets/reward_curve.png)

The agent improved from an average of around -8.7 (essentially failing immediately, comparable to random play) up to a best reward of 8.8 by episode ~6,200, after which it plateaued for the remainder of training.

**Honest note:** this is a work-in-progress result. Flappy Bird is a notoriously hard sparse-reward environment for vanilla DQN, and the current agent has learned to survive noticeably longer than random play but has not yet learned to reliably navigate multiple pipes. Next steps to improve this are listed below.

## Project Structure

```
.
├── agent.py                # Main Agent class: training & testing loop
├── dqn.py                  # Q-network architecture
├── experience_replay.py    # Replay memory buffer
├── game_flappy_bird.py     # Load a trained model and watch it play
├── parameters.yaml         # Hyperparameter configurations
├── requirements.txt
├── runs/                   # Saved models (.pt) and training logs (.log)
└── assets/                 # README images/gifs
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train the agent:**
```bash
python agent.py flappybirdv0 --train
```

**Watch the trained agent play:**
```bash
python game_flappy_bird.py
```

**Test the trained agent (via agent.py, renders automatically):**
```bash
python agent.py flappybirdv0
```

## Hyperparameters

Configured in `parameters.yaml` under the `flappybirdv0` key:

| Parameter | Value |
|---|---|
| Learning rate (alpha) | 0.001 |
| Discount factor (gamma) | 0.99 |
| Epsilon (init → min) | 1.0 → 0.05 |
| Epsilon decay | 0.9995 |
| Replay memory size | 100,000 |
| Mini-batch size | 32 |
| Network sync rate | 10 steps |
| Reward threshold (episode cap) | 1000 |

## Next Steps

- Train for more episodes / tune epsilon decay to escape the current plateau
- Experiment with Double DQN or Dueling DQN to improve stability
- Try reward shaping to give denser learning signal
- Train on GPU for faster iteration on hyperparameters

## Tech Stack

- Python, PyTorch
- Gymnasium + flappy-bird-gymnasium
- PyYAML