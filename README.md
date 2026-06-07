# CartPole Q-Learning Agent

A reinforcement learning agent that learns to balance a pole using Q-Learning.

## How it works
- Environment: CartPole-v1 (OpenAI Gymnasium)
- Algorithm: Q-Learning with epsilon-greedy exploration
- State space is discretized into bins for each observation
- Agent improves over 5000 episodes

## Results
Agent consistently achieves 200/200 reward after training.

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install gymnasium numpy
```

## Train
```bash
python agent.py
```

## Watch it play
```bash
python play.py
```
