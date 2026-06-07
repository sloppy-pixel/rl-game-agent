import gymnasium as gym
import numpy as np

BINS = [
    np.linspace(-4.8, 4.8, 20),
    np.linspace(-4, 4, 20),
    np.linspace(-0.418, 0.418, 20),
    np.linspace(-4, 4, 20),
]

def discretize(obs):
    return tuple(np.digitize(obs[i], BINS[i]) for i in range(4))

q_table = np.load("q_table.npy")
env = gym.make("CartPole-v1", render_mode="human")

for ep in range(5):
    obs, _ = env.reset()
    total_reward = 0
    while True:
        action = np.argmax(q_table[discretize(obs)])
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode {ep+1}, Reward: {total_reward}")
            break

env.close()