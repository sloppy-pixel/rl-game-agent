import gymnasium as gym
import numpy as np

# Discretize continuous observation space into bins
BINS = [
    np.linspace(-4.8, 4.8, 20),
    np.linspace(-4, 4, 20),
    np.linspace(-0.418, 0.418, 20),
    np.linspace(-4, 4, 20),
]

def discretize(obs):
    return tuple(np.digitize(obs[i], BINS[i]) for i in range(4))

# Q-table: 11^4 states x 2 actions
q_table = np.zeros([21] * 4 + [2])

# Hyperparameters
alpha = 0.1       # learning rate
gamma = 0.99      # discount factor
epsilon = 1.0     # exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 5000

env = gym.make("CartPole-v1")

for ep in range(episodes):
    obs, _ = env.reset()
    state = discretize(obs)
    total_reward = 0

    while True:
        # Epsilon-greedy action
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize(next_obs)

        # Q-learning update
        best_next = np.max(q_table[next_state])
        q_table[state][action] += alpha * (reward + gamma * best_next - q_table[state][action])

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if ep % 100 == 0:
        print(f"Episode {ep}, Reward: {total_reward:.1f}, Epsilon: {epsilon:.2f}")

np.save("q_table.npy", q_table)
print("Training done. Q-table saved.")
env.close()