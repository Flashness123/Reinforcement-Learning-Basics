import gymnasium as gym
import numpy as np

def train_episode(env, q_values, learning_rate, discount_factor, epsilon):
    terminated = False
    truncated = False
    state, info = env.reset(seed=21)
    total_reward = 0
    
    while not (terminated or truncated):
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            best_value = np.max(q_values[state])
            best_actions = np.flatnonzero(q_values[state] == best_value)
            action = np.random.choice(best_actions)

        next_state, reward, terminated, truncated, info = env.step(action)
        # print(next_state, reward, terminated, truncated, info)

        new_q_value = reward + discount_factor*max(q_values[next_state])
        q_values[state][action] += learning_rate*new_q_value 

        state = next_state
        total_reward += reward
    return total_reward

if __name__ == "__main__":
    env = gym.make('CliffWalking-v1')
    q_values = np.zeros((env.observation_space.n, env.action_space.n))
    
    discount_factor = 0.9
    epsilon = 0.1
    learning_rate = 0.01
    num_episodes = 10000
    for episode in range(num_episodes):

        reward = train_episode(env, q_values, learning_rate, discount_factor, epsilon)
        if episode % 100 == 0:
            print(f"Episode: {episode}, Reward: {reward}")

    print(q_values)
    env.close()