import torch
from torch import nn
import gymnasium as gym


class QNetwork(nn.Module):
    def __init__(self, input_state: int, num_actions: int):
        super().__init__()

        self.qnetwork = nn.Sequential(
            nn.Linear(input_state, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, state):
        return self.qnetwork(state)


if __name__ == "__main__":
    print("Start")

    env = gym.make('CliffWalking-v1')
    input_state = env.observation_space.shape[0]
    num_actions = env.action_space.n

    network = QNetwork(input_state, num_actions)

    observation, info = env.reset(seed=21)
    state = torch.tensor(observation, dtype=torch.float32)

    with torch.no_grad():
        q_values = network(state)

    action = torch.argmax(q_values).item()

    next_observation, reward, terminated, truncated, info = env.step(action)
    bellman_target = reward + discount*

    print("Observation:", observation)
    print("Q-values:", q_values)
    print("Selected action:", action)

    env.close()