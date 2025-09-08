import gymnasium as gym
import ale_py
from torch import nn
import torch
import sys

#create neural network
class Agent(nn.Module):

    def __init__(self):
        self.score = 0.0
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(5, 1, kernel_size=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(31416, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32) / 255.0
        x = x.unsqueeze(0).unsqueeze(0)
        output = self.model(x).argmax().item()
        return output

gym.register_envs(ale_py)

env = gym.make("ALE/SpaceInvaders-v5", obs_type='grayscale', render_mode="human")

def load_checkpoint(filename) -> Agent:
    checkpoint = torch.load(filename)

    #load first without mutation
    agent = Agent()
    agent.load_state_dict(checkpoint['model_state_dict'])
    return agent

agent = load_checkpoint(sys.argv[1])

obs, info = env.reset()
while True:
    action = agent.forward(obs)
    env.step(action)
