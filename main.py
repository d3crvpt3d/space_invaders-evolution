import gymnasium as gym
import ale_py
from torch import nn
import torch
import sys

mutation_spread = 0.125
mutation_rate = 0.1

#create neural network
class Agent(nn.Module):

    score: int = 0

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear( 210*160, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )

    def forward(self, x):
        x = self.flatten(x)
        output = self.model(x).argmax().item()
        return output

    def mutate(self, mutation_rate) :
        with torch.no_grad():
            for param in self.model.parameters():
                mask = torch.rand_like(param) < mutation_rate # % of weights
                noise = torch.randn_like(param) * mutation_spread # impact of mutation
                param[mask] += noise[mask]

dorf = []

#save/load checkpoints
def save_checkpoint(dorf1, generation, suffix):
    dorf1.sort(lambda x: x.score)
    agent: Agent = dorf1[0]
    torch.save({
        'model_state_dict': agent.state_dict()
        }, 'it'+str(generation)+'_score'+str(dorf1[0].score)+suffix)

def load_checkpoint(filename, cdorf):
    checkpoint = torch.load(filename)

    #load first without mutation
    cdorf.append(Agent())
    cdorf[0].model.load_dict_state(checkpoint['model_state_dict'])
    
    for i in range(99):
        cdorf.append(Agent())
        cdorf[i].model.load_dict_state(checkpoint['model_state_dict'])
        cdorf[i].mutate(0.01, 0.001)

gym.register_envs(ale_py)

env = gym.make("ALE/SpaceInvaders-v5", obs_type="grayscale")#render_mode="human"

#load if checkpoint is given
if len(sys.argv) == 2:
    load_checkpoint(sys.argv[1], dorf)
else:
    for _ in range(100):
        dorf.append(Agent())

#training
iteration = 0
while True:#each iteration
    iteration = iteration + 1
    #each agent
    for cAgent in dorf:

        obs, info = env.reset()
        end = False
        while not end:
            print(info) #debug
            #exit(1) #debug
            action = cAgent.forward(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            cAgent.score = cAgent.score + reward

            end = terminated or truncated

    #save_checkpoint
    if (iteration % 50) == 0:
        save_checkpoint(dorf,
                        iteration,
                        '_spaceInvaders.model')

    #breed and mutate
    #TODO






















