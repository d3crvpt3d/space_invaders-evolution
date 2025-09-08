import gymnasium as gym
import ale_py
from torch import nn
import torch
import sys

mutation_spread = 0.01
mutation_rate = 0.1
num_agents = 32
per_gen_steps = 100
seed = torch.randint(1,9999999, (1,)).item()

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
            nn.Linear(31416, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32) / 255.0
        x = x.unsqueeze(0)
        output = self.model(x).argmax().item()
        return output

    def mutate(self) :
        with torch.no_grad():
            for param in self.model.parameters():
                mask = torch.rand_like(param) < mutation_rate # % of weights
                noise = torch.randn_like(param) * mutation_spread # impact of mutation
                param[mask] += noise[mask]

dorf: list[Agent] = []

#save/load checkpoints
def save_checkpoint(dorf1, generation, suffix):
    dorf1.sort(key=lambda x: x.score, reverse=True)
    agent: Agent = dorf1[0]
    torch.save({
        'generation': generation,
        'model_state_dict': agent.state_dict()
        }, 'models/it'+str(generation)+'_score'+str(dorf1[0].score)+str(seed)+suffix)

def load_checkpoint(filename, cdorf):
    global iteration
    checkpoint = torch.load(filename)

    iteration = int(checkpoint['generation'])

    #load first without mutation
    cdorf.append(Agent())
    cdorf[0].load_state_dict(checkpoint['model_state_dict'])
    
    for i in range(1, num_agents):
        cdorf.append(Agent())
        cdorf[i].load_state_dict(checkpoint['model_state_dict'])
        cdorf[i].mutate()

def create_next_generation(population) -> list[Agent]:
    population.sort(key=lambda agent: agent.score, reverse=True)

    new_population: list[Agent] = []

    #copy best
    best = Agent()
    with torch.no_grad():
        for i, param in enumerate(best.model.parameters()):
            parent_param = list(dorf[0].model.parameters())[i].data
            param.data.copy_(parent_param)

    new_population.append(best)

    #generate others based on best
    with torch.no_grad():
        for _ in range(1,100):

            child_agent = Agent()
            for i, child_param in enumerate(child_agent.model.parameters()):
                parent_param = list(best.model.parameters())[i].data
                child_param.data.copy_(parent_param)

            child_agent.mutate()

            new_population.append(child_agent)

    return new_population

gym.register_envs(ale_py)

env = gym.make("ALE/SpaceInvaders-v5", obs_type='grayscale')#render_mode="human"

iteration = 0
#load if checkpoint is given
if len(sys.argv) == 2:
    load_checkpoint(sys.argv[1], dorf)
else:
    for _ in range(num_agents):
        dorf.append(Agent())

#training
while True:#each iteration

    iteration = iteration + 1 #skip first save
    #each agent
    for cAgent in dorf:

        obs, info = env.reset()
        for _ in range(per_gen_steps):
            action = cAgent.forward(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            cAgent.score = cAgent.score + float(reward)

            if terminated or truncated:
                break

    #save_checkpoint
    if (iteration % 10) == 0:
        save_checkpoint(dorf,
                        iteration,
                        '_spaceInvaders.model')
        print(f"Saved Evolution {iteration}")

    print(f"Generation: {iteration}, Score: {dorf[0].score}")

    #breed and mutate
    dorf = create_next_generation(dorf)
    
    #reset score
    for cagent in dorf:
        cagent.score = 0.0
    
