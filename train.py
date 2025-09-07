import gymnasium as gym
import ale_py
from torch import nn
import torch
import sys

mutation_spread = 0.125
mutation_rate = 0.1
num_agents = 20

#create neural network
class Agent(nn.Module):

    score = 0.0

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear( 210*160, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32) / 255.0
        x = x.flatten()
        output = self.model(x).argmax().item()
        return output

    def mutate(self, mutation_rate) :
        with torch.no_grad():
            for param in self.model.parameters():
                mask = torch.rand_like(param) < mutation_rate # % of weights
                noise = torch.randn_like(param) * mutation_spread # impact of mutation
                param[mask] += noise[mask]

dorf: list[Agent] = []

#save/load checkpoints
def save_checkpoint(dorf1, generation, suffix):
    dorf1.sort(lambda x: x.score)
    agent: Agent = dorf1[0]
    torch.save({
        'generation': generation,
        'model_state_dict': agent.state_dict()
        }, 'models/it'+str(generation)+'_score'+str(dorf1[0].score)+suffix)

def load_checkpoint(filename, cdorf):
    global iteration
    checkpoint = torch.load(filename)

    iteration = int(checkpoint['generation'])

    #load first without mutation
    cdorf.append(Agent())
    cdorf[0].model.load_dict_state(checkpoint['model_state_dict'])
    
    for i in range(num_agents - 1):
        cdorf.append(Agent())
        cdorf[i].model.load_dict_state(checkpoint['model_state_dict'])
        cdorf[i].mutate(0.1, 0.001)

def create_next_generation(population, mut_rate):
    population.sort(key=lambda agent: agent.score, reverse=True)
    best_10 = population[:10]
    
    new_population = []
    for _ in range(num_agents):
        new_agent = Agent()
        
        # Crossover: Parameter-weise mischen
        with torch.no_grad():
            new_params = list(new_agent.model.parameters())
            for param_idx, param in enumerate(new_params):
                # Sammle entsprechende Parameter von allen parents
                parent_params = [list(parent.model.parameters())[param_idx] for parent in best_10]
                
                # Mische nur Parameter mit gleicher Shape
                for parent_param in parent_params:
                    mask = torch.rand_like(param) < 0.1  # 10% chance pro parent
                    param.data = torch.where(mask, parent_param.data, param.data)
        
        new_agent.mutate(mut_rate)
        new_population.append(new_agent)
    
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
    iteration = iteration + 1
    
    print(f"Generation: {iteration}")
    
    #each agent
    for cAgent in dorf:

        obs, info = env.reset()
        end = False
        for _ in range(num_agents):
            action = cAgent.forward(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            cAgent.score = cAgent.score + float(reward)

            if terminated or truncated:
                continue

    #save_checkpoint
    if (iteration % 10) == 0:
        save_checkpoint(dorf,
                        iteration,
                        '_spaceInvaders.model')
        print(f"Saved Evolution {iteration}")

    #breed and mutate
    dorf.sort(key=lambda x: x.score) #get elite in top 10

    best_10 = dorf[:10]
    dorf = dorf[:10]

    dorf = create_next_generation(dorf, 0.1)
    #reset score
    for cagent in dorf:
        cagent.score = 0.0



