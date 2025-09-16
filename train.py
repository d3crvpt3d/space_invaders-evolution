import gymnasium as gym
import ale_py
from torch import nn
import torch
import sys
from copy import deepcopy

mutation_spread = 0.1
mutation_rate = 0.3
num_agents = 32
num_best = num_agents // 10
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

    def mutate(self) :
        with torch.no_grad():
            for param in self.model.parameters():
                mask = torch.rand_like(param) < mutation_rate # % of weights
                noise = torch.randn_like(param) * mutation_spread # impact of mutation
                param[mask] += noise[mask]

dorf: list[Agent] = []

#save/load checkpoints
def save_checkpoint(best: list[Agent], generation, suffix):
    
    torch.save({
        'generation': generation,
        'num_agents': num_agents,
        'seed': seed,
        'per_gen_steps': per_gen_steps,
        'models_state_dicts': [cbest.state_dict() for cbest in best]
        }, 'models/it'+str(generation)+'_score'+str(int(best[0].score))+'_'+str(seed)+suffix)

def load_checkpoint(filename) -> tuple:

    checkpoint = torch.load(filename)

    cgeneration = int(checkpoint['generation'])
    cnum_agents = int(checkpoint['num_agents'])
    cseed = int(checkpoint['seed'])
    cper_gen_steps = int(checkpoint['per_gen_steps'])
    best_model_states = checkpoint.get('models_state_dicts')

    #load best without mutation
    cloaded_agents: list[Agent] = []
    for state_dict in best_model_states:
        curr = Agent()
        curr.load_state_dict(state_dict)
        cloaded_agents.append(curr)
    
    return cloaded_agents, cnum_agents, cnum_agents // 10, cgeneration, cseed, cper_gen_steps
    

def create_next_generation(best: list[Agent]) -> list[Agent]:
    #already sorted before

    new_population: list[Agent] = []

    num_best_all = num_agents // 10 * 9
    per_best = num_best_all // num_best
    num_new = num_agents - (per_best * num_best)

    with torch.no_grad():
        #copy best 10%
        for i in range(num_best):
            new_population.append(deepcopy(best[i]))
    
        #for each "elite" generate child agents to 90%
        for best_idx in range(num_best):
            for _ in range(per_best): 
                child_agent = deepcopy(best[best_idx])
                child_agent.mutate() 
                new_population.append(child_agent)
    
        #generate ~10-11% fresh agents (fill up)
        for _ in range(num_new):
                new_population.append(Agent())

    return new_population

gym.register_envs(ale_py)

env = gym.make("ALE/SpaceInvaders-v5", obs_type='grayscale')#render_mode="human"

iteration: int = 0
#load if checkpoint is given
if len(sys.argv) == 2:
    dorf, num_agents, num_best, iteration, seed, per_gen_steps = load_checkpoint(sys.argv[1])
else:
    for _ in range(num_agents):
        dorf.append(Agent())

#training
while True:#each iteration

    iteration = iteration + 1 #skip first save
    #each agent
    for cAgent in dorf:

        obs, info = env.reset()
        for _ in range(int(per_gen_steps)):
            action = cAgent.forward(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            cAgent.score = cAgent.score + float(reward)

            if terminated or truncated:
                break

    dorf.sort(key=lambda agent: agent.score, reverse=True)

    print(f"Generation: {iteration}, Score: {dorf[0].score}")

    #save_checkpoint
    if (iteration % 10) == 0:
        save_checkpoint(dorf[:num_best],
                        iteration,
                        '_spaceInvaders.model')
        print(f"Saved Evolution {iteration}")
        per_gen_steps *= 1.25
        mutation_spread = 10.0 / (100.0 + float(iteration)) #decrease spread

    #breed and mutate
    dorf = create_next_generation(dorf[:num_best])
    
    #reset score
    for cagent in dorf:
        cagent.score = 0.0
