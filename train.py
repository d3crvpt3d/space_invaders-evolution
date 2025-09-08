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
    
    for i in range(1, num_agents - 1):
        cdorf.append(Agent())
        cdorf[i].load_state_dict(checkpoint['model_state_dict'])
        cdorf[i].mutate()

def create_next_generation(population: list[Agent], num_elite: int) -> list[Agent]:
    # 1. Sortiere die Population nach Score in absteigender Reihenfolge
    population.sort(key=lambda agent: agent.score, reverse=True)

    # 2. Wähle die Elite aus (die Champions)
    elite = population[:num_elite]

    # 3. Die neue Population beginnt mit den unveränderten Champions
    new_population = elite[:] # Wichtig: Kopie der Liste, nicht nur Referenz

    # 4. Bestimme, wie viele neue Agenten (Kinder) erzeugt werden müssen
    num_children_to_create = len(population) - num_elite

    # 5. Erzeuge den Rest der Population durch Crossover und Mutation
    for _ in range(num_children_to_create):
        # Erstelle einen neuen, leeren Agenten
        child_agent = Agent()
        
        # Führe das Crossover durch (Parameter-Mittelung der Elite)
        with torch.no_grad():
            # Gehe durch jeden Parametersatz (Gewichte und Biases)
            for i, child_param in enumerate(child_agent.model.parameters()):
                # Sammle die entsprechenden Parameter von allen Elite-Eltern
                parent_params = [list(parent.model.parameters())[i].data for parent in elite]
                
                # Bilde den Durchschnitt der Elterngewichte
                stacked_params = torch.stack(parent_params)
                average_params = torch.mean(stacked_params, dim=0)
                
                # Weise dem Kind die durchschnittlichen Gewichte zu
                child_param.data.copy_(average_params)

        # 6. Mutiere das neu erstellte Kind, um Diversität zu schaffen
        child_agent.mutate()
        
        # Füge das Kind zur neuen Population hinzu
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
    dorf = create_next_generation(dorf, 2)
    
    #reset score
    for cagent in dorf:
        cagent.score = 0.0
    
