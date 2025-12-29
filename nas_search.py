import numpy as np
import torch
import torch.optim as optim
import random
import copy
import shutil
from torch.utils.data import DataLoader, Subset

from core.models import NAS_GHM_Model
from core.training import train_model
from core.hardware import estimate_performance
from dataset import GraspNetHeatmapDataset

# --- setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POPULATION_SIZE = 10
GENERATIONS = 5
ELITISM = 2
TRAIN_BATCHES = 20
TRAIN_EPOCHS = 1

# --- search space
SEARCH_SPACE = {
    'enc_ch':    [16, 32, 48, 64, 96, 128],
    'enc_depth': [1, 2, 3],
    'enc_k':     [3, 5],
    'enc_bits':  [4, 8],
    'btl_ch':    [64, 96, 128, 256],
    'dec_ch':    [16, 32, 64, 96]
}

def random_genome():
    """
    generate a random but monotonic growing genome using the defined search space.
    :return: valid architecture.
    """
    return {
        'enc_ch':    sorted([random.choice(SEARCH_SPACE['enc_ch']) for _ in range(3)]),
        'enc_depth': [random.choice(SEARCH_SPACE['enc_depth']) for _ in range(3)],
        'enc_k':     [random.choice(SEARCH_SPACE['enc_k']) for _ in range(3)],
        'enc_bits':  [random.choice(SEARCH_SPACE['enc_bits']) for _ in range(3)],
        'btl_ch':    random.choice(SEARCH_SPACE['btl_ch']),
        'dec_ch':    sorted([random.choice(SEARCH_SPACE['dec_ch']) for _ in range(3)], reverse=True)
    }

def mutate(genome):
    """mutates single gene."""
    g = copy.deepcopy(genome)
    key = random.choice(list(SEARCH_SPACE.keys()))

    if key == 'btl_ch':
        g[key] = random.choice(SEARCH_SPACE[key])
    else:
        idx = random.randint(0, len(g[key])-1)
        g[key][idx] = random.choice(SEARCH_SPACE[key])

        # keep constraints
        if key == 'enc_ch': g[key] = sorted(g[key])
        if key == 'dec_ch': g[key] = sorted(g[key], reverse=True)
    return g

def crossover(g1, g2):
    """uniform crossover with equal probabilities."""
    child = {}
    for k in g1.keys():
        child[k] = g1[k] if random.random() > 0.5 else g2[k]

    # keep constraints
    child['enc_ch'] = sorted(child['enc_ch'])
    child['dec_ch'] = sorted(child['dec_ch'], reverse=True)
    return child

def calculate_fitness(loss, fps, params_m):
    """
    fitness = accuracy * efficiency.
    efficiency = FPS * params.

    params smoothens out efficiency when FPS is identical.
    """
    # accuracy component (converts loss to a maximizing factor)
    accuracy_score = 1.0 / (loss + 1e-6)

    # efficiency component (smooth to avoid param extremes and generic FINN FPS prediction)
    param_factor = 10.0 / np.log(params_m + 2.0)
    efficiency_score = fps * param_factor

    return accuracy_score * efficiency_score


def evaluate_fitness(genome, loader, generation, individual_id):
    print(f"   Testing Genome {individual_id}...")

    # 1. build model (copied logic from core/utils.py)
    try:
        model = NAS_GHM_Model(genome).to(DEVICE)
    except Exception as e:
        print(f"setup failed: {e}")
        return 0, 0, 999, 0

    # 2. hardware estimation
    build_tag = f"gen{generation}_id{individual_id}"

    hw_metrics = estimate_performance(model, build_name=build_tag)
    fps = hw_metrics['fps'] if hw_metrics else 0.0

    # calculate params
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    # clean up
    shutil.rmtree(f"build_{build_tag}", ignore_errors=True)

    # invalid model for hardware filters out
    if fps == 0:
        return 0, 0, 999, params_m

    # 3. proxy train
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss = train_model(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=DEVICE,
        epochs=TRAIN_EPOCHS,
        save_path=None,
        max_batches=TRAIN_BATCHES,
        print_every=0
    )

    # 4. fitness
    fitness = calculate_fitness(loss, fps, params_m)

    print(f"      -> FPS:{fps:.0f} | params:{params_m:.2f}M | loss:{loss:.3f} | score:{fitness:.4f}")
    return fitness, fps, loss, params_m


# --- main loop
if __name__ == "__main__":
    # load data
    full_ds = GraspNetHeatmapDataset("data/graspnet", camera='kinect', downsample_factor=8)

    indices = list(range(len(full_ds)))
    random.shuffle(indices)
    search_loader = DataLoader(Subset(full_ds, indices[:1000]), batch_size=4, shuffle=True) # use 1000 images

    # initialize pop
    population = [random_genome() for _ in range(POPULATION_SIZE)]
    print(f"initialized {POPULATION_SIZE} models.")

    best_ever_fitness = -1
    best_ever_genome = None

    for gen in range(GENERATIONS):
        print(f"GENERATION {gen+1}/{GENERATIONS}")

        scored_pop = []
        for i, genome in enumerate(population):
            fit, fps, loss, params = evaluate_fitness(genome, search_loader, gen, i)
            scored_pop.append((fit, genome))

            if fit > best_ever_fitness:
                best_ever_fitness = fit
                best_ever_genome = genome
                with open("best_nas_genome.txt", "w") as f: f.write(str(genome))
                print("      🌟 new GLOBAL BEST saved.")

        scored_pop.sort(key=lambda x: x[0], reverse=True)
        print(f"   gen {gen+1} top score: {scored_pop[0][0]:.4f}")

        # evolution
        next_gen = [scored_pop[i][1] for i in range(ELITISM)]

        while len(next_gen) < POPULATION_SIZE:
            parent1 = random.choice(scored_pop[:len(scored_pop)//2])[1]
            parent2 = random.choice(scored_pop[:len(scored_pop)//2])[1]

            child = crossover(parent1, parent2)
            if random.random() < 0.3: child = mutate(child)
            next_gen.append(child)

        population = next_gen

    print("EVOLUTION COMPLETE")
    print(f"best genome: {best_ever_genome}")