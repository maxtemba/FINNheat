import torch
import torch.optim as optim
import random
import copy
import numpy as np
import shutil
import os
from torch.utils.data import DataLoader, Subset

# --- IMPORTS ---
from core.models import NAS_GHM_Model
from core.training import train_model  # <--- UPDATED: Uses the unified training function
from core.hardware import estimate_performance
from dataset import GraspNetHeatmapDataset

# --- SETTINGS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POPULATION_SIZE = 10     # How many models per generation
GENERATIONS = 5        # How many generations to evolve
ELITISM = 2              # Keep top 2 models unchanged
TRAIN_BATCHES = 20       # "X images" (batches) to train on
TRAIN_EPOCHS = 1         # "Y epochs" to train

# --- GENOME HELPERS ---
SEARCH_SPACE = {
    'enc_ch':    [16, 24, 32, 48, 64],
    'enc_depth': [1, 2, 3],
    'enc_k':     [3, 5],
    'enc_bits':  [4, 8],
    'btl_ch':    [64, 96, 128],
    'dec_ch':    [16, 32, 64]
}

def random_genome():
    return {
        'enc_ch':    sorted([random.choice(SEARCH_SPACE['enc_ch']) for _ in range(3)]),
        'enc_depth': [random.choice(SEARCH_SPACE['enc_depth']) for _ in range(3)],
        'enc_k':     [random.choice(SEARCH_SPACE['enc_k']) for _ in range(3)],
        'enc_bits':  [random.choice(SEARCH_SPACE['enc_bits']) for _ in range(3)],
        'btl_ch':    random.choice(SEARCH_SPACE['btl_ch']),
        'dec_ch':    sorted([random.choice(SEARCH_SPACE['dec_ch']) for _ in range(2)], reverse=True)
    }

def mutate(genome):
    """Randomly changes one gene."""
    g = copy.deepcopy(genome)
    key = random.choice(list(SEARCH_SPACE.keys()))

    if key in ['btl_ch']:
        g[key] = random.choice(SEARCH_SPACE[key])
    else:
        # Lists (Encoder/Decoder settings)
        idx = random.randint(0, len(g[key])-1)
        g[key][idx] = random.choice(SEARCH_SPACE[key])
        if 'ch' in key: g[key] = sorted(g[key], reverse=('dec' in key))
    return g

def crossover(g1, g2):
    """Mixes two genomes."""
    child = {}
    for k in g1.keys():
        child[k] = g1[k] if random.random() > 0.5 else g2[k]

    # Ensure sorted channels
    child['enc_ch'] = sorted(child['enc_ch'])
    child['dec_ch'] = sorted(child['dec_ch'], reverse=True)
    return child

# --- FITNESS FUNCTION ---
def evaluate_fitness(genome, loader, generation, individual_id):
    print(f"   Testing Genome {individual_id}...")

    # 1. Build
    try:
        model = NAS_GHM_Model(genome).to(DEVICE)
    except:
        return 0, 0, 999, 0 # Failed build

    # 2. Hardware Estimate (FPS & Params)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    # Create a specific build tag
    build_tag = f"gen{generation}_id{individual_id}"

    # Run Estimator
    hw_metrics = estimate_performance(model, build_name=build_tag)
    fps = hw_metrics['fps'] if hw_metrics else 50

    # 🧹 CLEANUP: Delete the build folder immediately to save space
    build_dir = f"build_{build_tag}"
    if os.path.exists(build_dir):
        try:
            shutil.rmtree(build_dir)
        except OSError:
            pass # Ignore errors if file is locked

    # 3. Proxy Train (Accuracy Estimate)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # UPDATED: Use the unified train_model function
    loss = train_model(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=DEVICE,
        epochs=TRAIN_EPOCHS,
        save_path=None,        # Don't save weights during search
        max_batches=TRAIN_BATCHES,
        print_every=0          # Keep console clean
    )

    # 4. Calculate Fitness Score
    # We want HIGH FPS, LOW Params, LOW Loss.
    fitness = (fps / (params_m + 0.01)) / (loss + 0.01)

    print(f"      -> FPS:{fps:.0f} | Params:{params_m:.2f}M | Loss:{loss:.3f} | Fitness:{fitness:.2f}")
    return fitness, fps, loss, params_m

# --- MAIN EVOLUTION LOOP ---
if __name__ == "__main__":
    # Load Data (Small subset for speed)
    full_ds = GraspNetHeatmapDataset("data/graspnet", camera='kinect', downsample_factor=8)
    indices = list(range(len(full_ds)))
    random.shuffle(indices)

    # We use a specific subset for the entire search so comparisons are fair
    search_loader = DataLoader(Subset(full_ds, indices[:1000]), batch_size=4, shuffle=True)

    # 1. Initialize Population
    population = [random_genome() for _ in range(POPULATION_SIZE)]
    print(f"🌱 Initialized {POPULATION_SIZE} random models.")

    best_ever_fitness = -1
    best_ever_genome = None

    for gen in range(GENERATIONS):
        print(f"\n⚡ GENERATION {gen+1}/{GENERATIONS}")

        # 2. Evaluate All
        scored_pop = []
        for i, genome in enumerate(population):
            fitness, fps, loss, params = evaluate_fitness(genome, search_loader, gen, i)
            scored_pop.append((fitness, genome))

            # Save Global Best
            if fitness > best_ever_fitness:
                best_ever_fitness = fitness
                best_ever_genome = genome
                with open("best_nas_genome.txt", "w") as f: f.write(str(genome))
                print("      🌟 New Global Best Found!")

        # 3. Selection (Sort by fitness)
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        print(f"   Gen {gen+1} Best Fitness: {scored_pop[0][0]:.2f}")

        # 4. Evolution (Next Gen)
        next_gen = []

        # Elitism: Keep top K
        for i in range(ELITISM):
            next_gen.append(scored_pop[i][1])

        # Breeding: Fill rest
        while len(next_gen) < POPULATION_SIZE:
            # Tournament Selection
            parent1 = random.choice(scored_pop[:len(scored_pop)//2])[1]
            parent2 = random.choice(scored_pop[:len(scored_pop)//2])[1]

            child = crossover(parent1, parent2)
            if random.random() < 0.3: # 30% Mutation chance
                child = mutate(child)
            next_gen.append(child)

        population = next_gen

    print("\n🏁 EVOLUTION COMPLETE")
    print(f"Best Genome: {best_ever_genome}")