import numpy as np
import torch
import torch.optim as optim
import random
import copy
import shutil
import csv
from torch.utils.data import DataLoader, Subset

from core.models import NAS_GHM_Model
from core.training import train_model
from core.hardware import estimate_performance
from dataset import GraspNetHeatmapDataset

# --- setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POPULATION_SIZE = 20
GENERATIONS = 400
ELITISM = 2
TRAIN_BATCHES = 20
TRAIN_EPOCHS = 3
LOG_FILE = "nas_search_log.csv"

# --- hardware contains (Kria KV260)
HARDWARE_LIMITS = {
    "LUT":  117120,
    "BRAM": 288,
    "DSP":  1248
}

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
    """generate a random but monotonic growing genome."""
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
        if key == 'enc_ch': g[key] = sorted(g[key])
        if key == 'dec_ch': g[key] = sorted(g[key], reverse=True)
    return g

def crossover(g1, g2):
    """uniform crossover."""
    child = {}
    for k in g1.keys():
        child[k] = g1[k] if random.random() > 0.5 else g2[k]
    child['enc_ch'] = sorted(child['enc_ch'])
    child['dec_ch'] = sorted(child['dec_ch'], reverse=True)
    return child

def calculate_fitness(loss, fps, params_m):
    """fitness = accuracy * efficiency"""
    accuracy_score = 1.0 / (loss + 1e-6)
    param_factor = 10.0 / np.log(params_m + 2.0)
    efficiency_score = fps * param_factor
    return accuracy_score * efficiency_score

def evaluate_fitness(genome, loader, generation, individual_id):
    print(f"   Testing Genome {individual_id}...")

    # 1. build model
    try:
        model = NAS_GHM_Model(genome).to(DEVICE)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
    except Exception as e:
        print(f"      -> Setup failed: {e}")
        return 0, 0, 999, 0, 0, 0

    # 2. hardware estimation
    build_tag = f"gen{generation}_id{individual_id}"
    hw_metrics = estimate_performance(model, build_name=build_tag)

    # clean up
    shutil.rmtree(f"build_{build_tag}", ignore_errors=True)

    # filter 1: estimator failure
    if not hw_metrics or hw_metrics['fps'] == 0:
        print(f"      -> Failed to estimate hardware.")
        return 0, 0, 999, params_m, 0, 0

    # filter 2: hardware constraints
    lut_usage = hw_metrics['lut']
    bram_usage = hw_metrics['bram']
    dsp_usage = hw_metrics['dsp']

    fits_hardware = (
            lut_usage <= HARDWARE_LIMITS['LUT'] and
            bram_usage <= HARDWARE_LIMITS['BRAM'] and
            dsp_usage <= HARDWARE_LIMITS['DSP']
    )

    if not fits_hardware:
        print(f"      -> ❌ HARDWARE OVERFLOW | LUT: {lut_usage:.0f} (Max {HARDWARE_LIMITS['LUT']}) | BRAM: {bram_usage} (Max {HARDWARE_LIMITS['BRAM']})")
        return 0, hw_metrics['fps'], 999, params_m, lut_usage, bram_usage

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
    fps = hw_metrics['fps']
    fitness = calculate_fitness(loss, fps, params_m)

    print(f"      -> ✅ Valid | FPS:{fps:.0f} | LUT:{lut_usage/1000:.1f}k | BRAM:{bram_usage} | loss:{loss:.3f} | score:{fitness:.4f}")
    return fitness, fps, loss, params_m, lut_usage, bram_usage


# --- main loop
if __name__ == "__main__":
    full_ds = GraspNetHeatmapDataset("data/graspnet", camera='kinect', downsample_factor=8)
    indices = list(range(len(full_ds)))
    random.shuffle(indices)
    search_loader = DataLoader(Subset(full_ds, indices[:1000]), batch_size=4, shuffle=True)

    population = [random_genome() for _ in range(POPULATION_SIZE)]
    print(f"initialized {POPULATION_SIZE} models.")

    print(f"Logging results to: {LOG_FILE}")
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['generation', 'id', 'fitness', 'fps', 'loss', 'params', 'lut', 'bram',
                         'enc_ch', 'enc_depth', 'enc_k', 'enc_bits', 'btl_ch', 'dec_ch'])

    best_ever_fitness = -1
    best_ever_genome = None

    for gen in range(GENERATIONS):
        print(f"GENERATION {gen+1}/{GENERATIONS}")
        scored_pop = []

        for i, genome in enumerate(population):
            fit, fps, loss, params, lut, bram = evaluate_fitness(genome, search_loader, gen, i)
            scored_pop.append((fit, genome))

            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    gen + 1, i, fit, fps, loss, params, lut, bram,
                    genome['enc_ch'], genome['enc_depth'], genome['enc_k'],
                    genome['enc_bits'], genome['btl_ch'], genome['dec_ch']
                ])

            if fit > best_ever_fitness:
                best_ever_fitness = fit
                best_ever_genome = genome
                with open("best_nas_genome.txt", "w") as f: f.write(str(genome))
                print("      🌟 new GLOBAL BEST saved.")

        scored_pop.sort(key=lambda x: x[0], reverse=True)
        print(f"   gen {gen+1} top score: {scored_pop[0][0]:.4f}")

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