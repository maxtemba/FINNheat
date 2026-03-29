import os
import numpy as np
import torch
import torch.optim as optim
import random
import csv
from torch.utils.data import DataLoader, Subset

# new core imports
from core.models import NAS_GHM_Model
from core.training import train_model
from core.hardware import estimate_performance
from core.dataset import GraspNetHeatmapDataset
from core.evolution import random_genome, mutate, crossover, calculate_fitness
from core.predictor import load_predictor, apply_calibration

# --- setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
POPULATION_SIZE = 20
GENERATIONS     = 60
ELITISM         = 2
TRAIN_BATCHES   = 100
TRAIN_EPOCHS    = 10
LOG_FILE = "nas_search_log.csv"

# --- hardware limits (Kria KV260)
# no explicit margin — calibrated predictor already overpredicts fitting designs by ~5-10%,
# which acts as an implicit safety buffer without making the search space empty
HARDWARE_LIMITS = {
    "LUT":  117120,
    "BRAM": 288,
    "DSP":  1248,
}

hw_predictors = {}
for _p in [2, 3, 4]:
    _pred = load_predictor(f"scripts/outputs/predictors/hw_predictor_p{_p}.pkl")
    if _pred:
        hw_predictors[_p] = _pred
        print(f"loaded hw predictor p={_p} (trained on {_pred['n_train']} runs)")
if not hw_predictors:
    print("no hw predictors found — using raw finn estimates")

def evaluate_fitness(genome, loader, val_loader, generation, individual_id):
    print(f"   Testing Genome {individual_id}...")

    # 1. build model
    try:
        model = NAS_GHM_Model(genome).to(DEVICE)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
    except Exception as e:
        print(f"      -> Setup failed: {e}")
        return 0, 0, 999, 0, 0, 0, 0

    # 2. hardware estimation
    parallelism = genome.get('parallelism', 2)
    build_tag = f"gen{generation}_id{individual_id}"
    hw_metrics = estimate_performance(model, build_name=build_tag, parallelism=parallelism)

    # filter 1: estimator failure
    if not hw_metrics or hw_metrics['fps'] == 0:
        print(f"      -> Failed to estimate hardware.")
        return 0, 0, 999, params_m, 0, 0, 0

    # filter 2: hardware constraints
    hw_predictor = hw_predictors.get(parallelism)
    if hw_predictor:
        cal = apply_calibration(hw_predictor, genome, hw_metrics)
        lut_usage  = cal['lut']  if cal else hw_metrics['lut']
        bram_usage = cal['bram'] if cal else hw_metrics['bram']
        dsp_usage  = cal['dsp']  if cal else hw_metrics['dsp']
        hw_label   = "(cal)" if cal else "(est)"
    else:
        lut_usage  = hw_metrics['lut']
        bram_usage = hw_metrics['bram']
        dsp_usage  = hw_metrics['dsp']
        hw_label   = "(est)"

    fits_hardware = (
            lut_usage <= HARDWARE_LIMITS['LUT'] and
            bram_usage <= HARDWARE_LIMITS['BRAM'] and
            dsp_usage <= HARDWARE_LIMITS['DSP']
    )

    if not fits_hardware:
        print(f"      -> HARDWARE OVERFLOW {hw_label} | LUT: {lut_usage:.0f} (Max {HARDWARE_LIMITS['LUT']}) | BRAM: {bram_usage} (Max {HARDWARE_LIMITS['BRAM']}) | DSP: {dsp_usage} (Max {HARDWARE_LIMITS['DSP']})")
        return 0, hw_metrics['fps'], 999, params_m, lut_usage, bram_usage, dsp_usage

    # 3. proxy train
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    loss = train_model(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=DEVICE,
        epochs=TRAIN_EPOCHS,
        save_path=None,
        max_batches=TRAIN_BATCHES,
        print_every=0,
        val_loader=val_loader,
    )

    # 4. fitness
    fps = hw_metrics['fps']
    fitness = calculate_fitness(loss, fps, params_m)

    print(f"      -> Valid | FPS:{fps:.0f} | LUT:{lut_usage/1000:.1f}k | BRAM:{bram_usage} | DSP:{dsp_usage} | loss:{loss:.3f} | score:{fitness:.4f}")
    return fitness, fps, loss, params_m, lut_usage, bram_usage, dsp_usage


# --- main loop
if __name__ == "__main__":
    img_cache = "data/graspnet_img_cache" if os.path.exists("data/graspnet_img_cache") else None
    full_ds = GraspNetHeatmapDataset("data/graspnet", camera='kinect', downsample_factor=8, img_cache_dir=img_cache)
    indices = list(range(len(full_ds)))
    random.shuffle(indices)
    search_loader = DataLoader(Subset(full_ds, indices[:800]),  batch_size=4, shuffle=True,
                               num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader    = DataLoader(Subset(full_ds, indices[800:1000]), batch_size=4, shuffle=False,
                               num_workers=2, pin_memory=True, persistent_workers=True)

    population = [random_genome() for _ in range(POPULATION_SIZE)]
    print(f"initialized {POPULATION_SIZE} models.")

    print(f"Logging results to: {LOG_FILE}")
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['generation', 'id', 'fitness', 'fps', 'loss', 'params', 'lut', 'bram', 'dsp',
                         'enc_ch', 'enc_depth', 'enc_k', 'enc_bits', 'btl_ch', 'dec_ch', 'parallelism'])

    best_ever_fitness = -1
    best_ever_genome = None

    for gen in range(GENERATIONS):
        print(f"GENERATION {gen+1}/{GENERATIONS}")
        scored_pop = []

        for i, genome in enumerate(population):
            fit, fps, loss, params, lut, bram, dsp = evaluate_fitness(genome, search_loader, val_loader, gen, i)
            scored_pop.append((fit, genome))

            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    gen + 1, i, fit, fps, loss, params, lut, bram, dsp,
                    genome['enc_ch'], genome['enc_depth'], genome['enc_k'],
                    genome['enc_bits'], genome['btl_ch'], genome['dec_ch'],
                    genome.get('parallelism', 2)
                ])

            if fit > best_ever_fitness:
                best_ever_fitness = fit
                best_ever_genome = genome
                with open("genomes/best_nas_genome.txt", "w") as f: f.write(str(genome))
                print("      new global best saved.")

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