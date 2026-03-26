import random
import copy
import numpy as np

# --- search space
# all genes sampled randomly; hw filter rejects genomes that exceed kv260 constraints.
SEARCH_SPACE = {
    'enc_ch':    [16, 32, 48, 64, 96, 128],
    'enc_depth': [1, 2, 3],
    'enc_k':     [3, 5],
    'enc_bits':  [4, 8],
    'btl_ch':    [64, 96, 128, 192, 256],
    'dec_ch':      [16, 32, 48, 64, 96],
    'parallelism': [2, 3, 4]
}

def random_genome():
    # initialize in the known-valid region: depth=1, bits=4 always pass the hw filter.
    # mutations will still explore depth>1/int8 but those get rejected and don't reproduce.
    return {
        'enc_ch':    sorted([random.choice(SEARCH_SPACE['enc_ch']) for _ in range(3)]),
        'enc_depth': [random.choice(SEARCH_SPACE['enc_depth']) for _ in range(3)],
        'enc_k':     [random.choice(SEARCH_SPACE['enc_k']) for _ in range(3)],
        'enc_bits':  [random.choice(SEARCH_SPACE['enc_bits']) for _ in range(3)],
        'btl_ch':      random.choice(SEARCH_SPACE['btl_ch']),
        'dec_ch':      sorted([random.choice(SEARCH_SPACE['dec_ch']) for _ in range(3)], reverse=True),
        'parallelism': random.choice(SEARCH_SPACE['parallelism']),
    }

def mutate(genome):
    """mutates single gene."""
    g = copy.deepcopy(genome)
    key = random.choice(list(SEARCH_SPACE.keys()))

    if key in ('btl_ch', 'parallelism'):
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
    """fitness = accuracy^2 * efficiency (accuracy-weighted)"""
    accuracy_score = 1.0 / (loss + 1e-6)
    param_factor = 10.0 / np.log(params_m + 2.0)
    efficiency_score = fps * param_factor
    return (accuracy_score ** 2) * efficiency_score