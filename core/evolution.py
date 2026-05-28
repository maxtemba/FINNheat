import random
import copy
import numpy as np

# search space, all genes sampled uniformly
# hw filter rejects genomes that exceed kv260 limits
SEARCH_SPACE = {
    'enc_ch':    [16, 32, 48, 64, 96, 128],
    'enc_depth': [1, 2, 3],
    'enc_k':     [3, 5],
    'enc_bits':  [4, 8],
    'btl_ch':    [64, 96, 128, 192, 256],
    'dec_ch':      [16, 32, 48, 64, 96],
    'parallelism': [2, 3, 4]
}


def _normalize(g):
    # enc_ch ascending, dec_ch descending (invariant used downstream)
    g['enc_ch'] = sorted(g['enc_ch'])
    g['dec_ch'] = sorted(g['dec_ch'], reverse=True)
    return g


def random_genome():
    # depth=1, bits=4 always pass the hw filter
    # mutations explore depth>1 / int8 but get rejected and don't reproduce
    return _normalize({
        'enc_ch':    [random.choice(SEARCH_SPACE['enc_ch']) for _ in range(3)],
        'enc_depth': [random.choice(SEARCH_SPACE['enc_depth']) for _ in range(3)],
        'enc_k':     [random.choice(SEARCH_SPACE['enc_k']) for _ in range(3)],
        'enc_bits':  [random.choice(SEARCH_SPACE['enc_bits']) for _ in range(3)],
        'btl_ch':      random.choice(SEARCH_SPACE['btl_ch']),
        'dec_ch':      [random.choice(SEARCH_SPACE['dec_ch']) for _ in range(3)],
        'parallelism': random.choice(SEARCH_SPACE['parallelism']),
    })


def mutate(genome):
    # mutate one random gene
    g = copy.deepcopy(genome)
    key = random.choice(list(SEARCH_SPACE.keys()))

    if key in ('btl_ch', 'parallelism'):
        g[key] = random.choice(SEARCH_SPACE[key])
    else:
        idx = random.randint(0, len(g[key])-1)
        g[key][idx] = random.choice(SEARCH_SPACE[key])
    return _normalize(g)


def crossover(g1, g2):
    # uniform crossover
    child = {k: (g1[k] if random.random() > 0.5 else g2[k]) for k in g1.keys()}
    return _normalize(child)


def calculate_fitness(loss, fps, params_m):
    # fitness = accuracy^2 * efficiency, accuracy-weighted
    accuracy_score = 1.0 / (loss + 1e-6)
    param_factor = 10.0 / np.log(params_m + 2.0)
    efficiency_score = fps * param_factor
    return (accuracy_score ** 2) * efficiency_score
