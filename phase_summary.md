# FINNheat — Phase Summary (Dec 2025 → Mar 2026)

## Problem → Investigation → Fix Flow

### 1. Overfitting Discovery
- Midway results used 5,000 randomly sampled images from 50K dataset
- Dataset has ~150 scenes → random sampling leaked scenes into both train/val
- Models saw every scene → severe overfitting, inflated accuracy
  - **→ Fix:** scene-based train/val split to prevent data leakage
  - **→ Fix:** added validation loss tracking (was train-only before)

### 2. Hardware Estimator Inaccuracy
- FINN+ built-in estimates were ~90% higher than real synthesis results
- NAS was accepting genomes that wouldn't actually fit, or rejecting ones that would
  - **→ Fix:** trained Random Forest predictor on real synthesis calibration data
  - **→ Result:** deviation reduced to ~5%, but NAS search space now more constrained
  - **→ Note:** RF good for LUTs, but degrades BRAM/DSP accuracy vs raw FINN+

### 3. Fitness Function Imbalance
- Original fitness: accuracy × FPS → FPS dominated, NAS favored fast-but-bad models
  - **→ Fix:** changed to accuracy² × FPS
  - **→ Pending:** log(FPS) scaling for further balancing

### 4. Loss Function Corrections
- cls loss was accidentally missing from total loss during midway evaluation
- Adding cls loss back caused loc loss to spike (cls gradient overwhelmed loc)
  - **→ Fix:** reweighted loss: reduced cls + reg importance, prioritized loc
  - **→ Fix:** reworked focal loss to match original HGGD paper (higher gamma, raised threshold)

### 5. Parallelism Bottleneck
- Default FINN+ parallelism was 4×4 (SMD + PES) → only supported 4-bit quantization
- 4-bit models stuck at high val loss with very limited search space
  - **→ Fix:** added parallelism gene to NAS genome (2×, 3×, 4×)
  - **→ Fix:** trained 3 separate RF predictors for each parallelism level

### 6. Decoder Kernel Experiment
- Tested 3×3 decoder kernel (was 1×1)
- 3×3 blurred the sparse signal → reverted to 1×1

### 7. Training Infrastructure
- Switched to AdamW + CosineAnnealingLR
- Added .npy image cache → 2× faster data loading
- Added GPU opts: pin_memory, persistent_workers, cudnn.benchmark
- Proxy training: 3 epochs / 20 batches for NAS speed
- Full training: 20 epochs, full dataset, heatmap callback every 5 epochs

---

## NAS Result → Synthesis → Training → Evaluation Flow

```
NAS Search (33 generations)
  │
  ├─ best genome: gen33_id7 (fitness 512.82)
  │    enc_ch=[16,16,64]  enc_bits=[8,4,4]  parallelism=3
  │
  ▼
FINN+ Synthesis (2026-03-28)
  │  FPS: 6.79 | 229.7 MHz | timing PASS
  │  LUTs: 69.2% | BRAMs: 19.8% | DSPs: 3.5%
  │  ✓ Fits on KV260
  │
  ▼
Full Training (20 epochs, full dataset)
  │  best val loss: 0.4784 (epoch 7)
  │  best val loc:  0.3054
  │  val cls ~0.75 (plateau), val reg ~0.045 (plateau)
  │
  ▼
LocalNet Evaluation (2 epochs, 100 scenes)
  │  valid_center: 0.01  ← BOTTLENECK
  │  rot_0.25: 0.291
  │  grasp_1.00: 0.084
  │
  ▼
Diagnosis: backbone valid_center too low (1%)
  → LocalNet can't learn without upstream grasp centers
```

---

## Earlier Training Runs (for comparison)

| Run | Genome | Bits | Epochs | Best Val Loss | Status |
|-----|--------|------|--------|---------------|--------|
| medium_model | [32,32,48] | 8,8,8 | 27/40 logged | 2.4644 (ep17) | Plateau, overfitting gap |
| nas_4bit | [16,32,48] | 4,4,4 | 26/40 logged | 2.5352 (ep25) | Stuck, limited search space |
| gen33_30ep | [16,16,64] | 8,4,4 | 30/30 | 0.5069 (ep25) | Good convergence |
| gen33_full | [16,16,64] | 8,4,4 | 18/20 logged | 0.4784 (ep7) | Best result, used for eval |

---

## Code Changes Summary

### FINNheat (NAS pipeline)
- `core/models.py` — decoder kernel kept at 1×1
- `core/training.py` — focal loss reworked, AdamW+CosineAnnealingLR, val tracking, CSV log
- `core/evolution.py` — accuracy²×FPS fitness, parallelism gene
- `core/hardware.py` — calibrated estimator, RF predictor (.pkl)
- `core/dataset.py` / `core/heatmap_generator.py` — .npy cache, scene-based split
- `nas_search.py` — GPU opts, proxy training 3ep/20batch, parallelism gene
- `scripts/train_genome.py` — full training config with heatmap callbacks
- `scripts/synthesize_hardware_genome.py` / `predict_hardware_genome.py` — FINN+ synthesis + ML predictor

### HGGDfinn (HGGD integration)
- Fixed double-sigmoid bug on loc output
- Fixed `load_nas_model_from_weights` (was hardcoding enc_bits=[8,8,8])
- Added `GHHAnchorWrapper`: spatial permute, upsample to 640×360, depth mask (0.1–1.1m), edge suppression
- Fixed output channel order: NAS → HGGD mapping
- Relaxed center-group threshold: 0.02m → 0.05m (matches 45×80 cell resolution)
- Added `ghh_model.sh` central config for model paths

---

## Data Files

| File | Contents |
|------|----------|
| `medium_model_epochs.csv` | Per-epoch train/val loss (27 epochs) |
| `nas_4bit_model_epochs.csv` | Per-epoch train/val loss (26 epochs) |
| `gen33_30ep_epochs.csv` | Per-epoch losses with loc/cls/reg breakdown (30 epochs) |
| `gen33_full_retrain_epochs.csv` | Per-epoch losses with loc/cls/reg breakdown (18 epochs) |
| `synthesis_kv260.csv` | FINN+ synthesis results and KV260 utilization |
| `localnet_evaluation.csv` | Grasp evaluation at 5 threshold levels |
| `genome_comparison.csv` | All genomes side-by-side with best results |
| `hardware_predictor_notes.csv` | RF predictor findings |
