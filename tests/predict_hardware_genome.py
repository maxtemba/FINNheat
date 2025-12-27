import sys
import os

# --- PATH SETUP ---
# Go up one level from 'tests/' to find 'core'
sys.path.append("..")

import torch
from core.models import NAS_GHM_Model
from core.hardware import estimate_performance

# ==============================================================================
# 1. DEFINE YOUR GENOME (Test Subject)
# ==============================================================================
my_genome = {
    'enc_ch':    [32, 64, 64],
    'enc_depth': [1, 1, 1],
    'enc_k':     [3, 3, 3],
    'enc_bits':  [8, 4, 4],
    'btl_ch':    128,
    'dec_ch':    [64, 32]
}

# ==============================================================================
# 2. MAIN SCRIPT
# ==============================================================================
def main():
    print(f"\n🧪 Testing Genome:\n{my_genome}\n")

    # 1. Build Model
    try:
        model = NAS_GHM_Model(my_genome)
        print("✅ Model Built Successfully")
    except Exception as e:
        print(f"❌ Model Build Failed: {e}")
        return

    # 2. Run Hardware Estimator
    print("📊 Running FINN Hardware Estimator...")

    # We use a unique build name to avoid overwriting other tests
    metrics = estimate_performance(model, build_name="manual_test_genome")

    if metrics:
        print("\n" + "="*40)
        print("       HARDWARE RESULTS")
        print("="*40)
        print(f"🚀 FPS:      {metrics['fps']:.2f}")
        print(f"🐢 Latency:  {metrics['latency']} cycles")
        print(f"🧱 LUTs:     {metrics['lut']}")
        print(f"💾 BRAMs:    {metrics['bram']}")
        print(f"⚡ DSPs:     {metrics['dsp']}")
        print("="*40)
    else:
        print("\n❌ Estimation Failed (Check logs)")
        sys.exit(1)

if __name__ == "__main__":
    main()