import sys

# init finn+ settings before any finn imports
from finn.util.settings import initialize_dummy_settings
initialize_dummy_settings()

import config
from core.hardware import estimate_performance
from core.models import load_nas_model


def main():
    print("starting hardware verification for the best genome...")

    # build model (weights not needed for estimation)
    try:
        model = load_nas_model(config.ACTIVE_GENOME, device='cpu')
        print("model built successfully.")
    except Exception as e:
        print(f"setup failed: {e}")
        return

    # run hardware estimator
    print("running FINN estimator...")

    metrics = estimate_performance(model, build_name="verify_best_genome")

    if metrics:
        print("\nHardware Estimation Results:")
        print("----------------------------")
        print(f"FPS:      {metrics['fps']:.2f}")
        print(f"Latency:  {metrics['latency']} cycles")
        print(f"LUTs:     {metrics['lut']}")
        print(f"BRAMs:    {metrics['bram']}")
        print(f"DSPs:     {metrics['dsp']}")
        print("----------------------------")
    else:
        print("estimation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
