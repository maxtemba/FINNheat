import numpy as np

# feature names in fixed order — must match genome_to_features() exactly
FEATURE_NAMES = [
    "enc_ch_0", "enc_ch_1", "enc_ch_2",
    "enc_depth_0", "enc_depth_1", "enc_depth_2",
    "enc_k_0", "enc_k_1", "enc_k_2",
    "enc_bits_0", "enc_bits_1", "enc_bits_2",
    "btl_ch",
    "dec_ch_0", "dec_ch_1", "dec_ch_2",
    "pred_lut", "pred_bram", "pred_dsp",
]


def genome_to_features(genome, finn_pred=None):
    # flattens genome + finn estimates into a fixed-order feature vector (19,)
    # finn_pred fields default to 0.0 if None (genome-only fallback)
    p = finn_pred or {}
    return np.array([
        genome["enc_ch"][0],    genome["enc_ch"][1],    genome["enc_ch"][2],
        genome["enc_depth"][0], genome["enc_depth"][1], genome["enc_depth"][2],
        genome["enc_k"][0],     genome["enc_k"][1],     genome["enc_k"][2],
        genome["enc_bits"][0],  genome["enc_bits"][1],  genome["enc_bits"][2],
        genome["btl_ch"],
        genome["dec_ch"][0],    genome["dec_ch"][1],    genome["dec_ch"][2],
        p.get("lut",  0.0),
        p.get("bram", 0.0),
        p.get("dsp",  0.0),
    ], dtype=np.float32)


def load_predictor(path):
    # loads saved predictor bundle; returns None on any failure
    try:
        import joblib
        bundle = joblib.load(path)
        print(f"loaded hw predictor from {path} (trained on {bundle['n_train']} runs)")
        return bundle
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"warning: failed to load hw predictor from {path}: {e}")
        return None


def apply_calibration(predictor, genome, finn_pred=None):
    # applies calibrated rf prediction to a genome + finn estimates
    # returns {lut, bram, dsp} or None on error
    try:
        x = genome_to_features(genome, finn_pred).reshape(1, -1)
        return {
            "lut":  max(0.0, float(predictor["lut"].predict(x)[0])),
            "bram": max(0.0, float(predictor["bram"].predict(x)[0])),
            "dsp":  max(0.0, float(predictor["dsp"].predict(x)[0])),
        }
    except Exception as e:
        print(f"warning: calibration predict failed: {e}")
        return None
