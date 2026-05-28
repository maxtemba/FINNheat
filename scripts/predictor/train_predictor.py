import os
import json
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
import joblib

import config
from core.predictor import genome_to_features, FEATURE_NAMES

PARALLELISM = 4   # change to match the data file being trained
DATA_FILE   = config.calibration_data_path(PARALLELISM)
SAVE_PATH   = config.predictor_path(PARALLELISM)


def main():
    # load calibration data
    if not os.path.exists(DATA_FILE):
        print(f"calibration data not found at {DATA_FILE}")
        return

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    all_runs  = data.get("runs", [])
    # include overflow runs, synth resource data is valid even when implementation fails
    ok_runs   = [r for r in all_runs if r["status"] in ("ok", "overflow") and r.get("actual") is not None]
    n_total   = len(all_runs)
    n_ok      = len(ok_runs)
    n_overflow = sum(1 for r in ok_runs if r["status"] == "overflow")
    n_failed  = n_total - n_ok

    print(f"calibration data: {n_total} total  |  {n_ok} usable ({n_overflow} overflow)  |  {n_failed} failed/skipped")

    if n_ok < 3:
        print(f"not enough data to train (need at least 3 ok runs, have {n_ok})")
        return
    if n_ok < 5:
        print(f"warning: only {n_ok} ok runs, predictor will be unreliable below 20 samples")

    # feature matrix and targets
    X      = np.array([genome_to_features(r["genome"], r["predicted"]) for r in ok_runs])
    y_lut  = np.array([r["actual"]["lut"]  for r in ok_runs])
    y_bram = np.array([r["actual"]["bram"] for r in ok_runs])
    y_dsp  = np.array([r["actual"]["dsp"]  for r in ok_runs])
    ids    = [r["genome_id"] for r in ok_runs]

    print(f"\nfeature matrix: {X.shape[0]} samples x {X.shape[1]} features")

    # loocv evaluation
    if n_ok < 4:
        print(f"\nwarning: loocv with n={n_ok} is degenerate, results not statistically meaningful")

    print(f"\nrunning loocv (n={n_ok})...")
    loo = LeaveOneOut()
    preds = {"lut": [], "bram": [], "dsp": []}
    actuals = {"lut": y_lut.tolist(), "bram": y_bram.tolist(), "dsp": y_dsp.tolist()}

    rf_params = dict(n_estimators=100, max_features='sqrt', random_state=42)

    for train_idx, test_idx in loo.split(X):
        rf_l = RandomForestRegressor(**rf_params).fit(X[train_idx], y_lut[train_idx])
        rf_b = RandomForestRegressor(**rf_params).fit(X[train_idx], y_bram[train_idx])
        rf_d = RandomForestRegressor(**rf_params).fit(X[train_idx], y_dsp[train_idx])
        preds["lut"].append(float(rf_l.predict(X[test_idx])[0]))
        preds["bram"].append(float(rf_b.predict(X[test_idx])[0]))
        preds["dsp"].append(float(rf_d.predict(X[test_idx])[0]))

    # per-run loocv table
    print(f"\n{'id':<28} {'act_lut':>9} {'pred_lut':>9} {'lut_err%':>9}  {'act_bram':>8} {'pred_bram':>9}  {'act_dsp':>7} {'pred_dsp':>8}")
    print("-" * 100)
    for i, gid in enumerate(ids):
        lut_pct  = abs(preds["lut"][i]  - actuals["lut"][i])  / max(actuals["lut"][i],  1) * 100
        bram_pct = abs(preds["bram"][i] - actuals["bram"][i]) / max(actuals["bram"][i], 1) * 100
        dsp_pct  = abs(preds["dsp"][i]  - actuals["dsp"][i])  / max(actuals["dsp"][i],  1) * 100
        print(f"{gid:<28} {actuals['lut'][i]:>9.0f} {preds['lut'][i]:>9.0f} {lut_pct:>8.1f}%"
              f"  {actuals['bram'][i]:>8.0f} {preds['bram'][i]:>9.1f}"
              f"  {actuals['dsp'][i]:>7.0f} {preds['dsp'][i]:>8.1f}")

    # summary metrics
    print()
    for target in ("lut", "bram", "dsp"):
        a = np.array(actuals[target])
        p = np.array(preds[target])
        mae  = np.mean(np.abs(p - a))
        mape = np.mean(np.abs(p - a) / np.maximum(a, 1)) * 100
        maxe = np.max(np.abs(p - a))
        print(f"loocv {target:<4}  mae={mae:.1f}  mape={mape:.1f}%  max_err={maxe:.1f}")

    # train final models on all data
    print(f"\ntraining final models on all {n_ok} samples...")
    rf_lut  = RandomForestRegressor(**rf_params).fit(X, y_lut)
    rf_bram = RandomForestRegressor(**rf_params).fit(X, y_bram)
    rf_dsp  = RandomForestRegressor(**rf_params).fit(X, y_dsp)

    # feature importances
    for name, rf in [("lut", rf_lut), ("bram", rf_bram), ("dsp", rf_dsp)]:
        top = sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda x: -x[1])[:5]
        top_str = "  ".join(f"{n}={v:.3f}" for n, v in top)
        print(f"top features ({name}): {top_str}")

    # save bundle
    os.makedirs(config.PREDICTORS_DIR, exist_ok=True)
    bundle = {
        "lut":           rf_lut,
        "bram":          rf_bram,
        "dsp":           rf_dsp,
        "n_train":       n_ok,
        "feature_names": FEATURE_NAMES,
        "trained_at":    datetime.utcnow().isoformat(),
    }
    joblib.dump(bundle, SAVE_PATH)
    print(f"\npredictor saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
