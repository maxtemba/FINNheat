import sys
import os
import glob
import json
import shutil
import subprocess
import tempfile
import time
from datetime import datetime

# init finn+ settings before any finn imports
from finn.util.settings import initialize_dummy_settings
initialize_dummy_settings()

import config

# override dummy paths with real vivado/vitis installations
os.environ["XILINX_VIVADO"] = config.XILINX_VIVADO_PATH
os.environ["VITIS_HLS"]     = config.VITIS_HLS_PATH
os.environ["XILINX_VITIS"]  = config.XILINX_VITIS_PATH
os.environ["VIVADO_PATH"]   = config.XILINX_VIVADO_PATH
os.environ["FINN_RTLLIB"]   = config.FINN_RTLLIB_PATH

from core.models import load_nas_model
from core.hardware import estimate_performance, synthesize_performance, _parse_synth_utilization_rpt

# ~50 calibration genomes covering the search space:
# size progression, depth, int4/int8, kernel size, bottleneck, mixed quant, asymmetric depth, large/overflow
GENOMES = [
    # baselines (original 10)
    # g00: existing best genome (baseline reference)
    {"id": "g00_best",          "genome": {"enc_ch": [16, 32, 32], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [8, 8, 4], "btl_ch": 64,  "dec_ch": [32, 32, 32]}},
    # g01-g02: size sweep INT4
    {"id": "g01_int4_small",    "genome": {"enc_ch": [16, 16, 32], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 4], "btl_ch": 64,  "dec_ch": [32, 16, 16]}},
    {"id": "g02_int4_medium",   "genome": {"enc_ch": [32, 48, 64], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 4], "btl_ch": 64,  "dec_ch": [64, 48, 32]}},
    # g03-g04: size sweep INT8
    {"id": "g03_int8_small",    "genome": {"enc_ch": [16, 16, 32], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [8, 8, 8], "btl_ch": 64,  "dec_ch": [32, 16, 16]}},
    {"id": "g04_int8_medium",   "genome": {"enc_ch": [32, 48, 64], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [8, 8, 8], "btl_ch": 64,  "dec_ch": [64, 48, 32]}},
    # g05-g06: mixed quant
    {"id": "g05_mixed_4_8_8",   "genome": {"enc_ch": [16, 32, 48], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [4, 8, 8], "btl_ch": 64,  "dec_ch": [48, 32, 16]}},
    {"id": "g06_mixed_8_4_4",   "genome": {"enc_ch": [16, 32, 48], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [8, 4, 4], "btl_ch": 64,  "dec_ch": [48, 32, 16]}},
    # g07: kernel 5
    {"id": "g07_k5_int4",       "genome": {"enc_ch": [16, 32, 32], "enc_depth": [1, 1, 1], "enc_k": [5, 5, 5], "enc_bits": [4, 4, 4], "btl_ch": 64,  "dec_ch": [32, 32, 16]}},
    # g08: deep encoder
    {"id": "g08_deep_int4",     "genome": {"enc_ch": [16, 32, 32], "enc_depth": [2, 2, 3], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 4], "btl_ch": 64,  "dec_ch": [32, 16, 16]}},
    # g09: wide bottleneck
    {"id": "g09_wide_btl",      "genome": {"enc_ch": [16, 32, 32], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 4], "btl_ch": 128, "dec_ch": [16, 16, 16]}},

    # size x depth x int4
    {"id": "g10_tiny_d2_int4",  "genome": {"enc_ch": [16, 16, 32], "enc_depth": [2, 2, 2], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 4], "btl_ch": 64,  "dec_ch": [32, 16, 16]}},
    {"id": "g11_tiny_d3_int4",  "genome": {"enc_ch": [16, 16, 32], "enc_depth": [3, 3, 3], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 4], "btl_ch": 64,  "dec_ch": [32, 16, 16]}},
    {"id": "g12_small_d1_int4", "genome": {"enc_ch": [16, 32, 48], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 4], "btl_ch": 64,  "dec_ch": [48, 32, 16]}},
    {"id": "g13_small_d2_int4", "genome": {"enc_ch": [16, 32, 48], "enc_depth": [2, 2, 2], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 4], "btl_ch": 64,  "dec_ch": [48, 32, 16]}},
    {"id": "g14_small_d3_int4", "genome": {"enc_ch": [16, 32, 48], "enc_depth": [3, 3, 3], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 4], "btl_ch": 64,  "dec_ch": [48, 32, 16]}},
    {"id": "g15_med_d2_int4",   "genome": {"enc_ch": [32, 48, 64], "enc_depth": [2, 2, 2], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 4], "btl_ch": 64,  "dec_ch": [64, 48, 32]}},
    {"id": "g16_med_d3_int4",   "genome": {"enc_ch": [32, 48, 64], "enc_depth": [3, 3, 3], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 4], "btl_ch": 64,  "dec_ch": [64, 48, 32]}},
    {"id": "g17_large_d1_int4", "genome": {"enc_ch": [32, 64, 96], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 4], "btl_ch": 96,  "dec_ch": [96, 64, 32]}},
    # size x depth x int8
    {"id": "g19_tiny_d2_int8",  "genome": {"enc_ch": [16, 16, 32], "enc_depth": [2, 2, 2], "enc_k": [3, 3, 3], "enc_bits": [8, 8, 8], "btl_ch": 64,  "dec_ch": [32, 16, 16]}},
    {"id": "g20_tiny_d3_int8",  "genome": {"enc_ch": [16, 16, 32], "enc_depth": [3, 3, 3], "enc_k": [3, 3, 3], "enc_bits": [8, 8, 8], "btl_ch": 64,  "dec_ch": [32, 16, 16]}},
    {"id": "g21_small_d1_int8", "genome": {"enc_ch": [16, 32, 48], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [8, 8, 8], "btl_ch": 64,  "dec_ch": [48, 32, 16]}},
    {"id": "g22_small_d2_int8", "genome": {"enc_ch": [16, 32, 48], "enc_depth": [2, 2, 2], "enc_k": [3, 3, 3], "enc_bits": [8, 8, 8], "btl_ch": 64,  "dec_ch": [48, 32, 16]}},
    {"id": "g23_med_d1_int8",   "genome": {"enc_ch": [32, 48, 64], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [8, 8, 8], "btl_ch": 64,  "dec_ch": [64, 48, 32]}},
    {"id": "g24_med_d2_int8",   "genome": {"enc_ch": [32, 48, 64], "enc_depth": [2, 2, 2], "enc_k": [3, 3, 3], "enc_bits": [8, 8, 8], "btl_ch": 64,  "dec_ch": [64, 48, 32]}},

    # kernel size 5 sweep
    {"id": "g25_tiny_k5_int4",  "genome": {"enc_ch": [16, 16, 32], "enc_depth": [1, 1, 1], "enc_k": [5, 5, 5], "enc_bits": [4, 4, 4], "btl_ch": 64,  "dec_ch": [32, 16, 16]}},
    {"id": "g26_small_k5_int4", "genome": {"enc_ch": [16, 32, 48], "enc_depth": [1, 1, 1], "enc_k": [5, 5, 5], "enc_bits": [4, 4, 4], "btl_ch": 64,  "dec_ch": [48, 32, 16]}},
    {"id": "g27_med_k5_int4",   "genome": {"enc_ch": [32, 48, 64], "enc_depth": [1, 1, 1], "enc_k": [5, 5, 5], "enc_bits": [4, 4, 4], "btl_ch": 64,  "dec_ch": [64, 48, 32]}},
    {"id": "g28_small_k5_int8", "genome": {"enc_ch": [16, 32, 48], "enc_depth": [1, 1, 1], "enc_k": [5, 5, 5], "enc_bits": [8, 8, 8], "btl_ch": 64,  "dec_ch": [48, 32, 16]}},
    {"id": "g29_mixed_k_int4",  "genome": {"enc_ch": [16, 32, 48], "enc_depth": [1, 1, 1], "enc_k": [3, 5, 3], "enc_bits": [4, 4, 4], "btl_ch": 64,  "dec_ch": [48, 32, 16]}},

    # bottleneck sweep (fixed small encoder to isolate effect)
    {"id": "g30_btl64",         "genome": {"enc_ch": [16, 32, 48], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 4], "btl_ch": 64,  "dec_ch": [48, 32, 16]}},
    {"id": "g31_btl96",         "genome": {"enc_ch": [16, 32, 48], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 4], "btl_ch": 96,  "dec_ch": [48, 32, 16]}},
    {"id": "g32_btl128",        "genome": {"enc_ch": [16, 32, 48], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 4], "btl_ch": 128, "dec_ch": [48, 32, 16]}},
    {"id": "g33_btl192",        "genome": {"enc_ch": [16, 32, 48], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 4], "btl_ch": 192, "dec_ch": [48, 32, 16]}},
    {"id": "g34_btl256",        "genome": {"enc_ch": [16, 32, 48], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 4], "btl_ch": 256, "dec_ch": [48, 32, 16]}},

    # all 6 mixed quantization patterns on medium encoder
    {"id": "g35_q448_med",      "genome": {"enc_ch": [32, 48, 64], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 8], "btl_ch": 64,  "dec_ch": [64, 48, 32]}},
    {"id": "g36_q484_med",      "genome": {"enc_ch": [32, 48, 64], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [4, 8, 4], "btl_ch": 64,  "dec_ch": [64, 48, 32]}},
    {"id": "g37_q844_med",      "genome": {"enc_ch": [32, 48, 64], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [8, 4, 4], "btl_ch": 64,  "dec_ch": [64, 48, 32]}},
    {"id": "g38_q884_med",      "genome": {"enc_ch": [32, 48, 64], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [8, 8, 4], "btl_ch": 64,  "dec_ch": [64, 48, 32]}},
    {"id": "g39_q488_med",      "genome": {"enc_ch": [32, 48, 64], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [4, 8, 8], "btl_ch": 64,  "dec_ch": [64, 48, 32]}},
    {"id": "g40_q848_med",      "genome": {"enc_ch": [32, 48, 64], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [8, 4, 8], "btl_ch": 64,  "dec_ch": [64, 48, 32]}},

    # asymmetric depth patterns
    {"id": "g41_depth_321",     "genome": {"enc_ch": [16, 32, 48], "enc_depth": [3, 2, 1], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 4], "btl_ch": 64,  "dec_ch": [48, 32, 16]}},
    {"id": "g42_depth_123",     "genome": {"enc_ch": [16, 32, 48], "enc_depth": [1, 2, 3], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 4], "btl_ch": 64,  "dec_ch": [48, 32, 16]}},
    {"id": "g43_depth_213",     "genome": {"enc_ch": [16, 32, 48], "enc_depth": [2, 1, 3], "enc_k": [3, 3, 3], "enc_bits": [4, 4, 4], "btl_ch": 64,  "dec_ch": [48, 32, 16]}},
    {"id": "g44_depth_311_int8","genome": {"enc_ch": [16, 32, 48], "enc_depth": [3, 1, 1], "enc_k": [3, 3, 3], "enc_bits": [8, 8, 8], "btl_ch": 64,  "dec_ch": [48, 32, 16]}},

    # g45-g49 removed: enc_ch up to 128 or depth 3 + k5 crashes hls c-synthesis (confirmed on g49)
    # g17 kept (depth 1, k3, borderline but should complete hls)
]

PARALLELISM    = 4   # change to 2 or 3 to collect those levels
GENOME_FILE    = os.path.join(config.GENOMES_DIR, "calibration_active.txt")
OUTPUT_FILE    = config.calibration_data_path(PARALLELISM)
# synthesis timeouts:
#   synth step (synth_1) finishes in <10 min, we read the utilization report immediately
#   design overflows (lut > limit) -> kill right away
#   design fits -> wait up to POST_SYNTH_TIMEOUT for placement to finish
HARD_CAP_MINS       = 90       # absolute max (synth ~10 min + placement up to 60 min + margin)
PRE_SYNTH_TIMEOUT   = 45 * 60  # kill if synth step has not finished in 45 min
POST_SYNTH_TIMEOUT  = 60 * 60  # allow up to 60 min for placement on valid (tight) designs
POLL_INTERVAL       = 30       # seconds between checks
LUT_LIMIT           = config.HARDWARE_LIMITS["LUT"]   # kv260 lut capacity


def _synth_sentinel_exists():
    # vivado writes this when synth_1 finishes, before impl_1 (placement) starts
    pattern = os.path.join(
        config.FINN_TMP_DIR, "synth_out_of_context_*",
        "results_finn_design_wrapper", "vivadocompile",
        "vivadocompile.runs", "synth_1", "__synthesis_is_complete__"
    )
    return bool(glob.glob(pattern))


def _kill_proc_group(proc):
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except ProcessLookupError:
        pass
    proc.wait()


def run_synthesis_with_timeout(genome, build_name):
    # synthesis runs in a fresh subprocess to avoid finn/vivado fork deadlocks
    # once synthesis completes we read the utilization report immediately:
    #   overflow (lut > limit) -> kill right away, no wasted placement time
    #   fits -> wait up to POST_SYNTH_TIMEOUT for placement to finish
    worker  = os.path.join(os.path.dirname(__file__), "_synth_worker.py")
    out_tmp = tempfile.mktemp(suffix=".json")
    python  = sys.executable
    cmd     = [python, worker, repr(genome), build_name, out_tmp, str(PARALLELISM)]

    proc = subprocess.Popen(cmd, cwd=os.path.dirname(__file__), preexec_fn=os.setsid)

    start              = time.time()
    synth_done_at      = None
    kill_reason        = None
    synth_overflow_data = None

    while proc.poll() is None:
        time.sleep(POLL_INTERVAL)
        elapsed = time.time() - start

        # detect when synthesis (synth_1) finishes, read utilization report
        if synth_done_at is None and _synth_sentinel_exists():
            synth_done_at = time.time()
            res = _parse_synth_utilization_rpt()
            if res and res["lut"] > LUT_LIMIT:
                print(f"  synthesis done ({elapsed/60:.1f} min), overflows ({res['lut']} LUTs > {LUT_LIMIT}), killing")
                synth_overflow_data = res
                kill_reason = "overflow"
                break
            else:
                lut = res["lut"] if res else "?"
                print(f"  synthesis done ({elapsed/60:.1f} min), fits ({lut} LUTs), waiting for placement...")

        if synth_done_at is not None and (time.time() - synth_done_at) > POST_SYNTH_TIMEOUT:
            print(f"placement timeout after {POST_SYNTH_TIMEOUT//60} min, killing")
            kill_reason = "post_synth_timeout"
            break

        if synth_done_at is None and elapsed > PRE_SYNTH_TIMEOUT:
            print(f"synthesis step timeout after {PRE_SYNTH_TIMEOUT//60} min, killing")
            kill_reason = "pre_synth_timeout"
            break

        if elapsed > HARD_CAP_MINS * 60:
            print(f"hard timeout after {HARD_CAP_MINS} min, killing")
            kill_reason = "hard_cap"
            break

    if kill_reason is not None:
        _kill_proc_group(proc)
        overflow = synth_overflow_data if kill_reason == "overflow" else _parse_synth_utilization_rpt()
        build_dir = os.path.join(config.BUILDS_DIR, f"build_{build_name}")
        shutil.rmtree(build_dir, ignore_errors=True)
        shutil.rmtree(config.FINN_TMP_DIR, ignore_errors=True)
        return overflow

    # process finished normally, read result written by worker
    if os.path.exists(out_tmp):
        with open(out_tmp) as f:
            result = json.load(f)
        os.remove(out_tmp)
        return result
    return None


def write_genome(genome):
    with open(GENOME_FILE, "w") as f:
        f.write(repr(genome))


def main():
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    os.makedirs(config.GENOMES_DIR, exist_ok=True)

    # load existing runs for resume support
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            data = json.load(f)
        existing_ids = {r["genome_id"] for r in data.get("runs", [])}
        runs = data["runs"]
        print(f"loaded {len(runs)} existing runs from {OUTPUT_FILE}")
    else:
        existing_ids = set()
        runs = []
        data = {"runs": runs}

    total = len(GENOMES)
    for idx, entry in enumerate(GENOMES):
        gid    = entry["id"]
        genome = entry["genome"]

        if gid in existing_ids:
            print(f"[{idx+1}/{total}] skipping {gid} (already collected)")
            continue

        print(f"\n{'='*60}")
        print(f"[{idx+1}/{total}] genome: {gid}")
        print(f"  {genome}")
        print(f"{'='*60}")

        # write genome to file so load_nas_model picks it up
        write_genome(genome)
        model = load_nas_model(GENOME_FILE, device='cpu')

        # estimation (fast)
        print(f"running estimator...")
        try:
            pred = estimate_performance(model, build_name=f"calib_est_{gid}", parallelism=PARALLELISM)
        except Exception as e:
            print(f"estimation exception: {e}")
            pred = None

        if pred is None:
            print(f"estimation failed, skipping synthesis")
            runs.append({"genome_id": gid, "genome": genome, "predicted": None, "actual": None,
                         "status": "estimation_failed", "timestamp": datetime.utcnow().isoformat()})
            with open(OUTPUT_FILE, "w") as f:
                json.dump(data, f, indent=2)
            continue

        print(f"  estimated: FPS={pred['fps']:.2f}  LUT={pred['lut']}  BRAM={pred['bram']}  DSP={pred['dsp']}")

        # synthesis (slow, 30-120 min per genome, killed after 90 min on hang)
        print(f"running synthesis (hard cap: {HARD_CAP_MINS} min)...")
        act = run_synthesis_with_timeout(genome, build_name=f"calib_synth_{gid}")

        if act is None:
            print(f"synthesis failed")
            runs.append({"genome_id": gid, "genome": genome,
                         "predicted": {"lut": pred["lut"], "bram": pred["bram"], "dsp": pred["dsp"], "fps": pred["fps"]},
                         "actual": None, "status": "synthesis_failed",
                         "timestamp": datetime.utcnow().isoformat()})
        elif act.get("overflow"):
            # design overflowed the device, implementation failed but synth resource data is valid
            print(f"  overflow:  LUT={act['lut']}  BRAM={act['bram']}  DSP={act['dsp']}  (exceeds KV260 limits)")
            runs.append({"genome_id": gid, "genome": genome,
                         "predicted": {"lut": pred["lut"], "bram": pred["bram"], "dsp": pred["dsp"], "fps": pred["fps"]},
                         "actual": {"lut": act["lut"], "lut_ram": act.get("lut_ram", 0),
                                    "bram": act["bram"], "dsp": act["dsp"]},
                         "status": "overflow", "timestamp": datetime.utcnow().isoformat()})
        else:
            fmax = round(1000.0 / act["clk_period_ns"], 1) if act.get("clk_period_ns") else None
            print(f"  actual:    FPS={act['fps_estimate']:.2f}  LUT={act['lut']}  BRAM={act['bram']}  DSP={act['dsp']}  fmax={fmax} MHz")
            runs.append({"genome_id": gid, "genome": genome,
                         "predicted": {"lut": pred["lut"], "bram": pred["bram"], "dsp": pred["dsp"], "fps": pred["fps"]},
                         "actual": {"lut": act["lut"], "lut_ram": act.get("lut_ram", 0), "ff": act.get("ff", 0),
                                    "bram": act["bram"], "dsp": act["dsp"], "fps": act["fps_estimate"],
                                    "fmax_mhz": fmax, "timing_wns_ns": act.get("timing_wns_ns")},
                         "status": "ok", "timestamp": datetime.utcnow().isoformat()})

        # save incrementally after each genome
        with open(OUTPUT_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"saved progress -> {OUTPUT_FILE}")

    # summary table
    print(f"\n{'='*60}")
    print("COLLECTION COMPLETE")
    print(f"{'='*60}")
    ok       = [r for r in runs if r["status"] == "ok"]
    overflow = [r for r in runs if r["status"] == "overflow"]
    failed   = [r for r in runs if r["status"] not in ("ok", "overflow")]
    print(f"genomes: {total}  ok: {len(ok)}  overflow: {len(overflow)}  failed: {len(failed)}")
    if ok:
        hdr = f"{'id':<28} {'pred_lut':>10} {'act_lut':>10} {'lut_err%':>9} {'pred_fps':>9} {'act_fps':>9}"
        print(hdr)
        print("-" * len(hdr))
        for r in ok:
            p, a = r["predicted"], r["actual"]
            lut_err = (a["lut"] - p["lut"]) / max(a["lut"], 1) * 100
            fps_err = (a["fps"] - p["fps"]) / a["fps"] * 100 if a.get("fps") else float("nan")
            print(f"{r['genome_id']:<28} {p['lut']:>10.0f} {a['lut']:>10.0f} {lut_err:>8.1f}% {p['fps']:>9.2f} {a['fps']:>9.2f}")
    print(f"\nresults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
