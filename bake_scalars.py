import numpy as np
import time
import concurrent.futures
from dispersion_calibrator import DispersionCalibrator
from tqdm import tqdm

# 1. Define the Granular Grid
HCP_BINS = np.round(np.arange(-5.0, 30.1, 0.1), 1)
DIST_BINS = np.round(np.arange(10.0, 260.0, 10.0), 1)
LIES = ['fairway', 'rough', 'sand']

def calibrate_single_point(args):
    """Worker function for parallel processing."""
    lie, hcp, dist = args
    calibrator = DispersionCalibrator()

    import sys, os
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    try:
        # Pass the lie argument into your calibrator
        scalar = calibrator.calibrate_shot(hcp, dist, lie=lie, handedness='R')
    finally:
        sys.stdout = old_stdout

    return lie, hcp, dist, scalar

def bake_master_matrices():
    print("Baking Multi-Surface Master Scalar Matrices...")
    print(f"HCP Points: {len(HCP_BINS)} | Distance Points: {len(DIST_BINS)} | Surfaces: {len(LIES)}")
    print(f"Total Optimizations Required: {len(HCP_BINS) * len(DIST_BINS) * len(LIES)}\n")

    # Create the task list with the new lie dimension
    tasks = [(lie, hcp, dist) for lie in LIES for hcp in HCP_BINS for dist in DIST_BINS]

    # Initialize a dictionary to hold a matrix for each lie
    scalar_matrices = {lie: np.zeros((len(HCP_BINS), len(DIST_BINS))) for lie in LIES}

    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = tqdm(executor.map(calibrate_single_point, tasks), total=len(tasks), desc="Calibrating")

        for lie, hcp, dist, scalar in results:
            hcp_idx = np.where(HCP_BINS == hcp)[0][0]
            dist_idx = np.where(DIST_BINS == dist)[0][0]
            scalar_matrices[lie][hcp_idx, dist_idx] = scalar

    print(f"\nOptimization Complete in {(time.time() - start_time):.1f} seconds.")

    # Save all arrays into a single compressed archive
    np.savez_compressed('optimal_scalars_multi.npz',
                        hcp_bins=HCP_BINS,
                        dist_bins=DIST_BINS,
                        fairway=scalar_matrices['fairway'],
                        rough=scalar_matrices['rough'],
                        sand=scalar_matrices['sand'])

    print("File saved: optimal_scalars_multi.npz")

if __name__ == "__main__":
    bake_master_matrices()