import numpy as np
from data_model import DispersionDataModel

def run_verification_suite():
    print("==================================================")
    print("   DISPERSION ENGINE - MASTER VERIFICATION SUITE  ")
    print("==================================================")

    data_model = DispersionDataModel()

    if not data_model.calibrated:
        print("\n[!] WARNING: Engine is not calibrated. Did you run bake_scalars.py?")
        return

    test_scenarios = []
    LIES = ['fairway', 'rough', 'sand']

    for hcp in data_model.broadie_hcp_bins:
        for dist in data_model.broadie_dist_bins:
            for lie in LIES:
                # Flag the proxy logic so you know it's being tested
                label_lie = "sand (proxy)" if lie == 'sand' and float(dist) > 40.0 else lie

                test_scenarios.append({
                    "hcp": float(hcp),
                    "dist": float(dist),
                    "lie": lie,
                    "label": f"Broadie Scenario: HCP {hcp} @ {dist}y [{label_lie.upper()}]"
                })

    for scenario in test_scenarios:
        hcp = scenario["hcp"]
        distance = scenario["dist"]
        lie = scenario["lie"]
        label = scenario["label"]

        print(f"\n[{label}]")
        print("-" * 50)

        # 1. Generate shots directly from the Master Engine
        num_test_shots = 10000

        shots = data_model.generate_shots(
            hcp=hcp,
            distance=distance,
            lie=lie,
            num_shots=num_test_shots,
            handedness='R',
            return_cartesian=True
        )

        dx = shots[:, 0]
        dy = shots[:, 1]

        # =========================================================
        # REFERENCE SHIFT: Account for Stagner Distance Loss
        # =========================================================
        if lie == 'sand':
            clamped_dist = data_model._clamp_to_max_distance(hcp, distance)
            distance_penalty = data_model._get_bunker_distance_loss(hcp, clamped_dist)
        else:
            distance_penalty = 0.0

        actual_center_y = distance - distance_penalty

        # ---------------------------------------------------------
        # TEST 1: BROADIE'S MEDIAN LEAVE (Spread around the center)
        # ---------------------------------------------------------
        # We measure spread relative to the actual center, not the pin,
        # to isolate the variance math from the mean shift math.
        distances_to_center = np.sqrt(dx**2 + (dy - actual_center_y)**2)
        actual_median = np.median(distances_to_center)

        expected_ratio = data_model.get_leave_ratio(hcp, distance, lie=lie)
        expected_median = expected_ratio * distance
        b_diff = abs(actual_median - expected_median)

        tolerance = max(0.5, expected_median * 0.025)
        b_status = "PASS" if b_diff <= tolerance else "FAIL"

        if b_status == "FAIL":
            print("1. BROADIE DATA (Median Leave Accuracy)")
            print(f"   Target Median: {expected_median:.1f} yds")
            print(f"   Actual Median: {actual_median:.1f} yds")
            print(f"   Status:        [{b_status}] (Variance: {b_diff:.2f} yds | Tol: {tolerance:.2f})\n")

        # ---------------------------------------------------------
        # TEST 2: SHOT SCOPE (Directional Miss Tendency)
        # ---------------------------------------------------------
        # Measure short/long skew relative to the center of the bubble
        shots_short = np.sum(dy < actual_center_y)
        pct_short = (shots_short / num_test_shots) * 100

        # Updated threshold to reflect the new dynamic awful shot mixture
        ss_status = "PASS" if pct_short > 50.0 else "FAIL"

        if ss_status == "FAIL":
            print("2. SHOT SCOPE DATA (Short/Long Miss Skew)")
            print(f"   Shots Short:   {pct_short:.1f}%")
            print(f"   Shots Long:    {100 - pct_short:.1f}%")
            print(f"   Status:        [{ss_status}] (Expected mild short-bias not met)\n")

        # ---------------------------------------------------------
        # TEST 3: AERODYNAMIC & HANDEDNESS SPIN BIAS
        # ---------------------------------------------------------
        # Quadrants MUST be calculated against the actual center!
        left_long = np.sum((dx < 0) & (dy > actual_center_y))
        right_short = np.sum((dx > 0) & (dy < actual_center_y))

        right_long = np.sum((dx > 0) & (dy > actual_center_y))
        left_short = np.sum((dx < 0) & (dy < actual_center_y))

        expected_high_density = left_long + right_short
        expected_low_density = right_long + left_short

        bias_ratio = expected_high_density / (expected_low_density + 1)

        if distance < 50.0:
            required_ratio = 0.90
            req_label = "Neutral/Short"
        else:
            required_ratio = 1.40
            req_label = "Full-Swing Bias"

        spin_status = "PASS" if bias_ratio > required_ratio else "FAIL"

        if spin_status == "FAIL":
            print("3. AERODYNAMIC CORRELATION (RH Pull-Draw / Push-Fade Bias)")
            print(f"   Typical Misses (Left-Long / Right-Short): {expected_high_density:,}")
            print(f"   Rare Misses (Right-Long / Left-Short):    {expected_low_density:,}")
            print(f"   Status:        [{spin_status}] (Shape ratio: {bias_ratio:.1f}x | Expected {req_label}: > {required_ratio:.1f})\n")

if __name__ == "__main__":
    run_verification_suite()