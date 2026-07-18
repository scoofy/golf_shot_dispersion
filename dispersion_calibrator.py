import numpy as np
from scipy.optimize import minimize_scalar
from data_model import DispersionDataModel, SEED

class DispersionCalibrator:
    def __init__(self):
        # We only need the Data Model now!
        self.data_model = DispersionDataModel()

    def _simulation_error(self, tuning_scalar, hcp, distance, lie, handedness, correlation_override):
        """
        Objective function for the optimizer: minimize the difference between
        the simulated median leave and the Broadie target median leave.
        """
        # 1. Generate a sample of test shots utilizing the test scalar
        shots = self.data_model.generate_shots(
            hcp=hcp,
            distance=distance,
            lie=lie,
            num_shots=5000,
            handedness=handedness,
            return_cartesian=True,
            tuning_scalar_override=tuning_scalar,
            correlation_override=correlation_override,
            seed=SEED
        )

        dx = shots[:, 0]
        dy = shots[:, 1]

        # 2. Shift the reference center for penal lies
        if lie == 'sand':
            clamped_dist = self.data_model._clamp_to_max_distance(hcp, distance)
            distance_penalty = self.data_model._get_bunker_distance_loss(hcp, clamped_dist)
        else:
            distance_penalty = 0.0

        actual_center_y = distance - distance_penalty

        # 3. Calculate actual median spread around the TRUE center
        distances_to_center = np.sqrt(dx**2 + (dy - actual_center_y)**2)
        actual_median = np.median(distances_to_center)

        # 4. Fetch the target median
        expected_ratio = self.data_model.get_leave_ratio(hcp, distance, lie=lie)
        target_median = expected_ratio * distance

        # Return the absolute difference for the optimizer to minimize
        return abs(actual_median - target_median)

    def calibrate_shot(self, hcp, distance, lie='fairway', handedness='R', correlation_override=None):
        # Pass lie into the args tuple for minimize_scalar
        result = minimize_scalar(
            self._simulation_error,
            bounds=(0.5, 4.0),
            args=(hcp, distance, lie, handedness, correlation_override),
            method='bounded',
            options={'xatol': 1e-3}
        )
        return result.x

if __name__ == "__main__":
    calibrator = DispersionCalibrator()

    # Test cases to verify multi-surface calibration is working
    test_cases = [
        (15.0, 125.0, 'fairway'),
        (15.0, 125.0, 'rough'),
        (15.0, 125.0, 'sand')
    ]

    print("Testing Multi-Surface Calibration:\n" + "="*35)
    for hcp, dist, lie in test_cases:
        best_scalar = calibrator.calibrate_shot(hcp, dist, lie=lie)
        print(f"HCP {hcp:4.1f} @ {dist:5.1f} yds ({lie.rjust(7)}) -> Optimal Scalar: {best_scalar:.4f}")