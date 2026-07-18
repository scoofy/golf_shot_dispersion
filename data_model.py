import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, PchipInterpolator
from scipy.stats import norm, qmc

SEED = 42

bd_hcp_anchors = [-5, 0, 7, 11, 15, 20.5, 26, 30.5, 35]
bd_long_game_miss_rates = [0.0071, 0.0147, 0.0212, 0.0306, 0.0422, 0.0537, 0.067, 0.081, 0.0964]
bd_short_game_miss_rates = [0.0029, 0.004, 0.01, 0.0153, 0.02, 0.0253, 0.03, 0.0343, 0.0382]

hd_dist_dict_yards_men = {
    'Driver': {
         5: 261,
        15: 236,
        25: 204,
        -5: 299,
    },
    '3-Wood': {
         5: 234,
        15: 215,
        25: 178,
        -5: 273,
    },
    '3-Hybrid': {
         5: 216,
        15: 197,
        25: 162,
        -5: 246, # <-- Only estimated datapoint (41% between 3w and 4i)
    },
    '4-Iron': {
         5: 201,
        15: 186,
        25: 151,
        -5: 228,
    },
    '5-Iron': {
         5: 183,
        15: 169,
        25: 143,
        -5: 218,
    },
    '6-Iron': {
         5: 172,
        15: 162,
        25: 137,
        -5: 216,
    },
    '7-Iron': {
         5: 164,
        15: 154,
        25: 132,
        -5: 194,
    },
    '8-Iron': {
         5: 153,
        15: 146,
        25: 122,
        -5: 180,
    },
    '9-Iron': {
         5: 139,
        15: 136,
        25: 108,
        -5: 166,
    },
    'PW': {
         5: 126,
        15: 121,
        25: 90,
        -5: 153,
    },
    'GW': {
         5: 109,
        15: 104,
        25: 79,
        -5: 135,
    },
    'SW': {
         5: 86,
        15: 84,
        25: 80,
        -5: 115,
    },
    'LW': {
         5: 71,
        15: 75,
        25: 49,
        -5: 95,
    },
}

ssd_directional_miss_percent_by_hcp_from_100_yards = {
    0: {
        'right' : 16, # miss percentage
        'left'  : 20,
        'long'  : 23,
        'short' : 41,
    },
    5: {
        'right' : 14,
        'left'  : 21,
        'long'  : 22,
        'short' : 43,
    },
    10: {
        'right' : 16,
        'left'  : 21,
        'long'  : 20,
        'short' : 43,
    },
    15: {
        'right' : 16,
        'left'  : 20,
        'long'  : 23,
        'short' : 41,
    },
    20: {
        'right' : 17,
        'left'  : 20,
        'long'  : 22,
        'short' : 41,
    },
    25: {
        'right' : 18,
        'left'  : 19,
        'long'  : 22,
        'short' : 41,
    },
}

ssd_tee_shot_dispersion_from_fairway_center = {
    0: {
        'left' : 46.3, # Lateral Miss (yards)
        'right': 45.0, # Lateral Miss (yards)
    },
    5: {
        'left' : 47.1, # Lateral Miss (yards)
        'right': 49.4, # Lateral Miss (yards)
    },
    15: {
        'left' : 50.5, # Lateral Miss (yards)
        'right': 52.6, # Lateral Miss (yards)
    },
    25: {
        'left' : 53.8, # Lateral Miss (yards)
        'right': 55.4, # Lateral Miss (yards)
    },
}

class DispersionDataModel:
    """
    Ingests Shot Scope and Broadie data to dynamically generate
    pure polar covariance matrices for any given Handicap and Distance.
    """
    def __init__(self):
        # ==========================================
        # 1. SHOT SCOPE DATA (Interpolated 1D Arrays)
        # ==========================================
        self.ss_hcp_keys = [0, 5, 10, 15, 20, 25]
        self.ss_driver_avg = [266, 242, 228, 214, 202, 195]
        self.ss_driver_good = [281, 265, 253, 240, 228, 223]
        self.ss_short_miss_pct = [41.0, 43.0, 43.0, 41.0, 41.0, 41.0]

        # ==========================================
        # 2. BROADIE DATA (Interpolated 2D Matrix)
        # ==========================================
        self.broadie_hcp_bins = [-5, 7, 15, 26, 35]
        self.broadie_dist_bins = [15, 40, 80, 125, 175]

        self.fairway_ratios = np.array([
            [0.11, 0.09, 0.06, 0.05, 0.06], # -5 HCP (Pro)
            [0.13, 0.13, 0.10, 0.09, 0.10], # 7 HCP
            [0.17, 0.16, 0.13, 0.13, 0.14], # 15 HCP
            [0.20, 0.20, 0.16, 0.16, 0.18], # 26 HCP
            [0.22, 0.23, 0.19, 0.19, 0.23], # 35 HCP
        ])

        # New Rough Data (from BROADIE_ROUGH_MEDIAN_LEAVE_DATA)
        self.rough_ratios = np.array([
            [0.17, 0.13, 0.09, 0.08, 0.09], # -5 HCP
            [0.19, 0.16, 0.12, 0.10, 0.13], # 7 HCP
            [0.26, 0.22, 0.16, 0.13, 0.18], # 15 HCP
            [0.32, 0.27, 0.20, 0.18, 0.25], # 26 HCP
            [0.36, 0.30, 0.23, 0.25, 0.34], # 35 HCP
        ])

        # Sand Data processing (up to 40 yards, converted feet to ratio)
        # Ratio = (feet / 3) / yards
        self.sand_dist_bins = [10, 20, 30, 40]
        self.sand_ratios = np.array([
            [(7/3)/10,  (8/3)/20,  (11/3)/30, (16/3)/40], # -5 HCP
            [(17/3)/10, (18/3)/20, (20/3)/30, (34/3)/40], # 7 HCP
            [(19/3)/10, (24/3)/20, (31/3)/30, (52/3)/40], # 15 HCP
            [(22/3)/10, (29/3)/20, (41/3)/30, (62/3)/40], # 26 HCP
            [(26/3)/10, (34/3)/20, (48/3)/30, (67/3)/40], # 35 HCP
        ])

        # Interpolators
        self.interp_fairway = RegularGridInterpolator((self.broadie_hcp_bins, self.broadie_dist_bins), self.fairway_ratios, bounds_error=False, fill_value=None)
        self.interp_rough = RegularGridInterpolator((self.broadie_hcp_bins, self.broadie_dist_bins), self.rough_ratios, bounds_error=False, fill_value=None)
        self.interp_sand_short = RegularGridInterpolator((self.broadie_hcp_bins, self.sand_dist_bins), self.sand_ratios, bounds_error=False, fill_value=None)

        # ==========================================
        # 3. FAIRWAY BUNKER DISTANCE LOSS (Lou Stagner)
        # ==========================================
        self.stagner_hcp_bins = [0, 5, 10, 15, 20]
        self.stagner_dist_bins = [105, 114, 125, 135, 145, 155, 165, 175, 185, 195, 205]

        # Matrix of Target Yardage - Actual Median Yardage = Distance Loss
        # Rows = HCP, Cols = Target Distance
        stagner_loss_matrix = np.array([
        # 0 HCP
        [6, 5, 9, 6, 10, 15, 20, 28, 36, 47, 53],
        # 5 HCP
        [6, 8, 8, 11, 14, 19, 26, 27, 41, 58, 66],
        # 10 HCP
        [9, 9, 13, 16, 20, 24, 35, 47, 57, 74, 76],
        # 15 HCP
        [13, 14, 15, 18, 24, 36, 39, 56, 72, 82, 94],
        # 20 HCP
        [17, 21, 29, 30, 29, 44, 54, 63, 75, 90, 101]
        ])
        self.bunker_loss_interp = RegularGridInterpolator(
            (self.stagner_hcp_bins, self.stagner_dist_bins),
            stagner_loss_matrix,
            bounds_error=False,
            fill_value=None # Extrapolates gracefully beyond edges
        )

        # ==========================================
        # 3. OPTIMAL TUNING SCALARS (Pre-calibrated .npz Load)
        # ==========================================
        is_baking = 'bake_scalars.py' in __import__('sys').argv[0]

        try:
            # Load the new compressed archive
            archive = np.load('optimal_scalars_multi.npz')
            self.scalar_hcp_bins = archive['hcp_bins']
            self.scalar_dist_bins = archive['dist_bins']

            # Create a dictionary of interpolators
            self.scalar_interpolators = {
                'fairway': RegularGridInterpolator((self.scalar_hcp_bins, self.scalar_dist_bins), archive['fairway'], bounds_error=False, fill_value=None),
                'rough': RegularGridInterpolator((self.scalar_hcp_bins, self.scalar_dist_bins), archive['rough'], bounds_error=False, fill_value=None),
                'sand': RegularGridInterpolator((self.scalar_hcp_bins, self.scalar_dist_bins), archive['sand'], bounds_error=False, fill_value=None)
            }
            self.calibrated = True
        except FileNotFoundError:
            if not is_baking:
                print("WARNING: Multi-surface calibration archive not found. Defaulting to tuning_scalar=1.0.")
            self.calibrated = False

        # ==========================================
        # 4. CLUB DISTANCE INTERPOLATORS (Hireko Data)
        # ==========================================
        self._init_club_data()

    def _get_dynamic_fairway_split(self, hcp, distance):
        """
        Calculates the dynamic lateral-to-depth variance ratio using
        Shot Scope's 100-yard quadrant data and Driver dispersion data.
        """
        # 1. Evaluate the 100-yard anchor (R_100)
        hcp_keys_100 = sorted(ssd_directional_miss_percent_by_hcp_from_100_yards.keys())

        left_pcts  = [ssd_directional_miss_percent_by_hcp_from_100_yards[k]['left'] for k in hcp_keys_100]
        right_pcts = [ssd_directional_miss_percent_by_hcp_from_100_yards[k]['right'] for k in hcp_keys_100]
        long_pcts  = [ssd_directional_miss_percent_by_hcp_from_100_yards[k]['long'] for k in hcp_keys_100]
        short_pcts = [ssd_directional_miss_percent_by_hcp_from_100_yards[k]['short'] for k in hcp_keys_100]

        lat_100 = np.interp(hcp, hcp_keys_100, left_pcts) + np.interp(hcp, hcp_keys_100, right_pcts)
        depth_100 = np.interp(hcp, hcp_keys_100, long_pcts) + np.interp(hcp, hcp_keys_100, short_pcts)

        R_100 = lat_100 / (lat_100 + depth_100)

        # 2. Evaluate the Driver anchor (R_driver)
        hcp_keys_driver = sorted(ssd_tee_shot_dispersion_from_fairway_center.keys())

        left_disp = [ssd_tee_shot_dispersion_from_fairway_center[k]['left'] for k in hcp_keys_driver]
        right_disp = [ssd_tee_shot_dispersion_from_fairway_center[k]['right'] for k in hcp_keys_driver]

        lat_driver_spread = np.interp(hcp, hcp_keys_driver, left_disp) + np.interp(hcp, hcp_keys_driver, right_disp)

        avg_dist = self._interpolate_1d(hcp, self.ss_driver_avg)
        good_dist = self._interpolate_1d(hcp, self.ss_driver_good)

        # Assuming (good_dist - avg_dist) is 1 standard deviation, the full spread is 4 sigma
        depth_driver_spread = (good_dist - avg_dist) * 4.0

        R_driver = lat_driver_spread / (lat_driver_spread + depth_driver_spread)

        # 3. Interpolate the target distance
        if distance <= 100.0:
            return R_100
        elif distance >= avg_dist:
            return R_driver
        else:
            # Linear interpolation between 100 yards and the Driver average distance
            progress = (distance - 100.0) / (avg_dist - 100.0)
            return R_100 + progress * (R_driver - R_100)

    def _get_bunker_distance_loss(self, hcp, distance):
        """
        Calculates the median distance lost when hitting from a bunker.
        Smoothly ramps up the penalty between 50 and 105 yards.
        """
        lookup_hcp = np.clip(hcp, self.stagner_hcp_bins[0], self.stagner_hcp_bins[-1])
        lookup_dist = np.clip(distance, self.stagner_dist_bins[0], self.stagner_dist_bins[-1])

        # Get the raw penalty from the Stagner matrix
        raw_penalty = float(self.bunker_loss_interp((lookup_hcp, lookup_dist)))

        # Do not apply fairway bunker distance loss to short greenside splashes.
        # Ramp the penalty from 0% at 50 yards to 100% at 105 yards.
        if distance < 50.0:
            return 0.0
        elif distance < 105.0:
            ramp_factor = (distance - 50.0) / 55.0
            return raw_penalty * ramp_factor

        return raw_penalty

    def _init_club_data(self):
        """
        Cleans the raw Hireko dictionary, imputes missing data,
        and builds monotonic interpolators for every club.
        """
        raw_data = hd_dist_dict_yards_men.copy()

        # Remove the rogue 'CLUB' key
        raw_data.pop('CLUB', None)

        # Load into pandas and transpose so Handicaps are the index (rows) and Clubs are columns
        df = pd.DataFrame(raw_data).sort_index()

        # Impute the missing -5 HCP 3-Hybrid using the proportional gap (~246 yards)
        if pd.isna(df.loc[-5, '3-Hybrid']):
            df.loc[-5, '3-Hybrid'] = 246.0

        self.club_interpolators = {}

        # PchipInterpolator is perfect here because it prevents polynomial overshoot
        # while requiring strictly increasing x-values (which our sorted [-5, 5, 15, 25] index provides)
        for club in df.columns:
            self.club_interpolators[club] = PchipInterpolator(df.index, df[club])

    def get_player_bag(self, hcp: float) -> dict:
        """
        Generates a complete dictionary of expected yardages for a specific handicap.
        Run this once at the start of the program for a specific player.
        """
        # Clamp handicap to the trained bounds of the data (-5 to 25)
        clamped_hcp = max(-5.0, min(float(hcp), 25.0))

        bag = {}
        for club, interp in self.club_interpolators.items():
            bag[club] = float(interp(clamped_hcp))

        # Sort the bag by distance (descending) for easier reading
        return dict(sorted(bag.items(), key=lambda item: item[1], reverse=True))

    def get_recommended_club(self, hcp: float, target_distance: float) -> tuple[str, float]:
        """
        Returns the (club_name, club_distance) closest to the target while still being larger.
        """
        bag = self.get_player_bag(hcp)

        # Filter for clubs that go AT LEAST the target distance
        valid_clubs = {club: dist for club, dist in bag.items() if dist >= target_distance}

        if not valid_clubs:
            # The target is further than their longest club! Return the Driver.
            longest_club = max(bag, key=bag.get)
            return longest_club, bag[longest_club]

        # Select the shortest club that can still clear the target
        best_club = min(valid_clubs, key=valid_clubs.get)
        return best_club, valid_clubs[best_club]

    def get_leave_ratio(self, hcp, distance, lie='fairway'):
        """
        Calculates the base median leave ratio, applying a synthetic expansion
        for penal lies based on Tour data and handicap dampening.
        """
        # 1. Fetch the baseline fairway leave ratio using your existing interpolator
        lookup_hcp = np.clip(hcp, self.broadie_hcp_bins[0], self.broadie_hcp_bins[-1])
        lookup_dist = np.clip(distance, self.broadie_dist_bins[0], self.broadie_dist_bins[-1])
        base_ratio = float(self.interp_fairway((lookup_hcp, lookup_dist)))

        if lie == 'fairway':
            return base_ratio


    def get_leave_ratio(self, hcp, distance, lie='fairway'):
        """
        Calculates the base median leave ratio, applying a synthetic expansion
        for penal lies based on Tour data and handicap dampening.
        """
        '''
        median_leave_model.py output:

        Bin (Yds)    | Rough Penalty   | Sand Penalty
        ---------------------------------------------
        10-30        | 1.45            | 1.52
        30-70        | 1.42            | 1.40
        70-130       | 1.75            | 2.45
        130-170      | 1.78            | 1.97
        170-250      | 1.83            | 2.37
        '''

        lookup_hcp = np.clip(hcp, self.broadie_hcp_bins[0], self.broadie_hcp_bins[-1])

        if lie == 'fairway':
            lookup_dist = np.clip(distance, self.broadie_dist_bins[0], self.broadie_dist_bins[-1])
            return float(self.interp_fairway((lookup_hcp, lookup_dist)))

        elif lie == 'rough':
            lookup_dist = np.clip(distance, self.broadie_dist_bins[0], self.broadie_dist_bins[-1])
            return float(self.interp_rough((lookup_hcp, lookup_dist)))

        elif lie == 'sand':
            if distance <= 40.0:
                lookup_dist = np.clip(distance, self.sand_dist_bins[0], self.sand_dist_bins[-1])
                return float(self.interp_sand_short((lookup_hcp, lookup_dist)))
            else:
                # Use shotlink fairway base + tour multiplier scaling for longer sand shots
                lookup_dist = np.clip(distance, self.broadie_dist_bins[0], self.broadie_dist_bins[-1])
                base_ratio = float(self.interp_fairway((lookup_hcp, lookup_dist)))

                hinge_dists = [20.0, 50.0, 100.0, 150.0, 210.0]
                tour_mults = [1.52, 1.40, 2.45, 1.97, 2.37]
                raw_multiplier = np.interp(distance, hinge_dists, tour_mults)

                hcp_dampening = np.clip(1.0 - (hcp / 52.0), 0.5, 1.0)
                final_multiplier = 1.0 + ((raw_multiplier - 1.0) * hcp_dampening)
                return base_ratio * final_multiplier

    def _interpolate_1d(self, hcp, values):
        return np.interp(hcp, self.ss_hcp_keys, values)

    def _clamp_to_max_distance(self, hcp, requested_distance):
        """
        Ensures the target distance does not exceed the golfer's physical limits,
        based on Shot Scope's 'Well Struck' driver data.
        """
        max_possible_distance = np.interp(hcp, self.ss_hcp_keys, self.ss_driver_good)

        if requested_distance > max_possible_distance:
            return float(max_possible_distance)

        return float(requested_distance)

    def get_polar_parameters(self, hcp, distance, lie='fairway', handedness='R', tuning_scalar_override=None, correlation_override=None):
        """
        Calculates the exact dist_1, dist_2, and P parameters based on the data.
        """
        # 0. Enforce Data-Driven Reality
        clamped_distance = self._clamp_to_max_distance(hcp, distance)

        # 1. FETCH PRE-CALIBRATED OPTIMAL SCALAR FOR SPECIFIC LIE
        if tuning_scalar_override is not None:
            tuning_scalar = tuning_scalar_override
        elif self.calibrated:
            lookup_dist = np.clip(clamped_distance, self.scalar_dist_bins.min(), self.scalar_dist_bins.max())
            lookup_hcp = np.clip(hcp, self.scalar_hcp_bins.min(), self.scalar_hcp_bins.max())

            # Select the correct interpolator, fallback to fairway if something weird is passed
            interpolator = self.scalar_interpolators.get(lie, self.scalar_interpolators['fairway'])
            tuning_scalar = float(interpolator((lookup_hcp, lookup_dist)))
        else:
            tuning_scalar = 1.0

        # 2. BASE ERROR (Broadie Median Leave Matrix)
        leave_ratio = self.get_leave_ratio(hcp, clamped_distance, lie)
        median_error_yards = leave_ratio * clamped_distance

        # 3. MIXTURE PROBABILITY (Catastrophic Tail)
        # Sourced empirically from Broadie's "Awful Shot Scorecard" and total shot counts.
        if clamped_distance < 65.0:
            # Short Game Catastrophic Miss Rates (< 65 yards)
            miss_rates = bd_short_game_miss_rates
        else:
            # Long Game Catastrophic Miss Rates (>= 65 yards)
            miss_rates = bd_long_game_miss_rates

        dynamic_miss_rate = np.interp(hcp, bd_hcp_anchors, miss_rates)
        P_primary = 1.0 - dynamic_miss_rate
        P_miss = dynamic_miss_rate

        ''' Previous theory that never really worked:

        MAX_P_MISS = 0.15

        # 3. MIXTURE PROBABILITY (Shot Scope Short Miss %b vs Catastrophic Tail)
        'note here, the existing theory was weighting miss hits'
        'far too high we set our our miss hits at 0.15 max here.'
        'Datamining, sure, but theory is just not matching the data.'

        total_short_miss_rate = self._interpolate_1d(hcp, self.ss_short_miss_pct) / 100.0

        # The primary Gaussian naturally produces short shots (symmetric variance).
        # We estimate this baseline using the average of the other three quadrants (Left, Right, Long).
        # Since the 4 quadrants sum to 1.0, the non-short portion is (1.0 - total_short_miss_rate).
        symmetric_baseline = (1.0 - total_short_miss_rate) / 3.0

        # The probability of a true mechanical 'mishit' (dist_2) is only the excess short misses
        data_P_miss = total_short_miss_rate - symmetric_baseline
        max_P_miss = MAX_P_MISS
        P_miss = min(data_P_miss, max_P_miss)

        # We must not let Shot Scope's general "short" stat be unreasonable.
        # A 15% weight turns dist_2 into a true heavy tail rather than a bimodal blob.
        P_primary = 1.0 - P_miss
        '''

        # 4. MISS OFFSET (Standard Deviation Paradigm)
        avg_dist = self._interpolate_1d(hcp, self.ss_driver_avg)
        good_dist = self._interpolate_1d(hcp, self.ss_driver_good)

        # Treat 'good_dist' as exactly +1 standard deviation from the mean
        sigma_power = good_dist - avg_dist

        # Calculate the Coefficient of Variation (variance as a % of total distance)
        power_cv = sigma_power / avg_dist

        # Apply the CV to the specific club/distance being hit
        sigma_club = clamped_distance * power_cv

        # Define a catastrophic mishit as a -3 sigma event
        # (You can tune this multiplier: 2.5 is a mild mishit, 4.0 is a complete top)
        miss_offset_yards = sigma_club * 3.0

        ''' Previous theory that never really worked:

        # 4. MISS OFFSET (Shot Scope Driver Efficiency Ratio)
        avg_dist = self._interpolate_1d(hcp, self.ss_driver_avg)
        good_dist = self._interpolate_1d(hcp, self.ss_driver_good)

        # Protect against division by zero in extreme edge cases
        safe_P_miss = max(P_miss, 0.01)

        # Calculate the true yardage gap between a well-struck shot and a mishit
        true_driver_miss_offset = (good_dist - avg_dist) / safe_P_miss

        # Convert to an efficiency loss ratio to apply uniformly across all clubs
        efficiency_loss = true_driver_miss_offset / good_dist
        miss_offset_yards = clamped_distance * efficiency_loss
        '''

        # 5. ERROR BUDGET ALLOCATION (Dynamic Variance Delta)

        # A. Calculate the baseline empirical Fairway variance
        fairway_leave = self.get_leave_ratio(hcp, clamped_distance, lie='fairway')
        fairway_variance = (fairway_leave * clamped_distance * tuning_scalar) ** 2

        # B. Calculate the dynamic fairway split using Shot Scope data
        fairway_lateral_ratio = self._get_dynamic_fairway_split(hcp, clamped_distance)
        base_lateral_variance = fairway_variance * fairway_lateral_ratio

        if lie == 'fairway':
            lateral_variance = base_lateral_variance
            depth_variance = fairway_variance * (1.0 - fairway_lateral_ratio)
        else:
            # C. Fetch the actual penal error magnitude using Broadie's Rough/Sand tables
            penal_leave = self.get_leave_ratio(hcp, clamped_distance, lie=lie)
            penal_variance = (penal_leave * clamped_distance * tuning_scalar) ** 2

            # D. The Variance Delta: Hold lateral dispersion relatively steady to the fairway,
            # and assign all the new chaotic penal variance directly to the depth axis.
            # (We scale lateral slightly by 1.1x to account for minor clubface twisting in thick grass/sand)
            lateral_variance = base_lateral_variance * 1.1
            depth_variance = penal_variance - lateral_variance

            # E. Safety catch: If a specific distance/HCP matrix combo behaves weirdly,
            # ensure depth variance never drops below a 50% minimum share.
            depth_variance = max(depth_variance, penal_variance * 0.5)

        # 6. ANGULAR ERROR (Theta)
        lateral_std = np.sqrt(lateral_variance)

        theta_std_primary = lateral_std / clamped_distance
        theta_var_primary = theta_std_primary ** 2

        # A mishit loses face control, so angular std dev actually increases (1.5x).
        # The polar conversion (X = R * sin(theta)) natively forms the funnel.
        theta_std_miss = theta_std_primary * 1.5
        theta_var_miss = theta_std_miss ** 2

        # 7. DISTANCE ERROR (Radius)
        base_depth_std = np.sqrt(depth_variance)

        # Relax the primary constraint so the core can breathe and widen
        r_std_primary = base_depth_std * 0.95
        r_var_primary = r_std_primary ** 2

        # [UPDATED] Use Shot Scope's efficiency loss to define the top of the mishit zone.
        miss_upper_bound = max(1.0, clamped_distance - miss_offset_yards)

        # Size the standard deviation so +/- 2 sigmas span perfectly from 0 to the upper bound.
        r_std_miss = miss_upper_bound / 4.0
        r_var_miss = r_std_miss ** 2

        # 8. AERODYNAMIC CORRELATION (Spin Bias / Tilt)
        base_correlation = correlation_override if correlation_override is not None else -0.35

        # Aerodynamics (pull-draw/push-fade) do not apply to short pitches or bunker splashes.
        if clamped_distance < 50.0:
            decay_factor = max(0.0, (clamped_distance - 15.0) / 35.0)
            base_correlation *= decay_factor

        correlation = base_correlation if handedness.upper() == 'R' else -base_correlation

        # Apply the curve ONLY to the primary, well-struck shots
        cov_rt_primary = correlation * r_std_primary * theta_std_primary

        # Mishits lack stable spin. Force strict statistical independence (orthogonal).
        cov_rt_miss = 0.0

        eps = 1e-8

        # 9. APPLY FAIRWAY BUNKER DISTANCE LOSS (Mean Shift)
        actual_distance_mean = clamped_distance
        if lie == 'sand':
            distance_penalty = self._get_bunker_distance_loss(hcp, clamped_distance)
            actual_distance_mean -= distance_penalty

        # 10. BUILD PURE POLAR DICTIONARIES
        dist_1_polar = {
            'mean': [actual_distance_mean, 0.0],
            'cov': [[r_var_primary + eps, cov_rt_primary],
                    [cov_rt_primary, theta_var_primary + eps]]
        }

        dist_2_polar = {
            # [UPDATED] Center the mishits halfway between the golfer and the miss ceiling
            'mean': [miss_upper_bound / 2.0, 0.0],
            'cov': [[r_var_miss + eps, cov_rt_miss],
                    [cov_rt_miss, theta_var_miss + eps]]
        }

        return dist_1_polar, dist_2_polar, P_primary

    def _apply_aerodynamic_squish(self, hcp, polar_shots, clamped_target):
        """
        Compresses the long-tail of a distribution against the golfer's physical limits
        using a 'leaky' boundary based on power variance.
        """
        r = polar_shots[:, 0]

        # 1. Fetch Shot Scope Baselines
        good_dist = self._interpolate_1d(hcp, self.ss_driver_good)
        avg_dist = self._interpolate_1d(hcp, self.ss_driver_avg)

        # 2. Define the absolute physical ceiling dynamically.
        # We take the gap between their average and their 'good' shots, and project it forward.
        # A 2.5x multiplier pushes the wall safely to the 99th percentile distance.
        power_variance = good_dist - avg_dist
        abs_max_dist = good_dist + (power_variance * 2.5)

        # 3. Isolate shots that overshot the intended target
        overshoot = r - clamped_target
        long_mask = overshoot > 0

        max_allowance = abs_max_dist - clamped_target
        bleed_rate = 0.15

        if max_allowance > 0:
            base_squish = max_allowance * np.tanh(overshoot[long_mask] / max_allowance)
            leaky_squish = base_squish + (overshoot[long_mask] * bleed_rate)
            r[long_mask] = clamped_target + leaky_squish
        else:
            r[long_mask] = clamped_target + (overshoot[long_mask] * bleed_rate)

        polar_shots[:, 0] = r
        return polar_shots

    def _apply_fat_shot_squish(self, polar_shots):
        """
        Prevents negative distances. A heavily chunked shot has a physical floor.
        Uses an exponential decay for shots dropping below 5 yards to smoothly
        compress the infinite Gaussian tail against a 0.5-yard physical limit.
        """
        r = polar_shots[:, 0]

        # Isolate the tail that is approaching or breaching the zero-boundary
        short_mask = r < 5.0

        if np.any(short_mask):
            # Smoothly asymptote towards 0.5 yards instead of going negative.
            # Example: A mathematical -15y miss compresses to roughly 0.55y.
            r[short_mask] = 4.5 * np.exp((r[short_mask] - 5.0) / 4.5) + 0.5

        polar_shots[:, 0] = r
        return polar_shots

    def _generate_golden_mixture(self, dist_1, dist_2, P_primary, num_shots, seed=SEED):
        """
        Applies Quasi-Monte Carlo to a two-part Gaussian Mixture Model.
        """
        if num_shots == 0:
            return np.empty((0, 2))

        num_core = int(round(num_shots * P_primary))
        num_miss = num_shots - num_core

        shots = []

        # 2. Process the Core Shots (dist_1)
        if num_core > 0:
            sampler_1 = qmc.Halton(d=2, scramble=True, seed=seed)
            u_1 = np.clip(sampler_1.random(n=num_core), 1e-6, 1.0 - 1e-6)
            z_1 = norm.ppf(u_1)

            L_1 = np.linalg.cholesky(dist_1['cov'])

            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                core_shots = z_1 @ L_1.T + dist_1['mean']
            shots.append(core_shots)

        # 3. Process the Miss Shots (dist_2)
        if num_miss > 0:
            # Offset the seed so the miss array doesn't copy the exact spatial pattern of the core array
            miss_seed = (seed + 1) if seed is not None else None
            sampler_2 = qmc.Halton(d=2, scramble=True, seed=miss_seed)
            u_2 = np.clip(sampler_2.random(n=num_miss), 1e-6, 1.0 - 1e-6)

            # [NEW] Hybrid Distribution: Uniform Radius, Gaussian Angle

            # 1. Recover the upper bound from the mean we anchored in get_polar_parameters
            miss_upper_bound = dist_2['mean'][0] * 2.0

            # 2. Radius (Column 0): Pure Uniform distribution
            # We skip norm.ppf() and scale the raw Halton sequence directly to the distance ceiling
            r_miss = u_2[:, 0] * miss_upper_bound

            # 3. Angle (Column 1): Standard Gaussian distribution
            # We keep norm.ppf() here so the lateral spread maintains its funnel shape
            theta_miss = norm.ppf(u_2[:, 1]) * np.sqrt(dist_2['cov'][1][1]) + dist_2['mean'][1]

            miss_shots = np.column_stack((r_miss, theta_miss))
            shots.append(miss_shots)

        final_shots = np.vstack(shots)
        np.random.shuffle(final_shots)

        return final_shots

    def generate_shots(self, hcp, distance, lie='fairway', num_shots=100, handedness='R', return_cartesian=True, tuning_scalar_override=None, correlation_override=None, seed=SEED):
        """
        The master entry point. Given a shot profile, generates a highly accurate,
        clump-free distribution array of simulated shots.

        Returns:
            np.ndarray: An array of shape (N, 2).
                        If return_cartesian is True, columns are [X (lateral), Y (distance)].
                        If False, columns are [R (radius), Theta (angle)].
        """
        # explicitly pass keyword arguments to prevent positional mismatches
        dist_1, dist_2, P_primary = self.get_polar_parameters(
            hcp=hcp,
            distance=distance,
            lie=lie,
            handedness=handedness,
            correlation_override=correlation_override,
            tuning_scalar_override=tuning_scalar_override,
        )

        polar_shots = self._generate_golden_mixture(dist_1, dist_2, P_primary, num_shots, seed=seed)

        if num_shots == 0:
            return np.empty((0, 2))

        # --- APPLY THE PHYSICS BOUNDARY SQUISHES ---
        polar_shots = self._apply_aerodynamic_squish(hcp, polar_shots, dist_1['mean'][0])
        polar_shots = self._apply_fat_shot_squish(polar_shots) # <-- Stop backwards shots

        if return_cartesian:
            r = polar_shots[:, 0]
            theta = polar_shots[:, 1]

            x = r * np.sin(theta)
            y = r * np.cos(theta)

            return np.column_stack((x, y))

        return polar_shots