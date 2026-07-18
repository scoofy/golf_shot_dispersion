import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from scipy.stats import gaussian_kde

from data_model import DispersionDataModel

# ==========================================
# CONFIGURATION
# ==========================================
TEST_DISTANCE = 10          # Target distance in yards
TEST_LIE = 'sand'
NUM_SHOTS = 1000            # 2,500 shots per HCP keeps initialization fast but contours smooth

HCPS = list(range(-5, 36))  # -5 to 35
SEED = 0

# The statistical probabilities for 2D standard deviations
SIGMA_LEVELS = [1.0, 2.0, 3.0]
SIGMA_COLORS = ['darkred', 'red', 'lightcoral'] # Inner (1σ), Middle (2σ), Outer (3σ)

# Generate evenly spaced RGBA colors based on viridis for the handicaps
cmap = plt.get_cmap('viridis')
rgba_colors = cmap(np.linspace(0, 1, len(HCPS)))
COLORS = [mcolors.to_hex(color) for color in rgba_colors]

def plot_interactive_sigma_contours(distance):
    print(f"Initializing KDE Topography for {len(HCPS)} Handicaps...")
    print("This will take approximately 5-10 seconds...")

    data_model = DispersionDataModel()

    fig, ax = plt.subplots(figsize=(14, 10))
    plt.subplots_adjust(bottom=0.25) # Increased from 0.2 to make room for stacked legends

    # Plot Targets
    ax.plot(0, 0, 'go', markersize=10, label='Golfer (Origin)', zorder=5)
    ax.plot(0, distance, 'kX', markersize=14, label='Intended Target', zorder=5)

    artists_by_hcp = {hcp: [] for hcp in HCPS}
    color_map = list(reversed(SIGMA_COLORS)) # contours plot low-to-high density
    club_by_hcp = {} # Cache selected clubs for terminal output and dynamic UI

    # ==========================================
    # DATA GENERATION & KDE EVALUATION LOOP
    # ==========================================
    print(f"\n--- Selected Clubs for {distance}y Target ---")
    for hcp, color in zip(HCPS, COLORS):
        # 1. Fetch the appropriate club using local distance parameter (not global)
        club_name, _ = data_model.get_recommended_club(hcp, distance)
        club_by_hcp[hcp] = club_name
        print(f"HCP {hcp:3d}: {club_name}")

        # 2. Generate shots aimed EXACTLY at the target distance
        dist_1, dist_2, P_primary = data_model.get_polar_parameters(hcp, distance, handedness='R')

        shots = data_model.generate_shots(
            hcp=hcp,
            distance=distance,
            lie=TEST_LIE,
            num_shots=NUM_SHOTS,
            handedness='R',
            return_cartesian=True
        )

        x = shots[:, 0]
        y = shots[:, 1]

        # 2. Fit the KDE (Silence the Apple Accelerate vecdot warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            kde = gaussian_kde(shots.T)

            # 3. Create a LOCAL grid to evaluate the KDE (ensures smooth contours for small distributions)
            x_margin = max(5.0, (x.max() - x.min()) * 0.2)
            y_margin = max(5.0, (y.max() - y.min()) * 0.2)

            xmin, xmax = x.min() - x_margin, x.max() + x_margin
            ymin, ymax = y.min() - y_margin, y.max() + y_margin

            X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([X.ravel(), Y.ravel()])

            # Evaluate density
            Z = np.reshape(kde(positions).T, X.shape)

        # 4. Calculate exactly which density thresholds correspond to the Sigma volumes
        Z_flat = Z.ravel()
        Z_sorted = np.sort(Z_flat)[::-1]
        Z_cumsum = np.cumsum(Z_sorted) / Z_sorted.sum()

        contour_levels = []
        for k in SIGMA_LEVELS:
            target_volume = 1.0 - np.exp(-(k**2) / 2.0)
            idx = np.searchsorted(Z_cumsum, target_volume)
            if idx < len(Z_sorted):
                contour_levels.append(Z_sorted[idx])
            else:
                contour_levels.append(Z_sorted[-1])

        contour_levels = sorted(contour_levels)

        # 5. Plot Scatter (using Handicap Viridis Color)
        sample_idx = np.random.choice(NUM_SHOTS, size=min(500, NUM_SHOTS), replace=False)
        sc = ax.scatter(x[sample_idx], y[sample_idx], color=color, alpha=0.3, s=8, zorder=1)
        artists_by_hcp[hcp].append(sc)

        # 6. Plot Contours (using fixed Sigma Colors)
        cs = ax.contour(X, Y, Z, levels=contour_levels, colors=color_map, linewidths=2.0, zorder=3, alpha=0.85)

        # Store contour collections for interactive toggling (Cross-version compatibility)
        if hasattr(cs, 'collections'):
            # Matplotlib < 3.8
            for collection in cs.collections:
                artists_by_hcp[hcp].append(collection)
        else:
            # Matplotlib >= 3.8 (QuadContourSet is itself an Artist)
            artists_by_hcp[hcp].append(cs)

    # ==========================================
    # STYLING & BASE UI
    # ==========================================
    base_title = f"True 2D Dispersion Density\nTarget: {distance} Yards ({TEST_LIE.upper()})"
    ax.set_title(f"{base_title}\nDisplay: ALL", fontsize=16, fontweight='bold')
    ax.set_xlabel("Lateral Miss (Yards) [Right is +]")
    ax.set_ylabel("Total Distance (Yards)")

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(distance, color='black', alpha=0.2, linestyle='--')
    ax.axvline(0, color='black', alpha=0.2, linestyle='--')

    # Matplotlib automatically set the limits to the widest element (HCP 35) during plotting.
    # Locking datalim ensures the camera doesn't zoom in/out as we cycle.
    ax.set_aspect('equal', 'datalim')

    # ==========================================
    # LEGENDS
    # ==========================================
    # LEGEND 1: Density Boundaries (Top)
    sigma_handles = []
    for sigma, color in zip(SIGMA_LEVELS, SIGMA_COLORS):
        prob = (1.0 - np.exp(-(sigma**2) / 2.0)) * 100
        label = f"{sigma}σ ({prob:.1f}%)"
        sigma_handles.append(mlines.Line2D([], [], color=color, linewidth=3, label=label))

    leg_sigma = ax.legend(handles=sigma_handles, loc='upper center', bbox_to_anchor=(0.5, -0.08),
              ncol=3, title="Mathematical Density Boundaries", fontsize=10, title_fontsize=11, framealpha=0.9)
    ax.add_artist(leg_sigma) # Add so the second legend doesn't overwrite it

    # LEGEND 2: Skill Levels (Bottom)
    hcp_handles = []
    legend_hcps = [h for h in HCPS if h % 5 == 0]

    for hcp in legend_hcps:
        idx = HCPS.index(hcp)
        club_name = club_by_hcp[hcp] # Using the cache from the previous edit
        label_text = f'HCP {hcp} ({club_name})'
        hcp_handles.append(mlines.Line2D([], [], color=COLORS[idx], marker='o',
                                         linestyle='None', markersize=8, label=label_text))

    ax.legend(handles=hcp_handles, loc='upper center', bbox_to_anchor=(0.5, -0.18),
                     ncol=5, title="Skill Level Reference [PRESS SPACE TO CYCLE]",
                     fontsize=10, title_fontsize=11, framealpha=0.9)

    # ==========================================
    # INTERACTIVE KEYPRESS LOGIC
    # ==========================================
    view_state = {'idx': -1}

    def on_key(event):
        if event.key not in [' ', 'space', 'right']:
            return

        view_state['idx'] += 1
        if view_state['idx'] >= len(HCPS):
            view_state['idx'] = -1

        target_hcp = "ALL" if view_state['idx'] == -1 else HCPS[view_state['idx']]

        for h, artists in artists_by_hcp.items():
            is_visible = (target_hcp == "ALL" or h == target_hcp)
            for artist in artists:
                artist.set_visible(is_visible)

        if target_hcp == "ALL":
            ax.set_title(f"{base_title}\nDisplay: ALL", fontsize=16, fontweight='bold', color='black')
        else:
            club_name = club_by_hcp[target_hcp]
            ax.set_title(f"{base_title}\nDisplay: HCP {target_hcp} ONLY — {club_name}",
                         fontsize=16, fontweight='bold', color=COLORS[view_state['idx']])

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)

    print("Initialization complete! Opening interactive window...")
    plt.show()

if __name__ == "__main__":
    plot_interactive_sigma_contours(distance=TEST_DISTANCE)