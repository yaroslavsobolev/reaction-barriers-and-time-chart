"""
Eyring Equation Landscape: Activation Energy vs Reaction Half-Life

A wall-worthy visualization showing the relationship between activation free energy
and reaction half-life across different temperatures, based on transition state theory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
h = 6.62607015e-34  # Planck constant (J·s)
R = 8.314462618     # Gas constant (J/(mol·K))

# Unit conversions
KCAL_TO_J = 4184.0  # kcal/mol to J/mol
KJ_TO_J = 1000.0    # kJ/mol to J/mol
HARTREE_TO_J = 4.359744e-18 * 6.02214076e23  # Hartree to J/mol

# Time conversions (to seconds)
MILLISECOND = 1e-3
SECOND = 1
MINUTE = 60
HOUR = 3600
DAY = 86400
WEEK = 604800
MONTH = 2592000  # 30 days
YEAR = 31557600  # Average year

# Configure matplotlib for publication-quality, Tufte-style plots
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman']
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 0.8
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 8
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['text.usetex'] = False

def eyring_rate_constant(dG_J_per_mol, T_K):
    """Calculate rate constant from Eyring equation."""
    prefactor = (k_B * T_K) / h
    return prefactor * np.exp(-dG_J_per_mol / (R * T_K))

def rate_to_halflife(k):
    """Convert first-order rate constant to half-life."""
    return np.log(2) / k

# Create figure with specific aspect ratio suitable for printing
fig, host = plt.subplots(figsize=(8.5, 10))

# adjust so that top margin is smaller, and bottom is larger
fig.subplots_adjust(top=0.93, bottom=0.2, left=0.09, right=0.92)

# Energy range in kJ/mol (primary unit for calculations)
dG_kJ = np.linspace(0, 200, 1000)
dG_J = dG_kJ * KJ_TO_J

# Temperature range
temps_C = np.arange(-270, 301, 10)
# extend from 300 to 1000 at every 50 degrees
temps_C = np.concatenate((temps_C, np.arange(300, 1001, 50)))
temps_K = temps_C + 273.15

# Dictionary for temperature label positions (temperature: x-position in kJ/mol)
# Customize these positions as needed to avoid overlap and optimize readability
# The angle will be automatically calculated based on the isotherm slope at each position

def find_E_such_that_half_life(T_C, target_half_life_s):
    T_K = T_C + 273.15
    """Find the activation energy (kJ/mol) that gives the target half-life at temperature T_K."""
    from scipy.optimize import bisect

    def half_life_diff(dG_kJ):
        dG_J = dG_kJ * KJ_TO_J
        k = eyring_rate_constant(dG_J, T_K)
        t_half = rate_to_halflife(k)
        return t_half - target_half_life_s

    # Use bisection method to find root
    return bisect(half_life_diff, 0, 200)

right_pos = 180
targ_half_life = 1e9
temp_label_positions = {}
for temp in np.arange(-270, 121, 10):
    pos = find_E_such_that_half_life(temp, targ_half_life)
    temp_label_positions[temp] = pos
for temp in np.arange(140, 301, 20):
    temp_label_positions[temp] = right_pos
for temp in np.arange(400, 1001, 100):
    temp_label_positions[temp] = right_pos

temp_label_positions[140] = 175
temp_label_positions[1000] = 195

# Plot Eyring lines for each temperature
for i, (T_C, T_K) in enumerate(zip(temps_C, temps_K)):
    k = eyring_rate_constant(dG_J, T_K)
    t_half = rate_to_halflife(k)

    # Make lines slightly thicker for labeled temperatures
    if T_C == 20:
        linewidth = 1.5
        alpha = 1
    elif T_C % 20 == 0:
        linewidth = 0.9
        alpha = 0.8
    else:
        linewidth = 0.5
        alpha = 0.6

    host.plot(dG_kJ, t_half, color='black', linewidth=linewidth,
              alpha=alpha, zorder=2)

    # Add temperature labels directly on the lines
    if T_C % 20 == 0 and T_C in temp_label_positions:
        # Get the custom x-position for this temperature
        label_x = temp_label_positions[T_C]
        label_idx = np.argmin(np.abs(dG_kJ - label_x))
        label_y = t_half[label_idx] - 0.6 * t_half[label_idx]  # Slightly below the line

        # Calculate angle of the line for text rotation at this specific position
        # Use log space for y since axis is logarithmic
        if label_idx > 10 and label_idx < len(dG_kJ) - 10:
            # Use points further apart for better angle calculation
            y1 = t_half[label_idx - 10]
            y2 = t_half[label_idx + 10]
            x1 = dG_kJ[label_idx - 10]
            x2 = dG_kJ[label_idx + 10]

            # Calculate in log space to match visual appearance
            if y1 > 0 and y2 > 0:
                y1_log = np.log10(y1)
                y2_log = np.log10(y2)

                # Calculate angle
                dy_log = y2_log - y1_log
                dx = x2 - x1

                # Rough aspect ratio correction (log scale spans ~18 decades over 200 kJ)
                aspect_correction = 200 / 18  # kJ per decade
                angle = np.degrees(np.arctan2(dy_log * aspect_correction, dx))
            else:
                angle = -45
        else:
            angle = -45

        # Place the text label
        if T_C == 20:
            # make font bold
            host.text(label_x, label_y, f'{int(T_C)}°C',
                        rotation=angle, va='bottom', ha='center',
                        fontsize=10, fontweight='bold', color='black', alpha=1,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                 edgecolor='none', alpha=1),
                        zorder=4)
        else:
            host.text(label_x, label_y, f'{int(T_C)}°C'.replace('-', '−'),
                      rotation=angle, va='bottom', ha='center',
                      fontsize=9, color='black', alpha=0.9,
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                               edgecolor='none', alpha=1),
                      zorder=4)

# Add horizontal reference lines for time scales
time_refs = [
    (MILLISECOND, '1 ms', 0.3),
    (SECOND, '1 second', 0.3),
    (MINUTE, '1 minute', 0.3),
    (HOUR, '1 hour', 0.3),
    (DAY, '1 day', 0.3),
    (WEEK, '1 week', 0.3),
    (MONTH, '1 month', 0.3),
    (YEAR, '1 year', 0.3),
    (100*YEAR, '1 century', 0.3),
]

for time_val, label, alpha in time_refs:
    host.axhline(time_val, color='black', linestyle='--', linewidth=0.7,
                 alpha=alpha, zorder=3)
    # Add labels on the right side
    host.text(202, time_val, label, va='center', ha='left',
              fontsize=9, style='italic', alpha=0.7)

# Configure main axis
host.set_ylabel('Half-life of first-order reaction, $t_{1/2} = \\ln(2)/k$ (seconds)', fontsize=11, fontweight='normal')
host.set_yscale('log')
host.set_ylim(1e-6, 1e10)
host.set_xlim(0, 200)

# Add very pale gridlines (both horizontal and vertical)
host.grid(True, which='major', axis='y', linestyle=':', linewidth=0.4,
          alpha=0.3, color='gray')
host.grid(True, which='major', axis='x', linestyle=':', linewidth=0.7,
          alpha=0.5, color='gray')

# Set the main x-axis label and ticks (kJ/mol)
host.set_xlabel('(kJ/mol)', fontsize=11)
host.set_xticks(np.arange(0, 201, 10))

# Create additional x-axes for kcal/mol and Hartree
ax_kcal = host.twiny()
ax_hartree = host.twiny()

# Position the additional axes below the main plot
ax_kcal.spines['bottom'].set_position(('outward', 40))
ax_kcal.spines['bottom'].set_visible(True)
ax_kcal.spines['top'].set_visible(False)
ax_kcal.xaxis.set_ticks_position('bottom')
ax_kcal.xaxis.set_label_position('bottom')

ax_hartree.spines['bottom'].set_position(('outward', 80))
ax_hartree.spines['bottom'].set_visible(True)
ax_hartree.spines['top'].set_visible(False)
ax_hartree.xaxis.set_ticks_position('bottom')
ax_hartree.xaxis.set_label_position('bottom')

# Set up kcal/mol axis (every 1 kcal/mol)
# 1 kcal/mol = 4.184 kJ/mol
ax_kcal.set_xlim(0, 200)  # Same as host in kJ/mol
max_kcal = int(np.ceil(200 / 4.184))
kcal_ticks_values = np.arange(0, max_kcal + 1, 1)
kcal_ticks_positions = kcal_ticks_values * 4.184
# Filter to keep only ticks within range
mask_kcal = kcal_ticks_positions <= 200
kcal_ticks_values = kcal_ticks_values[mask_kcal]
kcal_ticks_positions = kcal_ticks_positions[mask_kcal]
ax_kcal.set_xticks(kcal_ticks_positions)
ax_kcal.set_xticklabels([f'{int(x)}' for x in kcal_ticks_values], fontsize=9)
ax_kcal.set_xlabel('(kcal/mol)', fontsize=12)

# Set up Hartree axis (every 0.001 Hartree)
# 1 Hartree = 2625.5 kJ/mol
ax_hartree.set_xlim(0, 200)  # Same as host in kJ/mol
hartree_to_kJ = 2625.5/1000
max_hartree_ticks = int(np.ceil(200 / (hartree_to_kJ * 0.001)))
hartree_ticks_values = np.arange(0, max_hartree_ticks + 1, 1) * 5
hartree_ticks_positions = hartree_ticks_values * hartree_to_kJ
# Filter to keep only ticks within range
mask_hartree = hartree_ticks_positions <= 200
hartree_ticks_values = hartree_ticks_values[mask_hartree]
hartree_ticks_positions = hartree_ticks_positions[mask_hartree]
ax_hartree.set_xticks(hartree_ticks_positions)
ax_hartree.set_xticklabels([f'{x:.0f}' for x in hartree_ticks_values], fontsize=11)
ax_hartree.set_xlabel('Gibbs free energy of activation $\\Delta G^\\ddagger$ (milliHartree)', fontsize=12)

# Title with proper spacing
fig.suptitle('Reaction rate as a function of activation free energy and temperature', fontsize=16, fontweight='bold', y=0.99)
plt.text(0.5, 0.942,
         'Based on Eyring-Polanyi-Evans equation from transition state theory: $k = \\frac{k_B T}{h} \\kappa e^{-\\Delta G^\\ddagger / k_b T}$, $\\kappa=1$',
         ha='center', fontsize=11, style='italic', transform=fig.transFigure)

# Attribution and metadata (Tufte-style: small, unobtrusive, but complete)
attribution = (
    'Plotted by Yaroslav I. Sobolev. Ulsan, Republic of Korea. 2025.\n'
    'github.com/yaroslavsobolev/reaction-barriers-and-time-chart'
)

fig.text(0.99, 0.01, attribution, ha='right', fontsize=8.5,
         style='italic', color='dimgray', wrap=True, transform=fig.transFigure)

# hide top and right axis spines
host.spines['top'].set_visible(False)
host.spines['right'].set_visible(False)
axs = [host, ax_kcal, ax_hartree]
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


# Save with high quality
plt.savefig('eyring_landscape.pdf', dpi=300,# bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('eyring_landscape.png', dpi=300,# bbox_inches='tight',
            facecolor='white', edgecolor='none')

# plt.show()