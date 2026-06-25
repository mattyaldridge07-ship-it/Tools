"""
Conjugate thermal analysis and transient diving simulator for Autonomous Underwater Vehicles (AUVs).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
import argparse
import warnings
warnings.filterwarnings('ignore')

# Physical Constants
G0_ms2      = 9.80665       # m/s²

# Color Palette
BG      = '#0f0f0e'
GOLD    = '#b8920a'
MOSS    = '#4a7a4b'
RED     = '#c04040'
CYAN    = '#40b0c0'
PAPER   = '#f0ede8'
DIM     = '#8a8a7a'
GREY    = '#3a3a38'
BLUE    = '#4080c0'
COLORS  = [GOLD, CYAN, MOSS, RED, BLUE]

# Vessel Geometry & Material Properties
D_OUT_m        = 0.32       # 320 mm outer diameter (standard AUV class)
T_WALL_m       = 0.008      # 8 mm wall thickness
L_VESSEL_m     = 2.0        # 2.0 m cylinder length
A_CONTACT_m2   = 0.05       # 0.05 m² (500 cm²) electronics contact footprint

# Thermal Interface Material properties
R_TIM_spec     = 0.5e-4     # K·m²/W  (0.5 K·cm²/W)
R_TIM_W        = R_TIM_spec / A_CONTACT_m2  # 0.001 K/W

# Seawater properties at ~10°C
RHO_SW         = 1025.0     # kg/m³
CP_SW          = 3993.0     # J/kgK
K_SW           = 0.60       # W/mK
MU_SW          = 1.08e-3    # Pa·s
PR_SW          = 7.2        # Prandtl number

# Materials Database
MATERIALS = {
    'titanium': {
        'name': 'Titanium (Ti-6Al-4V)',
        'k': 6.7,           # W/mK
        'rho': 4430.0,      # kg/m³
        'Cp': 526.0,        # J/kgK
    },
    'aluminium': {
        'name': 'Aluminium (Al-6061)',
        'k': 167.0,
        'rho': 2700.0,
        'Cp': 896.0,
    }
}

# 1. Depth-Temperature Profile
def seawater_temp_at_depth(depth_m):
    """
    Computes seawater temperature as a function of depth.
    Surface (0-50m): 15°C. Thermocline (50-500m): drops to 4°C. Deep (>500m): 4°C.
    """
    if depth_m <= 50.0:
        return 15.0
    elif depth_m <= 500.0:
        return 15.0 - 11.0 * (depth_m - 50.0) / 450.0
    else:
        return 4.0

# 2. External Convection (Churchill-Bernstein)
def churchill_bernstein_convection(V_ms, D_outer_m):
    """
    Churchill-Bernstein correlation for convective heat transfer over a cylinder.
    """
    # Clip velocity to prevent Re=0
    V_eff = max(0.01, V_ms)
    Re = RHO_SW * V_eff * D_outer_m / MU_SW
    
    # Churchill-Bernstein
    term1 = 0.62 * Re**0.5 * PR_SW**(1.0/3.0)
    term2 = (1.0 + (0.4 / PR_SW)**(2.0/3.0))**0.25
    term3 = (1.0 + (Re / 282000.0)**(5.0/8.0))**(4.0/5.0)
    
    Nu = 0.3 + (term1 / term2) * term3
    h_ext = Nu * K_SW / D_outer_m
    return h_ext, Re

# 3. Steady-State Resistance Network
def solve_steady_state(P_elec_W, V_ms, depth_m, mat_name='titanium'):
    """
    Solves steady-state junction and wall temperatures.
    """
    mat = MATERIALS[mat_name]
    T_sw_C = seawater_temp_at_depth(depth_m)
    T_sw_K = T_sw_C + 273.15
    
    # Cylindrical wall resistance
    r_o = D_OUT_m / 2.0
    r_i = r_o - T_WALL_m
    R_wall = np.log(r_o / r_i) / (2.0 * np.pi * mat['k'] * L_VESSEL_m)
    
    # External resistance
    h_ext, _ = churchill_bernstein_convection(V_ms, D_OUT_m)
    A_ext = np.pi * D_OUT_m * L_VESSEL_m
    R_ext = 1.0 / (h_ext * A_ext)
    
    # Junction temperature
    T_junction_K = T_sw_K + P_elec_W * (R_TIM_W + R_wall + R_ext)
    T_hull_outer_K = T_sw_K + P_elec_W * R_ext
    T_hull_inner_K = T_hull_outer_K + P_elec_W * R_wall
    
    status = "OK"
    if T_junction_K - 273.15 > 85.0:
        status = "OVERHEATING"
        
    return {
        'T_junction_C':    T_junction_K - 273.15,
        'T_hull_inner_C':  T_hull_inner_K - 273.15,
        'T_hull_outer_C':  T_hull_outer_K - 273.15,
        'R_wall':          R_wall,
        'R_ext':           R_ext,
        'R_TIM':           R_TIM_W,
        'status':          status
    }

# 4. Transient Diving Simulator
def transient_diving_ode(t, y, P_elec_W, V_ms, mat, M_hull_kg, M_internal_kg):
    """
    ODE system for AUV thermal transient during a dive cycle.
    y = [T_hull_K, T_internal_K]
    """
    T_hull, T_internal = y
    
    # Seawater depth trajectory: dive at 1 m/s to 300m (t=0 to 300s),
    # cruise at 300m (t=300 to 11100s), rise at 1 m/s (t=11100 to 11400s)
    if t < 300.0:
        depth = 1.0 * t
    elif t < 11100.0:
        depth = 300.0
    elif t < 11400.0:
        depth = 300.0 - 1.0 * (t - 11100.0)
    else:
        depth = 0.0
        
    T_sw_C = seawater_temp_at_depth(depth)
    T_sw_K = T_sw_C + 273.15
    
    # Convection properties
    h_ext, _ = churchill_bernstein_convection(V_ms, D_OUT_m)
    A_ext = np.pi * D_OUT_m * L_VESSEL_m
    R_ext = 1.0 / (h_ext * A_ext)
    
    # Internal conduction (mounted electronics through TIM)
    Q_internal = (T_internal - T_hull) / R_TIM_W
    Q_external = (T_hull - T_sw_K) / R_ext
    
    # ODE rates
    dT_hull_dt = (Q_internal - Q_external) / (M_hull_kg * mat['Cp'])
    dT_internal_dt = (P_elec_W - Q_internal) / (M_internal_kg * 500.0)  # electronics mass
    
    return [dT_hull_dt, dT_internal_dt]

def run_diving_cycle(mat_name='titanium', P_elec_W=200.0, V_ms=1.5):
    mat = MATERIALS[mat_name]
    r_o = D_OUT_m / 2.0
    r_i = r_o - T_WALL_m
    
    # Mass calculations
    V_metal = np.pi * (r_o**2 - r_i**2) * L_VESSEL_m
    M_hull = V_metal * mat['rho']
    M_internal = 2.5  # kg internal electronics board weight
    
    t_eval = np.linspace(0.0, 11400.0, 500)
    sol = solve_ivp(
        transient_diving_ode, [0.0, 11400.0], [288.15, 288.15],
        args=(P_elec_W, V_ms, mat, M_hull, M_internal),
        method='RK45',
        t_eval=t_eval,
        rtol=1e-5, atol=1e-4
    )
    
    return sol.t, sol.y[0] - 273.15, sol.y[1] - 273.15

# 5. Print Summary Report
SCENARIOS = [
    # Power [W], Speed [m/s], Depth [m], Material
    (200.0,  1.5,  10.0,  'titanium', 'Surface Transit'),
    (200.0,  1.5,  300.0, 'titanium', 'Mid-depth Cruise'),
    (400.0,  1.5,  300.0, 'titanium', 'High Power Dive'),
    (200.0,  1.5,  300.0, 'aluminium', 'Aluminium Alternative'),
]

def print_report():
    print()
    print("=" * 105)
    print("  AUV PRESSURE VESSEL THERMAL SIZING CALCULATOR — STEADY STATE RESULTS")
    print("=" * 105)
    print(f"  {'Scenario':25}  {'Power [W]':>10}  {'Speed [m/s]':>12}  "
          f"{'Depth [m]':>10}  {'T_junction':>12}  {'R_ext [K/W]':>12}  {'Status':>8}")
    print("-" * 105)
    
    for P, V, depth, mat, label in SCENARIOS:
        r = solve_steady_state(P, V, depth, mat)
        print(f"  {label:25}  {P:10.1f}  {V:12.1f}  "
              f"{depth:10.1f}  {r['T_junction_C']:10.1f}°C  {r['R_ext']:12.4f}  {r['status']:>8}")
              
    print("=" * 105)
    print()

# 6. Plotting
def style_ax(ax, title):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(GREY)
        sp.set_linewidth(0.6)
    ax.tick_params(colors=DIM, labelsize=9)
    ax.xaxis.label.set_color(DIM)
    ax.yaxis.label.set_color(DIM)
    ax.set_title(title, color=PAPER, fontsize=10, fontweight='bold',
                 pad=8, fontfamily='monospace')
    ax.grid(True, color='#2a2a28', linewidth=0.5, linestyle='--', alpha=0.6)

def plot_all(output_path):
    fig = plt.figure(figsize=(18, 22), facecolor=BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.38,
                            left=0.08, right=0.96, top=0.93, bottom=0.05)

    fig.text(0.5, 0.975, 'AUV ELECTRONICS HULL THERMAL CALCULATOR',
             ha='center', va='top', color=PAPER, fontsize=13, fontweight='bold',
             fontfamily='monospace')
    fig.text(0.5, 0.960,
             'Conduction & forced convection  ·  Churchill-Bernstein seawater flow  ·  Dive profile transient  ·  Condensation risk',
             ha='center', va='top', color=DIM, fontsize=8.5, fontfamily='monospace')
    
    # Baseline calculations
    depths = np.linspace(0.0, 1000.0, 50)
    T_j_v0 = []
    T_j_v1 = []
    T_j_v3 = []
    
    for d in depths:
        T_j_v0.append(solve_steady_state(200.0, 0.05, d, 'titanium')['T_junction_C'])
        T_j_v1.append(solve_steady_state(200.0, 1.0, d, 'titanium')['T_junction_C'])
        T_j_v3.append(solve_steady_state(200.0, 3.0, d, 'titanium')['T_junction_C'])

    # Panel 1: T_junction vs Depth
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'JUNCTION TEMPERATURE VS DEPTH (Power = 200W)')
    ax1.plot(depths, T_j_v0, color=RED, linewidth=2.0, label='Hovering (0.05 m/s)')
    ax1.plot(depths, T_j_v1, color=GOLD, linewidth=2.0, label='Transit (1.0 m/s)')
    ax1.plot(depths, T_j_v3, color=CYAN, linewidth=1.5, linestyle='--', label='Dash (3.0 m/s)')
    ax1.axhline(85.0, color=RED, linestyle=':', alpha=0.8, label='Junction Limit (85°C)')
    
    ax1.set_xlabel('Operation Depth  [m]', fontsize=9)
    ax1.set_ylabel('Junction Temperature  [°C]', fontsize=9)
    ax1.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # Panel 2: Max Power vs Speed
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'MAXIMUM ALLOWABLE POWER VS VEHICLE SPEED')
    
    speeds = np.linspace(0.1, 4.0, 30)
    for i, d_ref in enumerate([10.0, 100.0, 500.0]):
        max_P = []
        for v in speeds:
            # find max power such that T_junction == 85°C (358.15 K)
            def f_P(p):
                return solve_steady_state(p, v, d_ref, 'titanium')['T_junction_C'] - 85.0
            try:
                max_P.append(brentq(f_P, 10.0, 2000.0))
            except ValueError:
                max_P.append(np.nan)
        ax2.plot(speeds, max_P, color=COLORS[i], linewidth=2.0, label=f'Depth {d_ref:.0f} m')
        
    ax2.set_xlabel('Vehicle Speed  [m/s]', fontsize=9)
    ax2.set_ylabel('Max Allowable Electronics Power  [W]', fontsize=9)
    ax2.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # Panel 3: Transient Dive Cycle
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'TRANSIENT TEMPERATURES DURING DIVE CYCLE (Titanium, 200W)')
    
    t_div, T_h_div, T_j_div = run_diving_cycle('titanium', 200.0, 1.5)
    
    ax3.plot(t_div/60.0, T_j_div, color=GOLD, linewidth=2.0, label='Electronics Junction')
    ax3.plot(t_div/60.0, T_h_div, color=CYAN, linewidth=1.5, linestyle='--', label='Hull Temperature')
    
    # Plot Seawater profile on twin axis
    ax3_r = ax3.twinx()
    ax3_r.set_facecolor(BG)
    # Re-calculate depths along the time vector
    depth_traj = []
    for t in t_div:
        if t < 300.0:
            depth_traj.append(1.0 * t)
        elif t < 11100.0:
            depth_traj.append(300.0)
        elif t < 11400.0:
            depth_traj.append(300.0 - 1.0 * (t - 11100.0))
        else:
            depth_traj.append(0.0)
    ax3_r.plot(t_div/60.0, depth_traj, color=GREY, linewidth=1.0, alpha=0.3, label='Depth Profile')
    ax3_r.set_ylabel('Depth  [m]', color=DIM, fontsize=9)
    ax3_r.tick_params(colors=DIM, labelsize=9)
    ax3_r.invert_yaxis()
    ax3_r.spines['right'].set_color(GREY)
    
    ax3.set_xlabel('Mission Time  [min]', fontsize=9)
    ax3.set_ylabel('Temperature  [°C]', fontsize=9)
    ax3.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper left')

    # Panel 4: Surfacing Condensation Risk
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, 'SURFACING CONDENSATION RISK')
    
    # Surfaced transient: starting from 4°C deep-water hull temp, warming in 15°C humid air
    T_air_C = 15.0
    T_dew_C = 12.0  # dew point at 15°C, 80% RH
    M_hull_Ti = 69.4
    Cp_Ti = 526.0
    h_air = 15.0  # typical natural convection in air
    A_ext = np.pi * D_OUT_m * L_VESSEL_m
    
    t_surf = np.linspace(0.0, 3600.0, 200)  # 1 hour
    T_hull_surf_Ti = T_air_C - (T_air_C - 4.0) * np.exp(-h_air * A_ext * t_surf / (M_hull_Ti * Cp_Ti))
    
    # Aluminium comparison
    M_hull_Al = (np.pi * (0.16**2 - 0.152**2) * 2.0) * 2700.0  # ~42.3 kg
    Cp_Al = 896.0
    T_hull_surf_Al = T_air_C - (T_air_C - 4.0) * np.exp(-h_air * A_ext * t_surf / (M_hull_Al * Cp_Al))
    
    ax4.plot(t_surf/60.0, T_hull_surf_Ti, color=RED, linewidth=2.0, label='Titanium Hull')
    ax4.plot(t_surf/60.0, T_hull_surf_Al, color=GOLD, linewidth=1.5, linestyle='-.', label='Aluminium Hull')
    ax4.axhline(T_dew_C, color=CYAN, linestyle='--', linewidth=1.2, label='Air Dew Point (12°C)')
    
    # Highlight condensation zones
    ax4.fill_between(t_surf/60.0, T_hull_surf_Ti, T_dew_C, where=T_hull_surf_Ti < T_dew_C, alpha=0.1, color=RED, label='Condensation (Ti)')
    
    ax4.set_xlabel('Time Post-Surfacing  [min]', fontsize=9)
    ax4.set_ylabel('Hull outer Temperature  [°C]', fontsize=9)
    ax4.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='lower right')

    # Panel 5: Resistance Breakdown Pie Chart
    ax5 = fig.add_subplot(gs[2, 0])
    style_ax(ax5, 'THERMAL RESISTANCE NETWORK BREAKDOWN')
    
    r_steady = solve_steady_state(200.0, 1.5, 300.0, 'titanium')
    res_vals = [r_steady['R_TIM'], r_steady['R_wall'], r_steady['R_ext']]
    res_labels = [f'TIM pad\n{r_steady["R_TIM"]:.5f} K/W', f'Wall conduction\n{r_steady["R_wall"]:.5f} K/W', f'External convection\n{r_steady["R_ext"]:.5f} K/W']
    
    wedge_props = {'width': 0.6, 'edgecolor': BG, 'linewidth': 2}
    ax5.pie(res_vals, labels=res_labels, colors=[GOLD, CYAN, RED], wedgeprops=wedge_props,
            textprops={'color': PAPER, 'fontsize': 8, 'fontfamily': 'monospace'},
            startangle=90, autopct='%1.0f%%', pctdistance=0.75, labeldistance=1.2)

    # Panel 6: Power/Speed Thermal Envelope Map
    ax6 = fig.add_subplot(gs[2, 1])
    style_ax(ax6, 'THERMAL COMPLIANCE ENVELOPE (300m Depth)')
    
    PP, VV = np.meshgrid(np.linspace(50.0, 600.0, 30), np.linspace(0.1, 4.0, 30))
    ENV_MAP = np.zeros_like(PP)
    for i in range(PP.shape[0]):
        for j in range(PP.shape[1]):
            r = solve_steady_state(PP[i, j], VV[i, j], 300.0, 'titanium')
            ENV_MAP[i, j] = r['T_junction_C']
            
    cf = ax6.contourf(PP, VV, ENV_MAP, levels=np.linspace(10.0, 120.0, 12), cmap='plasma')
    cs = ax6.contour(PP, VV, ENV_MAP, levels=[85.0], colors='white', linewidths=1.2, linestyles='--')
    ax6.clabel(cs, fmt='Junction Limit (85°C)', fontsize=8, colors='white')
    
    cb = plt.colorbar(cf, ax=ax6, pad=0.01)
    cb.set_label('Junction Temp  [°C]', color=DIM, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=DIM, labelsize=8)
    cb.outline.set_edgecolor(GREY)
    
    ax6.set_xlabel('Payload Electronics Power  [W]', fontsize=9)
    ax6.set_ylabel('Vehicle Speed  [m/s]', fontsize=9)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"Plot saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AUV electronics hull thermal calculator')
    parser.add_argument('--point', action='store_true', help='Print single design point metrics table')
    args = parser.parse_args()

    if args.point:
        print_report()
    else:
        print_report()
        plot_all('auv_thermal.png')
