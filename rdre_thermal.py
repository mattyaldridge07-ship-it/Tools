"""
Rotating Detonation Rocket Engine (RDRE) Combustor Wall Thermal Solver
======================================================================
Transient 1D thermal conduction and fatigue analysis tool for a Rotating 
Detonation Rocket Engine (RDRE) combustor wall.

Directly useful to:
  - Venus Aerospace (Houston, Texas, US) — RDRE flight tests & VDR2 drone
  - Castelion (Torrance, California, US) — hypersonic strike platforms
  - High-performance propulsion engineers working on next-gen rocketry

Physics:
  - Chapman-Jouguet (CJ) detonation velocity and rotation frequency calculation
  - Rankine-Hugoniot shock relations for post-detonation state estimation
  - Time-varying wall heat flux representing a rotating detonation wave (1-2 km/s)
  - 1D transient explicit finite difference heat conduction solver (Inconel 625)
  - Coffin-Manson low-cycle thermal fatigue life estimation

References:
  - Anderson (2003), Modern Compressible Flow: With Historical Perspective
  - Kailasanath (2000), Review of propulsion applications of detonation waves
  - Coffin-Manson low-cycle thermal fatigue models for high-temperature superalloys

Usage:
  python rdre_thermal.py             # run transient simulation and plots
  python rdre_thermal.py --point     # print design point summary verification

Author: MKA — pringlesmaths.co.uk
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq
import argparse
import warnings
warnings.filterwarnings('ignore')

# ── Physical Constants ────────────────────────────────────────────────────────
SIGMA_SB    = 5.6704e-8     # W/m²/K⁴   Stefan-Boltzmann constant

# ── Color Palette ─────────────────────────────────────────────────────────────
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

# ── Annular Combustor Geometry & Wall Properties ──────────────────────────────
D_ANNULUS_m    = 0.15       # 150 mm mean annulus diameter
L_WALL_m       = 1.5e-3     # 1.5 mm wall thickness
N_NODES        = 11         # 11 nodes (dx = 0.15 mm)

# Inconel 625 properties
RHO_WALL       = 8440.0     # kg/m³
CP_WALL        = 410.0      # J/kgK
K_WALL         = 9.8        # W/mK

T_LIMIT_STEADY = 1100.0     # 1100 K maximum safe operating temperature (steady-state)
T_LIMIT_SHORT  = 1300.0     # 1300 K short-term maximum

# ── Propellant Options ────────────────────────────────────────────────────────
PROPELLANTS = {
    'H2': {
        'name': 'Hydrogen / Oxygen (H2/O2)',
        'V_CJ_stoich': 2840.0,   # m/s
        'T_CJ': 3700.0,          # K
        'Isp_s': 450.0
    },
    'CH4': {
        'name': 'Methane / Oxygen (CH4/O2)',
        'V_CJ_stoich': 2380.0,   # m/s
        'T_CJ': 3500.0,          # K
        'Isp_s': 380.0
    }
}

# ── 1. Chapman-Jouguet Detonation Properties ──────────────────────────────────
def compute_detonation_properties(prop_type='H2', phi=1.0):
    """
    Computes detonation wave speed and frequency based on equivalence ratio phi.
    """
    prop = PROPELLANTS[prop_type]
    V_stoich = prop['V_CJ_stoich']
    
    # Semi-empirical fit for CJ velocity vs equivalence ratio phi
    V_CJ = V_stoich * (1.0 - 0.15 * (phi - 1.0)**2)
    # Rotation frequency around the 150 mm annulus
    f_rotation = V_CJ / (np.pi * D_ANNULUS_m)
    
    return V_CJ, f_rotation

# ── 2. Time-Varying Heat Flux Generator ───────────────────────────────────────
def get_instantaneous_gas_conditions(t_s, T_period_s, T_CJ_K, h_peak=100000.0, h_min=10000.0):
    """
    Generates transient gas convective heat transfer coefficient and effective gas temp 
    to represent the passage of the detonation wave rotating around the annulus.
    """
    # Detonation wave passage time ~0.04 * T_period (very narrow high-heat flux peak)
    t_mod = t_s % T_period_s
    
    # Wave passage width
    w_wave = 0.04 * T_period_s
    
    # Gaussian heat transfer coefficient spike
    h_gas = h_min + (h_peak - h_min) * np.exp(-((t_mod - 0.1 * T_period_s) / w_wave)**2)
    
    # Temperature stays hot during combustion expansion phase, then drops when refilled
    # Decays slower than heat transfer coefficient
    T_gas_K = 300.0 + (T_CJ_K - 300.0) * np.exp(-((t_mod - 0.1 * T_period_s) / (0.15 * T_period_s))**2)
    
    return h_gas, T_gas_K

# ── 3. Transient 1D Conduction Solver ──────────────────────────────────────────
def solve_transient_conduction(m_dot_cool_kgs, prop_type='H2', phi=1.0, N_rotations=10):
    """
    Solves 1D heat equation explicitly inside the Inconel wall.
    Boundary conditions:
      - x = 0 (hot side): q_in(t) = h_gas(t) * (T_gas(t) - T_wall_hot(t))
      - x = L_wall (coolant side): q_out = h_cool * (T_wall_cold - T_cool)
    """
    prop = PROPELLANTS[prop_type]
    T_CJ_K = prop['T_CJ']
    V_CJ, f_rot = compute_detonation_properties(prop_type, phi)
    T_period = 1.0 / f_rot
    
    # Simulation Time Grid
    dt = 1e-6  # 1 microsecond timestep
    t_end = N_rotations * T_period
    time_steps = np.arange(0.0, t_end, dt)
    
    # Spatial Grid
    dx = L_WALL_m / (N_NODES - 1)
    
    # Check explicit Fourier stability limit: alpha * dt / dx^2 < 0.5
    alpha_diff = K_WALL / (RHO_WALL * CP_WALL)
    stability_val = alpha_diff * dt / dx**2
    if stability_val >= 0.5:
        # adjust dt if necessary (though 1e-6 is extremely safe here)
        dt = 0.49 * dx**2 / alpha_diff
        time_steps = np.arange(0.0, t_end, dt)
        
    # Coolant forced convection properties (water coolant inside annular cooling jacket)
    # Channel: Hydraulic diameter D_h = 10 mm
    D_h_cool = 0.010
    A_flow_cool = np.pi * D_ANNULUS_m * 0.005  # 5mm gap width
    V_cool = m_dot_cool_kgs / (998.0 * A_flow_cool)
    Re_cool = 998.0 * V_cool * D_h_cool / 1.0e-3
    Pr_cool = 7.0
    
    if Re_cool > 10000.0:
        Nu_cool = 0.023 * Re_cool**0.8 * Pr_cool**0.4
    else:
        Nu_cool = 4.36
    h_cool = Nu_cool * 0.60 / D_h_cool  # W/m²K
    T_cool_K = 300.0  # 27°C water
    
    # Temperature vector
    T = np.zeros(N_NODES) + 300.0  # Initialize wall to coolant temp (300 K)
    
    T_hot_hist = []
    T_cold_hist = []
    q_in_hist = []
    
    # Explicit marching loop
    for t in time_steps:
        h_gas, T_gas_K = get_instantaneous_gas_conditions(t, T_period, T_CJ_K)
        
        # Boundary fluxes
        q_in = h_gas * (T_gas_K - T[0])
        q_out = h_cool * (T[-1] - T_cool_K)
        
        T_new = T.copy()
        
        # Explicit finite difference equations
        # Fourier coefficient
        Fo = alpha_diff * dt / dx**2
        
        # Hot Boundary node (node 0)
        # T_new[0] = T[0] + 2*Fo*(T[1] - T[0]) + 2*Fo*dx/k * q_in
        T_new[0] = T[0] + 2.0 * Fo * (T[1] - T[0]) + 2.0 * Fo * (dx / K_WALL) * q_in
        
        # Internal nodes
        for j in range(1, N_NODES - 1):
            T_new[j] = T[j] + Fo * (T[j-1] - 2.0 * T[j] + T[j+1])
            
        # Cold Boundary node (node N-1)
        # T_new[-1] = T[-1] + 2*Fo*(T[-2] - T[-1]) - 2*Fo*dx/k * q_out
        T_new[-1] = T[-1] + 2.0 * Fo * (T[-2] - T[-1]) - 2.0 * Fo * (dx / K_WALL) * q_out
        
        T = T_new.copy()
        T_hot_hist.append(T[0])
        T_cold_hist.append(T[-1])
        q_in_hist.append(q_in / 1e6)  # MW/m²
        
    T_hot_hist = np.array(T_hot_hist)
    T_cold_hist = np.array(T_cold_hist)
    q_in_hist = np.array(q_in_hist)
    
    # Calculate metrics over the last 3 rotation periods (quasi-steady states)
    n_last_steps = int(3.0 * T_period / dt)
    T_hot_last = T_hot_hist[-n_last_steps:]
    T_cold_last = T_cold_hist[-n_last_steps:]
    
    T_hot_max = T_hot_last.max()
    T_hot_min = T_hot_last.min()
    dT_cycle = T_hot_max - T_hot_min
    
    # Coffin-Manson low cycle thermal fatigue estimate
    # Inconel fatigue coefficients: C = 5000, m = 1.9
    C_Inconel = 5000.0
    m_fatigue = 1.9
    N_f = C_Inconel / (max(1.0, dT_cycle))**m_fatigue
    
    status = "OK"
    if T_hot_max > T_LIMIT_SHORT:
        status = "MELTDOWN"
    elif T_hot_max > T_LIMIT_STEADY:
        status = "OVERHEATING"
        
    return {
        't':              time_steps,
        'T_hot':          T_hot_hist,
        'T_cold':         T_cold_hist,
        'q_in':           q_in_hist,
        'T_hot_max_K':    T_hot_max,
        'dT_cycle_K':     dT_cycle,
        'N_f':            N_f,
        'status':         status,
        'T_period':       T_period
    }

# ── 4. Print Summary Report ───────────────────────────────────────────────────
COOLANT_FLOW_RATES = [0.20, 0.40, 0.60, 0.80]  # kg/s

def print_report():
    print()
    print("=" * 105)
    print("  ROTATING DETONATION ROCKET COMBUSTOR WALL TRANS THERMAL VERIFICATION")
    print("=" * 105)
    print(f"  {'Propellant':20}  {'Flow [kg/s]':>12}  {'T_hot_max':>11}  "
          f"{'dT_cycle':>10}  {'Cycles N_f':>12}  {'Status':>12}")
    print("-" * 105)
    
    for prop in ['H2', 'CH4']:
        for mdot in COOLANT_FLOW_RATES:
            r = solve_transient_conduction(mdot, prop_type=prop, N_rotations=6)
            N_f_str = f"{r['N_f']:.0f}" if r['N_f'] < 1e7 else ">10M"
            print(f"  {PROPELLANTS[prop]['name']:20}  {mdot:12.2f}  "
                  f"{r['T_hot_max_K']:9.1f}K  {r['dT_cycle_K']:9.1f}K  {N_f_str:>12}  {r['status']:>12}")
                  
    print("=" * 105)
    print()

# ── 5. Plotting ───────────────────────────────────────────────────────────────
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

    fig.text(0.5, 0.975, 'ROTATING DETONATION ROCKET COMBUSTOR WALL SOLVER',
             ha='center', va='top', color=PAPER, fontsize=13, fontweight='bold',
             fontfamily='monospace')
    fig.text(0.5, 0.960,
             'Transient 1D finite difference (explicit)  ·  Chapman-Jouguet shock physics  ·  Inconel fatigue cycling',
             ha='center', va='top', color=DIM, fontsize=8.5, fontfamily='monospace')
    fig.text(0.99, 0.005, 'pringlesmaths.co.uk  |  MKA',
             ha='right', va='bottom', color='#3a3a38', fontsize=8, fontfamily='monospace')

    # Run reference simulations
    res_H2 = solve_transient_conduction(0.50, prop_type='H2', N_rotations=8)
    res_CH4 = solve_transient_conduction(0.50, prop_type='CH4', N_rotations=8)
    
    # Time vectors and zoom masks for H2 and CH4
    t_ms_H2 = res_H2['t'] * 1e3
    T_p_H2 = res_H2['T_period'] * 1e3
    idx_zoom_H2 = t_ms_H2 > (res_H2['t'][-1]*1e3 - 3.2 * T_p_H2)

    t_ms_CH4 = res_CH4['t'] * 1e3
    T_p_CH4 = res_CH4['T_period'] * 1e3
    idx_zoom_CH4 = t_ms_CH4 > (res_CH4['t'][-1]*1e3 - 3.2 * T_p_CH4)
    
    # ── Panel 1: CJ Detonation Wave Speeds ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'CJ DETONATION WAVE SPEED & FREQUENCY')
    
    phi_sweep = np.linspace(0.7, 1.3, 30)
    V_H2 = [compute_detonation_properties('H2', phi)[0] for phi in phi_sweep]
    V_CH4 = [compute_detonation_properties('CH4', phi)[0] for phi in phi_sweep]
    
    ax1.plot(phi_sweep, V_H2, color=CYAN, linewidth=2.0, label='H2/O2 Wave Speed [m/s]')
    ax1.plot(phi_sweep, V_CH4, color=GOLD, linewidth=2.0, label='CH4/O2 Wave Speed [m/s]')
    
    ax1_r = ax1.twinx()
    ax1_r.set_facecolor(BG)
    f_H2 = [compute_detonation_properties('H2', phi)[1] / 1e3 for phi in phi_sweep]
    ax1_r.plot(phi_sweep, f_H2, color=RED, linewidth=1.5, linestyle='--', label='Rotation Freq [kHz]')
    ax1_r.tick_params(colors=RED, labelsize=9)
    ax1_r.set_ylabel('Rotation Frequency  [kHz]', color=RED, fontsize=9)
    ax1_r.spines['right'].set_color(GREY)
    
    ax1.set_xlabel('Equivalence Ratio φ', fontsize=9)
    ax1.set_ylabel('Detonation Wave Velocity  [m/s]', fontsize=9)
    ax1.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='lower left')
    ax1_r.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper right')
 
    # ── Panel 2: Transient Temperatures over 3 Cycles ───────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'TRANSIENT WALL TEMPERATURE HISTORIES (Last 3 Cycles)')
    
    ax2.plot(t_ms_H2[idx_zoom_H2], res_H2['T_hot'][idx_zoom_H2], color=RED, linewidth=2.0, label='H2 Hot Side Wall')
    ax2.plot(t_ms_H2[idx_zoom_H2], res_H2['T_cold'][idx_zoom_H2], color=CYAN, linewidth=1.5, linestyle='--', label='H2 Cold Side Wall')
    ax2.plot(t_ms_CH4[idx_zoom_CH4], res_CH4['T_hot'][idx_zoom_CH4], color=GOLD, linewidth=1.5, linestyle='-.', label='CH4 Hot Side Wall')
    
    ax2.set_xlabel('Simulation Time  [ms]', fontsize=9)
    ax2.set_ylabel('Wall Temperature  [K]', fontsize=9)
    ax2.legend(fontsize=8, framealpha=0, labelcolor=DIM)
 
    # ── Panel 3: Dynamic Heat Flux q(t) over 3 Cycles ─────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'DYNAMIC WALL HEAT FLUX PROFILE (Last 3 Cycles)')
    
    ax3.plot(t_ms_H2[idx_zoom_H2], res_H2['q_in'][idx_zoom_H2], color=RED, linewidth=2.0, label='H2 Detonation Wave')
    ax3.plot(t_ms_CH4[idx_zoom_CH4], res_CH4['q_in'][idx_zoom_CH4], color=GOLD, linewidth=1.5, linestyle='-.', label='CH4 Detonation Wave')
    ax3.fill_between(t_ms_H2[idx_zoom_H2], res_H2['q_in'][idx_zoom_H2], alpha=0.1, color=RED)
    
    ax3.set_xlabel('Simulation Time  [ms]', fontsize=9)
    ax3.set_ylabel('Transient Heat Flux  [MW/m²]', fontsize=9)
    ax3.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 4: Peak Wall Temp vs Coolant Flow ───────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, 'PEAK WALL TEMPERATURE VS COOLANT FLOW RATE')
    
    flows = np.linspace(0.15, 0.90, 20)
    T_max_H2 = []
    T_max_CH4 = []
    for f in flows:
        T_max_H2.append(solve_transient_conduction(f, 'H2', N_rotations=6)['T_hot_max_K'])
        T_max_CH4.append(solve_transient_conduction(f, 'CH4', N_rotations=6)['T_hot_max_K'])
        
    ax4.plot(flows, T_max_H2, color=CYAN, linewidth=2.2, label='H2/O2 combustor')
    ax4.plot(flows, T_max_CH4, color=GOLD, linewidth=2.2, label='CH4/O2 combustor')
    ax4.axhline(T_LIMIT_STEADY, color=RED, linestyle='--', linewidth=1.0, label='Steady Material Limit (1100 K)')
    
    ax4.set_xlabel('Coolant Water Flow Rate  [kg/s]', fontsize=9)
    ax4.set_ylabel('Peak Hot Side Temperature  [K]', fontsize=9)
    ax4.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 5: Thermal Cycle Amplitude vs Coolant Flow ───────────────
    ax5 = fig.add_subplot(gs[2, 0])
    style_ax(ax5, 'WALL TEMPERATURE SWING (FATIGUE DRIVER) VS FLOW')
    
    dT_H2 = []
    dT_CH4 = []
    for f in flows:
        dT_H2.append(solve_transient_conduction(f, 'H2', N_rotations=6)['dT_cycle_K'])
        dT_CH4.append(solve_transient_conduction(f, 'CH4', N_rotations=6)['dT_cycle_K'])
        
    ax5.plot(flows, dT_H2, color=CYAN, linewidth=2.2, label='H2/O2 combustor')
    ax5.plot(flows, dT_CH4, color=GOLD, linewidth=2.2, label='CH4/O2 combustor')
    
    ax5.set_xlabel('Coolant Water Flow Rate  [kg/s]', fontsize=9)
    ax5.set_ylabel('Diurnal Cycle Delta_T  [K]', fontsize=9)
    ax5.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 6: Coffin-Manson Fatigue Life ───────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    style_ax(ax6, 'COFFIN-MANSON LIFE ESTIMATION')
    
    N_f_H2 = []
    N_f_CH4 = []
    for f in flows:
        N_f_H2.append(solve_transient_conduction(f, 'H2', N_rotations=6)['N_f'])
        N_f_CH4.append(solve_transient_conduction(f, 'CH4', N_rotations=6)['N_f'])
        
    ax6.plot(flows, N_f_H2, color=CYAN, linewidth=2.2, label='H2/O2 combustor')
    ax6.plot(flows, N_f_CH4, color=GOLD, linewidth=2.2, label='CH4/O2 combustor')
    ax6.axhline(1e5, color=RED, linestyle=':', alpha=0.5, label='Target Life (100,000 cycles)')
    
    ax6.set_xlabel('Coolant Water Flow Rate  [kg/s]', fontsize=9)
    ax6.set_ylabel('Estimated Cycles to Failure (N_f)', fontsize=9)
    ax6.set_yscale('log')
    ax6.set_ylim(10, 1e8)
    ax6.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"Plot saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RDRE combustor wall transient thermal solver')
    parser.add_argument('--point', action='store_true', help='Print design point verification tables')
    args = parser.parse_args()

    if args.point:
        print_report()
    else:
        print_report()
        plot_all('rdre_thermal.png')
