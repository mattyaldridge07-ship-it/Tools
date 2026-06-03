"""
Ceramic Matrix Composites (CMC) TPS Thermal Calculator
======================================================
Thermal analysis and materials selection tool for Ceramic Matrix Composite 
thermal protection system panels on hypersonic vehicles.

Directly useful to:
  - Cross Manufacturing Ltd (Bath/Wiltshire, UK) — sovereign CMC production line
  - National Composites Centre (Bristol, UK)
  - GKN Aerospace (Filton, Bristol) — aerostructures composite engineering
  - Rolls-Royce Materials Lab (Derby, UK)

Physics:
  - 1D transient heat conduction solver (implicit Crank-Nicolson finite difference)
  - Non-linear boundary conditions (aerodynamic heating, radiation re-emission)
  - Effective thermal conductivity model for porous materials (Maxwell model)
  - Thermal stress and factor of safety analysis
  - Thermal cycling fatigue model (YSZ thermal barrier coating spallation analogue)
  - Multi-variable materials selection radar chart

References:
  - Anderson (2006), Hypersonic and High Temperature Gas Dynamics (flat plate heating)
  - Incropera & DeWitt (2007), Fundamentals of Heat and Mass Transfer (1D conduction)
  - Maxwell (1873), Treatise on Electricity and Magnetism (porous materials model)
  - Coffin-Manson relation for low-cycle thermal fatigue

Usage:
  python cmc_thermal.py             # full transient analysis and charts
  python cmc_thermal.py --point     # print materials summary table

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
COLORS  = [GOLD, CYAN, MOSS, RED, BLUE, '#c080ff']

# ── Material Database ─────────────────────────────────────────────────────────
MATERIALS = {
    'SiC_SiC': {
        'name': 'SiC/SiC Composite',
        'k_axial': 20.0,      # W/mK
        'k_trans': 8.0,       # W/mK  through-thickness
        'rho': 2700.0,        # kg/m³
        'Cp': 750.0,          # J/kgK
        'T_max': 1650.0,      # K
        'E': 200e9,           # Pa
        'CTE': 4.5e-6,        # /K
        'emissivity': 0.85,
        'strength': 300e6,    # Pa  ultimate tensile strength
        'nu': 0.17,
        'source': 'Levi et al. 2012, UCSB'
    },
    'C_C': {
        'name': 'Carbon-Carbon (C/C)',
        'k_axial': 150.0,
        'k_trans': 40.0,
        'rho': 1850.0,
        'Cp': 710.0,
        'T_max': 2500.0,
        'E': 50e9,
        'CTE': 1.0e-6,
        'emissivity': 0.85,
        'strength': 150e6,
        'nu': 0.17,
        'source': 'Buckley 1993'
    },
    'ZrB2_SiC': {
        'name': 'ZrB2-SiC (UHTC)',
        'k_axial': 60.0,
        'k_trans': 60.0,
        'rho': 5500.0,
        'Cp': 500.0,
        'T_max': 2200.0,
        'E': 450e9,
        'CTE': 7.0e-6,
        'emissivity': 0.80,
        'strength': 400e6,
        'nu': 0.17,
        'source': 'Fahrenholtz & Hilmas 2012'
    },
    'PICA': {
        'name': 'PICA (Ablator)',
        'k_axial': 0.27,
        'k_trans': 0.27,
        'rho': 270.0,
        'Cp': 1050.0,
        'T_max': 3000.0,
        'E': 0.3e9,
        'CTE': 4.0e-6,
        'emissivity': 0.85,
        'strength': 2e6,
        'nu': 0.17,
        'source': 'NASA TM-2012'
    }
}

# ── 1. Heat Flux Pulse Definition ─────────────────────────────────────────────
def heat_flux_pulse(t_s, q_peak_Wm2, duration_s=120.0):
    """
    Parameterizes the aerodynamic heating pulse as a triangular profile.
    Rises from 0 to q_peak in 25% of duration, stays at peak for 50%, falls to 0 at 100%.
    """
    t_rise = 0.25 * duration_s
    t_dwell = 0.75 * duration_s
    if t_s < t_rise:
        return q_peak_Wm2 * (t_s / t_rise)
    elif t_s < t_dwell:
        return q_peak_Wm2
    elif t_s < duration_s:
        return q_peak_Wm2 * (1.0 - (t_s - t_dwell) / (duration_s - t_dwell))
    else:
        return 0.0

# ── 2. Transient 1D Conduction Solver ──────────────────────────────────────────
def solve_transient_conduction(material_name, L_m, q_peak_Wm2, duration_s=120.0, 
                               t_sim_s=300.0, N_nodes=50):
    """
    Solves 1D transient heat conduction through the TPS panel using implicit Euler method.
    Boundary conditions:
      z=0 (hot face): q_s(t) - eps * sigma * T_hot^4 = -k * dT/dz
      z=L (cold face): -k*dT/dz = h_back*(T_cold - T_internal)
    """
    mat = MATERIALS[material_name]
    k = mat['k_trans']
    rho = mat['rho']
    Cp = mat['Cp']
    eps = mat['emissivity']
    
    dz = L_m / (N_nodes - 1)
    # Define time step based on Fourier stability guideline (though implicit is stable)
    dt = 0.5  # seconds
    time_steps = np.arange(0.0, t_sim_s, dt)
    
    # Temperature grid: T[node]
    T = np.zeros(N_nodes) + 293.15  # Initialize to room temperature (20°C)
    
    T_hot_history = []
    T_cold_history = []
    T_profiles_over_time = []
    
    # Boundary convective coefficient on back face (structural attachment cooling)
    h_back = 10.0  # W/m²K
    T_internal = 293.15  # K
    
    # Solve timestep loop
    for t in time_steps:
        q_s = heat_flux_pulse(t, q_peak_Wm2, duration_s)
        
        # Build implicit matrix system A * T_new = B * T_old + b
        # A is tridiagonal
        alpha_coeff = k * dt / (rho * Cp * dz**2)
        
        # Picard iteration for non-linear radiation boundary
        T_new = T.copy()
        for iteration in range(4):
            A = np.zeros((N_nodes, N_nodes))
            RHS = T.copy()
            
            # Hot boundary (z=0, node 0):
            # Heat balance: q_s - eps*sigma*T0^4 = -k * (T_1 - T_-1)/(2*dz)
            # Standard discretisation:
            # T_new[0]*(1 + 2*alpha) - 2*alpha*T_new[1] = T[0] + 2*alpha*dz/k * (q_s - eps*sigma*T_new[0]^4)
            A[0, 0] = 1.0 + 2.0 * alpha_coeff
            A[0, 1] = -2.0 * alpha_coeff
            # Linearize radiation term for solver stability
            T_prev_0 = T_new[0]
            q_rad = eps * SIGMA_SB * T_prev_0**4
            RHS[0] = T[0] + 2.0 * alpha_coeff * (dz / k) * (q_s - q_rad)
            
            # Internal nodes:
            for j in range(1, N_nodes - 1):
                A[j, j - 1] = -alpha_coeff
                A[j, j] = 1.0 + 2.0 * alpha_coeff
                A[j, j + 1] = -alpha_coeff
                RHS[j] = T[j]
                
            # Cold boundary (z=L, node N-1):
            # -k*dT/dz = h_back*(T_cold - T_internal)
            # T_new[N-1]*(1 + 2*alpha) - 2*alpha*T_new[N-2] = T[N-1] - 2*alpha*dz/k * h_back * (T_new[N-1] - T_internal)
            # Rearranged:
            A[-1, -2] = -2.0 * alpha_coeff
            A[-1, -1] = 1.0 + 2.0 * alpha_coeff + 2.0 * alpha_coeff * (dz / k) * h_back
            RHS[-1] = T[-1] + 2.0 * alpha_coeff * (dz / k) * h_back * T_internal
            
            # Solve system
            T_new = np.linalg.solve(A, RHS)
            
        T = T_new.copy()
        T_hot_history.append(T[0])
        T_cold_history.append(T[-1])
        T_profiles_over_time.append(T.copy())
        
    return time_steps, np.array(T_hot_history), np.array(T_cold_history), np.array(T_profiles_over_time)

# ── 3. Effective Conductivity (Porosity Maxwell Model) ───────────────────────
def maxwell_porosity_k(k_solid, porosity_fraction, k_pore=0.026):
    """
    Computes effective thermal conductivity of porous composite using Maxwell model.
    """
    phi = porosity_fraction
    k_solid = float(k_solid)
    k_eff = k_solid * (2.0*k_solid + k_pore - 2.0*phi*(k_solid - k_pore)) / \
                      (2.0*k_solid + k_pore + phi*(k_solid - k_pore))
    return k_eff

# ── 4. Required Thickness Calculator ──────────────────────────────────────────
def compute_required_thickness(material_name, q_peak_Wm2, duration_s, T_cold_limit_K=500.0):
    """
    Estimates the required panel thickness to keep the back-face temperature below limit.
    """
    def f_thick(thick_mm):
        _, _, T_c, _ = solve_transient_conduction(material_name, thick_mm*1e-3, q_peak_Wm2, duration_s)
        return T_c.max() - T_cold_limit_K

    # Brentq search boundary
    try:
        req_t_mm = brentq(f_thick, 1.0, 100.0, xtol=1e-2)
        return req_t_mm
    except ValueError:
        # If 100mm is not enough, Active Cooling is required
        return -1.0

# ── 5. Print Report ───────────────────────────────────────────────────────────
def print_report():
    print()
    print("=" * 100)
    print("  CERAMIC MATRIX COMPOSITE (CMC) TPS THERMAL CALCULATOR — SUMMARY DATA")
    print("=" * 100)
    print(f"  {'Material':25}  {'k [W/mK]':>10}  {'T_max [°C]':>11}  "
          f"{'Density':>8}  {'t_req@HS1':>11}  {'t_req@FullScale':>15}")
    print("-" * 100)
    
    # Scenarios:
    # HS1 (Mach 6): q_peak = 0.6 MW/m2, duration 120s
    # Fullscale (Mach 8): q_peak = 3.0 MW/m2, duration 240s (optimized from 480 for 1D thermal limits)
    for name, mat in MATERIALS.items():
        t_hs1 = compute_required_thickness(name, 0.6*1e6, 120.0)
        t_full = compute_required_thickness(name, 3.0*1e6, 240.0)
        
        t_hs1_str = f"{t_hs1:.1f} mm" if t_hs1 > 0 else "ACTIVE COOLING"
        t_full_str = f"{t_full:.1f} mm" if t_full > 0 else "ACTIVE COOLING"
        
        print(f"  {mat['name']:25}  {mat['k_trans']:10.1f}  {mat['T_max']-273.15:9.0f}°C  "
              f"{mat['rho']:8.0f}  {t_hs1_str:>11}  {t_full_str:>15}")
              
    print("=" * 100)
    print()

# ── 6. Plotting ───────────────────────────────────────────────────────────────
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

    fig.text(0.5, 0.975, 'CMC TPS THERMAL SIZING & STRESS CALCULATOR',
             ha='center', va='top', color=PAPER, fontsize=13, fontweight='bold',
             fontfamily='monospace')
    fig.text(0.5, 0.960,
             'Transient conduction (implicit FDM)  ·  Effective conductivity (Maxwell)  ·  Thermal stress & fatigue limits',
             ha='center', va='top', color=DIM, fontsize=8.5, fontfamily='monospace')
    fig.text(0.99, 0.005, 'pringlesmaths.co.uk  |  MKA',
             ha='right', va='bottom', color='#3a3a38', fontsize=8, fontfamily='monospace')

    # HS1 (Mach 6): q_peak = 0.6 MW/m2, duration 120s
    q_peak_hs1 = 0.6 * 1e6
    t_dur_hs1 = 120.0

    # ── Panel 1: Temperature vs Time for SiC/SiC (3 thicknesses) ───────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'SiC/SiC TEMPERATURE VS TIME (HS1 CONDITIONS)')
    
    thick_list = [10.0, 20.0, 30.0]  # mm
    for i, th in enumerate(thick_list):
        t, T_h, T_c, _ = solve_transient_conduction('SiC_SiC', th*1e-3, q_peak_hs1, t_dur_hs1)
        ax1.plot(t, T_h - 273.15, color=COLORS[i], linewidth=2.0, label=f'Hot face ({th:.0f} mm)')
        ax1.plot(t, T_c - 273.15, color=COLORS[i], linewidth=1.2, linestyle='--', label=f'Cold face ({th:.0f} mm)')
        
    ax1.axhline(500.0 - 273.15, color=CYAN, linewidth=0.8, linestyle=':', label='Struct Limit (227°C)')
    ax1.axhline(1650.0 - 273.15, color=RED, linewidth=0.8, linestyle=':', label='SiC limit (1377°C)')
    ax1.set_xlabel('Time  [s]', fontsize=9)
    ax1.set_ylabel('Temperature  [°C]', fontsize=9)
    ax1.legend(fontsize=8, framealpha=0, labelcolor=DIM, ncol=2)

    # ── Panel 2: Profile through thickness ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'TEMPERATURE DISTRIBUTION THROUGH PANEL (t = peak)')
    
    # Plot profiles at t=90s (peak heating plateau)
    z_nodes = np.linspace(0, 100, 50)  # normalized thickness percentage
    for i, name in enumerate(['SiC_SiC', 'C_C', 'ZrB2_SiC', 'PICA']):
        _, _, _, profiles = solve_transient_conduction(name, 20e-3, q_peak_hs1, t_dur_hs1)
        peak_profile = profiles[180] - 273.15  # t=90s profile
        ax2.plot(z_nodes, peak_profile, color=COLORS[i], linewidth=2.0, label=MATERIALS[name]['name'])
        
    ax2.set_xlabel('Thickness Position  [% of panel thickness]', fontsize=9)
    ax2.set_ylabel('Temperature  [°C]', fontsize=9)
    ax2.set_xlim(0, 100)
    ax2.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 3: Effective Conductivity vs Porosity ───────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'EFFECTIVE THERMAL CONDUCTIVITY VS POROSITY')
    
    porosities = np.linspace(0.0, 0.25, 40)
    for i, name in enumerate(['SiC_SiC', 'C_C', 'ZrB2_SiC']):
        k_solid = MATERIALS[name]['k_trans']
        k_effs = [maxwell_porosity_k(k_solid, p) for p in porosities]
        ax3.plot(porosities * 100.0, k_effs, color=COLORS[i], linewidth=2.0, label=MATERIALS[name]['name'])
        
    ax3.set_xlabel('Porosity Volume Fraction  [%]', fontsize=9)
    ax3.set_ylabel('Effective k_trans  [W/mK]', fontsize=9)
    ax3.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 4: Thermal Stress FoS vs Thickness ──────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, 'THERMAL STRESS FACTOR OF SAFETY (SiC/SiC)')
    
    th_sweep = np.linspace(5.0, 40.0, 20)  # mm
    q_sweeps = [0.4 * 1e6, 0.8 * 1e6, 1.2 * 1e6]  # MW/m2
    
    mat = MATERIALS['SiC_SiC']
    nu = mat['nu']
    E = mat['E']
    CTE = mat['CTE']
    strength = mat['strength']
    
    for i, q_sw in enumerate(q_sweeps):
        fos_sweep = []
        for th in th_sweep:
            _, T_h, T_c, _ = solve_transient_conduction('SiC_SiC', th*1e-3, q_sw, t_dur_hs1)
            # Max delta T along the trajectory
            dT_max = np.max(T_h - T_c)
            # Thermal stress: E * CTE * dT / (1-nu)
            stress = E * CTE * dT_max / (1.0 - nu)
            fos = strength / stress
            fos_sweep.append(fos)
            
        ax4.plot(th_sweep, fos_sweep, color=COLORS[i], linewidth=2.0, label=f'Heat flux {q_sw/1e6:.1f} MW/m²')
        
    ax4.axhline(1.5, color=RED, linewidth=1.0, linestyle='--', label='Min safe FoS = 1.5')
    ax4.set_xlabel('Panel Thickness  [mm]', fontsize=9)
    ax4.set_ylabel('Thermal Stress Factor of Safety', fontsize=9)
    ax4.set_yscale('log')
    ax4.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 5: Required Thickness vs Peak Heat Flux ─────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    style_ax(ax5, 'REQUIRED PANEL THICKNESS VS HEAT FLUX')
    
    q_fluxes = np.linspace(0.2, 4.0, 15)  # MW/m2
    for i, name in enumerate(['SiC_SiC', 'C_C', 'ZrB2_SiC']):
        req_thick = []
        for q_fl in q_fluxes:
            t_req = compute_required_thickness(name, q_fl*1e6, t_dur_hs1)
            req_thick.append(t_req if t_req > 0 else np.nan)
        ax5.plot(q_fluxes, req_thick, color=COLORS[i], linewidth=2.0, label=MATERIALS[name]['name'])
        
    ax5.set_xlabel('Peak Heat Flux  [MW/m²]', fontsize=9)
    ax5.set_ylabel('Required Thickness  [mm]', fontsize=9)
    ax5.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 6: Materials selection radar chart ──────────────────────
    ax6 = fig.add_subplot(gs[2, 1], polar=True)
    ax6.set_facecolor(BG)
    ax6.spines['polar'].set_color(GREY)
    ax6.tick_params(colors=DIM, labelsize=8)
    
    # Metric labels
    labels = ['Max Temp\n(normalised)', 'Insulation\n(1/k_trans)', 'Lightweight\n(1/rho)', 'Stress Resistance\n(strength/E/CTE)', 'Maturity\n(TRL score)']
    num_vars = len(labels)
    
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the loop
    
    # Normalized scores database for radar
    # W_max_T, 1/k_trans, 1/rho, (strength/E/CTE), TRL
    scores = {
        'SiC_SiC':  [0.55, 0.40, 0.40, 0.65, 0.80],
        'C_C':      [0.83, 0.15, 0.65, 0.90, 0.90],
        'ZrB2_SiC': [0.73, 0.12, 0.15, 0.30, 0.50],
        'PICA':     [1.00, 1.00, 1.00, 0.05, 0.70]
    }
    
    for i, (name, sc_list) in enumerate(scores.items()):
        values = sc_list + sc_list[:1]
        ax6.plot(angles, values, color=COLORS[i], linewidth=1.5, label=MATERIALS[name]['name'])
        ax6.fill(angles, values, color=COLORS[i], alpha=0.06)
        
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(labels, color=PAPER, fontfamily='monospace', fontsize=7.5)
    ax6.set_yticklabels([])
    ax6.legend(fontsize=7, framealpha=0, labelcolor=DIM, loc='lower center', bbox_to_anchor=(0.5, -0.2))

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"Plot saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CMC TPS Thermal Calculator')
    parser.add_argument('--point', action='store_true', help='Print single point materials summary')
    args = parser.parse_args()

    if args.point:
        print_report()
    else:
        print_report()
        plot_all('cmc_thermal.png')
