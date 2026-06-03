"""
Tokamak Divertor Thermal Exhaust Calculator
===========================================
Steady-state thermal-hydraulic and fatigue analysis of a tokamak divertor 
heat exhaust monoblock (tungsten tile with CuCrZr cooling tube).

Directly useful to:
  - Tokamak Energy (Milton Park, UK) — compact spherical tokamaks (ST40)
  - First Light Fusion (Oxford, UK) — inertial fusion target chamber design
  - UK Atomic Energy Authority (Culham, UK) — STEP fusion power plant programme

Physics:
  - 1D radial and Cartesian thermal resistance circuit
  - Dittus-Boelter correlation for turbulent convective heat transfer
  - Subcooled boiling onset prediction (McNaught correction / Bergles-Rohsenow analogue)
  - Darcy-Weisbach pressure drop and Blasius turbulent friction factor
  - Coffin-Manson low-cycle thermal fatigue model for CuCrZr copper alloy
  - Comparative context of heat flux levels (fusion vs aerospace)

References:
  - ITER Design Specifications & divertor monoblock design guidelines
  - Incropera & DeWitt (2007), Fundamentals of Heat and Mass Transfer
  - McNaught (1982), Correlation of subcooled boiling heat transfer
  - Coffin-Manson low-cycle fatigue relations for copper alloys

Usage:
  python divertor_thermal.py             # full parametric sweeps and plots
  python divertor_thermal.py --point     # print scenario summary tables

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

# ── Design Parameters ─────────────────────────────────────────────────────────
W_TILE_m       = 0.028      # Width of tungsten monoblock (28 mm)
H_TILE_m       = 0.028      # Height of tungsten monoblock (28 mm)
T_TILE_m       = 0.012      # Thickness of tungsten monoblock (12 mm, axial length)
D_TUBE_OUT_m   = 0.012      # Tube outer diameter (12 mm)
D_TUBE_IN_m    = 0.009      # Tube inner diameter (9 mm)
T_BOND_m       = 0.5e-3     # Thickness of copper/HIP interface bond layer (0.5 mm)

K_W            = 170.0      # W/mK  Tungsten thermal conductivity
K_CU           = 350.0      # W/mK  CuCrZr copper alloy thermal conductivity

L_CASSETTE_m   = 1.5        # m     Axial length of divertor cooling cassette channel

# ── Coolant Properties ────────────────────────────────────────────────────────
COOLANTS = {
    'water': {
        'name': 'Pressurised Water',
        'p_Pa': 4.0e6,           # 4 MPa (40 bar)
        'T_in_K': 373.15,        # 100°C
        'T_sat_K': 523.55,       # Saturation temperature at 4 MPa (~250.4°C)
        'rho': 958.0,            # kg/m³
        'mu': 2.8e-4,            # Pa·s
        'k': 0.68,               # W/mK
        'Cp': 4216.0,            # J/kgK
        'Pr': 1.73,
        'T_limit_bulk_K': 523.15 # 250°C
    },
    'helium': {
        'name': 'Helium Gas',
        'p_Pa': 10.0e6,          # 10 MPa (100 bar)
        'T_in_K': 573.15,        # 300°C
        'T_sat_K': 10.0,         # Supercritical, no boiling
        'rho': 5.6,              # kg/m³
        'mu': 2.5e-5,            # Pa·s
        'k': 0.19,               # W/mK
        'Cp': 5193.0,            # J/kgK
        'Pr': 0.68,
        'T_limit_bulk_K': 873.15 # 600°C
    }
}

# ── 1. Thermal Hydraulic Performance Solver ───────────────────────────────────
def solve_divertor_thermal(q_surf_MWm2, V_cool_ms, coolant_type='water'):
    """
    Computes steady-state temperatures, pressure drop, and fatigue for a single monoblock.
    """
    cool = COOLANTS[coolant_type]
    q_surface_Wm2 = q_surf_MWm2 * 1e6
    
    # Flow parameters
    rho = cool['rho']
    mu = cool['mu']
    k_f = cool['k']
    Pr = cool['Pr']
    d_i = D_TUBE_IN_m
    d_o = D_TUBE_OUT_m
    L = T_TILE_m
    
    Re = rho * V_cool_ms * d_i / mu
    
    # Convective heat transfer coefficient (Dittus-Boelter)
    if Re > 10000.0:
        Nu = 0.023 * Re**0.8 * Pr**0.4
    else:
        Nu = 4.36  # laminar, constant heat flux limit
    h_cool = Nu * k_f / d_i
    
    # 1D Thermal Resistance Circuit
    # Conduction from plasma surface (top face) to outer tube wall:
    # A = frontal heat flux receiving area
    A_rec = W_TILE_m * T_TILE_m  # 0.028 * 0.012 m²
    t_W = H_TILE_m / 2.0 - d_o / 2.0  # Distance from top surface to tube outer wall = 14mm - 6mm = 8mm
    
    R_cond_W = t_W / (K_W * A_rec)
    R_cond_bond = T_BOND_m / (K_CU * A_rec)
    
    # Conduction through copper tube wall
    R_tube = np.log(d_o / d_i) / (2.0 * np.pi * K_CU * L)
    
    # Convective resistance
    R_cool = 1.0 / (h_cool * np.pi * d_i * L)
    
    # Total heat rate per monoblock
    Q_block_W = q_surface_Wm2 * A_rec
    
    # Temperatures
    T_cool_in = cool['T_in_K']
    T_surf = T_cool_in + Q_block_W * (R_cond_W + R_cond_bond + R_tube + R_cool)
    T_Cu_interface = T_cool_in + Q_block_W * (R_tube + R_cool)
    T_wet_wall = T_cool_in + Q_block_W * R_cool
    
    # Subcooled boiling onset McNaught correction (for water only)
    T_ONB = np.nan
    boiling_status = "Single Phase"
    if coolant_type == 'water':
        # McNaught correction for boiling onset
        T_sat = cool['T_sat_K']
        T_ONB = T_sat + q_surface_Wm2 / (0.00155 * Re**0.461 * Pr**0.301 * h_cool)
        if T_wet_wall > T_ONB:
            boiling_status = "Nucleate Boiling"
            # Subcooled boiling enhances heat transfer, making this 1D temperature overestimate conservative
            
    # Pressure Drop along the cassette (1.5 m)
    f_friction = 0.316 * Re**(-0.25) if Re > 4000.0 else 64.0 / Re
    dP_Pa = f_friction * (L_CASSETTE_m / d_i) * (rho * V_cool_ms**2 / 2.0)
    
    # Pumping power
    V_dot_m3s = V_cool_ms * np.pi / 4.0 * d_i**2
    P_pump_W = V_dot_m3s * dP_Pa
    
    # Coffin-Manson low-cycle fatigue life for CuCrZr alloy
    # dT_Cu = temperature swing at copper interface during plasma pulse
    dT_Cu = T_Cu_interface - T_cool_in
    C_Cu = 3000.0
    m_CM = 1.7
    N_f = C_Cu / (max(1.0, dT_Cu))**m_CM if dT_Cu > 0 else np.inf
    
    # Limits verification
    status = "OK"
    if T_surf > 3422.0 + 273.15:
        status = "W MELTDOWN"
    elif T_surf > 1300.0 + 273.15:
        status = "W RECRYSTALLISATION"
    elif T_Cu_interface > 500.0 + 273.15:
        status = "Cu SOFTENING"
        
    return {
        'T_surf_C':           T_surf - 273.15,
        'T_Cu_C':             T_Cu_interface - 273.15,
        'T_wet_C':            T_wet_wall - 273.15,
        'T_ONB_C':            T_ONB - 273.15 if not np.isnan(T_ONB) else np.nan,
        'h_cool_Wm2K':        h_cool,
        'dP_bar':             dP_Pa / 1e5,
        'P_pump_W':           P_pump_W,
        'N_f':                N_f,
        'boiling_status':     boiling_status,
        'status':             status
    }

# ── 2. Summary Report ─────────────────────────────────────────────────────────
SCENARIOS = [
    # q [MW/m2], V [m/s], Coolant, Label
    (5.0,   5.0,  'water',  'A) ST40 (Current)'),
    (20.0,  10.0, 'water',  'B) ITER-Class'),
    (50.0,  15.0, 'water',  'C) Demo-Class'),
    (100.0, 20.0, 'water',  'D) Reactor-Class'),
]

def print_report():
    print()
    print("=" * 105)
    print("  TOKAMAK DIVERTOR EXHAUST THERMAL CALCULATOR — SCENARIO VERIFICATION")
    print("=" * 105)
    print(f"  {'Scenario':20}  {'Heat Flux':>10}  {'Velocity':>9}  "
          f"{'T_surf':>9}  {'T_Cu':>9}  {'dP':>9}  {'Fatigue N_f':>13}  {'Status':>12}")
    print("-" * 105)
    
    for q_fl, V, cool, label in SCENARIOS:
        r = solve_divertor_thermal(q_fl, V, coolant_type=cool)
        N_f_str = f"{r['N_f']:.0f}" if r['N_f'] < 1e6 else ">1M"
        print(f"  {label:20}  {q_fl:6.1f} MW/m²  {V:5.1f} m/s  "
              f"{r['T_surf_C']:7.1f}°C  {r['T_Cu_C']:7.1f}°C  {r['dP_bar']:7.2f} bar  "
              f"{N_f_str:>13}  {r['status']:>12}")
              
    print("=" * 105)
    print()

# ── 3. Plotting ───────────────────────────────────────────────────────────────
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

    fig.text(0.5, 0.975, 'TOKAMAK DIVERTOR THERMAL EXHAUST DESIGN SUITE',
             ha='center', va='top', color=PAPER, fontsize=13, fontweight='bold',
             fontfamily='monospace')
    fig.text(0.5, 0.960,
             'Convective Dittus-Boelter flow  ·  ONB Bergles-Rohsenow correction  ·  Darcy pressure drop  ·  CuCrZr fatigue limits',
             ha='center', va='top', color=DIM, fontsize=8.5, fontfamily='monospace')
    fig.text(0.99, 0.005, 'pringlesmaths.co.uk  |  MKA',
             ha='right', va='bottom', color='#3a3a38', fontsize=8, fontfamily='monospace')

    # Scenarios values
    V_sweep = np.linspace(1.0, 25.0, 50)
    q_design = 20.0  # ITER-class reference heat flux (20 MW/m²)

    # ── Panel 1: Temperature vs Coolant Velocity ───────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'TEMPERATURES VS COOLANT VELOCITY (q = 20 MW/m²)')
    T_surf_list = []
    T_Cu_list = []
    T_wet_list = []
    for V in V_sweep:
        r = solve_divertor_thermal(q_design, V, 'water')
        T_surf_list.append(r['T_surf_C'])
        T_Cu_list.append(r['T_Cu_C'])
        T_wet_list.append(r['T_wet_C'])
        
    ax1.plot(V_sweep, T_surf_list, color=RED, linewidth=2.0, label='Tungsten Tile Surface')
    ax1.plot(V_sweep, T_Cu_list, color=GOLD, linewidth=2.0, label='CuCrZr Tube Interface')
    ax1.plot(V_sweep, T_wet_list, color=CYAN, linewidth=1.5, linestyle='--', label='Wet Wall Interface')
    
    ax1.axhline(1300.0, color=RED, linestyle=':', alpha=0.7, label='W Recrystallisation Limit (1300°C)')
    ax1.axhline(500.0, color=GOLD, linestyle=':', alpha=0.7, label='CuCrZr Softening Limit (500°C)')
    ax1.set_xlabel('Coolant Velocity  [m/s]', fontsize=9)
    ax1.set_ylabel('Temperature  [°C]', fontsize=9)
    ax1.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 2: Required Velocity vs Applied Heat Flux ─────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'REQUIRED COOLANT VELOCITY VS APPLIED HEAT FLUX')
    
    q_sweep = np.linspace(1.0, 80.0, 40)
    V_req_list = []
    for q in q_sweep:
        # Solve for V such that T_Cu_interface == 500°C (773.15 K)
        # Using bisection
        def f_V(v):
            return solve_divertor_thermal(q, v, 'water')['T_Cu_C'] - 500.0
        try:
            V_sol = brentq(f_V, 0.5, 50.0)
        except ValueError:
            V_sol = np.nan
        V_req_list.append(V_sol)
        
    ax2.plot(q_sweep, V_req_list, color=GOLD, linewidth=2.5, label='Required velocity for Cu limit')
    ax2.fill_between(q_sweep, V_req_list, 30.0, alpha=0.1, color=MOSS, label='Safe operating window')
    
    # Mark scenario design points
    for q_fl, V, _, label in SCENARIOS[:3]:
        ax2.plot(q_fl, V, 'o', color=RED, ms=7)
        ax2.annotate(label.split(') ')[1], xy=(q_fl, V), xytext=(8, -3),
                     textcoords='offset points', fontsize=8, color=RED, fontfamily='monospace')
                     
    ax2.set_xlabel('Applied Heat Flux  [MW/m²]', fontsize=9)
    ax2.set_ylabel('Required Coolant Velocity  [m/s]', fontsize=9)
    ax2.set_ylim(0, 30)
    ax2.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 3: Pressure Drop and Pumping Power ────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'CASSETTE PRESSURE DROP & PUMPING POWER')
    
    dP_list = []
    P_pump_list = []
    for V in V_sweep:
        r = solve_divertor_thermal(q_design, V, 'water')
        dP_list.append(r['dP_bar'])
        P_pump_list.append(r['P_pump_W'])
        
    ax3.plot(V_sweep, dP_list, color=RED, linewidth=2.0, label='Pressure Drop [bar]')
    
    ax3_r = ax3.twinx()
    ax3_r.set_facecolor(BG)
    ax3_r.plot(V_sweep, P_pump_list, color=CYAN, linewidth=1.5, linestyle='--', label='Pumping Power [W]')
    ax3_r.tick_params(colors=CYAN, labelsize=9)
    ax3_r.set_ylabel('Pumping Power  [W]', color=CYAN, fontsize=9)
    ax3_r.spines['right'].set_color(GREY)
    
    ax3.set_xlabel('Coolant Velocity  [m/s]', fontsize=9)
    ax3.set_ylabel('Pressure Drop  [bar]', fontsize=9)
    ax3.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper left')
    ax3_r.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='lower right')

    # ── Panel 4: Onset of Nucleate Boiling Map ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, 'ONSET OF NUCLEATE BOILING (ONB) MAP')
    
    # 2D Grid of heat flux vs velocity
    Q_g, V_g = np.meshgrid(np.linspace(1.0, 50.0, 30), np.linspace(1.0, 20.0, 30))
    BOIL_MAP = np.zeros_like(Q_g)
    for i in range(Q_g.shape[0]):
        for j in range(Q_g.shape[1]):
            r = solve_divertor_thermal(Q_g[i, j], V_g[i, j], 'water')
            if r['T_surf_C'] > 1300.0:
                BOIL_MAP[i, j] = 2  # Recrystallization
            elif r['boiling_status'] == "Nucleate Boiling":
                BOIL_MAP[i, j] = 1  # Boiling
            else:
                BOIL_MAP[i, j] = 0  # Single Phase
                
    cf = ax4.contourf(Q_g, V_g, BOIL_MAP, levels=[-0.5, 0.5, 1.5, 2.5], 
                      colors=['#0f4a2a', '#b8920a', '#c04040'], alpha=0.8)
    
    ax4.set_xlabel('Applied Heat Flux  [MW/m²]', fontsize=9)
    ax4.set_ylabel('Coolant Velocity  [m/s]', fontsize=9)
    
    # Annotations
    ax4.text(10.0, 15.0, 'Single Phase', color=PAPER, fontsize=8, fontfamily='monospace', fontweight='bold')
    ax4.text(25.0, 5.0, 'Nucleate Boiling', color=PAPER, fontsize=8, fontfamily='monospace', fontweight='bold')
    ax4.text(40.0, 15.0, 'Tungsten Limit\nExceeded', color=PAPER, fontsize=8, fontfamily='monospace', fontweight='bold')

    # ── Panel 5: Coffin-Manson Fatigue Life ─────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    style_ax(ax5, 'COFFIN-MANSON FATIGUE LIFE')
    
    N_f_sweep = []
    for q in q_sweep:
        r = solve_divertor_thermal(q, 12.0, 'water')  # constant 12 m/s velocity
        N_f_sweep.append(r['N_f'])
        
    ax5.plot(q_sweep, N_f_sweep, color=RED, linewidth=2.2, label='CuCrZr Fatigue Life')
    ax5.axhline(20000, color=GOLD, linestyle='--', linewidth=1.0, label='Required Lifetime (20,000 cycles)')
    
    ax5.set_xlabel('Applied Heat Flux  [MW/m²]', fontsize=9)
    ax5.set_ylabel('Cycles to Failure (N_f)', fontsize=9)
    ax5.set_yscale('log')
    ax5.set_ylim(10, 1e7)
    ax5.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 6: Heat Flux Comparison (Aerospace vs Fusion) ─────────────
    ax6 = fig.add_subplot(gs[2, 1])
    style_ax(ax6, 'COMPARATIVE HEAT FLUX CONTEXT')
    
    contexts = [
        'SABRE Precooler', 'Shuttle Re-entry', 'Re-entry (Mach 8)', 
        'Scramjet Combustor', 'ITER Divertor', 'DEMO Divertor / Rocket Throat'
    ]
    heat_fluxes = [0.1, 0.5, 3.0, 15.0, 20.0, 50.0]  # MW/m²
    
    bars = ax6.barh(contexts, heat_fluxes, color=COLORS * 2, alpha=0.7, height=0.55)
    ax6.tick_params(colors=DIM, labelsize=8)
    
    for bar, val in zip(bars, heat_fluxes):
        width = bar.get_width()
        ax6.text(width + 1.0, bar.get_y() + bar.get_height()/2.0, f'{val:.1f} MW/m²',
                 ha='left', va='center', color=PAPER, fontsize=8, fontfamily='monospace')
                 
    ax6.set_xlabel('Heat Flux  [MW/m²]', fontsize=9)
    ax6.set_xlim(0, 60)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"Plot saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokamak Divertor Thermal Exhaust Calculator')
    parser.add_argument('--point', action='store_true', help='Print single design point report')
    args = parser.parse_args()

    if args.point:
        print_report()
    else:
        print_report()
        plot_all('divertor_thermal.png')
