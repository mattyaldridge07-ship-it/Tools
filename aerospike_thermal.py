"""
Linear Aerospike Rocket Spike Thermal Calculator
================================================
Steady-state conjugate heat transfer and sizing tool for the regenerative 
cooling channels of a linear aerospike rocket engine central spike.

Directly useful to:
  - Polaris Spaceplanes (Bremen, Germany) — linear aerospike spaceplanes (MIRA)
  - Nammo (Norway/Germany) — rocket motor supply chain
  - Reaction Engines alumni (UK) — compact cooling engineering

Physics:
  - Rocket expansion thermodynamics (LOX/kerosene isentropic gas expansion)
  - External gas convection using the Bartz correlation (property correction factors)
  - 1D wall thermal conduction through Inconel alloy
  - Dittus-Boelter correlation for internal forced convection (kerosene coolant)
  - Darcy-Weisbach channel pressure drop and Blasius turbulent friction factor
  - Multi-variable trade study (channel geometry vs wall temperature & pressure drop)

References:
  - Bartz (1957), A simple equation for rapid estimation of rocket nozzle convective heat transfer
  - Huzel & Huang (1992), Modern Engineering of Liquid-Propellant Rocket Engines
  - Incropera & DeWitt (2007), Fundamentals of Heat and Mass Transfer

Usage:
  python aerospike_thermal.py             # full parametric sweeps and plots
  python aerospike_thermal.py --point     # print optimization tables

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
G0_ms2      = 9.80665       # m/s²
R_UNIV      = 8314.46       # J/(kmol·K)

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

# ── Rocket Parameters (LOX/RP-1 Kerosene) ────────────────────────────────────
Pc_Pa          = 3.5e6      # 3.5 MPa chamber pressure
Tc_K           = 3500.0     # Chamber flame temperature
GAMMA_G        = 1.2        # Specific heat ratio of combustion products
MW_G           = 22.0       # g/mol molecular weight of products
R_G            = R_UNIV / (MW_G * 1e-3)  # J/kgK specific gas constant (~377.9)
D_t_m          = 0.025      # 25 mm throat diameter equivalent
L_SPIKE_m      = 0.15       # 150 mm spike length
T_WALL_m       = 1.5e-3     # 1.5 mm wall thickness (Inconel 718)
K_INCONEL      = 14.0       # W/mK  Inconel thermal conductivity

T_MAT_LIMIT_K  = 1100.0     # 1100 K maximum safe operating temperature (steady-state)

# ── Coolant Properties (Kerosene RP-1) ────────────────────────────────────────
RHO_KEROSENE   = 810.0      # kg/m³
CP_KEROSENE    = 2000.0     # J/kgK
K_KEROSENE     = 0.13       # W/mK
MU_KEROSENE    = 1.5e-3     # Pa·s
PR_KEROSENE    = 23.0       # Prandtl number for liquid kerosene at ~300 K

# ── 1. Combustion Gas Expansion ───────────────────────────────────────────────
def gas_properties_along_spike(x_m):
    """
    Computes local gas conditions along the spike using 1D isentropic expansion approximation.
    Mach increases linearly from 1.0 (throat, x=0) to 3.0 (tip, x=0.15).
    """
    M = 1.0 + 2.0 * (x_m / L_SPIKE_m)
    
    # Isentropic relations
    T_gas_K = Tc_K * (1.0 + (GAMMA_G - 1.0)/2.0 * M**2)**(-1)
    p_gas_Pa = Pc_Pa * (1.0 + (GAMMA_G - 1.0)/2.0 * M**2)**(-GAMMA_G / (GAMMA_G - 1.0))
    rho_gas_kgm3 = p_gas_Pa / (R_G * T_gas_K)
    V_gas_ms = M * np.sqrt(GAMMA_G * R_G * T_gas_K)
    
    # Area ratio A/At
    A_ratio = (1.0 / M) * ((2.0 / (GAMMA_G + 1.0)) * (1.0 + (GAMMA_G - 1.0)/2.0 * M**2))**((GAMMA_G + 1.0) / (2.0 * (GAMMA_G - 1.0)))
    
    return M, T_gas_K, p_gas_Pa, rho_gas_kgm3, V_gas_ms, A_ratio

# ── 2. Convective Heat Transfer (Bartz Correlation) ──────────────────────────
def bartz_heat_transfer_coefficient(M, T_gas_K, T_w_K, A_ratio):
    """
    Bartz equation for rapid estimation of rocket nozzle convective heat transfer coefficient.
    """
    R_c = 0.05  # m  throat curvature radius approximation
    
    # Characteristic values
    # Gas viscosity (Sutherland-type power law for combustion products)
    mu_gas = 4.5e-5  # Pa·s
    Cp_gas = GAMMA_G * R_G / (GAMMA_G - 1.0)  # ~2267 J/kgK
    Pr_gas = 0.70
    
    c_star = np.sqrt(GAMMA_G * R_G * Tc_K) / (GAMMA_G * np.sqrt((2.0 / (GAMMA_G + 1.0))**((GAMMA_G + 1.0) / (GAMMA_G - 1.0))))
    
    # Property correction factor sigma
    sigma = (0.5 * (T_w_K / Tc_K) * (1.0 + (GAMMA_G - 1.0)/2.0 * M**2) + 0.5)**(-0.68) * \
            (1.0 + (GAMMA_G - 1.0)/2.0 * M**2)**(-0.12)
            
    # Bartz formula
    h_g = (0.026 / D_t_m**0.2) * (mu_gas**0.2 * Cp_gas / Pr_gas**0.6) * (Pc_Pa * G0_ms2 / c_star)**0.8 * \
          (D_t_m / R_c)**0.1 * (1.0 / A_ratio)**0.9 * sigma
          
    # Adiabatic wall temperature
    r_recovery = Pr_gas**0.33  # turbulent boundary layer
    T_aw_K = Tc_K * (1.0 + r_recovery * (GAMMA_G - 1.0)/2.0 * M**2) / (1.0 + (GAMMA_G - 1.0)/2.0 * M**2)
    
    return h_g, T_aw_K

# ── 3. Conjugate Heat Transfer Marching Solver ────────────────────────────────
def solve_conjugate_heat_transfer(m_dot_cool_kgs, w_c_mm=1.5, h_c_mm=3.0, N_channels=24):
    """
    Marches along the spike to compute gas properties, external heat flux, Inconel wall
    temperatures, coolant bulk temperature rise, and channel pressure drop.
    """
    w_c = w_c_mm * 1e-3
    h_c = h_c_mm * 1e-3
    D_h = 2.0 * w_c * h_c / (w_c + h_c)
    A_flow = w_c * h_c
    
    # Marching grid along spike length
    N_stations = 50
    x_vec = np.linspace(0.0, L_SPIKE_m, N_stations)
    dx = x_vec[1] - x_vec[0]
    
    # Inlet coolant conditions
    T_cool_K = 300.0  # 27°C kerosene inlet
    
    # Pre-allocate arrays
    T_hot_vec = []
    T_cold_vec = []
    T_cool_vec = []
    q_ext_vec = []
    dP_total_Pa = 0.0
    
    # Perimeter of cooling channels (width of heated wall in contact with coolant)
    # Annular sector approximation: total width of linear aerospike section cooled by Nc channels
    perimeter = w_c * N_channels
    
    # Marching loop
    for x in x_vec:
        M, T_gas_K, p_gas_Pa, rho_gas_kgm3, V_gas_ms, A_ratio = gas_properties_along_spike(x)
        
        # Coolant flow velocity and Reynolds
        V_cool = m_dot_cool_kgs / (RHO_KEROSENE * N_channels * A_flow)
        Re_cool = RHO_KEROSENE * V_cool * D_h / MU_KEROSENE
        
        # Dittus-Boelter heat transfer coefficient (heating)
        if Re_cool > 10000.0:
            Nu_cool = 0.023 * Re_cool**0.8 * PR_KEROSENE**0.4
        else:
            Nu_cool = 4.36  # laminar limit
        h_cool = Nu_cool * K_KEROSENE / D_h
        
        # Solve wall temperature iteratively (due to non-linear Bartz dependency on T_w)
        T_hot_temp = 900.0
        for _ in range(5):
            h_g, T_aw_K = bartz_heat_transfer_coefficient(M, T_gas_K, T_hot_temp, A_ratio)
            # 1D steady conduction resistance network
            # q = U * (T_aw - T_cool)
            # 1/U = 1/h_g + t_wall/k_wall + 1/h_cool
            U = 1.0 / (1.0/h_g + T_WALL_m/K_INCONEL + 1.0/h_cool)
            q_ext = U * (T_aw_K - T_cool_K)
            
            T_hot_temp = T_aw_K - q_ext / h_g
            
        T_hot_K = T_hot_temp
        T_cold_K = T_cool_K + q_ext / h_cool
        
        T_hot_vec.append(T_hot_K)
        T_cold_vec.append(T_cold_K)
        T_cool_vec.append(T_cool_K)
        q_ext_vec.append(q_ext / 1e6)  # MW/m²
        
        # Temperature rise of coolant bulk fluid
        # dT_cool = q_ext * dA_heat / (m_dot * Cp)
        # dA_heat = perimeter * dx
        dT_cool = (q_ext * perimeter * dx) / (m_dot_cool_kgs * CP_KEROSENE)
        T_cool_K += dT_cool
        
        # Friction factor (Blasius)
        f_friction = 0.316 * Re_cool**(-0.25) if Re_cool > 4000.0 else 64.0 / Re_cool
        dp_step = f_friction * (dx / D_h) * (RHO_KEROSENE * V_cool**2 / 2.0)
        dP_total_Pa += dp_step
        
    return {
        'x_m':             x_vec,
        'T_hot_K':         np.array(T_hot_vec),
        'T_cold_K':        np.array(T_cold_vec),
        'T_cool_K':        np.array(T_cool_vec),
        'q_ext_MWm2':      np.array(q_ext_vec),
        'dP_bar':          dP_total_Pa / 1e5,
        'T_hot_max_K':     max(T_hot_vec)
    }

# ── 4. Sizing Sizer Report ────────────────────────────────────────────────────
GEOMETRIES = [
    # w_c [mm], h_c [mm], N_c, m_dot [kg/s]
    (1.0, 2.0, 24, 0.25, 'Narrow Channel'),
    (1.5, 3.0, 24, 0.25, 'Standard Channel (Default)'),
    (2.0, 4.0, 24, 0.25, 'Wide Channel'),
    (1.5, 3.0, 24, 0.15, 'Low Coolant Flow'),
]

def print_report():
    print()
    print("=" * 105)
    print("  LINEAR AEROSPIKE SPIKE THERMAL CALCULATOR — REGENT CHANNEL GEOMETRIES")
    print("=" * 105)
    print(f"  {'Configuration':30}  {'w_c [mm]':>8}  {'h_c [mm]':>8}  "
          f"{'m_dot [kg/s]':>12}  {'T_hot_max':>11}  {'Pressure Drop':>15}  {'Status':>10}")
    print("-" * 105)
    
    for wc, hc, Nc, mdot, label in GEOMETRIES:
        r = solve_conjugate_heat_transfer(mdot, wc, hc, Nc)
        status = "SAFE" if r['T_hot_max_K'] < T_MAT_LIMIT_K else "MELTDOWN"
        print(f"  {label:30}  {wc:8.1f}  {hc:8.1f}  "
              f"{mdot:12.2f}  {r['T_hot_max_K']:9.1f}K  {r['dP_bar']:13.2f} bar  {status:>10}")
              
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
    fig = plt.figure(figsize=(18, 20), facecolor=BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.38,
                            left=0.08, right=0.96, top=0.93, bottom=0.05)

    fig.text(0.5, 0.975, 'LINEAR AEROSPIKE spike thermal management',
             ha='center', va='top', color=PAPER, fontsize=13, fontweight='bold',
             fontfamily='monospace')
    fig.text(0.5, 0.960,
             'Conjugate heat transfer  ·  Bartz gas convection  ·  Inconel wall conduction  ·  Regenerative cooling trades',
             ha='center', va='top', color=DIM, fontsize=8.5, fontfamily='monospace')
    fig.text(0.99, 0.005, 'pringlesmaths.co.uk  |  MKA',
             ha='right', va='bottom', color='#3a3a38', fontsize=8, fontfamily='monospace')

    # Baseline solutions
    r_std = solve_conjugate_heat_transfer(0.25, 1.5, 3.0, 24)
    r_low_flow = solve_conjugate_heat_transfer(0.15, 1.5, 3.0, 24)
    r_wide = solve_conjugate_heat_transfer(0.25, 2.0, 4.0, 24)
    
    x = r_std['x_m'] * 1e3  # mm

    # ── Panel 1: Gas Expansion Mach & Temp ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'GAS EXPANSION ALONG THE SPIKE')
    
    machs = [gas_properties_along_spike(x_val)[0] for x_val in r_std['x_m']]
    temps = [gas_properties_along_spike(x_val)[1] for x_val in r_std['x_m']]
    
    ax1.plot(x, machs, color=GOLD, linewidth=2.0, label='Local Mach')
    
    ax1_r = ax1.twinx()
    ax1_r.set_facecolor(BG)
    ax1_r.plot(x, temps, color=RED, linewidth=1.5, linestyle='--', label='Gas Temperature')
    ax1_r.tick_params(colors=RED, labelsize=9)
    ax1_r.set_ylabel('Gas Temperature  [K]', color=RED, fontsize=9)
    ax1_r.spines['right'].set_color(GREY)
    
    ax1.set_xlabel('Spike Distance  [mm]', fontsize=9)
    ax1.set_ylabel('Mach Number', fontsize=9)
    ax1.set_xlim(0, L_SPIKE_m*1e3)
    ax1.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper left')
    ax1_r.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper right')

    # ── Panel 2: External heat flux q_ext ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'EXTERNAL PLUME HEAT FLUX DISTRIBUTION')
    ax2.plot(x, r_std['q_ext_MWm2'], color=RED, linewidth=2.2, label='Baseline Flow (0.25 kg/s)')
    ax2.plot(x, r_low_flow['q_ext_MWm2'], color=RED, linewidth=1.5, linestyle=':', label='Low Flow (0.15 kg/s)')
    ax2.fill_between(x, r_std['q_ext_MWm2'], alpha=0.1, color=RED)
    
    ax2.set_xlabel('Spike Distance  [mm]', fontsize=9)
    ax2.set_ylabel('Applied Heat Flux  [MW/m²]', fontsize=9)
    ax2.set_xlim(0, L_SPIKE_m*1e3)
    ax2.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 3: Inconel Wall Temperatures ────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'INCONEL WALL TEMPERATURE DISTRIBUTION')
    ax3.plot(x, r_std['T_hot_K'], color=RED, linewidth=2.0, label='Hot Side Wall (0.25 kg/s)')
    ax3.plot(x, r_low_flow['T_hot_K'], color=RED, linewidth=1.5, linestyle=':', label='Hot Side Wall (0.15 kg/s)')
    ax3.plot(x, r_std['T_cold_K'], color=CYAN, linewidth=1.5, linestyle='--', label='Cold Side Wall')
    
    ax3.axhline(T_MAT_LIMIT_K, color=GOLD, linestyle='--', linewidth=1.0, label='Inconel safe limit (1100 K)')
    ax3.set_xlabel('Spike Distance  [mm]', fontsize=9)
    ax3.set_ylabel('Temperature  [K]', fontsize=9)
    ax3.set_xlim(0, L_SPIKE_m*1e3)
    ax3.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 4: Coolant Bulk Temperature Rise ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, 'REGEN COOLANT BULK FLUID HEATING')
    ax4.plot(x, r_std['T_cool_K'] - 273.15, color=CYAN, linewidth=2.0, label='Kerosene Temp (0.25 kg/s)')
    ax4.plot(x, r_low_flow['T_cool_K'] - 273.15, color=CYAN, linewidth=1.5, linestyle=':', label='Kerosene Temp (0.15 kg/s)')
    
    ax4.set_xlabel('Spike Distance  [mm]', fontsize=9)
    ax4.set_ylabel('Coolant Bulk Temperature  [°C]', fontsize=9)
    ax4.set_xlim(0, L_SPIKE_m*1e3)
    ax4.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 5: Parametric Sweep: Channel Width vs Max Wall T ─────────
    ax5 = fig.add_subplot(gs[2, 0])
    style_ax(ax5, 'CHANNEL WIDTH SENSITIVITY')
    
    width_sweep = np.linspace(0.5, 3.0, 25)  # mm
    T_max_sweep = []
    dP_sweep = []
    
    for w in width_sweep:
        r = solve_conjugate_heat_transfer(0.25, w_c_mm=w, h_c_mm=3.0)
        T_max_sweep.append(r['T_hot_max_K'])
        dP_sweep.append(r['dP_bar'])
        
    ax5.plot(width_sweep, T_max_sweep, color=RED, linewidth=2.0, label='Max Wall Temp [K]')
    ax5.axhline(T_MAT_LIMIT_K, color=GOLD, linestyle=':', alpha=0.8)
    
    ax5_r = ax5.twinx()
    ax5_r.set_facecolor(BG)
    ax5_r.plot(width_sweep, dP_sweep, color=CYAN, linewidth=1.5, linestyle='--', label='Pressure Drop [bar]')
    ax5_r.tick_params(colors=CYAN, labelsize=9)
    ax5_r.set_ylabel('Total Coolant Pressure Drop  [bar]', color=CYAN, fontsize=9)
    ax5_r.spines['right'].set_color(GREY)
    
    ax5.set_xlabel('Coolant Channel Width  [mm]', fontsize=9)
    ax5.set_ylabel('Peak Wall Temperature  [K]', fontsize=9)
    ax5.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper left')
    ax5_r.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='lower right')

    # ── Panel 6: Coolant Mass Flow Sizing Sweep ────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    style_ax(ax6, 'MINIMUM REQUIRED COOLANT FLOW')
    
    m_flow_sweep = np.linspace(0.08, 0.40, 30)
    T_max_flow = []
    for m in m_flow_sweep:
        r = solve_conjugate_heat_transfer(m, w_c_mm=1.5, h_c_mm=3.0)
        T_max_flow.append(r['T_hot_max_K'])
        
    ax6.plot(m_flow_sweep, T_max_flow, color=GOLD, linewidth=2.5, label='Peak Wall Temp vs Flow')
    ax6.axhline(T_MAT_LIMIT_K, color=RED, linestyle='--', linewidth=1.0, label='Inconel safe limit (1100 K)')
    
    # Solve exact crossover point
    def f_crossover(m):
        return solve_conjugate_heat_transfer(m, w_c_mm=1.5, h_c_mm=3.0)['T_hot_max_K'] - T_MAT_LIMIT_K
    try:
        m_limit = brentq(f_crossover, 0.05, 0.5)
        ax6.axvline(m_limit, color=MOSS, linestyle=':', linewidth=1.0)
        ax6.text(m_limit + 0.01, 1200, f'm_min = {m_limit:.2f} kg/s', color=MOSS, fontsize=8, fontfamily='monospace')
    except ValueError:
        pass
        
    ax6.set_xlabel('Kerosene Mass Flow  [kg/s]', fontsize=9)
    ax6.set_ylabel('Peak Wall Temperature  [K]', fontsize=9)
    ax6.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"Plot saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Aerospike central spike sizing calculator')
    parser.add_argument('--point', action='store_true', help='Print single design point optimization tables')
    args = parser.parse_args()

    if args.point:
        print_report()
    else:
        print_report()
        plot_all('aerospike_thermal.png')
