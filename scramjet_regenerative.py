"""
Regeneratively Cooled Scramjet Combustor — 1D Thermal Solver
============================================================
Simulates the coupled heat transfer between a supersonic combustion gas path 
and a supercritical liquid hydrogen (LH2) regenerative cooling jacket.

Directly relevant to:
  - Rolls-Royce HVX (Bristol/Derby, UK) — hypersonic scramjet propulsion
  - Hermeus (Atlanta, Georgia, US) — Chimera turboramjet/scramjet ducts
  - Reaction Engines successors / Frazer-Nash (INVICTUS high-speed cooling)
  - Defense Science and Technology Laboratory (Dstl) / DART hypersonics

Physics:
  - Supersonic gas convection: Eckert's reference temperature/enthalpy method
  - Wall heat conduction: radial resistance network across copper alloy (GRCop-84)
  - Coolant convection: Sieder-Tate correlation for turbulent supercritical channel flow
  - Temperature-dependent supercritical hydrogen thermodynamic properties
  - 1D marching integration along the combustor axial length

References:
  - Eckert (1955), Engineering Relations for Friction and Heat Transfer to Surfaces in High Velocity Flow
  - Sieder & Tate (1936), Heat Transfer and Pressure Drop of Liquids in Tubes
  - NASA TM-103271 (GRCop-84 Copper alloy mechanical/thermal properties)

Usage:
  python scramjet_regenerative.py             # Run default simulation and plots
  python scramjet_regenerative.py --point     # Print design point summary tables

Author: MKA — pringlesmaths.co.uk
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import warnings
warnings.filterwarnings('ignore')

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
COLORS  = [GOLD, CYAN, MOSS, RED, BLUE, '#c080ff', '#ff8040']

# ── Physical Constants ────────────────────────────────────────────────────────
GAMMA_GAS   = 1.35          # Combustion gas specific heat ratio
R_GAS       = 295.0         # J/(kg·K) combustion gas constant
SIGMA_SB    = 5.6704e-8     # W/m²/K⁴ Stefan-Boltzmann

# ── Combustor Channel Geometry ────────────────────────────────────────────────
L_COMBUSTOR_m  = 1.0        # 1.0 m combustor length
W_Duct_m       = 0.20       # 200 mm duct width
H_Duct_m       = 0.10       # 100 mm duct height
PERIMETER_m    = 2 * (W_Duct_m + H_Duct_m)
A_FLOW_GAS_m2  = W_Duct_m * H_Duct_m

# ── Cooling Jacket Microchannels (100 parallel channels) ──────────────────────
N_CHANNELS     = 100
W_CH_m         = 0.002      # 2 mm channel width
H_CH_m         = 0.003      # 3 mm channel height
T_RIB_m        = 0.001      # 1 mm rib thickness
T_WALL_HOT_m   = 0.0015     # 1.5 mm hot-side wall thickness
A_FLOW_CH_m2   = W_CH_m * H_CH_m
PERIM_CH_m     = 2 * (W_CH_m + H_CH_m)
D_H_CH_m       = 4 * A_FLOW_CH_m2 / PERIM_CH_m  # Hydraulic diameter (2.4 mm)

# ── Materials Database ────────────────────────────────────────────────────────
MATERIALS = {
    'grcop84': {
        'name': 'GRCop-84 (NASA Copper Superalloy)',
        'k': 320.0,         # W/mK  very high conductivity
        'T_max_safe_K': 1000.0, # Safe temperature limit
    },
    'inconel718': {
        'name': 'Inconel 718 (Nickel Superalloy)',
        'k': 19.0,          # W/mK  low conductivity but high strength
        'T_max_safe_K': 1250.0,
    }
}

# ── Supercritical Hydrogen Properties (Temperature-Dependent at 50 bar) ───────
def hydrogen_cp(T_K):
    """Specific heat [J/kgK] of hydrogen at 50 bar.
    Ref: NIST Chemistry WebBook (approximate curve fit).
    """
    # Peak near pseudo-critical temp (~45 K at 50 bar)
    if T_K < 45.0:
        return 12000.0 + (T_K - 30.0) * 800.0
    elif T_K < 80.0:
        return 24000.0 - (T_K - 45.0) * 280.0
    else:
        return max(14200.0, 14200.0 + 10.0 * (T_K - 80.0))

def hydrogen_viscosity(T_K):
    """Dynamic viscosity [Pa·s] of hydrogen at 50 bar."""
    return 1e-6 * (1.5 + 0.045 * T_K)

def hydrogen_conductivity(T_K):
    """Thermal conductivity [W/mK] of hydrogen at 50 bar."""
    return 0.02 + 0.0006 * T_K

# ── Convective Heat Transfer Coefficients ──────────────────────────────────────
def eckert_gas_h(x_axial_m, M_gas, T_gas_static_K, p_gas_static_Pa):
    """
    Supersonic gas path convective heat transfer coefficient using Eckert Reference Temp.
    Ref: Eckert (1955).
    """
    T_w_guess_K = 800.0
    gamma_ratio = (GAMMA_GAS - 1.0) / 2.0
    r_recovery = 0.89  # Turbulent recovery factor (approx Pr^(1/3))
    
    # Stagnation and recovery temperatures
    T_gas_total_K = T_gas_static_K * (1.0 + gamma_ratio * M_gas**2)
    T_recovery_K = T_gas_static_K * (1.0 + r_recovery * gamma_ratio * M_gas**2)
    
    # Eckert reference temperature
    T_ref_K = T_gas_static_K + 0.5 * (T_w_guess_K - T_gas_static_K) + 0.22 * (T_recovery_K - T_gas_static_K)
    
    # Fluid properties at reference temp
    a_ref = np.sqrt(GAMMA_GAS * R_GAS * T_ref_K)
    rho_ref = p_gas_static_Pa / (R_GAS * T_ref_K)
    mu_ref = 1.7e-5 * (T_ref_K / 273.15)**0.76  # Sutherland law for combustion gas
    k_ref = 0.024 * (T_ref_K / 273.15)**0.82
    Pr_ref = 0.72
    
    # Reynolds number based on axial distance
    V_gas = M_gas * np.sqrt(GAMMA_GAS * R_GAS * T_gas_static_K)
    x_eff = max(0.02, x_axial_m)  # Avoid singularity at inlet
    Re_x = rho_ref * V_gas * x_eff / mu_ref
    
    # Colburn analogy / turbulent flat plate correlation
    Nu_x = 0.0296 * Re_x**0.8 * Pr_ref**(1.0/3.0)
    h_gas = Nu_x * k_ref / x_eff
    return h_gas, T_recovery_K

def coolant_convection_h(T_cool_K, m_dot_ch_kgs):
    """
    Convective heat transfer coefficient [W/m²K] inside cooling channels.
    Sieder-Tate correlation for turbulent duct flow.
    """
    rho_LH2 = 50.0  # kg/m³ supercritical density
    mu = hydrogen_viscosity(T_cool_K)
    k = hydrogen_conductivity(T_cool_K)
    cp = hydrogen_cp(T_cool_K)
    Pr = cp * mu / k
    
    G_cool = m_dot_ch_kgs / A_FLOW_CH_m2
    Re_d = G_cool * D_H_CH_m / mu
    
    # Turbulent Sieder-Tate correlation
    if Re_d > 4000.0:
        Nu = 0.023 * Re_d**0.8 * Pr**0.4
    else:
        Nu = 4.36  # laminar limit
        
    h_cool = Nu * k / D_H_CH_m
    return h_cool, Re_d

# ── 1D Marching Solver ────────────────────────────────────────────────────────
def solve_combustor_profile(m_dot_total_kgs, M_gas=2.5, T_gas_static_K=2600.0, 
                            p_gas_static_Pa=2.0e5, material_name='grcop84', n_steps=50):
    """
    Integrates axial heat transfer and fluid state along the scramjet combustor.
    Marching direction is parallel to coolant flow (co-flow setup).
    """
    dx = L_COMBUSTOR_m / n_steps
    x_positions = np.linspace(0.0, L_COMBUSTOR_m, n_steps)
    
    # Coolant properties
    m_dot_ch = m_dot_total_kgs / N_CHANNELS
    
    # State histories
    T_cool = np.zeros(n_steps)
    T_wall_h = np.zeros(n_steps)
    T_wall_c = np.zeros(n_steps)
    q_flux = np.zeros(n_steps)
    h_gas_hist = np.zeros(n_steps)
    h_cool_hist = np.zeros(n_steps)
    
    # Initial conditions
    T_cool_current = 40.0  # LH2 inlet temperature (K)
    mat = MATERIALS[material_name]
    
    for i, x in enumerate(x_positions):
        h_gas, T_recovery = eckert_gas_h(x, M_gas, T_gas_static_K, p_gas_static_Pa)
        h_cool, _ = coolant_convection_h(T_cool_current, m_dot_ch)
        
        # Radial resistance network: Q = U * Perimeter * dx * (T_recovery - T_cool)
        # 1/U = 1/h_gas + t_wall/k_wall + 1/h_cool
        R_gas = 1.0 / h_gas
        R_wall = T_WALL_HOT_m / mat['k']
        R_cool = 1.0 / h_cool
        R_total = R_gas + R_wall + R_cool
        
        q_local = (T_recovery - T_cool_current) / R_total  # W/m²
        
        # Node temperatures
        T_w_h_local = T_recovery - q_local * R_gas
        T_w_c_local = T_w_h_local - q_local * R_wall
        
        # Update coolant temperature for the next step (energy balance)
        # dq = m_dot * cp * dT
        perimeter_ch_total = N_CHANNELS * W_CH_m  # effectively cooled width
        Q_total_step = q_local * perimeter_ch_total * dx
        cp_current = hydrogen_cp(T_cool_current)
        dT_cool = Q_total_step / (m_dot_total_kgs * cp_current)
        
        T_cool[i] = T_cool_current
        T_wall_h[i] = T_w_h_local
        T_wall_c[i] = T_w_c_local
        q_flux[i] = q_local / 1e6  # Convert to MW/m²
        h_gas_hist[i] = h_gas
        h_cool_hist[i] = h_cool
        
        T_cool_current += dT_cool
        
    status = "OK"
    if T_wall_h.max() > mat['T_max_safe_K']:
        status = "OVERHEATED"
        
    return {
        'x':            x_positions,
        'T_cool':       T_cool,
        'T_wall_hot':   T_wall_h,
        'T_wall_cool':  T_wall_c,
        'q_flux':       q_flux,
        'h_gas':        h_gas_hist,
        'h_cool':       h_cool_hist,
        'T_recovery':   T_recovery,
        'status':       status
    }

# ── Print Report ──────────────────────────────────────────────────────────────
def print_report():
    print()
    print("=" * 100)
    print("  REGENERATIVELY COOLED SCRAMJET COMBUSTOR WALL SOLVER -- DESIGN POINT SUMMARY")
    print("=" * 100)
    print(f"  {'Material':15}  {'Coolant [kg/s]':>16}  {'T_wall_max':>12}  "
          f"{'T_cool_out':>12}  {'Peak q [MW/m2]':>16}  {'Status':>10}")
    print("-" * 100)
    
    for mat_name in ['grcop84', 'inconel718']:
        for mdot in [0.5, 1.0, 1.5, 2.0]:
            r = solve_combustor_profile(mdot, material_name=mat_name)
            T_wmx = r['T_wall_hot'].max()
            T_cout = r['T_cool'][-1]
            q_max = r['q_flux'].max()
            print(f"  {MATERIALS[mat_name]['name']:15}  {mdot:16.2f}  "
                  f"{T_wmx:11.1f}K  {T_cout:11.1f}K  {q_max:15.2f}  {r['status']:>10}")
    print("=" * 100)
    print()

# ── Plotting ──────────────────────────────────────────────────────────────────
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
    
    fig.text(0.5, 0.975, 'REGENERATIVELY COOLED SCRAMJET COMBUSTOR THERMAL ANALYSIS',
             ha='center', va='top', color=PAPER, fontsize=13, fontweight='bold',
             fontfamily='monospace')
    fig.text(0.5, 0.960,
             'Supersonic gas path boundary layers  ·  Eckert reference temperature  ·  GRCop-84 microchannel coolant profiles',
             ha='center', va='top', color=DIM, fontsize=8.5, fontfamily='monospace')
    fig.text(0.99, 0.005, 'pringlesmaths.co.uk  |  MKA',
             ha='right', va='bottom', color='#3a3a38', fontsize=8, fontfamily='monospace')
    
    # ── Colors ────────────────────────────────────────────────────────────────
    GOLD    = '#b8920a'
    RED     = '#c04040'
    CYAN    = '#40b0c0'
    MOSS    = '#4a7a4b'
    BLUE    = '#4080c0'
    GREY    = '#3a3a38'
    
    # Reference cases (1.0 kg/s total coolant flow)
    res_cu = solve_combustor_profile(1.0, material_name='grcop84')
    res_inc = solve_combustor_profile(1.0, material_name='inconel718')
    
    # ── Panel 1: Temperature Profiles along Duct (GRCop-84) ───────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'AXIAL TEMPERATURE DISTRIBUTION (GRCop-84, 1.0 kg/s)')
    ax1.plot(res_cu['x'], res_cu['T_wall_hot'], color=RED, linewidth=2.2, label='Wall Hot Side')
    ax1.plot(res_cu['x'], res_cu['T_wall_cool'], color=GOLD, linewidth=1.5, linestyle='--', label='Wall Cool Side')
    ax1.plot(res_cu['x'], res_cu['T_cool'], color=CYAN, linewidth=2.0, label='LH2 Coolant')
    ax1.axhline(res_cu['T_recovery'], color=RED, linestyle=':', alpha=0.5, label=f'Recovery Temp ({res_cu["T_recovery"]:.0f}K)')
    ax1.axhline(1000.0, color=RED, linestyle='-.', linewidth=1.0, label='GRCop-84 Safe Limit (1000K)')
    
    ax1.set_xlabel('Axial Position along Combustor  [m]', fontsize=9)
    ax1.set_ylabel('Temperature  [K]', fontsize=9)
    ax1.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='lower right')
    
    # ── Panel 2: Heat Flux Profile along Duct ─────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'CONVECTIVE HEAT FLUX DISTRIBUTION')
    ax2.plot(res_cu['x'], res_cu['q_flux'], color=RED, linewidth=2.2, label='GRCop-84 Wall')
    ax2.plot(res_inc['x'], res_inc['q_flux'], color=GOLD, linewidth=1.5, linestyle='-.', label='Inconel 718 Wall')
    
    ax2.set_xlabel('Axial Position along Combustor  [m]', fontsize=9)
    ax2.set_ylabel('Heat Flux  [MW/m²]', fontsize=9)
    ax2.legend(fontsize=8, framealpha=0, labelcolor=DIM)
    
    # ── Panel 3: Material Temperature Comparison ──────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'MATERIAL COMPARISON: GRCop-84 vs INCONEL 718')
    ax3.plot(res_cu['x'], res_cu['T_wall_hot'], color=CYAN, linewidth=2.2, label='GRCop-84 Hot Side')
    ax3.plot(res_inc['x'], res_inc['T_wall_hot'], color=RED, linewidth=2.2, label='Inconel 718 Hot Side')
    ax3.axhline(1000.0, color=CYAN, linestyle='--', alpha=0.5)
    ax3.axhline(1250.0, color=RED, linestyle='--', alpha=0.5)
    
    ax3.set_xlabel('Axial Position along Combustor  [m]', fontsize=9)
    ax3.set_ylabel('Wall Temperature  [K]', fontsize=9)
    ax3.legend(fontsize=8, framealpha=0, labelcolor=DIM)
    
    # ── Panel 4: Peak Wall Temp vs Coolant Flow ───────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, 'PEAK WALL TEMPERATURE VS COOLANT FLOW RATE')
    flows = np.linspace(0.4, 2.5, 20)
    T_max_cu = []
    T_max_inc = []
    
    for f in flows:
        T_max_cu.append(solve_combustor_profile(f, material_name='grcop84')['T_wall_hot'].max())
        T_max_inc.append(solve_combustor_profile(f, material_name='inconel718')['T_wall_hot'].max())
        
    ax4.plot(flows, T_max_cu, color=CYAN, linewidth=2.2, label='GRCop-84')
    ax4.plot(flows, T_max_inc, color=RED, linewidth=2.2, label='Inconel 718')
    ax4.axhline(1000.0, color=CYAN, linestyle=':', alpha=0.8, label='GRCop-84 safe limit')
    ax4.axhline(1250.0, color=RED, linestyle=':', alpha=0.8, label='Inconel safe limit')
    
    ax4.set_xlabel('Coolant Hydrogen Flow  [kg/s]', fontsize=9)
    ax4.set_ylabel('Peak Wall Temperature  [K]', fontsize=9)
    ax4.legend(fontsize=8, framealpha=0, labelcolor=DIM)
    
    # ── Panel 5: Coolant Temperature Rise vs Flow ─────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    style_ax(ax5, 'COOLANT OUTLET TEMPERATURE VS COOLANT FLOW')
    T_out_cu = []
    for f in flows:
        T_out_cu.append(solve_combustor_profile(f, material_name='grcop84')['T_cool'][-1])
        
    ax5.plot(flows, T_out_cu, color=BLUE, linewidth=2.2, label='GRCop-84')
    ax5.set_xlabel('Coolant Hydrogen Flow  [kg/s]', fontsize=9)
    ax5.set_ylabel('Coolant Outlet Temperature  [K]', fontsize=9)
    
    # ── Panel 6: Convective Coefficients along Duct ───────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    style_ax(ax6, 'HEAT TRANSFER COEFFICIENTS DISTRIBUTION (1.0 kg/s)')
    ax6.plot(res_cu['x'], res_cu['h_gas'], color=GOLD, linewidth=2.0, label='Gas Convection (h_gas)')
    ax6.plot(res_cu['x'], res_cu['h_cool'], color=CYAN, linewidth=2.0, label='Coolant Convection (h_cool)')
    ax6.set_yscale('log')
    
    ax6.set_xlabel('Axial Position along Combustor  [m]', fontsize=9)
    ax6.set_ylabel('HTC  [W/m²K]', fontsize=9)
    ax6.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"Plot saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regeneratively Cooled Scramjet Combustor Solver')
    parser.add_argument('--point', action='store_true', help='Print design point verification tables')
    args = parser.parse_args()

    if args.point:
        print_report()
    else:
        print_report()
        plot_all('scramjet_regenerative.png')
