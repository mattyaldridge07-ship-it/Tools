"""
Liquid Hydrogen (LH2) Propulsion & Thermal Budget Calculator
============================================================
Systems-level thermal-hydraulic and combustion analysis tool for liquid 
hydrogen-powered hypersonic aircraft.

Directly useful to:
  - Destinus (Payerne, Switzerland / Munich / London) — LH2 interceptor drones
  - Rolls-Royce HVX / Bristol site (UK) — combat hypersonics precoolers
  - Hermeus (Atlanta, US) — Chimera TBCC mode transition analysis

Physics:
  - H2 combustion thermodynamics (adiabatic flame temperature, equivalence ratio)
  - Rayleigh flow afterburner model (subsonic flow heat addition & pressure drop)
  - Leading-edge stagnation heating (Fay-Riddell / Detra-Kemp-Riddell model)
  - Precooler heat load integration (imported from sabre_precooler.py)
  - System-level LH2 mass flow budgeting (propulsion vs cooling matching)
  - Breguet range estimation with hydrogen-propulsion trades

References:
  - Incropera & DeWitt (2007), Fundamentals of Heat and Mass Transfer
  - Anderson (2003), Modern Compressible Flow: With Historical Perspective
  - Heiser & Pratt (1994), Hypersonic Airbreathing Propulsion

Usage:
  python lh2_propulsion_budget.py             # full envelope sweeps and plots
  python lh2_propulsion_budget.py --point     # print design point summary table

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

# ── Import Base Model ─────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from sabre_precooler import (
        isa_atmosphere, ram_conditions, precooler_performance,
        T_AIR_OUT_TARGET, LH2_TEMP, G0, CP_AIR, CP_H2, GAMMA, R_AIR
    )
except ImportError:
    # Fallback to local definitions if sabre_precooler isn't in path
    G0 = 9.80665
    CP_AIR = 1005.0
    CP_H2 = 14310.0
    LH2_TEMP = 20.3
    GAMMA = 1.4
    R_AIR = 287.058
    
    def isa_atmosphere(altitude_m):
        # basic isothermal/lapse model
        T = 216.65
        p = 22632.1 * np.exp(-G0 * (altitude_m - 11000.0) / (R_AIR * T)) if altitude_m > 11000.0 else 288.15 - 0.0065 * altitude_m
        p = max(p, 10.0)
        T = max(T, 150.0)
        rho = p / (R_AIR * T)
        a = np.sqrt(GAMMA * R_AIR * T)
        return T, p, rho, a

    def ram_conditions(M, altitude_m):
        T0, p0, rho0, a0 = isa_atmosphere(altitude_m)
        T_ram = T0 * (1.0 + (GAMMA - 1.0) / 2.0 * M**2)
        p_ram = p0 * (T_ram / T0) ** (GAMMA / (GAMMA - 1.0))
        V = M * a0
        return T_ram, p_ram, T0, p0, V

    def precooler_performance(M, altitude_m, humidity_fraction=0.003):
        # Mock precooler load
        T_ram, _, _, _, _ = ram_conditions(M, altitude_m)
        Q_MW = max(0.0, 0.05 * M * (T_ram - 250.0))
        return {'Q_MW': Q_MW, 'T_ram_K': T_ram}

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
R_NOSE_m          = 0.05      # 50 mm nose radius
A_NOSE_m2         = 0.02      # Effective nose cone cooling area
T_EXIT_LH2_K      = 250.0     # K   Hydrogen exit temperature from cooling channels
MASS_VEHICLE_kg   = 8000.0    # kg  HAPS/hypersonic cruise drone mass
L_D_CRUISE        = 4.5       # -   Aerodynamic Lift-to-Drag ratio
M_FUEL_CAP_kg     = 1500.0    # kg  Total LH2 fuel tank capacity

# ── 1. H2 Combustion Thermodynamics ───────────────────────────────────────────
def adiabatic_flame_temp_H2(T_in_K, phi):
    """
    Computes adiabatic flame temperature for hydrogen-air combustion.
    H2 + 0.5 O2 -> H2O  | LHV = 119.96 MJ/kg
    """
    FAR_stoich = 0.02915
    FAR = phi * FAR_stoich
    LHV_H2 = 119.96e6  # J/kg
    
    # Enthalpy balance: linear variable Cp model to account for dissociation at high T
    T_flame = T_in_K + phi * FAR_stoich * LHV_H2 / (CP_AIR * (1.0 + FAR))
    
    # Dissociation cap (empirical fit above 2500 K)
    if T_flame > 2400.0:
        T_flame = 2400.0 + 0.3 * (T_flame - 2400.0)
    return min(3200.0, T_flame)

# ── 2. Rayleigh Flow Afterburner Solver ───────────────────────────────────────
def solve_rayleigh_afterburner(M1, T1_K, p1_Pa, q_heat_Jkg, g=1.4):
    """
    Solves 1D Rayleigh flow afterburner exit Mach number and pressure drop.
    """
    T0_1 = T1_K * (1.0 + (g - 1.0) / 2.0 * M1**2)
    T0_2 = T0_1 + q_heat_Jkg / CP_AIR
    
    def f_M(M2):
        ratio = (M2 / M1)**2 * ((1.0 + g * M1**2) / (1.0 + g * M2**2))**2 * \
                ((1.0 + (g - 1.0) / 2.0 * M2**2) / (1.0 + (g - 1.0) / 2.0 * M1**2))
        return ratio - (T0_2 / T0_1)

    try:
        M2 = brentq(f_M, M1, 0.9999)
    except ValueError:
        M2 = 1.0  # Choked flow limit
        
    p2_p1 = (1.0 + g * M1**2) / (1.0 + g * M2**2)
    T2_T1 = p2_p1**2 * (M2 / M1)**2
    
    # Stagnation pressure loss
    p02_p01 = p2_p1 * ((1.0 + (g - 1.0) / 2.0 * M2**2) / (1.0 + (g - 1.0) / 2.0 * M1**2))**(g / (g - 1.0))
    
    return M2, T2_T1 * T1_K, p02_p01

# ── 3. LH2 Systems Sizing Sizer ───────────────────────────────────────────────
def solve_lh2_budget(M_flight, alt_km, phi=0.5):
    """
    Computes system mass flow balance: cooling flow vs propulsion flow.
    """
    alt_m = alt_km * 1e3
    T_atm, p_atm, rho_atm, a_atm = isa_atmosphere(alt_m)
    T_ram, _, _, _, V_inlet = ram_conditions(M_flight, alt_m)
    
    # Convective stagnation point heating on leading edge
    # Detra-Kemp-Riddell formulation
    rho_SL = 1.225
    q_nose_Wcm2 = 1.83e-4 / np.sqrt(R_NOSE_m) * np.sqrt(rho_atm / rho_SL) * (V_inlet**3)
    Q_nose_W = q_nose_Wcm2 * 1e4 * A_NOSE_m2
    
    # Precooler load (from sabre_precooler)
    precool_data = precooler_performance(M_flight, alt_m)
    Q_precool_W = precool_data['Q_MW'] * 1e6
    
    # Cooling mass flow required
    dT_LH2 = T_EXIT_LH2_K - LH2_TEMP
    m_dot_LH2_cool = (Q_nose_W + Q_precool_W) / (CP_H2 * dT_LH2)
    
    # Propulsion mass flow required
    # Thrust required for cruise: T = D = mg / (L/D)
    Thrust_req_N = MASS_VEHICLE_kg * G0 / L_D_CRUISE
    
    # Stoichiometric FAR
    FAR_stoich = 0.02915
    FAR = phi * FAR_stoich
    
    # Combustion temperature
    T_comb_flame = adiabatic_flame_temp_H2(T_ram, phi)
    
    # Ramjet specific impulse model (physical expansion back to ambient)
    Cp_comb = 1400.0  # J/kgK
    gamma_comb = 1.25
    
    # Expansion back to ambient
    # Afterburner inlet pressure ratio is reduced due to Rayleigh flow
    # Assume diffuser efficiency + Rayleigh pressure ratio:
    p0_afterburner = 0.85 * ram_conditions(M_flight, alt_m)[1]  # 85% diffuser recovery
    T_exit_static = T_comb_flame * (p_atm / p0_afterburner) ** ((gamma_comb - 1.0) / gamma_comb)
    V_exit = np.sqrt(max(0.0, 2.0 * Cp_comb * (T_comb_flame - T_exit_static)))
    
    # Thrust per unit air flow
    specific_thrust = (1.0 + FAR) * V_exit - V_inlet
    
    # Mass flows
    m_dot_air = Thrust_req_N / specific_thrust if specific_thrust > 0 else np.nan
    m_dot_LH2_prop = m_dot_air * FAR if not np.isnan(m_dot_air) else np.nan
    
    # Specific Impulse Isp = F / (mdot * g)
    Isp_s = Thrust_req_N / (m_dot_LH2_prop * G0) if m_dot_LH2_prop > 0 else np.nan
    
    # Match budget: total flow is max of cooling flow vs combustion burn flow
    # If cool > prop, we have excess fuel flow (cooling penalty)
    m_dot_total = np.nan
    cooling_penalty_pct = 0.0
    if not np.isnan(m_dot_LH2_prop):
        m_dot_total = max(m_dot_LH2_cool, m_dot_LH2_prop)
        cooling_penalty_pct = (m_dot_total - m_dot_LH2_prop) / m_dot_LH2_prop * 100.0
        
    # Breguet Range Estimation
    # Range = V * (L/D) * Isp * ln(M_wet / M_dry)
    M_wet = MASS_VEHICLE_kg
    M_dry = MASS_VEHICLE_kg - M_FUEL_CAP_kg
    range_km = np.nan
    if not np.isnan(Isp_s) and m_dot_total > 0:
        # Effective Isp reflects the excess fuel burned just for cooling
        Isp_eff = Thrust_req_N / (m_dot_total * G0)
        range_km = V_inlet * L_D_CRUISE * Isp_eff * np.log(M_wet / M_dry) / 1e3
        
    return {
        'M':                   M_flight,
        'alt_km':              alt_km,
        'Q_nose_MW':           Q_nose_W / 1e6,
        'Q_precool_MW':        Q_precool_W / 1e6,
        'm_dot_LH2_cool':      m_dot_LH2_cool,
        'm_dot_LH2_prop':      m_dot_LH2_prop,
        'm_dot_total':         m_dot_total,
        'cooling_penalty_pct': cooling_penalty_pct,
        'Isp_s':               Isp_s,
        'range_km':            range_km,
        'T_flame_K':           T_comb_flame
    }

# ── 4. Print Report ───────────────────────────────────────────────────────────
DESIGN_POINTS = [
    # Mach, Alt, Label
    (1.5,  12.0, 'Subsonic/Transonic climb'),
    (3.0,  18.0, 'Supersonic cruise'),
    (5.0,  25.0, 'Hypersonic cruise point'),
]

def print_report():
    print()
    print("=" * 105)
    print("  LH2 HYPERSONIC PROPULSION & THERMAL BUDGET CALCULATOR — DESIGN POINTS")
    print("=" * 105)
    print(f"  {'Design Point':28}  {'Mach':>5}  {'Alt [km]':>8}  "
          f"{'Q_total':>9}  {'m_cool':>8}  {'m_prop':>8}  {'Cooling%':>9}  {'Est Range':>9}")
    print("-" * 105)
    
    for M, alt, label in DESIGN_POINTS:
        r = solve_lh2_budget(M, alt)
        q_tot = r['Q_nose_MW'] + r['Q_precool_MW']
        cooling_pct = r['m_dot_LH2_cool'] / r['m_dot_total'] * 100.0 if r['m_dot_total'] > 0 else 0.0
        print(f"  {label:28}  {M:5.1f}  {alt:8.1f}  "
              f"{q_tot:7.1f}MW  {r['m_dot_LH2_cool']:6.3f}kg/s  {r['m_dot_LH2_prop']:6.3f}kg/s  "
              f"{cooling_pct:7.1f}%  {r['range_km']:7.0f}km")
              
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

    fig.text(0.5, 0.975, 'LH2 PROPULSION THERMAL BUDGET & RANGE PERFORMANCE',
             ha='center', va='top', color=PAPER, fontsize=13, fontweight='bold',
             fontfamily='monospace')
    fig.text(0.5, 0.960,
             'Systems integration  ·  Fay-Riddell leading edge heating  ·  Rayleigh combustion afterburner  ·  Breguet range matching',
             ha='center', va='top', color=DIM, fontsize=8.5, fontfamily='monospace')
    fig.text(0.99, 0.005, 'pringlesmaths.co.uk  |  MKA',
             ha='right', va='bottom', color='#3a3a38', fontsize=8, fontfamily='monospace')

    # Trajectory sweep: Mach 1 to 5.5
    mach_sweep = np.linspace(1.2, 5.5, 40)
    alt_profile = np.interp(mach_sweep, [1.0, 2.0, 3.0, 4.0, 5.0, 5.5], [10.0, 15.0, 19.0, 22.0, 25.0, 26.0])

    m_cool_list = []
    m_prop_list = []
    m_total_list = []
    q_nose_list = []
    q_precool_list = []
    range_list = []
    
    for M, alt in zip(mach_sweep, alt_profile):
        r = solve_lh2_budget(M, alt)
        m_cool_list.append(r['m_dot_LH2_cool'])
        m_prop_list.append(r['m_dot_LH2_prop'])
        m_total_list.append(r['m_dot_total'])
        q_nose_list.append(r['Q_nose_MW'])
        q_precool_list.append(r['Q_precool_MW'])
        range_list.append(r['range_km'])
        
    m_cool_list = np.array(m_cool_list)
    m_prop_list = np.array(m_prop_list)
    m_total_list = np.array(m_total_list)
    q_nose_list = np.array(q_nose_list)
    q_precool_list = np.array(q_precool_list)
    range_list = np.array(range_list)

    # ── Panel 1: LH2 split stacked area ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'LH2 CONSUMPTION BUDGET SPLIT')
    
    ax1.fill_between(mach_sweep, m_prop_list, color=MOSS, alpha=0.3, label='Propulsion Burn (Useful)')
    ax1.fill_between(mach_sweep, m_total_list, m_prop_list, color=RED, alpha=0.3, label='Cooling Excess Penalty')
    ax1.plot(mach_sweep, m_total_list, color=GOLD, linewidth=2.0, label='Total LH2 flow')
    ax1.plot(mach_sweep, m_prop_list, color=MOSS, linewidth=1.5, linestyle='--')
    
    ax1.set_xlabel('Flight Mach Number', fontsize=9)
    ax1.set_ylabel('LH2 Mass Flow  [kg/s]', fontsize=9)
    ax1.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 2: Leading edge and Precooler Loads ──────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'LEADING EDGE & INLET PRECOOLING LOADS')
    
    ax2.plot(mach_sweep, q_nose_list, color=CYAN, linewidth=2.0, label='Nose Heating [MW]')
    ax2.plot(mach_sweep, q_precool_list, color=GOLD, linewidth=2.0, label='Inlet Precooler [MW]')
    ax2.plot(mach_sweep, q_nose_list + q_precool_list, color=RED, linewidth=1.5, linestyle='--', label='Total Cooling Load')
    
    ax2.set_xlabel('Flight Mach Number', fontsize=9)
    ax2.set_ylabel('Heat Load  [MW]', fontsize=9)
    ax2.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 3: Adiabatic Flame Temp ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'ADIABATIC FLAME TEMP VS EQUIVALENCE RATIO')
    
    phi_sweep = np.linspace(0.2, 1.2, 40)
    for i, M_ref in enumerate([2.5, 4.0, 5.0]):
        T_ram_ref = ram_conditions(M_ref, 20.0 * 1e3)[0]
        T_flames = [adiabatic_flame_temp_H2(T_ram_ref, phi) for phi in phi_sweep]
        ax3.plot(phi_sweep, T_flames, color=COLORS[i], linewidth=2.0, label=f'Mach {M_ref:.1f}')
        
    ax3.axhline(2400.0, color=RED, linestyle=':', alpha=0.5, label='Dissociation Onset (~2400K)')
    ax3.set_xlabel('Equivalence Ratio φ', fontsize=9)
    ax3.set_ylabel('Adiabatic Flame Temperature  [K]', fontsize=9)
    ax3.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 4: Rayleigh afterburner ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, 'RAYLEIGH AFTERBURNER PROFILE')
    
    q_heat_sweep = np.linspace(0.1e6, 2.0e6, 45)  # J/kg
    M2_list = []
    p0_loss_list = []
    for q_h in q_heat_sweep:
        m2, t2, p0_loss = solve_rayleigh_afterburner(0.3, 800.0, 1e5, q_h)
        M2_list.append(m2)
        p0_loss_list.append(p0_loss)
        
    ax4.plot(q_heat_sweep/1e6, M2_list, color=GOLD, linewidth=2.0, label='Exit Mach (Inlet: 0.3)')
    
    ax4_r = ax4.twinx()
    ax4_r.set_facecolor(BG)
    ax4_r.plot(q_heat_sweep/1e6, p0_loss_list, color=RED, linewidth=1.5, linestyle='--', label='Stagnation P_ratio')
    ax4_r.tick_params(colors=RED, labelsize=9)
    ax4_r.set_ylabel('Stagnation Pressure Ratio  p₀₂/p₀₁', color=RED, fontsize=9)
    ax4_r.spines['right'].set_color(GREY)
    
    ax4.set_xlabel('Heat Added  [MJ/kg]', fontsize=9)
    ax4.set_ylabel('Exit Mach Number', fontsize=9)
    ax4.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper left')
    ax4_r.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='lower right')

    # ── Panel 5: Range Sensitivity ────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :])
    style_ax(ax5, 'BREGUET VEHICLE RANGE PERFORMANCE')
    
    ax5.plot(mach_sweep, range_list, color=CYAN, linewidth=2.5, label='Breguet Range with Thermal Budget')
    
    # Theoretical range if no cooling penalty applied (pure propulsion budget)
    ideal_range_list = []
    for M, alt in zip(mach_sweep, alt_profile):
        r = solve_lh2_budget(M, alt)
        ideal_r = np.nan
        if r['m_dot_LH2_prop'] > 0:
            Isp_ideal = Thrust_req_N = MASS_VEHICLE_kg * G0 / L_D_CRUISE / (r['m_dot_LH2_prop'] * G0)
            M_wet = MASS_VEHICLE_kg
            M_dry = MASS_VEHICLE_kg - M_FUEL_CAP_kg
            ideal_r = (M * np.sqrt(GAMMA * R_AIR * isa_atmosphere(alt*1e3)[0])) * L_D_CRUISE * Isp_ideal * np.log(M_wet / M_dry) / 1e6
        ideal_range_list.append(ideal_r)
        
    ax5.plot(mach_sweep, ideal_range_list, color=MOSS, linewidth=1.5, linestyle=':', label='Ideal (Zero Cooling Penalty)')
    ax5.fill_between(mach_sweep, range_list, ideal_range_list, alpha=0.08, color=RED, label='Thermal Drag Range Loss')
    
    ax5.set_xlabel('Flight Mach Number', fontsize=9)
    ax5.set_ylabel('Aircraft Cruise Range  [km]', fontsize=9)
    ax5.legend(fontsize=9, framealpha=0, labelcolor=DIM, loc='lower left')

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"Plot saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LH2 Propulsion & Thermal Budget Calculator')
    parser.add_argument('--point', action='store_true', help='Print design point verification table')
    args = parser.parse_args()

    if args.point:
        print_report()
    else:
        print_report()
        plot_all('lh2_thermal_budget.png')
