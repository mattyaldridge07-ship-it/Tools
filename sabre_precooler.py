"""
SABRE Precooler Performance Calculator
=======================================
A thermodynamic analysis tool for the Reaction Engines SABRE precooler
heat exchanger.

Calculates precooler performance across a Mach 0-5 flight envelope:
  - Inlet air conditions (ram temperature, pressure, density)
  - Heat exchanger effectiveness and heat load
  - LH2 fuel consumption for precooling
  - Frost risk index (moisture deposition on tubes)
  - Parametric plots over the full flight envelope

Physics:
  - ISA atmosphere model (0-80 km)
  - Isentropic compression to ram conditions
  - Crossflow heat exchanger NTU-effectiveness model
  - Frost formation: convective mass transfer analogy

Usage:
  python sabre_precooler.py           # full envelope plots
  python sabre_precooler.py --point   # single operating point

Author: PringlesMaths (MKA) — pringlesmaths.co.uk
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import argparse
import warnings
warnings.filterwarnings('ignore')

# ── Physical constants ────────────────────────────────────────────────────────
R_AIR   = 287.058     # J/(kg·K)  specific gas constant, air
GAMMA   = 1.4         # specific heat ratio, air (low-T)
CP_AIR  = 1005.0      # J/(kg·K)  specific heat, air at low T
CP_H2   = 14310.0     # J/(kg·K)  specific heat, LH2 (avg 20–250 K)
LH2_TEMP = 20.3       # K         boiling point of LH2 at 1 bar
G0      = 9.80665     # m/s²
R_UNIV  = 8314.46     # J/(kmol·K)
M_AIR   = 28.966      # kg/kmol   molar mass of air
M_H2O   = 18.015      # kg/kmol
# LH2 enthalpy: latent heat of vaporisation + sensible heat to outlet temp
# Simplified: use constant effective cp over the range 20 K → 250 K
H_LH2_EFFECTIVE = CP_H2  # J/(kg·K) — effective specific heat capacity for cooling

# ── SABRE/precooler design parameters
# Precooler only activates above this ram temperature
PRECOOLER_ACTIVATION_T = 400.0  # K  (~Mach 1.5 at low altitude) ────────────────────────────────────────
# Based on published SABRE data and open literature on the HTX test article
# Ref: Jivraj et al. 2007, Varvill 2008, ESA review documents
PRECOOLER_NTU    = 4.2      # Number of Transfer Units (crossflow HX, published ~4)
PRECOOLER_EFF_MAX = 0.985   # Maximum heat exchanger effectiveness (published >98%)
TUBE_OD          = 1.0e-3   # m  outer diameter of precooler tubes (~1 mm)
TUBE_WALL_T      = 0.05e-3  # m  wall thickness (very thin-walled)
AIR_MASS_FLOW_0  = 400.0    # kg/s  total air capture at design point (Mach 5, sea level equiv)
# Design point: Mach 5, ~26 km altitude (tropopause / stratosphere boundary)
DESIGN_MACH      = 5.0
DESIGN_ALT_KM    = 25.0
# Target precooler air outlet temperature (must be < compressor limit ~200 K)
T_AIR_OUT_TARGET = 250.0    # K  target precooler outlet temperature (compressor inlet)
# LH2 inlet temperature
T_LH2_IN         = LH2_TEMP

# ── ISA Atmosphere model ──────────────────────────────────────────────────────
def isa_atmosphere(altitude_m):
    """
    International Standard Atmosphere (ISA) up to 80 km.
    Returns (T [K], p [Pa], rho [kg/m³], a [m/s]).
    Layers: troposphere, tropopause, stratosphere 1, stratosphere 2,
            stratopause, mesosphere 1, mesosphere 2.
    """
    # Layer base altitudes (m), lapse rates (K/m), base temperatures (K), base pressures (Pa)
    layers = [
        (0,       11000, -0.0065, 288.15, 101325.0),
        (11000,   20000,  0.0,    216.65,  22632.1),
        (20000,   32000,  0.001,  216.65,   5474.89),
        (32000,   47000,  0.0028, 228.65,    868.019),
        (47000,   51000,  0.0,    270.65,    110.906),
        (51000,   71000, -0.0028, 270.65,     66.9389),
        (71000,   86000, -0.002,  214.65,      3.95642),
        (86000,  120000,  0.0,    186.87,      0.3734),
    ]
    h = np.clip(altitude_m, 0, 120000)
    T_base, p_base, lapse = None, None, None
    h_base = 0
    for (h0, h1, lr, T0, p0) in layers:
        if h <= h1:
            T_base, p_base, lapse, h_base = T0, p0, lr, h0
            break
    if T_base is None:
        T_base, p_base, lapse, h_base = 186.87, 0.3734, 0.0, 86000

    dh = h - h_base
    if abs(lapse) < 1e-10:
        T = T_base
        p = p_base * np.exp(-G0 * dh / (R_AIR * T_base))
    else:
        T = T_base + lapse * dh
        p = p_base * (T / T_base) ** (-G0 / (lapse * R_AIR))

    rho = p / (R_AIR * T)
    a   = np.sqrt(GAMMA * R_AIR * T)
    return T, p, rho, a


def ram_conditions(M, altitude_m):
    """
    Isentropic stagnation (ram) conditions ahead of precooler.
    Returns (T_ram [K], p_ram [Pa], T_static [K], p_static [Pa], V [m/s]).
    In the real SABRE, there's a normal shock at the intake lip, but for
    preliminary design we use isentropic stagnation (conservative — actual
    T_ram is slightly lower after the shock system).
    """
    T0, p0, rho0, a0 = isa_atmosphere(altitude_m)
    T_ram = T0 * (1 + (GAMMA - 1) / 2 * M**2)
    p_ram = p0 * (T_ram / T0) ** (GAMMA / (GAMMA - 1))
    V     = M * a0
    return T_ram, p_ram, T0, p0, V


def precooler_performance(M, altitude_m, humidity_fraction=0.003):
    """
    Full precooler thermodynamic analysis for a given Mach number and altitude.

    Returns a dict with all key performance parameters.

    NTU-effectiveness method for a crossflow heat exchanger (unmixed-unmixed):
        effectiveness = 1 - exp( (1/C_r) * NTU^0.22 * (exp(-C_r * NTU^0.78) - 1) )
    where C_r = C_min / C_max = (m_dot_air * cp_air) / (m_dot_LH2 * cp_LH2)

    For SABRE: LH2 is the working fluid on the cold side. The design is such
    that C_r ≈ 1 at the design point, i.e. the LH2 flow is matched to the
    air heat load.
    """
    T_ram, p_ram, T_static, p_static, V = ram_conditions(M, altitude_m)
    _, _, rho_static, a_static = isa_atmosphere(altitude_m)

    # ── Air mass flow rate ─────────────────────────────────────────────────
    # Scales with dynamic pressure × intake area (fixed geometry approximation)
    # Reference: design point Mach 5, 25 km
    T_ram_design, p_ram_design, _, _, _ = ram_conditions(DESIGN_MACH, DESIGN_ALT_KM * 1e3)
    _, _, rho_design, a_design = isa_atmosphere(DESIGN_ALT_KM * 1e3)
    rho_static, _, _, _ = isa_atmosphere(altitude_m), None, None, None
    rho_s = isa_atmosphere(altitude_m)[2]
    rho_d = isa_atmosphere(DESIGN_ALT_KM * 1e3)[2]
    # Mass flow scales approximately with rho * V (momentum flux)
    m_dot_air = AIR_MASS_FLOW_0 * (rho_s * M * a_static) / \
                (rho_d * DESIGN_MACH * a_design) * 0.82  # 0.82 = intake efficiency

    # ── Heat load ──────────────────────────────────────────────────────────
    # Air must be cooled from T_ram to T_AIR_OUT_TARGET
    # Q = m_dot_air * cp_air * (T_ram - T_out)
    # Clamp: if T_ram is already below target, no precooling needed
    delta_T_air = max(0.0, T_ram - T_AIR_OUT_TARGET)
    Q_required  = m_dot_air * CP_AIR * delta_T_air  # W

    # ── LH2 mass flow required ─────────────────────────────────────────────
    # Energy balance: Q = m_dot_LH2 * cp_LH2 * (T_LH2_out - T_LH2_in)
    # Assume LH2 exits at ~250 K (warm hydrogen, before combustion)
    T_LH2_out   = 250.0   # K
    dH_LH2      = CP_H2 * (T_LH2_out - T_LH2_IN)  # J/kg
    m_dot_LH2   = Q_required / dH_LH2 if dH_LH2 > 0 else 0.0

    # ── Heat exchanger effectiveness ───────────────────────────────────────
    C_air  = m_dot_air * CP_AIR
    C_LH2  = m_dot_LH2 * CP_H2 if m_dot_LH2 > 0 else C_air * 0.9
    C_min  = min(C_air, C_LH2)
    C_max  = max(C_air, C_LH2)
    C_r    = C_min / C_max if C_max > 0 else 1.0

    NTU    = PRECOOLER_NTU
    # Crossflow, both fluids unmixed:
    try:
        eff = 1 - np.exp((1/C_r) * NTU**0.22 * (np.exp(-C_r * NTU**0.78) - 1))
    except (OverflowError, ZeroDivisionError):
        eff = PRECOOLER_EFF_MAX
    eff = min(eff, PRECOOLER_EFF_MAX)

    # Actual air outlet temperature
    T_air_out_actual = T_ram - eff * (T_ram - T_LH2_IN) if T_ram > T_LH2_IN else T_ram

    # ── Frost risk index ───────────────────────────────────────────────────
    # Frost forms when tube wall temperature < dew point of incoming air
    # Wall temperature approximated as: T_wall ≈ T_LH2_IN + (T_LH2_out-T_LH2_IN)/2
    # (average along tube length — conservative)
    T_wall = (T_LH2_IN + T_LH2_out) / 2   # ~135 K cold side average

    # Saturation vapour pressure (Magnus formula, valid 200-373 K)
    # Extended to low T using Clausius-Clapeyron
    def p_sat(T_K):
        if T_K < 273.15:
            # Ice saturation (Murphey & Koop 2005 simplified)
            T_C = T_K - 273.15
            return 611.657 * np.exp(22.5452 * T_C / (272.55 + T_C)) if T_C > -80 else 1e-10
        else:
            T_C = T_K - 273.15
            return 611.657 * np.exp(17.368 * T_C / (238.83 + T_C))

    # humidity_fraction is in g/kg (grams of water vapour per kg dry air)
    # Convert to dimensionless mixing ratio (kg/kg)
    w = humidity_fraction / 1000.0   # kg/kg
    p_vs_inlet = p_sat(T_static)
    # Partial pressure of water vapour: p_v = w * p / (0.622 + w)
    p_v = w * p_static / (0.622 + w)
    # Relative humidity at inlet
    RH_inlet = p_v / p_vs_inlet if p_vs_inlet > 0 else 0.0

    # T_dew calculated below in frost section via bisection (more accurate)
    T_dew = 180.0  # placeholder, overwritten below

    # ── Frost risk: vapour pressure driving force (Sherwood analogy) ─────
    # Deposition occurs when ambient vapour pressure p_v > p_sat(T_wall)
    # Rate ∝ (p_v - p_sat_wall). Even if T_wall < T_dew, at very low ambient
    # humidity (stratosphere) p_v ≈ 0.01 Pa → negligible deposition.
    # Reference pressure 500 Pa gives risk = 1 (heavy icing at low altitude).
    FROST_REF_PRESSURE = 500.0   # Pa
    p_sat_wall = p_sat(max(T_wall, 100.0))
    deposition_driving_force = max(0.0, p_v - p_sat_wall)  # Pa
    frost_risk = min(1.0, deposition_driving_force / FROST_REF_PRESSURE)

    # Frost margin: T_wall - T_dew  (computed by bisection on p_sat)
    if p_v > 1e-8:
        T_lo_dp, T_hi_dp = 100.0, T_static
        for _ in range(50):
            T_mid_dp = (T_lo_dp + T_hi_dp) / 2.0
            if p_sat(T_mid_dp) > p_v:
                T_hi_dp = T_mid_dp
            else:
                T_lo_dp = T_mid_dp
        T_dew = (T_lo_dp + T_hi_dp) / 2.0
    else:
        T_dew = 100.0

    margin = T_wall - T_dew  # +ve = wall above dew point (thermodynamically safe)
    rho_v_inf  = p_v / (R_AIR / 0.622 * T_static) if T_static > 0 else 0.0
    rho_v_wall = p_sat_wall / (R_AIR / 0.622 * max(T_wall, 100.0))

    # ── Specific impulse penalty ───────────────────────────────────────────
    # LH2 used for precooling is not available for propulsion
    # Typical SABRE thrust ~667 kN at sea level, Isp ~3500 s air-breathing
    # Precooling fuel fraction = m_dot_LH2 / total LH2 burn rate
    ISP_AIRBREATHING   = 3500.0  # s  published SABRE target
    TOTAL_THRUST_KN    = 667.0   # kN  published design thrust
    # Total LH2 burn: thrust = m_dot_prop * Isp * g0
    m_dot_LH2_propulsion = TOTAL_THRUST_KN * 1e3 / (ISP_AIRBREATHING * G0)
    fuel_fraction_precooling = m_dot_LH2 / (m_dot_LH2 + m_dot_LH2_propulsion) \
                               if m_dot_LH2_propulsion > 0 else 0.0

    return {
        'M':                   M,
        'altitude_km':         altitude_m / 1e3,
        'T_ram_K':             T_ram,
        'T_static_K':          T_static,
        'p_ram_Pa':            p_ram,
        'p_static_Pa':         p_static,
        'V_ms':                V,
        'm_dot_air_kgs':       m_dot_air,
        'm_dot_LH2_kgs':       m_dot_LH2,
        'Q_MW':                Q_required / 1e6,
        'T_air_out_K':         T_air_out_actual,
        'T_wall_K':            T_wall,
        'T_dew_K':             T_dew,
        'effectiveness':       eff,
        'frost_risk':          frost_risk,
        'deposition_force':    deposition_driving_force * 1e6,  # scaled for display
        'RH_inlet_pct':        RH_inlet * 100,
        'fuel_frac_precooling': fuel_fraction_precooling * 100,
        'margin_K':            margin,
    }


def plot_full_envelope(output_path='sabre_precooler_analysis.png'):
    """
    Plot precooler performance across the full Mach 0.5 – 5 flight envelope
    for a representative trajectory (altitude scales with Mach).
    """
    # Representative cruise altitude trajectory: Mach 0.5 → 5, altitude 0 → 25 km
    # Based on published SABRE flight profile
    mach_vals = np.linspace(0.5, 5.0, 80)
    altitude_profile = np.interp(mach_vals,
                                  [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
                                  [0.0, 8.0, 15.0, 20.0, 23.0, 25.0]) * 1e3  # m

    # Also compute for sea level for comparison
    humidity_dry  = 0.003   # cruise (stratosphere, ~3 ppmv ≈ 0.003 g/kg)
    humidity_wet  = 10.0    # low-altitude humid day (~10 g/kg)

    results_cruise = [precooler_performance(M, h, humidity_dry)
                      for M, h in zip(mach_vals, altitude_profile)]
    results_sealevel = [precooler_performance(M, 0.0, humidity_wet)
                        for M in mach_vals]

    def extract(results, key):
        return np.array([r[key] for r in results])

    # ── Figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 20), facecolor='#0f0f0e')
    fig.patch.set_facecolor('#0f0f0e')

    GOLD  = '#b8920a'
    MOSS  = '#4a7a4b'
    PAPER = '#f0ede8'
    DIM   = '#8a8a7a'
    RED   = '#c04040'
    BLUE  = '#4080c0'
    CYAN  = '#40b0c0'

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.52, wspace=0.38,
                           left=0.08, right=0.96,
                           top=0.93, bottom=0.05)

    def style_ax(ax, title):
        ax.set_facecolor('#0f0f0e')
        for spine in ax.spines.values():
            spine.set_color('#3a3a38')
            spine.set_linewidth(0.6)
        ax.tick_params(colors=DIM, labelsize=9)
        ax.xaxis.label.set_color(DIM)
        ax.yaxis.label.set_color(DIM)
        ax.set_title(title, color=PAPER, fontsize=10.5, fontweight='bold',
                     pad=8, fontfamily='monospace')
        ax.grid(True, color='#2a2a28', linewidth=0.5, linestyle='--', alpha=0.7)

    # ── Panel 1: Ram temperature ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'RAM TEMPERATURE  vs  MACH')
    ax1.plot(mach_vals, extract(results_cruise, 'T_ram_K'),
             color=RED, linewidth=2.0, label='Cruise trajectory')
    ax1.plot(mach_vals, extract(results_sealevel, 'T_ram_K'),
             color=RED, linewidth=1.5, linestyle='--', alpha=0.6, label='Sea level')
    ax1.axhline(1000, color=RED, linewidth=0.8, linestyle=':', alpha=0.5)
    ax1.text(0.6, 1020, '~1000°C limit (Mach 5)', color=RED, fontsize=8,
             fontfamily='monospace', alpha=0.7)
    ax1.axhline(T_AIR_OUT_TARGET, color=MOSS, linewidth=0.8, linestyle=':', alpha=0.5)
    ax1.text(0.6, T_AIR_OUT_TARGET + 15, 'Precooler target', color=MOSS, fontsize=8,
             fontfamily='monospace', alpha=0.7)
    ax1.set_xlabel('Mach number', fontsize=9)
    ax1.set_ylabel('Temperature  [K]', fontsize=9)
    ax1.legend(fontsize=8, framealpha=0.0, labelcolor=DIM)

    # ── Panel 2: Heat load ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'PRECOOLER HEAT LOAD  [MW]')
    ax2.fill_between(mach_vals, extract(results_cruise, 'Q_MW'),
                     alpha=0.25, color=GOLD)
    ax2.plot(mach_vals, extract(results_cruise, 'Q_MW'),
             color=GOLD, linewidth=2.0, label='Cruise trajectory')
    ax2.plot(mach_vals, extract(results_sealevel, 'Q_MW'),
             color=GOLD, linewidth=1.5, linestyle='--', alpha=0.6, label='Sea level')
    ax2.set_xlabel('Mach number', fontsize=9)
    ax2.set_ylabel('Heat load  [MW]', fontsize=9)
    ax2.legend(fontsize=8, framealpha=0.0, labelcolor=DIM)

    # ── Panel 3: LH2 consumption ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'LH2 PRECOOLING FLOW RATE  [kg/s]')
    ax3.plot(mach_vals, extract(results_cruise, 'm_dot_LH2_kgs'),
             color=CYAN, linewidth=2.0, label='Precooling only (cruise)')
    ax3.fill_between(mach_vals, extract(results_cruise, 'm_dot_LH2_kgs'),
                     alpha=0.15, color=CYAN)
    ax_twin = ax3.twinx()
    ax_twin.set_facecolor('#0f0f0e')
    ax_twin.plot(mach_vals, extract(results_cruise, 'fuel_frac_precooling'),
                 color=MOSS, linewidth=1.5, linestyle='-.', label='% total LH2 budget')
    ax_twin.tick_params(colors=DIM, labelsize=9)
    ax_twin.yaxis.label.set_color(MOSS)
    ax_twin.set_ylabel('Precooling fuel fraction  [%]', fontsize=9, color=MOSS)
    ax_twin.spines['right'].set_color('#3a3a38')
    ax3.set_xlabel('Mach number', fontsize=9)
    ax3.set_ylabel('LH2 flow  [kg/s]', fontsize=9)

    # ── Panel 4: HX effectiveness + outlet temperature ───────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, 'HX EFFECTIVENESS  &  AIR OUTLET TEMPERATURE')
    ax4.plot(mach_vals, extract(results_cruise, 'effectiveness') * 100,
             color=MOSS, linewidth=2.0, label='Effectiveness [%]')
    ax4.set_xlabel('Mach number', fontsize=9)
    ax4.set_ylabel('Effectiveness  [%]', fontsize=9)
    ax4b = ax4.twinx()
    ax4b.set_facecolor('#0f0f0e')
    ax4b.plot(mach_vals, extract(results_cruise, 'T_air_out_K'),
              color=GOLD, linewidth=1.8, linestyle='--', label='T_air_out')
    ax4b.axhline(T_AIR_OUT_TARGET, color=RED, linewidth=0.7,
                 linestyle=':', alpha=0.5)
    ax4b.tick_params(colors=DIM, labelsize=9)
    ax4b.yaxis.label.set_color(GOLD)
    ax4b.set_ylabel('Air outlet temperature  [K]', fontsize=9, color=GOLD)
    ax4b.spines['right'].set_color('#3a3a38')
    ax4.legend(fontsize=8, framealpha=0.0, labelcolor=DIM, loc='lower left')

    # ── Panel 5: Frost risk across Mach-humidity space ───────────────────
    ax5 = fig.add_subplot(gs[2, :])
    style_ax(ax5, 'FROST RISK INDEX  —  MACH vs HUMIDITY  (Cruise Trajectory)')
    ax5.set_aspect('auto')
    M_grid   = np.linspace(0.5, 5.0, 60)
    hum_grid = np.linspace(0.001, 15.0, 40)   # g/kg: 0.001 (stratosphere) to 15 (humid troposphere)
    MM, HH = np.meshgrid(M_grid, hum_grid)
    FROST = np.zeros_like(MM)
    for i, hum in enumerate(hum_grid):
        for j, M in enumerate(M_grid):
            alt = np.interp(M, [0.5,1,2,3,4,5], [0,8,15,20,23,25]) * 1e3
            r = precooler_performance(M, alt, hum)
            FROST[i, j] = r['frost_risk']

    frost_cmap = LinearSegmentedColormap.from_list(
        'frost', [(0, '#0f4a2a'), (0.3, '#3a8a4a'), (0.6, '#c8a020'), (1.0, '#c83030')])
    cf = ax5.contourf(MM, HH, FROST,
                      levels=np.linspace(0, 1, 21), cmap=frost_cmap)
    cs = ax5.contour(MM, HH, FROST,
                     levels=[0.2, 0.4, 0.6, 0.8], colors='white',
                     linewidths=0.6, alpha=0.4)
    ax5.clabel(cs, fmt='%.1f', fontsize=8, colors='white')
    cb = plt.colorbar(cf, ax=ax5, pad=0.01)
    cb.set_label('Frost risk index  (0 = none, 1 = severe)', color=DIM, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=DIM, labelsize=8)
    cb.outline.set_edgecolor('#3a3a38')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=DIM)
    ax5.set_xlabel('Mach number', fontsize=10)
    ax5.set_ylabel('Specific humidity  [g H₂O / kg air]', fontsize=10)
    ax5.axvline(DESIGN_MACH, color=GOLD, linewidth=1.0, linestyle='--', alpha=0.6)
    ax5.text(DESIGN_MACH + 0.05, 13.5, 'Design\npoint', color=GOLD,
             fontsize=8, fontfamily='monospace', alpha=0.8)
    # Mark typical stratospheric humidity (very dry)
    ax5.axhline(0.003, color=CYAN, linewidth=0.8, linestyle=':', alpha=0.5)
    ax5.text(0.55, 0.003 + 0.2, 'Typical cruise humidity (0.003 g/kg)', color=CYAN,
             fontsize=8, fontfamily='monospace', alpha=0.7)

    # ── Panel 6: Altitude-Mach envelope with frost margin ────────────────
    ax6 = fig.add_subplot(gs[3, 0])
    style_ax(ax6, 'FROST MARGIN  vs  ALTITUDE  (Mach 5)')
    alts = np.linspace(0, 35000, 60)
    margins = [precooler_performance(5.0, h, 0.003)['margin_K'] for h in alts]
    t_dews  = [precooler_performance(5.0, h, 0.003)['T_dew_K']  for h in alts]
    t_walls = [precooler_performance(5.0, h, 0.003)['T_wall_K'] for h in alts]
    ax6.plot(np.array(alts)/1000, margins, color=GOLD, linewidth=2.0,
             label='T_wall − T_dew')
    ax6.axhline(0, color=RED, linewidth=0.8, linestyle='--', alpha=0.7)
    ax6.fill_between(np.array(alts)/1000, margins, 0,
                     where=np.array(margins) < 0,
                     color=RED, alpha=0.2, label='Frost deposition zone')
    ax6.fill_between(np.array(alts)/1000, margins, 0,
                     where=np.array(margins) >= 0,
                     color=MOSS, alpha=0.15, label='Frost-free zone')
    ax6.set_xlabel('Altitude  [km]', fontsize=9)
    ax6.set_ylabel('Frost margin  [K]', fontsize=9)
    ax6.legend(fontsize=8, framealpha=0.0, labelcolor=DIM)

    # ── Panel 7: Sensitivity — outlet T vs NTU ───────────────────────────
    ax7 = fig.add_subplot(gs[3, 1])
    style_ax(ax7, 'SENSITIVITY: OUTLET TEMPERATURE  vs  NTU  (Mach 5, 25 km)')
    ntu_vals = np.linspace(1.0, 8.0, 60)
    T_outs = []
    effs   = []
    T_ram_design, _, _, _, _ = ram_conditions(5.0, 25000)
    for ntu in ntu_vals:
        C_r_approx = 0.95
        try:
            e = 1 - np.exp((1/C_r_approx)*ntu**0.22*(np.exp(-C_r_approx*ntu**0.78)-1))
        except:
            e = 0.99
        e = min(e, 0.995)
        T_out = T_ram_design - e * (T_ram_design - T_LH2_IN)
        T_outs.append(T_out)
        effs.append(e * 100)
    ax7.plot(ntu_vals, T_outs, color=CYAN, linewidth=2.0, label='Air outlet T [K]')
    ax7.axhline(T_AIR_OUT_TARGET, color=RED, linewidth=0.8, linestyle='--', alpha=0.7)
    ax7.text(1.1, T_AIR_OUT_TARGET + 10, 'Compressor inlet limit',
             color=RED, fontsize=8, fontfamily='monospace', alpha=0.7)
    ax7.axvline(PRECOOLER_NTU, color=GOLD, linewidth=1.0, linestyle=':', alpha=0.7)
    ax7.text(PRECOOLER_NTU + 0.1, max(T_outs) * 0.6, f'NTU = {PRECOOLER_NTU}',
             color=GOLD, fontsize=8, fontfamily='monospace')
    ax7.set_xlabel('NTU  (Number of Transfer Units)', fontsize=9)
    ax7.set_ylabel('Air outlet temperature  [K]', fontsize=9)
    ax7.legend(fontsize=8, framealpha=0.0, labelcolor=DIM)

    # ── Header ─────────────────────────────────────────────────────────────
    fig.text(0.5, 0.975,
             'SABRE PRECOOLER — THERMODYNAMIC PERFORMANCE ANALYSIS',
             ha='center', va='top', color=PAPER,
             fontsize=14, fontweight='bold', fontfamily='monospace')
    fig.text(0.5, 0.960,
             'NTU-effectiveness model  |  ISA atmosphere  |  Isentropic ram compression  |  Frost: convective mass transfer analogy',
             ha='center', va='top', color=DIM,
             fontsize=8.5, fontfamily='monospace')
    fig.text(0.99, 0.005,
             'pringlesmaths.co.uk  |  MKA  |  For discussion purposes',
             ha='right', va='bottom', color='#3a3a38',
             fontsize=8, fontfamily='monospace')

    plt.savefig(output_path, dpi=160, bbox_inches='tight',
                facecolor='#0f0f0e', edgecolor='none')
    plt.close()
    print(f"Plot saved to {output_path}")
    return output_path


def print_single_point(M, altitude_km, humidity=0.003):
    """Print a detailed single-point analysis table."""
    r = precooler_performance(M, altitude_km * 1e3, humidity)
    print()
    print("=" * 60)
    print(f"  SABRE PRECOOLER  —  SINGLE OPERATING POINT")
    print("=" * 60)
    print(f"  Mach number         :  {r['M']:.2f}")
    print(f"  Altitude            :  {r['altitude_km']:.1f} km")
    print(f"  Specific humidity   :  {humidity*1000:.2f} g/kg")
    print("-" * 60)
    print(f"  Ram temperature     :  {r['T_ram_K']:.1f} K  ({r['T_ram_K']-273.15:.0f} °C)")
    print(f"  Static temperature  :  {r['T_static_K']:.1f} K")
    print(f"  Ram pressure        :  {r['p_ram_Pa']/1e5:.3f} bar")
    print(f"  Flight velocity     :  {r['V_ms']:.0f} m/s  ({r['V_ms']/340:.2f} × a₀)")
    print("-" * 60)
    print(f"  Air mass flow       :  {r['m_dot_air_kgs']:.1f} kg/s")
    print(f"  Heat load           :  {r['Q_MW']:.2f} MW")
    print(f"  HX effectiveness    :  {r['effectiveness']*100:.2f} %")
    print(f"  LH2 precool flow    :  {r['m_dot_LH2_kgs']:.2f} kg/s")
    print(f"  Air outlet temp     :  {r['T_air_out_K']:.1f} K  ({r['T_air_out_K']-273.15:.0f} °C)")
    print(f"  Fuel frac (precool) :  {r['fuel_frac_precooling']:.1f} % of total LH2")
    print("-" * 60)
    print(f"  Tube wall temp      :  {r['T_wall_K']:.1f} K")
    print(f"  Dew point (inlet)   :  {r['T_dew_K']:.1f} K")
    print(f"  Frost margin        :  {r['margin_K']:+.1f} K  "
          f"({'SAFE — no deposition' if r['margin_K'] > 0 else '⚠ FROST RISK'})")
    print(f"  Frost risk index    :  {r['frost_risk']:.3f}  "
          f"({'negligible' if r['frost_risk'] < 0.1 else 'low' if r['frost_risk'] < 0.3 else 'moderate' if r['frost_risk'] < 0.6 else 'HIGH'})")
    print(f"  Inlet humidity      :  {r['RH_inlet_pct']:.2f} % RH")
    print("=" * 60)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SABRE Precooler Performance Calculator')
    parser.add_argument('--point', action='store_true',
                        help='Print single design-point analysis instead of plots')
    parser.add_argument('--mach', type=float, default=5.0,
                        help='Mach number for single-point (default: 5.0)')
    parser.add_argument('--alt', type=float, default=25.0,
                        help='Altitude in km for single-point (default: 25.0)')
    parser.add_argument('--humidity', type=float, default=0.003,
                        help='Specific humidity g/kg (default: 0.003 g/kg = stratosphere)')
    parser.add_argument('--output', type=str,
                        default='sabre_precooler_analysis.png',
                        help='Output path for plot')
    args = parser.parse_args()

    if args.point:
        print_single_point(args.mach, args.alt, args.humidity)
        # Also print a few notable points
        print("Notable trajectory points:")
        for M, alt_km in [(1.0, 8.0), (2.0, 15.0), (3.0, 20.0), (4.0, 23.0), (5.0, 25.0)]:
            r = precooler_performance(M, alt_km * 1e3, 0.003)
            print(f"  Mach {M:.0f} @ {alt_km:.0f} km  |  "
                  f"T_ram={r['T_ram_K']:.0f} K  |  "
                  f"Q={r['Q_MW']:.1f} MW  |  "
                  f"LH2={r['m_dot_LH2_kgs']:.1f} kg/s  |  "
                  f"Frost={'⚠' if r['frost_risk'] > 0.3 else 'OK':2s}  {r['frost_risk']:.2f}")
    else:
        print("Generating SABRE precooler performance plots...")
        plot_full_envelope(args.output)
        print_single_point(DESIGN_MACH, DESIGN_ALT_KM, humidity=0.003)
