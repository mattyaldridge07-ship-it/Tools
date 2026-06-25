"""
Hypersonic air-breathing trajectory simulation and heat exchanger sizing optimization tool.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import CubicSpline
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import base model
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sabre_precooler import (
    isa_atmosphere, ram_conditions, precooler_performance,
    T_AIR_OUT_TARGET, T_LH2_IN, PRECOOLER_NTU,
    G0, CP_AIR, CP_H2, GAMMA, R_AIR,
    PRECOOLER_ACTIVATION_T
)

# Colour palette
GOLD   = '#b8920a'; MOSS  = '#4a7a4b'; PAPER = '#f0ede8'
DIM    = '#8a8a7a'; RED   = '#c04040'; BLUE  = '#4080c0'
CYAN   = '#40b0c0'; GREY  = '#3a3a38'; BG    = '#0f0f0e'

# ══════════════════════════════════════════════════════════════════════════════
#  TRAJECTORY DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def trajectory_ssto():
    """
    SSTO / SABRE-style horizontal takeoff to Mach 5 @ 25 km.
    Waypoints: (time_s, mach, altitude_km, humidity_gkg, phase_label)
    Based on standard single-stage-to-orbit (SSTO) hypersonic flight profiles.
    """
    waypoints = [
        #  t      M    alt    hum   phase
        (   0,  0.0,  0.00,  8.0,  'Ground roll'),
        (  60,  0.3,  0.05,  8.0,  'Rotate'),
        ( 180,  0.7,  3.0,   6.0,  'Subsonic climb'),
        ( 300,  0.9,  7.0,   4.0,  'Transonic acceleration'),
        ( 420,  1.2,  10.0,  2.0,  'Supersonic climb'),
        ( 600,  2.0,  14.0,  0.5,  'Supersonic cruise'),
        ( 780,  3.0,  18.0,  0.1,  'High supersonic'),
        ( 960,  4.0,  22.0,  0.01, 'Pre-hypersonic'),
        (1080,  5.0,  25.0,  0.003,'Hypersonic design point'),
    ]
    return waypoints, 'SSTO Horizontal Takeoff Profile'


def trajectory_tbcc():
    """
    Hypersonic cruise mode-transition profile.
    Critical phase: Mach 0.5 → 3, low altitude, humid atmosphere.
    The turbojet→ramjet transition occurs around Mach 2.5–3.
    Frost risk peaks during subsonic/transonic phase before altitude gain.
    """
    waypoints = [
        #  t      M    alt    hum   phase
        (   0,  0.5,  3.0,  10.0,  'Launch / drop'),
        (  30,  0.8,  5.0,   8.0,  'Turbojet acceleration'),
        (  90,  1.0,  7.0,   6.0,  'Transonic (shock ingestion)'),
        ( 150,  1.5,  9.0,   4.0,  'Low supersonic'),
        ( 240,  2.0, 11.0,   2.0,  'Pre-transition'),
        ( 330,  2.5, 13.0,   0.8,  'Mode transition zone'),
        ( 420,  3.0, 15.0,   0.3,  'Ramjet takeover'),
        ( 540,  3.5, 17.0,   0.1,  'Ramjet acceleration'),
        ( 660,  4.0, 20.0,   0.02, 'High supersonic ramjet'),
        ( 780,  5.0, 25.0,   0.003,'Hypersonic dash'),
    ]
    return waypoints, 'Supersonic/Hypersonic Mode-Transition Profile'


# ══════════════════════════════════════════════════════════════════════════════
#  TRAJECTORY INTEGRATOR
# ══════════════════════════════════════════════════════════════════════════════

def integrate_trajectory(waypoints, dt=5.0):
    """
    Integrate precooler performance along a trajectory.

    Parameters
    ----------
    waypoints : list of (t, M, alt_km, hum_gkg, label)
    dt        : time step in seconds

    Returns
    -------
    dict of numpy arrays, one value per time step
    """
    t_wp  = np.array([w[0] for w in waypoints])
    M_wp  = np.array([w[1] for w in waypoints])
    h_wp  = np.array([w[2] for w in waypoints])
    hm_wp = np.array([w[3] for w in waypoints])

    # Cubic spline interpolation for smooth trajectory
    cs_M   = CubicSpline(t_wp, M_wp,  bc_type='natural')
    cs_h   = CubicSpline(t_wp, h_wp,  bc_type='natural')
    cs_hum = CubicSpline(t_wp, hm_wp, bc_type='natural')

    t_end = t_wp[-1]
    t_vec = np.arange(0, t_end + dt, dt)

    M_vec   = np.clip(cs_M(t_vec),   0.1, 6.0)
    h_vec   = np.clip(cs_h(t_vec),   0.0, 40.0) * 1e3   # m
    hum_vec = np.clip(cs_hum(t_vec), 0.001, 20.0)

    # Per-timestep performance
    keys = ['T_ram_K','T_air_out_K','Q_MW','m_dot_LH2_kgs',
            'frost_risk','effectiveness','T_wall_K','T_dew_K',
            'margin_K','RH_inlet_pct','fuel_frac_precooling',
            'm_dot_air_kgs','V_ms','deposition_force']
    results = {k: np.zeros(len(t_vec)) for k in keys}
    results['t']   = t_vec
    results['M']   = M_vec
    results['alt_km'] = h_vec / 1e3

    for i, (t, M, h, hum) in enumerate(zip(t_vec, M_vec, h_vec, hum_vec)):
        r = precooler_performance(M, h, hum)
        for k in keys:
            results[k][i] = r[k]
        # Precooler only active when ram T exceeds activation threshold
        if r['T_ram_K'] <= PRECOOLER_ACTIVATION_T:
            results['m_dot_LH2_kgs'][i] = 0.0
            results['Q_MW'][i] = 0.0

    # Cumulative LH2 consumed (kg) — trapezoidal integration
    results['LH2_cumulative_kg'] = np.cumsum(
        results['m_dot_LH2_kgs'] * dt)

    # Mode transition flag: turbojet viable below ~1100 K ram temp
    # Above this temperature, compressor blades approach material limits
    # without precooling assistance
    results['turbojet_limit_K'] = np.full(len(t_vec), 1100.0)
    results['mode_transition_idx'] = np.argmax(results['T_ram_K'] > 800.0)

    # Frost alert: risk > 0.15 is operationally significant
    results['frost_alert'] = results['frost_risk'] > 0.15

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  HEAT EXCHANGER GEOMETRY BACK-CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def geometry_calculator(NTU_target=4.2, M_design=5.0, alt_design_km=25.0,
                        humidity=0.003):
    """
    Given a target NTU, back-calculate a viable tube bundle geometry and
    compute the aerodynamic pressure drop penalty.

    Model: crossflow compact heat exchanger, circular tubes.
    Air-side: external crossflow over tube bank (Zukauskas correlation)
    LH2-side: internal forced convection (Dittus-Boelter)

    Returns
    -------
    dict with geometry parameters and performance metrics
    """
    # Operating conditions
    r = precooler_performance(M_design, alt_design_km * 1e3, humidity)
    T_ram  = r['T_ram_K']
    p_ram  = r['p_ram_Pa']
    m_air  = r['m_dot_air_kgs']
    m_LH2  = max(r['m_dot_LH2_kgs'], 0.1)

    # Fluid properties at mean temperatures
    T_air_mean = (T_ram + T_AIR_OUT_TARGET) / 2.0      # ~730 K
    T_LH2_mean = (T_LH2_IN + 250.0) / 2.0              # ~135 K

    # Air properties at T_air_mean, p_ram (simplified power laws)
    mu_air  = 1.458e-6 * T_air_mean**1.5 / (T_air_mean + 110.4)  # Pa·s (Sutherland)
    k_air   = 0.0241 * (T_air_mean / 273.15)**0.82               # W/(m·K)
    Pr_air  = 0.72                                                  # ≈ constant for air
    rho_air = p_ram / (R_AIR * T_air_mean)

    # LH2 properties (approximate, ~100–200 K, 10 bar)
    mu_LH2  = 1.2e-5    # Pa·s  (liquid hydrogen, rough estimate)
    k_LH2   = 0.10      # W/(m·K)
    Pr_LH2  = 0.85
    rho_LH2 = 60.0      # kg/m³  (supercritical at ~10 bar)
    cp_LH2  = CP_H2

    # Design sweep: vary tube outer diameter
    # SABRE precooler tubes: ~0.5–2 mm OD, thin wall (~0.05 mm)
    d_outer_range = np.linspace(0.4e-3, 2.5e-3, 40)   # m
    t_wall        = 0.05e-3                              # m (fixed)
    sigma_wall    = 0.1                                  # tube solidity (fraction of frontal area occupied by tubes)
    pitch_ratio   = 1.25                                 # tube pitch / tube OD (St = Sl = pitch_ratio * d)

    results = []
    for d_o in d_outer_range:
        d_i = d_o - 2 * t_wall
        if d_i <= 0:
            continue

        A_tube_x = np.pi * d_o**2 / 4.0   # tube cross-sectional area
        pitch = pitch_ratio * d_o

        # Frontal area of precooler (m²) — sized to pass required air mass flow
        # at ram conditions without excessive blockage
        # V_air through precooler ≈ Mach 0.1–0.2 (subsonic in HX)
        V_hx = 0.15 * np.sqrt(GAMMA * R_AIR * T_air_mean)  # ~80 m/s
        A_frontal = m_air / (rho_air * V_hx * (1.0 - sigma_wall))

        # Reynolds number (air side, based on d_o)
        Re_air = rho_air * V_hx * d_o / mu_air

        # Zukauskas correlation for tube bank (Incropera, Ch 7)
        # Nu = C * Re^m * Pr^0.36 * (Pr/Pr_wall)^0.25
        # For Re = 10³–2×10⁵, aligned arrangement:
        if Re_air < 100:
            C, m_exp = 0.8, 0.40
        elif Re_air < 1000:
            C, m_exp = 0.51, 0.50
        elif Re_air < 2e5:
            C, m_exp = 0.27, 0.63
        else:
            C, m_exp = 0.021, 0.84

        Nu_air  = C * Re_air**m_exp * Pr_air**0.36
        h_air   = Nu_air * k_air / d_o    # W/(m²·K)

        # Dittus-Boelter for LH2 internal (heating: n=0.4)
        A_LH2_flow = np.pi * d_i**2 / 4.0
        # Number of tubes in parallel (approximate)
        n_tubes_parallel = max(1, int(A_frontal / pitch**2))
        V_LH2 = m_LH2 / (rho_LH2 * n_tubes_parallel * A_LH2_flow)
        Re_LH2 = rho_LH2 * V_LH2 * d_i / mu_LH2
        Re_LH2 = max(Re_LH2, 100.0)

        if Re_LH2 > 10000:
            Nu_LH2 = 0.023 * Re_LH2**0.8 * Pr_LH2**0.4
        else:
            Nu_LH2 = 3.66   # laminar, constant wall temp
        h_LH2 = Nu_LH2 * k_LH2 / d_i

        # Overall heat transfer coefficient (neglect wall conduction — thin wall)
        # 1/U = 1/h_air + (d_o/d_i)/h_LH2
        U = 1.0 / (1.0/h_air + (d_o/d_i)/h_LH2)   # W/(m²·K)

        # Required surface area from NTU definition
        C_min = min(m_air * CP_AIR, m_LH2 * cp_LH2)
        A_required = NTU_target * C_min / U          # m²

        # Tube length from A_required and number of tubes
        # A = n_tubes * π * d_o * L
        n_tubes_total = n_tubes_parallel
        L_tube = A_required / (n_tubes_total * np.pi * d_o)
        L_tube = max(L_tube, 0.01)

        # Pressure drop (air side, Fanning friction, tube bank)
        # Euler number correlation for aligned tube bank:
        Eu = 0.18 * (Re_air / 1000)**(-0.25) if Re_air > 100 else 0.5
        n_rows = max(1, int(L_tube / pitch))   # number of tube rows
        dp_air = Eu * n_rows * rho_air * V_hx**2 / 2.0   # Pa

        # Pressure drop as fraction of ram pressure
        dp_fraction = dp_air / p_ram

        # Specific surface area (m² per m³ of HX volume)
        V_HX_total = A_frontal * L_tube
        alpha = A_required / max(V_HX_total, 1e-6)   # m²/m³

        # Total HX mass estimate (tube mass only, thin wall)
        # Material: Inconel 718, rho ≈ 8200 kg/m³
        rho_metal = 8200.0
        V_metal   = n_tubes_total * np.pi * d_o * t_wall * L_tube
        mass_kg   = V_metal * rho_metal

        results.append({
            'd_outer_mm':     d_o * 1e3,
            'd_inner_mm':     d_i * 1e3,
            'n_tubes':        n_tubes_total,
            'L_tube_m':       L_tube,
            'U_Wm2K':         U,
            'h_air_Wm2K':     h_air,
            'h_LH2_Wm2K':     h_LH2,
            'Re_air':         Re_air,
            'Re_LH2':         Re_LH2,
            'A_required_m2':  A_required,
            'dp_Pa':          dp_air,
            'dp_fraction_pct':dp_fraction * 100,
            'alpha_m2m3':     alpha,
            'mass_kg':        mass_kg,
            'A_frontal_m2':   A_frontal,
            'NTU_achieved':   U * A_required / C_min,
            'V_hx_L':         A_frontal * L_tube * 1000,
        })

    return results, r


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def style_ax(ax, title, fig_bg=BG):
    ax.set_facecolor(fig_bg)
    for spine in ax.spines.values():
        spine.set_color(GREY); spine.set_linewidth(0.6)
    ax.tick_params(colors=DIM, labelsize=9)
    ax.xaxis.label.set_color(DIM)
    ax.yaxis.label.set_color(DIM)
    ax.set_title(title, color=PAPER, fontsize=10, fontweight='bold',
                 pad=8, fontfamily='monospace')
    ax.grid(True, color='#2a2a28', linewidth=0.5, linestyle='--', alpha=0.7)


def ctext(ax, x, y, s, **kw):
    ax.text(x, y, s, transform=ax.transAxes, **kw)


def plot_trajectory_results(res, title, waypoints, output_path):
    """Seven-panel mission time-series plot."""
    t   = res['t'] / 60.0   # minutes
    wt  = [w[0]/60.0 for w in waypoints]
    wl  = [w[4]      for w in waypoints]

    fig = plt.figure(figsize=(18, 22), facecolor=BG)
    gs  = gridspec.GridSpec(4, 2, figure=fig,
                            hspace=0.55, wspace=0.38,
                            left=0.08, right=0.96,
                            top=0.94, bottom=0.05)

    fig.text(0.5, 0.975, title.upper(),
             ha='center', va='top', color=PAPER,
             fontsize=13, fontweight='bold', fontfamily='monospace')
    fig.text(0.5, 0.961,
             'Time-resolved precooler performance  ·  NTU-effectiveness model  ·  ISA atmosphere',
             ha='center', va='top', color=DIM, fontsize=8.5, fontfamily='monospace')
    
    def add_waypoints(ax, y_pos=0.97):
        for wt_i, wl_i in zip(wt, wl):
            ax.axvline(wt_i, color=GOLD, linewidth=0.5, alpha=0.3, linestyle=':')

    # Panel 1: Trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'FLIGHT TRAJECTORY')
    ax1_r = ax1.twinx()
    ax1_r.set_facecolor(BG)
    ax1.plot(t, res['M'],      color=GOLD, linewidth=2.0, label='Mach')
    ax1_r.plot(t, res['alt_km'], color=CYAN, linewidth=1.6,
               linestyle='--', label='Altitude [km]')
    add_waypoints(ax1)
    ax1.set_xlabel('Mission time  [min]', fontsize=9)
    ax1.set_ylabel('Mach number', fontsize=9)
    ax1_r.set_ylabel('Altitude  [km]', fontsize=9, color=CYAN)
    ax1_r.tick_params(colors=CYAN, labelsize=9)
    ax1_r.spines['right'].set_color(GREY)
    ax1.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper left')
    ax1_r.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='lower right')

    # Annotate phase labels at waypoints
    for i, (wt_i, wl_i) in enumerate(zip(wt[::2], wl[::2])):
        ax1.annotate(wl_i, xy=(wt_i, res['M'][np.argmin(np.abs(res['t']/60-wt_i))]),
                     xytext=(0, 12), textcoords='offset points',
                     fontsize=6.5, color=GOLD, alpha=0.7,
                     fontfamily='monospace', ha='center',
                     arrowprops=dict(arrowstyle='-', color=GOLD, alpha=0.3, lw=0.5))

    # Panel 2: Ram temperature + compressor limit
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'RAM TEMPERATURE & COMPRESSOR LIMIT')
    ax2.plot(t, res['T_ram_K'],       color=RED,  linewidth=2.0, label='Ram T')
    ax2.plot(t, res['T_air_out_K'],   color=MOSS, linewidth=1.6, label='Precooler outlet T')
    ax2.axhline(1273.15, color=RED, linewidth=0.7, linestyle=':', alpha=0.5)
    ax2.text(t[-1]*0.02, 1290, '1000°C material limit', color=RED,
             fontsize=7.5, fontfamily='monospace', alpha=0.7)
    ax2.axhline(T_AIR_OUT_TARGET, color=MOSS, linewidth=0.7, linestyle=':', alpha=0.5)
    ax2.text(t[-1]*0.02, T_AIR_OUT_TARGET+15, 'Compressor inlet target',
             color=MOSS, fontsize=7.5, fontfamily='monospace', alpha=0.7)
    add_waypoints(ax2)
    ax2.set_xlabel('Mission time  [min]', fontsize=9)
    ax2.set_ylabel('Temperature  [K]', fontsize=9)
    ax2.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # Panel 3: Heat load + LH2 flow
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'HEAT LOAD  &  LH2 PRECOOLING FLOW')
    ax3.fill_between(t, res['Q_MW'], alpha=0.2, color=GOLD)
    ax3.plot(t, res['Q_MW'],          color=GOLD, linewidth=2.0, label='Heat load [MW]')
    ax3_r = ax3.twinx()
    ax3_r.set_facecolor(BG)
    ax3_r.plot(t, res['m_dot_LH2_kgs'], color=CYAN, linewidth=1.6,
               linestyle='--', label='LH2 flow [kg/s]')
    ax3_r.tick_params(colors=CYAN, labelsize=9)
    ax3_r.spines['right'].set_color(GREY)
    ax3_r.set_ylabel('LH2 flow rate  [kg/s]', fontsize=9, color=CYAN)
    add_waypoints(ax3)
    ax3.set_xlabel('Mission time  [min]', fontsize=9)
    ax3.set_ylabel('Heat load  [MW]', fontsize=9)
    ax3.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper left')
    ax3_r.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='lower right')

    # Panel 4: Cumulative LH2 consumed
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, 'CUMULATIVE LH2 CONSUMED  (PRECOOLING ONLY)')
    ax4.fill_between(t, res['LH2_cumulative_kg'], alpha=0.15, color=BLUE)
    ax4.plot(t, res['LH2_cumulative_kg'], color=BLUE, linewidth=2.0)
    # Annotate total
    total_lh2 = res['LH2_cumulative_kg'][-1]
    ax4.annotate(f'Total: {total_lh2:.0f} kg\n(precooling only)',
                 xy=(t[-1], total_lh2),
                 xytext=(-80, -30), textcoords='offset points',
                 color=BLUE, fontsize=8.5, fontfamily='monospace',
                 arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.0))
    add_waypoints(ax4)
    ax4.set_xlabel('Mission time  [min]', fontsize=9)
    ax4.set_ylabel('LH2 consumed  [kg]', fontsize=9)

    # Panel 5: Frost risk
    ax5 = fig.add_subplot(gs[2, 0])
    style_ax(ax5, 'FROST RISK INDEX  (0 = NONE,  1 = SEVERE)')

    # Colour fill by risk level
    frost_cmap = LinearSegmentedColormap.from_list(
        'fr', [(0,'#0f3a1a'), (0.15,'#2a7a3a'), (0.4,'#c8a020'), (1.0,'#c83030')])
    for i in range(len(t)-1):
        risk = res['frost_risk'][i]
        ax5.fill_between(t[i:i+2], 0, res['frost_risk'][i:i+2],
                         color=frost_cmap(risk), alpha=0.85)
    ax5.plot(t, res['frost_risk'], color='white', linewidth=0.8, alpha=0.5)
    ax5.axhline(0.15, color=GOLD, linewidth=0.8, linestyle='--', alpha=0.7)
    ax5.text(t[-1]*0.02, 0.17, 'Operational threshold (0.15)',
             color=GOLD, fontsize=7.5, fontfamily='monospace', alpha=0.8)

    # Shade frost-alert zones
    alert = res['frost_alert']
    if np.any(alert):
        ax5.fill_between(t, 0, 1, where=alert, alpha=0.08, color=RED)

    add_waypoints(ax5)
    ax5.set_ylim(0, 1.05)
    ax5.set_xlabel('Mission time  [min]', fontsize=9)
    ax5.set_ylabel('Frost risk index', fontsize=9)

    # Panel 6: Frost margin + dew point vs wall temp
    ax6 = fig.add_subplot(gs[2, 1])
    style_ax(ax6, 'FROST MARGIN  (T_WALL − T_DEW)')
    ax6.axhline(0, color=RED, linewidth=1.0, linestyle='--', alpha=0.6)
    ax6.fill_between(t, res['margin_K'], 0,
                     where=res['margin_K'] < 0,
                     color=RED, alpha=0.18, label='Frost zone (thermodynamic)')
    ax6.fill_between(t, res['margin_K'], 0,
                     where=res['margin_K'] >= 0,
                     color=MOSS, alpha=0.12, label='Safe zone')
    ax6.plot(t, res['margin_K'], color=GOLD, linewidth=1.8)
    add_waypoints(ax6)
    ax6.set_xlabel('Mission time  [min]', fontsize=9)
    ax6.set_ylabel('T_wall − T_dew  [K]', fontsize=9)
    ax6.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # Panel 7: HX effectiveness
    ax7 = fig.add_subplot(gs[3, :])
    style_ax(ax7, 'HX EFFECTIVENESS  &  PRECOOLER FUEL FRACTION  vs  MISSION TIME')
    ax7.plot(t, res['effectiveness'] * 100, color=MOSS,
             linewidth=2.0, label='HX effectiveness [%]')
    ax7_r = ax7.twinx()
    ax7_r.set_facecolor(BG)
    ax7_r.plot(t, res['fuel_frac_precooling'], color=GOLD, linewidth=1.6,
               linestyle='-.', label='Precooling % of LH2 budget')
    ax7_r.tick_params(colors=GOLD, labelsize=9)
    ax7_r.spines['right'].set_color(GREY)
    ax7_r.set_ylabel('Precooling fuel fraction  [%]', fontsize=9, color=GOLD)
    add_waypoints(ax7)
    ax7.set_xlabel('Mission time  [min]', fontsize=9)
    ax7.set_ylabel('HX effectiveness  [%]', fontsize=9)
    ax7.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper left')
    ax7_r.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper right')

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"  Trajectory plot saved: {output_path}")


def plot_geometry(geo_results, op_conds, output_path):
    """Four-panel heat exchanger geometry trade study plot."""
    d   = np.array([r['d_outer_mm']    for r in geo_results])
    U   = np.array([r['U_Wm2K']        for r in geo_results])
    A   = np.array([r['A_required_m2'] for r in geo_results])
    dp  = np.array([r['dp_fraction_pct'] for r in geo_results])
    L   = np.array([r['L_tube_m']      for r in geo_results])
    mass= np.array([r['mass_kg']       for r in geo_results])
    alph= np.array([r['alpha_m2m3']    for r in geo_results])
    V   = np.array([r['V_hx_L']        for r in geo_results])

    fig = plt.figure(figsize=(16, 14), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.48, wspace=0.38,
                            left=0.09, right=0.96, top=0.91, bottom=0.07)

    fig.text(0.5, 0.965,
             'PRECOOLER TUBE BUNDLE — GEOMETRY TRADE STUDY',
             ha='center', va='top', color=PAPER,
             fontsize=13, fontweight='bold', fontfamily='monospace')
    M_d = op_conds['M']
    h_d = op_conds['altitude_km']
    fig.text(0.5, 0.950,
             f'Design point: Mach {M_d:.1f} @ {h_d:.0f} km  |  '
             f'Target NTU = {PRECOOLER_NTU:.1f}  |  '
             f'Air-side Zukauskas correlation  |  LH2-side Dittus-Boelter',
             ha='center', va='top', color=DIM, fontsize=8.5,
             fontfamily='monospace')
    
    # Panel 1: U and A vs tube diameter
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'HEAT TRANSFER COEFFICIENT  &  SURFACE AREA  vs  TUBE Ø')
    ax1.plot(d, U / 1000, color=GOLD, linewidth=2.0, label='U  [kW/m²·K]')
    ax1_r = ax1.twinx()
    ax1_r.set_facecolor(BG)
    ax1_r.plot(d, A, color=CYAN, linewidth=1.6, linestyle='--', label='A_req  [m²]')
    ax1_r.tick_params(colors=CYAN, labelsize=9)
    ax1_r.spines['right'].set_color(GREY)
    ax1_r.set_ylabel('Required surface area  [m²]', fontsize=9, color=CYAN)
    ax1.set_xlabel('Tube outer diameter  [mm]', fontsize=9)
    ax1.set_ylabel('U  [kW/(m²·K)]', fontsize=9)
    ax1.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper right')
    ax1_r.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper left')

    # SABRE design point annotation
    sabre_d = 1.0
    idx_sabre = np.argmin(np.abs(d - sabre_d))
    ax1.axvline(sabre_d, color=RED, linewidth=0.8, linestyle=':', alpha=0.6)
    ax1.text(sabre_d + 0.05, U[idx_sabre]/1000 * 0.5,
             f'SABRE-reported\nd ≈ 1 mm',
             color=RED, fontsize=7.5, fontfamily='monospace', alpha=0.8)

    # Panel 2: Pressure drop vs tube diameter
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'PRESSURE DROP PENALTY  vs  TUBE Ø')
    ax2.fill_between(d, dp, alpha=0.18, color=RED)
    ax2.plot(d, dp, color=RED, linewidth=2.0)
    ax2.axhline(1.0, color=GOLD, linewidth=0.8, linestyle='--', alpha=0.6)
    ax2.text(d[0] + 0.05, 1.05, '1% ΔP/P_ram limit',
             color=GOLD, fontsize=8, fontfamily='monospace', alpha=0.8)
    ax2.axvline(sabre_d, color=GOLD, linewidth=0.8, linestyle=':', alpha=0.5)
    ax2.set_xlabel('Tube outer diameter  [mm]', fontsize=9)
    ax2.set_ylabel('ΔP / P_ram  [%]', fontsize=9)

    # Panel 3: Specific surface area (compactness)
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'HX COMPACTNESS  &  ESTIMATED MASS  vs  TUBE Ø')
    ax3.plot(d, alph, color=MOSS, linewidth=2.0, label='Spec. surface area [m²/m³]')
    ax3_r = ax3.twinx()
    ax3_r.set_facecolor(BG)
    ax3_r.plot(d, mass, color=GOLD, linewidth=1.6, linestyle='--',
               label='Estimated mass [kg]')
    ax3_r.tick_params(colors=GOLD, labelsize=9)
    ax3_r.spines['right'].set_color(GREY)
    ax3_r.set_ylabel('Estimated tube mass  [kg]', fontsize=9, color=GOLD)
    ax3.set_xlabel('Tube outer diameter  [mm]', fontsize=9)
    ax3.set_ylabel('Specific surface area  [m²/m³]', fontsize=9)
    ax3.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper right')
    ax3_r.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='lower right')

    # Panel 4: Pareto trade — compactness vs pressure drop
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, 'PARETO TRADE:  COMPACTNESS  vs  ΔP PENALTY')
    sc = ax4.scatter(dp, alph, c=d, cmap='plasma', s=60, zorder=5)
    ax4.set_xlabel('Pressure drop penalty  ΔP/P_ram  [%]', fontsize=9)
    ax4.set_ylabel('HX compactness  [m²/m³]', fontsize=9)
    cb = plt.colorbar(sc, ax=ax4, pad=0.01)
    cb.set_label('Tube Ø  [mm]', color=DIM, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=DIM, labelsize=8)
    cb.outline.set_edgecolor(GREY)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=DIM)

    # Mark operating region
    mask = dp < 1.5
    if np.any(mask):
        ax4.fill_between(dp[mask], 0, alph[mask], alpha=0.08, color=MOSS)
        ax4.text(0.05, 0.08, 'Viable operating region\n(ΔP < 1.5%)',
                 transform=ax4.transAxes, color=MOSS,
                 fontsize=8, fontfamily='monospace', alpha=0.9)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"  Geometry plot saved:   {output_path}")


def print_geometry_table(geo_results, n=8):
    """Print a summary table of key geometry configurations."""
    print()
    print("=" * 90)
    print("  HEAT EXCHANGER GEOMETRY TRADE — KEY CONFIGURATIONS")
    print(f"  {'d_o [mm]':>10}  {'U [kW/m^2-K]':>12}  {'A [m^2]':>8}  "
          f"{'dP [%]':>8}  {'L_tube [m]':>12}  {'Mass [kg]':>10}  {'alpha [m^2/m^3]':>10}")
    print("-" * 90)
    step = max(1, len(geo_results) // n)
    for r in geo_results[::step]:
        print(f"  {r['d_outer_mm']:>10.2f}  {r['U_Wm2K']/1000:>12.2f}  "
              f"{r['A_required_m2']:>8.1f}  {r['dp_fraction_pct']:>8.3f}  "
              f"{r['L_tube_m']:>12.3f}  {r['mass_kg']:>10.1f}  "
              f"{r['alpha_m2m3']:>10.0f}")
    print("=" * 90)

    # Find optimal design (max compactness for ΔP < 1%)
    viable = [r for r in geo_results if r['dp_fraction_pct'] < 1.0]
    if viable:
        best = max(viable, key=lambda r: r['alpha_m2m3'])
        print()
        print("  RECOMMENDED DESIGN (max compactness, ΔP < 1%):")
        print(f"    Tube outer diameter : {best['d_outer_mm']:.2f} mm")
        print(f"    Overall U           : {best['U_Wm2K']:.0f} W/(m²·K)")
        print(f"    Required surface A  : {best['A_required_m2']:.1f} m²")
        print(f"    Pressure drop       : {best['dp_fraction_pct']:.3f} %")
        print(f"    Tube length         : {best['L_tube_m']:.3f} m")
        print(f"    Specific surface α  : {best['alpha_m2m3']:.0f} m²/m³")
        print(f"    Estimated tube mass : {best['mass_kg']:.1f} kg")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SABRE Precooler — Trajectory Integrator & Geometry Calculator')
    parser.add_argument('--ssto', action='store_true',
                        help='SSTO horizontal-takeoff trajectory')
    parser.add_argument('--tbcc',  action='store_true',
                        help='Combined cycle mode-transition trajectory')
    parser.add_argument('--geometry', action='store_true',
                        help='Heat exchanger geometry back-calculator only')
    parser.add_argument('--all',      action='store_true',
                        help='Run all analyses (default)')
    args = parser.parse_args()

    run_ssto  = args.ssto or args.all or not any([args.ssto, args.tbcc, args.geometry])
    run_tbcc = args.tbcc  or args.all or not any([args.ssto, args.tbcc, args.geometry])
    run_geo  = args.geometry or args.all or not any([args.ssto, args.tbcc, args.geometry])

    if run_ssto:
        print("\n[1/3]  SSTO trajectory analysis...")
        wp, title = trajectory_ssto()
        res = integrate_trajectory(wp, dt=4.0)
        plot_trajectory_results(
            res, title, wp,
            'ssto_trajectory.png')
        print(f"       Mission duration : {res['t'][-1]/60:.1f} min")
        print(f"       Total LH2 (precooling) : {res['LH2_cumulative_kg'][-1]:.0f} kg")
        print(f"       Peak heat load   : {res['Q_MW'].max():.1f} MW  at Mach {res['M'][np.argmax(res['Q_MW'])]:.2f}")
        frost_pct = np.mean(res['frost_alert']) * 100
        print(f"       Time above frost threshold : {frost_pct:.1f}% of mission")

    if run_tbcc:
        print("\n[2/3]  Combined cycle mode-transition trajectory analysis...")
        wp, title = trajectory_tbcc()
        res = integrate_trajectory(wp, dt=3.0)
        plot_trajectory_results(
            res, title, wp,
            'tbcc_trajectory.png')
        print(f"       Mission duration : {res['t'][-1]/60:.1f} min")
        print(f"       Peak frost risk  : {res['frost_risk'].max():.3f}  at t={res['t'][np.argmax(res['frost_risk'])]/60:.1f} min")
        frost_pct = np.mean(res['frost_alert']) * 100
        print(f"       Time above frost threshold : {frost_pct:.1f}% of mission")
        mode_t = res['t'][res['mode_transition_idx']] / 60.0
        mode_M = res['M'][res['mode_transition_idx']]
        print(f"       Turbojet thermal limit crossed at : Mach {mode_M:.2f}  (t = {mode_t:.1f} min)")

    if run_geo:
        print("\n[3/3]  Geometry back-calculator...")
        geo_results, op_conds = geometry_calculator(
            NTU_target=PRECOOLER_NTU,
            M_design=5.0,
            alt_design_km=25.0)
        plot_geometry(geo_results, op_conds, 'hx_geometry.png')
        print_geometry_table(geo_results)

    print("\nDone. Output files:")
    if run_ssto:  print("  ssto_trajectory.png")
    if run_tbcc: print("  tbcc_trajectory.png")
    if run_geo:  print("  hx_geometry.png")
