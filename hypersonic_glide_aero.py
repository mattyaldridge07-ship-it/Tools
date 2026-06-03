"""
Hypersonic Glide Vehicle — Aerothermal Performance Calculator
=============================================================
Preliminary design tool for aerothermal analysis of hypersonic glide vehicles
during atmospheric re-entry / sustained hypersonic flight.

Directly relevant to:
  - Hypersonica HS1 (Mach 6, 300km range, glide vehicle, Norway test Feb 2026)
  - Castelion Blackbeard (solid-boost, hypersonic glide weapon)
  - Any boost-glide or sustained hypersonic vehicle requiring TPS sizing

Computes:
  1. 2D point-mass re-entry trajectory (lift/drag, gravity, ISA atmosphere)
  2. Stagnation point heat flux — Detra-Kemp-Riddell (DKR) correlation
  3. Convective heating distribution along body — Lees similarity solution
  4. Radiation equilibrium wall temperature
  5. Simple ablation mass loss model (PICA-type ablator)
  6. Required TPS thickness for a given ablator density

Physics models:
  - ISA atmosphere (0–80 km, multi-layer)
  - Isentropic relations for Mach number
  - DKR stagnation heating: q = 1.83e-4/sqrt(Rn) * sqrt(rho/rho_SL) * V^3
  - Radiation equilibrium: q_conv = sigma * eps * T_w^4
  - Point-mass equations of motion (no rotation, no bank angle)
  - Ablation: m_dot = q_w / (h_v + Cp_char * dT)

Known limitations / simplifications:
  - Point-mass trajectory (no attitude dynamics, no bank)
  - Laminar heating only (no turbulent transition model)
  - No real-gas effects (significant above Mach 8–10)
  - DKR correlation valid for laminar stagnation point only
  - Lees distribution approximate for sharp-nosed vehicles
  - No radiative heating from shock layer (important above ~10 km/s)
  - No aeroelastic deformation
  - Ablation model is simple char model, not detailed pyrolysis

Usage:
  python hypersonic_glide_aero.py            # both scenarios
  python hypersonic_glide_aero.py --hs1      # HS1 prototype scenario
  python hypersonic_glide_aero.py --full     # full-scale glide vehicle
  python hypersonic_glide_aero.py --point    # single-point table

Author: MKA — pringlesmaths.co.uk
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
import argparse, warnings
warnings.filterwarnings('ignore')

# ── Physical constants ────────────────────────────────────────────────────────
G0         = 9.80665    # m/s²    standard gravity
R_AIR      = 287.058    # J/kg/K  gas constant air
GAMMA      = 1.4        # —       specific heat ratio air
SIGMA_SB   = 5.6704e-8  # W/m²/K⁴ Stefan-Boltzmann
RHO_SL     = 1.225      # kg/m³   sea-level density (ISA)
RE         = 6.371e6    # m       Earth radius (for gravity variation)

# ── Ablator properties (PICA-type, approximate) ──────────────────────────────
H_ABLATION    = 2.0e7   # J/kg    effective heat of ablation (latent + pyrolysis)
CP_CHAR       = 1400.0  # J/kg/K  char specific heat
T_ABLATION    = 600.0   # K       onset of significant ablation
RHO_ABLATOR   = 270.0   # kg/m³   PICA density (~270 kg/m³)
EMISSIVITY    = 0.85    # —       surface emissivity (carbon-based TPS)

# ── Vehicle scenario definitions ─────────────────────────────────────────────
SCENARIOS = {
    'hs1': {
        'name':       'HS1 Prototype (Hypersonica, Feb 2026 test)',
        'mass_kg':    150.0,
        'S_ref_m2':   0.10,
        'CD':         0.40,
        'CL':         0.10,
        'R_nose_m':   0.05,
        'V0_ms':      1800.0,    # Mach 6 at ~15km
        'gamma0_deg': -5.0,
        'h0_m':       30000.0,
        'x0_m':       0.0,
        'L_body_m':   2.5,       # approximate body length
    },
    'full': {
        'name':       'Full-Scale Hypersonic Glide Vehicle',
        'mass_kg':    800.0,
        'S_ref_m2':   0.35,
        'CD':         0.28,
        'CL':         0.45,
        'R_nose_m':   0.025,
        'V0_ms':      2500.0,    # Mach 8+ at entry
        'gamma0_deg': -3.0,
        'h0_m':       50000.0,
        'x0_m':       0.0,
        'L_body_m':   4.0,
    }
}

# ── Colour palette (consistent with sabre_precooler.py) ──────────────────────
BG    = '#0f0f0e'
GOLD  = '#b8920a'
MOSS  = '#4a7a4b'
PAPER = '#f0ede8'
DIM   = '#8a8a7a'
RED   = '#c04040'
BLUE  = '#4080c0'
CYAN  = '#40b0c0'
GREY  = '#3a3a38'


def isa_atmosphere(h_m):
    """ISA atmosphere 0–80km. Returns (T[K], p[Pa], rho[kg/m³], a[m/s])."""
    layers = [
        (0,      11000, -6.5e-3, 288.15, 101325.0),
        (11000,  20000,  0.0,    216.65,  22632.1),
        (20000,  32000,  1.0e-3, 216.65,   5474.89),
        (32000,  47000,  2.8e-3, 228.65,    868.019),
        (47000,  51000,  0.0,    270.65,    110.906),
        (51000,  71000, -2.8e-3, 270.65,     66.9389),
        (71000,  86000, -2.0e-3, 214.65,      3.95642),
        (86000, 120000,  0.0,    186.87,      0.3734),
    ]
    h = np.clip(h_m, 0, 119999)
    for (h0, h1, lr, T0, p0) in layers:
        if h <= h1:
            dh = h - h0
            T  = T0 + lr * dh
            if abs(lr) < 1e-12:
                p = p0 * np.exp(-G0 * dh / (R_AIR * T0))
            else:
                p = p0 * (T / T0) ** (-G0 / (lr * R_AIR))
            rho = p / (R_AIR * T)
            a   = np.sqrt(GAMMA * R_AIR * max(T, 100.0))
            return T, p, rho, a
    return 186.87, 0.374, 0.374 / (R_AIR * 186.87), np.sqrt(GAMMA * R_AIR * 186.87)


def gravity(h_m):
    """Altitude-corrected gravitational acceleration [m/s²]."""
    return G0 * (RE / (RE + h_m)) ** 2


def mach_number(V_ms, h_m):
    """Mach number given velocity and altitude."""
    _, _, _, a = isa_atmosphere(h_m)
    return V_ms / max(a, 1.0)


def stagnation_heat_flux_DKR(V_ms, h_m, R_nose_m):
    """
    Stagnation point heat flux using Detra-Kemp-Riddell (1957) approximation.
    Valid for laminar, subsonic-to-hypersonic flight.

    q_s [W/m²] = (1.83e-4 / sqrt(R_n)) * sqrt(rho/rho_SL) * V^3
    (V in m/s, R_n in m, rho in kg/m³)

    Reference: Detra, Kemp & Riddell, Jet Propulsion, 1957.
    """
    _, _, rho, _ = isa_atmosphere(h_m)
    if V_ms < 500 or R_nose_m <= 0:
        return 0.0
    # DKR constant 1.83e-4 with V in m/s, Rn in m, rho in kg/m³ gives W/m² directly.
    # Do NOT multiply by 1e4 — that would incorrectly assume the output is W/cm².
    q_Wm2 = 1.83e-4 / np.sqrt(R_nose_m) * np.sqrt(rho / RHO_SL) * (V_ms ** 3)
    return q_Wm2   # W/m²


def radiation_equilibrium_T(q_conv_Wm2):
    """
    Wall temperature from radiation equilibrium (adiabatic surface).
    q_conv = sigma * eps * T_w^4  =>  T_w = (q_conv / (sigma*eps))^0.25
    """
    if q_conv_Wm2 <= 0:
        return 300.0
    return (q_conv_Wm2 / (SIGMA_SB * EMISSIVITY)) ** 0.25


def ablation_rate(q_net_Wm2, T_wall_K):
    """
    Simple char ablation model.
    m_dot [kg/(m²·s)] = q_net / (h_ablation + Cp_char * (T_wall - T_ref))
    Only active if T_wall > T_ablation.
    """
    if T_wall_K < T_ABLATION or q_net_Wm2 <= 0:
        return 0.0
    dT   = max(T_wall_K - T_ABLATION, 0.0)
    h_eff = H_ABLATION + CP_CHAR * dT
    return q_net_Wm2 / max(h_eff, 1e4)


def heating_distribution(s_over_Rn, q_stag_Wm2):
    """
    Heating distribution along body using Lees (1956) similarity solution.
    q(s) / q_stag ≈ sqrt(R_n / R(s)) * velocity_ratio_factor

    Simplified for a sphere-cone: use Lees' result for spherical nose
    transitioning to cone:
    q(theta) / q_stag = (sin(theta))^(1/2) * cos(theta) for sphere
    Parameterised as function of s/R_n (arc length / nose radius).

    Reference: Lees, JAS, 1956.
    """
    # Approximate Lees distribution (spherical-nose approximation)
    # q/q_s peaks at stagnation, falls as ~1/sqrt(s/Rn) for s >> Rn
    if s_over_Rn <= 0:
        return q_stag_Wm2
    # Sphere-to-cone transition model
    if s_over_Rn < 1.0:
        # On nose cap: slow decrease
        ratio = np.sqrt(np.sin(np.clip(s_over_Rn * np.pi / 2, 0, np.pi / 2)))
    else:
        # On conical body: faster decrease
        ratio = 0.5 / np.sqrt(max(s_over_Rn, 0.5))
    return q_stag_Wm2 * np.clip(ratio, 0.0, 1.0)


def equations_of_motion(t, state, sc):
    """
    2D point-mass equations of motion (no rotation, no bank, no wind).

    State: [V [m/s], gamma [rad], h [m], x [m]]
    sc:    vehicle scenario dict (mass, CD, CL, S_ref)

    dV/dt     = -D/m - g*sin(gamma)
    d(gamma)/dt = (L/m - g*cos(gamma)) / V
    dh/dt     = V*sin(gamma)
    dx/dt     = V*cos(gamma)
    """
    V, gam, h, x = state
    h     = max(h, 0.0)
    V     = max(V, 10.0)
    T, p, rho, a = isa_atmosphere(h)
    M     = V / max(a, 1.0)
    g     = gravity(h)
    q_dyn = 0.5 * rho * V**2    # dynamic pressure [Pa]
    D     = q_dyn * sc['CD'] * sc['S_ref_m2']
    L     = q_dyn * sc['CL'] * sc['S_ref_m2']
    m     = sc['mass_kg']
    dV    = -D / m - g * np.sin(gam)
    dgam  = (L / m - g * np.cos(gam)) / V
    dh    = V * np.sin(gam)
    dx    = V * np.cos(gam)
    return [dV, dgam, dh, dx]


def integrate_trajectory(sc, dt_max=0.5):
    """
    Integrate trajectory and compute aerothermal quantities at each step.
    Returns dict of time-series arrays.
    """
    V0    = sc['V0_ms']
    gam0  = np.radians(sc['gamma0_deg'])
    h0    = sc['h0_m']
    x0    = sc['x0_m']

    t_end = 600.0   # max integration time [s]

    def event_ground(t, y, sc): return y[2]         # h = 0
    def event_slow(t, y, sc):   return y[0] - 200.0  # V < 200 m/s
    event_ground.terminal = True;  event_ground.direction = -1
    event_slow.terminal   = True;  event_slow.direction   = -1

    sol = solve_ivp(
        equations_of_motion,
        [0, t_end],
        [V0, gam0, h0, x0],
        method='RK45',
        args=(sc,),
        events=[event_ground, event_slow],
        max_step=dt_max,
        rtol=1e-5, atol=1e-4,
        dense_output=False
    )

    t   = sol.t
    V   = sol.y[0]
    gam = sol.y[1]
    h   = np.maximum(sol.y[2], 0.0)
    x   = sol.y[3]

    # ── Derived quantities at each timestep ──────────────────────────────
    n = len(t)
    T_atm  = np.zeros(n); rho    = np.zeros(n); M_arr  = np.zeros(n)
    q_stag = np.zeros(n); T_wall = np.zeros(n); m_dot_abl = np.zeros(n)
    q_dyn  = np.zeros(n)

    for i in range(n):
        Ti, pi, rhoi, ai = isa_atmosphere(h[i])
        T_atm[i] = Ti
        rho[i]   = rhoi
        M_arr[i] = V[i] / max(ai, 1.0)
        q_s      = stagnation_heat_flux_DKR(V[i], h[i], sc['R_nose_m'])
        q_stag[i] = q_s
        T_w      = radiation_equilibrium_T(q_s)
        T_wall[i] = T_w
        m_dot_abl[i] = ablation_rate(q_s, T_w)
        q_dyn[i] = 0.5 * rhoi * V[i]**2

    # Cumulative heat load [MJ/m²]
    Q_total = np.cumsum(q_stag * np.gradient(t)) / 1e6

    # Cumulative ablation thickness [mm]
    m_abl_cum  = np.cumsum(m_dot_abl * np.gradient(t))  # kg/m²
    tps_thick  = m_abl_cum / RHO_ABLATOR * 1000          # mm

    return {
        't': t, 'V_ms': V, 'gamma_deg': np.degrees(gam),
        'h_km': h/1e3, 'x_km': x/1e3,
        'M': M_arr, 'T_atm_K': T_atm, 'rho': rho,
        'q_stag_MWm2': q_stag/1e6,
        'T_wall_K': T_wall,
        'm_dot_abl': m_dot_abl,
        'Q_total_MJm2': Q_total,
        'tps_thick_mm': tps_thick,
        'q_dyn_kPa': q_dyn/1e3,
    }


def style_ax(ax, title):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(GREY); sp.set_linewidth(0.6)
    ax.tick_params(colors=DIM, labelsize=9)
    ax.xaxis.label.set_color(DIM)
    ax.yaxis.label.set_color(DIM)
    ax.set_title(title, color=PAPER, fontsize=10, fontweight='bold',
                 pad=8, fontfamily='monospace')
    ax.grid(True, color='#2a2a28', linewidth=0.5, linestyle='--', alpha=0.6)


def plot_scenario(res, sc, output_path):
    """Six-panel aerothermal mission plot."""
    t = res['t']
    fig = plt.figure(figsize=(18, 20), facecolor=BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            hspace=0.52, wspace=0.38,
                            left=0.08, right=0.96,
                            top=0.93, bottom=0.05)

    fig.text(0.5, 0.975,
             sc['name'].upper() + '  —  AEROTHERMAL ANALYSIS',
             ha='center', va='top', color=PAPER,
             fontsize=12, fontweight='bold', fontfamily='monospace')
    fig.text(0.5, 0.961,
             'Point-mass trajectory  ·  DKR stagnation heating  ·  '
             'Radiation equilibrium T_wall  ·  Simple char ablation model',
             ha='center', va='top', color=DIM, fontsize=8.5,
             fontfamily='monospace')
    fig.text(0.99, 0.005, 'pringlesmaths.co.uk  |  MKA  |  Preliminary design only',
             ha='right', va='bottom', color='#3a3a38', fontsize=8,
             fontfamily='monospace')

    # ── Panel 1: Trajectory ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'TRAJECTORY')
    ax1.plot(res['x_km'], res['h_km'], color=GOLD, linewidth=2.0)
    ax1.set_xlabel('Range  [km]', fontsize=9)
    ax1.set_ylabel('Altitude  [km]', fontsize=9)
    # Mark max heating point
    idx_maxq = np.argmax(res['q_stag_MWm2'])
    ax1.plot(res['x_km'][idx_maxq], res['h_km'][idx_maxq],
             'o', color=RED, ms=8, zorder=5)
    ax1.annotate(f"Peak heating\n{res['h_km'][idx_maxq]:.0f}km, M{res['M'][idx_maxq]:.1f}",
                 xy=(res['x_km'][idx_maxq], res['h_km'][idx_maxq]),
                 xytext=(10, 10), textcoords='offset points',
                 color=RED, fontsize=8, fontfamily='monospace')

    # ── Panel 2: Velocity & Mach ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'VELOCITY  &  MACH NUMBER')
    ax2.plot(t, res['V_ms']/1000, color=GOLD, linewidth=2.0, label='V [km/s]')
    ax2r = ax2.twinx()
    ax2r.set_facecolor(BG)
    ax2r.plot(t, res['M'], color=CYAN, linewidth=1.6, linestyle='--', label='Mach')
    ax2r.tick_params(colors=CYAN, labelsize=9)
    ax2r.spines['right'].set_color(GREY)
    ax2r.set_ylabel('Mach number', fontsize=9, color=CYAN)
    ax2.set_xlabel('Time  [s]', fontsize=9)
    ax2.set_ylabel('Velocity  [km/s]', fontsize=9)
    ax2.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper right')
    ax2r.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='lower left')

    # ── Panel 3: Stagnation heat flux ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'STAGNATION HEAT FLUX  [MW/m²]')
    ax3.fill_between(t, res['q_stag_MWm2'], alpha=0.25, color=RED)
    ax3.plot(t, res['q_stag_MWm2'], color=RED, linewidth=2.0)
    peak_q = res['q_stag_MWm2'].max()
    ax3.annotate(f'Peak: {peak_q:.2f} MW/m²',
                 xy=(t[idx_maxq], peak_q),
                 xytext=(0, 12), textcoords='offset points',
                 color=RED, fontsize=9, fontfamily='monospace', ha='center')
    ax3.set_xlabel('Time  [s]', fontsize=9)
    ax3.set_ylabel('q_stag  [MW/m²]', fontsize=9)

    # ── Panel 4: Wall temperature ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, 'RADIATION EQUILIBRIUM WALL TEMPERATURE  [K]')
    ax4.plot(t, res['T_wall_K'], color=GOLD, linewidth=2.0, label='T_wall (rad. eq.)')
    ax4.axhline(1100, color=RED, linewidth=0.8, linestyle='--', alpha=0.6)
    ax4.text(t[0]+1, 1115, 'Inconel limit (1100 K)', color=RED,
             fontsize=8, fontfamily='monospace', alpha=0.8)
    ax4.axhline(2000, color=RED, linewidth=0.5, linestyle=':', alpha=0.4)
    ax4.text(t[0]+1, 2015, 'C-C composite limit (~2000 K)', color=RED,
             fontsize=8, fontfamily='monospace', alpha=0.6)
    ax4.axhline(T_ABLATION, color=MOSS, linewidth=0.8, linestyle=':', alpha=0.6)
    ax4.text(t[0]+1, T_ABLATION+30, 'Ablation onset', color=MOSS,
             fontsize=8, fontfamily='monospace', alpha=0.7)
    ax4.set_xlabel('Time  [s]', fontsize=9)
    ax4.set_ylabel('T_wall  [K]', fontsize=9)
    ax4.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 5: Cumulative heat load ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    style_ax(ax5, 'CUMULATIVE HEAT LOAD  [MJ/m²]')
    ax5.fill_between(t, res['Q_total_MJm2'], alpha=0.2, color=BLUE)
    ax5.plot(t, res['Q_total_MJm2'], color=BLUE, linewidth=2.0)
    total_Q = res['Q_total_MJm2'][-1]
    ax5.annotate(f'Total: {total_Q:.1f} MJ/m²',
                 xy=(t[-1], total_Q),
                 xytext=(-60, -20), textcoords='offset points',
                 color=BLUE, fontsize=9, fontfamily='monospace',
                 arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.8))
    ax5.set_xlabel('Time  [s]', fontsize=9)
    ax5.set_ylabel('Q  [MJ/m²]', fontsize=9)

    # ── Panel 6: TPS thickness required ──────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    style_ax(ax6, 'REQUIRED TPS THICKNESS  (PICA ablator)')
    ax6.fill_between(t, res['tps_thick_mm'], alpha=0.2, color=MOSS)
    ax6.plot(t, res['tps_thick_mm'], color=MOSS, linewidth=2.0)
    final_tps = res['tps_thick_mm'][-1]
    ax6.annotate(f'Required: {final_tps:.1f} mm',
                 xy=(t[-1], final_tps),
                 xytext=(-80, 10), textcoords='offset points',
                 color=MOSS, fontsize=9, fontfamily='monospace',
                 arrowprops=dict(arrowstyle='->', color=MOSS, lw=0.8))
    # Show ablation rate on right axis
    ax6r = ax6.twinx()
    ax6r.set_facecolor(BG)
    ax6r.plot(t, res['m_dot_abl'] * 1000, color=GOLD, linewidth=1.2,
              linestyle='-.', alpha=0.7, label='Abl. rate [g/m²/s]')
    ax6r.tick_params(colors=GOLD, labelsize=9)
    ax6r.spines['right'].set_color(GREY)
    ax6r.set_ylabel('Ablation rate  [g/m²/s]', fontsize=9, color=GOLD)
    ax6.set_xlabel('Time  [s]', fontsize=9)
    ax6.set_ylabel('TPS thickness  [mm]', fontsize=9)
    ax6r.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"  Plot saved: {output_path}")


def print_summary(res, sc):
    """Print a formatted engineering summary table."""
    peak_q   = res['q_stag_MWm2'].max()
    peak_T   = res['T_wall_K'].max()
    total_Q  = res['Q_total_MJm2'][-1]
    tps_req  = res['tps_thick_mm'][-1]
    range_km = res['x_km'][-1]
    t_flight = res['t'][-1]
    peak_M   = res['M'].max()
    peak_q_dyn = res['q_dyn_kPa'].max()

    print()
    print("=" * 66)
    print(f"  {sc['name']}")
    print("=" * 66)
    print(f"  Entry conditions:")
    print(f"    Velocity          :  {sc['V0_ms']:.0f} m/s "
          f"  (Mach {sc['V0_ms']/np.sqrt(GAMMA*R_AIR*216.65):.1f} at entry alt)")
    print(f"    Altitude          :  {sc['h0_m']/1000:.1f} km")
    print(f"    Flight path angle :  {sc['gamma0_deg']:.1f}°")
    print(f"    Nose radius       :  {sc['R_nose_m']*1000:.0f} mm")
    print("-" * 66)
    print(f"  Trajectory results:")
    print(f"    Flight time       :  {t_flight:.1f} s")
    print(f"    Range achieved    :  {range_km:.1f} km")
    print(f"    Peak Mach         :  {peak_M:.2f}")
    print(f"    Peak dynamic pres :  {peak_q_dyn:.1f} kPa")
    print("-" * 66)
    print(f"  Aerothermal results:")
    print(f"    Peak heat flux    :  {peak_q:.3f} MW/m²  ({peak_q*100:.0f} W/cm²)")
    print(f"    Peak wall temp    :  {peak_T:.0f} K  ({peak_T-273.15:.0f} °C)")
    print(f"    Total heat load   :  {total_Q:.1f} MJ/m²")
    print(f"    Required TPS      :  {tps_req:.1f} mm  (PICA, rho={RHO_ABLATOR} kg/m³)")
    # TPS mass estimate
    tps_mass = tps_req/1000 * RHO_ABLATOR * sc['S_ref_m2']
    print(f"    TPS mass estimate :  {tps_mass:.1f} kg  "
          f"({tps_mass/sc['mass_kg']*100:.1f}% of vehicle mass)")
    print("=" * 66)
    print()


def plot_heating_distribution(sc, output_path):
    """
    Plot stagnation heating distribution along body at peak heating condition.
    Shows how heat flux falls off from nose to tail.
    """
    V_peak   = sc['V0_ms'] * 0.9   # approximate peak heating velocity
    h_peak   = max(sc['h0_m'] - 15000, 10000)
    q_stag   = stagnation_heat_flux_DKR(V_peak, h_peak, sc['R_nose_m'])
    L_body   = sc['L_body_m']
    R_n      = sc['R_nose_m']

    s_vals = np.linspace(0, L_body, 200)
    q_dist = np.array([heating_distribution(s/R_n, q_stag) for s in s_vals])
    T_dist = np.array([radiation_equilibrium_T(q) for q in q_dist])

    return s_vals, q_dist/1e6, T_dist   # MW/m², K


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hypersonic Glide Vehicle Aerothermal Calculator')
    parser.add_argument('--hs1',   action='store_true',
                        help='HS1 prototype scenario only')
    parser.add_argument('--full',  action='store_true',
                        help='Full-scale glide vehicle scenario only')
    parser.add_argument('--point', action='store_true',
                        help='Print single-point stagnation heating tables')
    args = parser.parse_args()

    run_hs1  = args.hs1  or not any([args.hs1, args.full, args.point])
    run_full = args.full or not any([args.hs1, args.full, args.point])

    if args.point:
        print("\nStagnation heating at key conditions (DKR correlation):")
        print(f"{'Mach':>6}  {'Alt[km]':>8}  {'V[m/s]':>8}  "
              f"{'Rn=25mm':>12}  {'Rn=50mm':>12}  {'Rn=100mm':>12}")
        print("-" * 70)
        for M_val, alt_km in [(3,15),(4,20),(5,25),(6,28),(7,30),(8,35)]:
            _, _, _, a = isa_atmosphere(alt_km*1e3)
            V = M_val * a
            q25  = stagnation_heat_flux_DKR(V, alt_km*1e3, 0.025)/1e6
            q50  = stagnation_heat_flux_DKR(V, alt_km*1e3, 0.050)/1e6
            q100 = stagnation_heat_flux_DKR(V, alt_km*1e3, 0.100)/1e6
            print(f"  M{M_val}  {alt_km:>8.0f}  {V:>8.0f}  "
                  f"{q25:>11.2f}M  {q50:>11.2f}M  {q100:>11.2f}M  [MW/m²]")
        print()
    else:
        scenarios_to_run = []
        if run_hs1:  scenarios_to_run.append(('hs1',  SCENARIOS['hs1']))
        if run_full: scenarios_to_run.append(('full', SCENARIOS['full']))

        for key, sc in scenarios_to_run:
            tag = 'hs1' if key == 'hs1' else 'fullscale'
            print(f"\nComputing {sc['name']}...")
            res = integrate_trajectory(sc, dt_max=0.5)
            print_summary(res, sc)
            out_path = f'{tag}_trajectory.png'
            plot_scenario(res, sc, out_path)
