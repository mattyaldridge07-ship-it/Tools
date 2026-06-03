"""
Spacecraft Orbital Thermal Analysis
=====================================
First-order thermal design tool for small satellites in LEO.

Computes temperature variation over multiple orbital periods as a function of:
  - Orbital altitude and inclination
  - Surface coatings (absorptivity and emissivity)
  - Internal electronics dissipation
  - Heater power requirements

Directly relevant to:
  - SSTL (Surrey Satellite Technology Ltd, Guildford) — UK small satellite leader
  - Open Cosmos (Oxford) — small satellite startup
  - In-Space Missions (Alton, Hampshire)
  - Airbus Defence & Space (Stevenage) — full-size spacecraft

Physics:
  - Circular LEO orbital mechanics (eclipse fractions)
  - Solar, albedo, and Earth IR heat loads
  - Radiation to deep space (only heat rejection path)
  - Lumped capacitance transient model
  - Two-node extension: satellite body + solar panel

Known limitations:
  - Lumped capacitance (no spatial gradients or component-level analysis)
  - No multi-layer insulation (MLI) conduction model
  - Simplified eclipse model (circular orbit, fixed inclination)
  - No thermoelectric effects (heat pipes not modelled)
  - View factors not computed — uses simple area projections
  - No internal conduction between components

References:
  - Gilmore (2002), Spacecraft Thermal Control Handbook Vol. 1
  - ECSS-E-HB-31-01A, Spacecraft Thermal Design Handbook

Usage:
  python spacecraft_thermal.py              # full analysis, all coatings
  python spacecraft_thermal.py --cubesat    # 3U CubeSat scenario
  python spacecraft_thermal.py --sstl100    # SSTL-100 class scenario
  python spacecraft_thermal.py --point      # print steady-state table

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
SIGMA_SB    = 5.6704e-8     # W/m²/K⁴
GM_EARTH    = 3.986004e14   # m³/s²
R_EARTH     = 6.371e6       # m
S0          = 1361.0        # W/m²   solar constant
A_ALBEDO    = 0.30          # —      Earth albedo
T_EARTH_IR  = 255.0         # K      Earth effective IR temperature
T_SPACE     = 3.0           # K      deep space

# ── Surface coating database ─────────────────────────────────────────────────
# (alpha = solar absorptivity, eps = IR emissivity)
COATINGS = {
    'white_paint':        {'alpha': 0.20, 'eps': 0.85, 'label': 'White paint'},
    'black_paint':        {'alpha': 0.97, 'eps': 0.97, 'label': 'Black paint'},
    'aluminised_kapton':  {'alpha': 0.15, 'eps': 0.03, 'label': 'Alum. Kapton (MLI)'},
    'bare_aluminium':     {'alpha': 0.15, 'eps': 0.04, 'label': 'Bare aluminium'},
    'gold_plating':       {'alpha': 0.23, 'eps': 0.03, 'label': 'Gold plating'},
    'OSR':                {'alpha': 0.07, 'eps': 0.80, 'label': 'OSR (Qsil-13)'},
}

# ── Standard orbital scenarios ────────────────────────────────────────────────
SATELLITES = {
    'cubesat': {
        'name':     '3U CubeSat',
        'mass_kg':  2.0,
        'Cp':       800.0,      # J/kgK  effective (mix of Al structure + PCBs)
        'A_total':  0.06,       # m²     total surface area (0.1×0.1×0.34 m)
        'A_solar':  0.01,       # m²     projected to Sun
        'A_nadir':  0.01,       # m²     projected to Earth
        'A_rad':    0.05,       # m²     radiating area (all minus solar panel)
        'P_elec':   5.0,        # W      internal dissipation
        'alt_km':   550.0,
        'coating':  'white_paint',
        'T_min':    -20.0,      # °C     minimum electronics temperature
        'T_max':    60.0,       # °C     maximum electronics temperature
    },
    'sstl100': {
        'name':     'SSTL-100 Class',
        'mass_kg':  100.0,
        'Cp':       600.0,
        'A_total':  1.8,        # m²
        'A_solar':  0.3,        # m²
        'A_nadir':  0.25,       # m²
        'A_rad':    1.5,        # m²
        'P_elec':   80.0,       # W
        'alt_km':   650.0,
        'coating':  'OSR',
        'T_min':    -10.0,
        'T_max':    50.0,
    }
}

# ── Colour palette ────────────────────────────────────────────────────────────
BG='#0f0f0e'; GOLD='#b8920a'; MOSS='#4a7a4b'; PAPER='#f0ede8'
DIM='#8a8a7a'; RED='#c04040'; BLUE='#4080c0'; CYAN='#40b0c0'; GREY='#3a3a38'
COLORS = [GOLD, CYAN, MOSS, RED, BLUE, '#c080ff']


def orbital_params(alt_km):
    """Compute orbital period and eclipse fraction for circular orbit."""
    h = alt_km * 1e3
    r = R_EARTH + h
    T_orb = 2 * np.pi * np.sqrt(r**3 / GM_EARTH)   # s
    # Eclipse fraction (geometric shadow of Earth)
    # sin(rho) = R_earth / r, rho = Earth angular radius from satellite
    rho = np.arcsin(R_EARTH / r)
    # Eclipse occurs when satellite is in Earth's shadow
    # Assuming worst-case (beta=0, orbit perpendicular to Sun vector):
    # eclipse fraction ≈ rho/pi (fraction of orbit in shadow)
    # More precisely: f_eclipse = arccos(cos(rho)/cos(beta_orbit)) / pi
    # For worst-case beta=0:
    f_eclipse = np.arccos(0.0) / np.pi - rho / np.pi
    # Simplified: f_eclipse ≈ rho / pi (fraction spent in eclipse)
    f_eclipse = rho / np.pi
    f_sun = 1.0 - f_eclipse
    return T_orb, f_eclipse, f_sun


def heat_loads(t, T_orb, f_eclipse, sc, coating_name):
    """
    Compute instantaneous heat loads [W] at time t in the orbit.
    Eclipse model: first f_eclipse * T_orb is eclipse, remainder is sunlit.
    """
    c = COATINGS[coating_name]
    h = sc['alt_km'] * 1e3
    r = R_EARTH + h

    # Earth view factor (point source approximation)
    F_earth = (R_EARTH / r) ** 2

    # Determine if sunlit or eclipse at this time in orbit
    t_in_orbit = t % T_orb
    in_eclipse = t_in_orbit < (f_eclipse * T_orb)

    # Solar and albedo (only in sunlit phase)
    if not in_eclipse:
        Q_solar  = c['alpha'] * S0 * sc['A_solar']
        Q_albedo = c['alpha'] * S0 * A_ALBEDO * F_earth * sc['A_nadir']
    else:
        Q_solar  = 0.0
        Q_albedo = 0.0

    # Earth IR (always present)
    Q_IR = c['eps'] * SIGMA_SB * T_EARTH_IR**4 * F_earth * sc['A_nadir']

    # Internal dissipation (always present)
    Q_int = sc['P_elec']

    return Q_solar, Q_albedo, Q_IR, Q_int


def heat_rejection(T_K, sc, coating_name):
    """Radiative heat rejection to deep space [W]."""
    c = COATINGS[coating_name]
    return c['eps'] * SIGMA_SB * sc['A_rad'] * (T_K**4 - T_SPACE**4)


def ode_thermal(t, y, T_orb, f_eclipse, sc, coating_name):
    """Thermal ODE: dT/dt = (Q_in - Q_out) / (M*Cp)"""
    T_K = y[0]
    T_K = max(200.0, T_K)   # clip to avoid negative temperatures

    Q_solar, Q_albedo, Q_IR, Q_int = heat_loads(t, T_orb, f_eclipse, sc, coating_name)
    Q_in  = Q_solar + Q_albedo + Q_IR + Q_int
    Q_out = heat_rejection(T_K, sc, coating_name)

    C_eff = sc['mass_kg'] * sc['Cp']
    return [(Q_in - Q_out) / C_eff]


def simulate_orbits(sc, coating_name, n_orbits=8, T0_K=293.15):
    """Simulate n_orbits and return time-series."""
    T_orb, f_eclipse, f_sun = orbital_params(sc['alt_km'])
    t_end = n_orbits * T_orb

    sol = solve_ivp(
        ode_thermal, [0, t_end], [T0_K],
        args=(T_orb, f_eclipse, sc, coating_name),
        method='RK45',
        t_eval=np.linspace(0, t_end, n_orbits * 200),
        rtol=1e-5, atol=1e-4
    )

    T_C = sol.y[0] - 273.15
    return sol.t / 60.0, T_C, T_orb / 60.0   # minutes


def heater_power(sc, coating_name):
    """Minimum heater power to prevent T_min violation during eclipse."""
    T_orb, f_eclipse, f_sun = orbital_params(sc['alt_km'])
    T_min_K = sc['T_min'] + 273.15
    c = COATINGS[coating_name]
    h = sc['alt_km'] * 1e3; r = R_EARTH + h
    F_earth = (R_EARTH / r)**2
    Q_IR_eclipse = c['eps'] * SIGMA_SB * T_EARTH_IR**4 * F_earth * sc['A_nadir']
    Q_out_min    = heat_rejection(T_min_K, sc, coating_name)
    Q_heater     = max(0.0, Q_out_min - Q_IR_eclipse - sc['P_elec'])
    return Q_heater


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


def plot_analysis(sc_name, output_path):
    sc = SATELLITES[sc_name]
    T_orb, f_ecl, f_sun = orbital_params(sc['alt_km'])

    fig = plt.figure(figsize=(18, 20), facecolor=BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            hspace=0.52, wspace=0.38,
                            left=0.08, right=0.96,
                            top=0.93, bottom=0.05)

    fig.text(0.5, 0.975,
             f'SPACECRAFT THERMAL ANALYSIS  |  {sc["name"].upper()}'
             f'  |  {sc["alt_km"]:.0f} km LEO',
             ha='center', va='top', color=PAPER,
             fontsize=12, fontweight='bold', fontfamily='monospace')
    fig.text(0.5, 0.961,
             f'Orbital period: {T_orb/60:.1f} min  ·  '
             f'Eclipse fraction: {f_ecl*100:.1f}%  ·  '
             f'Sunlit: {f_sun*100:.1f}%  ·  '
             f'Internal dissipation: {sc["P_elec"]:.0f} W',
             ha='center', va='top', color=DIM,
             fontsize=8.5, fontfamily='monospace')
    fig.text(0.99, 0.005, 'pringlesmaths.co.uk  |  MKA',
             ha='right', va='bottom', color='#3a3a38',
             fontsize=8, fontfamily='monospace')

    # ── Panel 1: T vs time for 3 coatings ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1, f'TEMPERATURE vs TIME  —  3 COATINGS COMPARED')

    coatings_to_plot = ['white_paint', 'OSR', 'black_paint']
    T_orb_min = T_orb / 60.0

    for i, cname in enumerate(coatings_to_plot):
        t_min, T_C, _ = simulate_orbits(sc, cname, n_orbits=8)
        label = COATINGS[cname]['label']
        ax1.plot(t_min, T_C, color=COLORS[i], linewidth=2.0, label=label)

    # Shade eclipse intervals
    t_orb_m = T_orb / 60.0
    for orb in range(8):
        t_start = orb * t_orb_m
        t_ecl   = t_start + f_ecl * t_orb_m
        ax1.axvspan(t_start, t_ecl, alpha=0.06, color='white')

    ax1.axhline(sc['T_max'], color=RED, linewidth=0.8,
                linestyle='--', alpha=0.7)
    ax1.axhline(sc['T_min'], color=CYAN, linewidth=0.8,
                linestyle='--', alpha=0.7)
    ax1.text(1.0, sc['T_max'] + 1.5, f'T_max = {sc["T_max"]}°C',
             color=RED, fontsize=8, fontfamily='monospace')
    ax1.text(1.0, sc['T_min'] - 4, f'T_min = {sc["T_min"]}°C',
             color=CYAN, fontsize=8, fontfamily='monospace')

    ax1.set_xlabel('Time  [min]', fontsize=9)
    ax1.set_ylabel('Temperature  [°C]', fontsize=9)
    ax1.legend(fontsize=9, framealpha=0, labelcolor=DIM, loc='upper right')
    ax1.text(0.02, 0.05, '(shaded = eclipse)',
             transform=ax1.transAxes, color=DIM, fontsize=8,
             fontfamily='monospace', alpha=0.6)

    # ── Panel 2: T_max and T_min vs altitude ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2, 'T_MAX & T_MIN vs ALTITUDE  (3 coatings)')

    alts = np.linspace(300, 1000, 20)
    for i, cname in enumerate(['white_paint', 'OSR', 'black_paint']):
        Tmx_list, Tmn_list = [], []
        for alt in alts:
            sc_alt = dict(sc); sc_alt['alt_km'] = alt
            t_m, T_C, _ = simulate_orbits(sc_alt, cname, n_orbits=6)
            T_ss = T_C[len(T_C)//2:]   # last half = quasi-steady
            Tmx_list.append(T_ss.max())
            Tmn_list.append(T_ss.min())
        c = COLORS[i]
        ax2.plot(alts, Tmx_list, color=c, linewidth=2.0,
                 label=f'{COATINGS[cname]["label"]}')
        ax2.plot(alts, Tmn_list, color=c, linewidth=1.2,
                 linestyle='--', alpha=0.7)

    ax2.axhline(sc['T_max'], color=RED, linewidth=0.8,
                linestyle=':', alpha=0.6)
    ax2.axhline(sc['T_min'], color=CYAN, linewidth=0.8,
                linestyle=':', alpha=0.5)
    ax2.set_xlabel('Altitude  [km]', fontsize=9)
    ax2.set_ylabel('Temperature  [°C]', fontsize=9)
    ax2.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 3: Required radiator area vs P_elec ─────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3, 'REQUIRED RADIATOR AREA vs INTERNAL POWER')

    P_vals = np.linspace(5, 300, 40)
    T_max_K = sc['T_max'] + 273.15

    for i, cname in enumerate(['white_paint', 'OSR', 'bare_aluminium']):
        c_coeff = COATINGS[cname]
        # Steady state: Q_in = Q_out
        # In sunlit: Q_solar + Q_albedo + Q_IR + P_elec = eps*sigma*A_rad*T_max^4
        h = sc['alt_km'] * 1e3; r = R_EARTH + h
        F_e = (R_EARTH/r)**2
        Q_solar_ss = c_coeff['alpha'] * S0 * sc['A_solar']
        Q_alb_ss   = c_coeff['alpha'] * S0 * A_ALBEDO * F_e * sc['A_nadir']
        Q_IR_ss    = c_coeff['eps'] * SIGMA_SB * T_EARTH_IR**4 * F_e * sc['A_nadir']

        A_req = []
        for P in P_vals:
            Q_total = Q_solar_ss + Q_alb_ss + Q_IR_ss + P
            A = Q_total / (c_coeff['eps'] * SIGMA_SB * T_max_K**4)
            A_req.append(max(0.0, A))

        ax3.plot(P_vals, A_req, color=COLORS[i], linewidth=2.0,
                 label=COATINGS[cname]['label'])

    ax3.axhline(sc['A_total'], color=GOLD, linewidth=0.8,
                linestyle='--', alpha=0.7)
    ax3.text(P_vals[-1]*0.6, sc['A_total']+0.03,
             f'Total area = {sc["A_total"]} m²',
             color=GOLD, fontsize=8, fontfamily='monospace')
    ax3.set_xlabel('Internal power dissipation  [W]', fontsize=9)
    ax3.set_ylabel('Required radiator area  [m²]', fontsize=9)
    ax3.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 4: Heater power vs coating ─────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    style_ax(ax4, 'HEATER POWER vs ALTITUDE  (worst-case eclipse)')

    alts4 = np.linspace(300, 900, 20)
    for i, cname in enumerate(['white_paint', 'OSR', 'aluminised_kapton']):
        P_heat = []
        for alt in alts4:
            sc_alt = dict(sc); sc_alt['alt_km'] = alt
            P_heat.append(heater_power(sc_alt, cname))
        ax4.plot(alts4, P_heat, color=COLORS[i], linewidth=2.0,
                 label=COATINGS[cname]['label'])

    ax4.axhline(0, color=MOSS, linewidth=0.8, linestyle='--', alpha=0.5)
    ax4.set_xlabel('Altitude  [km]', fontsize=9)
    ax4.set_ylabel('Minimum heater power  [W]', fontsize=9)
    ax4.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 5: Heat load breakdown ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    style_ax(ax5, f'HEAT LOAD BREAKDOWN  ({COATINGS[sc["coating"]]["label"]})')

    T_orb_s, f_ecl_s, f_sun_s = orbital_params(sc['alt_km'])
    h = sc['alt_km']*1e3; r = R_EARTH + h; F_e = (R_EARTH/r)**2
    c_ref = COATINGS[sc['coating']]

    Q_s_sun  = c_ref['alpha'] * S0 * sc['A_solar']
    Q_alb    = c_ref['alpha'] * S0 * A_ALBEDO * F_e * sc['A_nadir']
    Q_ir     = c_ref['eps'] * SIGMA_SB * T_EARTH_IR**4 * F_e * sc['A_nadir']
    Q_int_v  = sc['P_elec']

    # Orbit-average
    Q_s_avg = Q_s_sun * f_sun_s
    Q_alb_avg = Q_alb * f_sun_s

    values = [Q_s_avg, Q_alb_avg, Q_ir, Q_int_v]
    labels = [f'Solar\n{Q_s_avg:.1f}W', f'Albedo\n{Q_alb_avg:.1f}W',
              f'Earth IR\n{Q_ir:.1f}W', f'Internal\n{Q_int_v:.1f}W']
    colors_pie = [GOLD, CYAN, MOSS, RED]
    wedge_props = {'width': 0.6, 'edgecolor': BG, 'linewidth': 2}
    ax5.pie(values, labels=labels, colors=colors_pie, wedgeprops=wedge_props,
            textprops={'color': PAPER, 'fontsize': 9, 'fontfamily': 'monospace'},
            startangle=90, autopct='%1.0f%%',
            pctdistance=0.75, labeldistance=1.15)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"  Plot saved: {output_path}")


def print_table(sc_name):
    sc = SATELLITES[sc_name]
    print(f"\n{'='*80}")
    print(f"  {sc['name']}  |  {sc['alt_km']:.0f} km LEO  |  {sc['P_elec']:.0f} W internal")
    print(f"{'='*80}")
    print(f"  {'Coating':22}  {'T_max':>8}  {'T_min':>8}  "
          f"{'ΔT':>8}  {'P_heat':>10}  {'OK?':>6}")
    print(f"  {'-'*74}")
    T_orb, f_ecl, _ = orbital_params(sc['alt_km'])
    for cname, cdata in COATINGS.items():
        t_m, T_C, _ = simulate_orbits(sc, cname, n_orbits=8)
        T_ss = T_C[len(T_C)//2:]
        Tmx = T_ss.max(); Tmn = T_ss.min()
        dT  = Tmx - Tmn
        P_h = heater_power(sc, cname)
        ok  = '✓' if Tmn >= sc['T_min'] and Tmx <= sc['T_max'] else '✗'
        print(f"  {cdata['label']:22}  {Tmx:>7.1f}°  {Tmn:>7.1f}°  "
              f"{dT:>7.1f}°  {P_h:>9.1f}W  {ok:>6}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Spacecraft Orbital Thermal Analysis')
    parser.add_argument('--cubesat', action='store_true',
                        help='3U CubeSat at 550 km')
    parser.add_argument('--sstl100', action='store_true',
                        help='SSTL-100 class at 650 km')
    parser.add_argument('--point',   action='store_true',
                        help='Print summary tables only')
    args = parser.parse_args()

    run_cubesat = args.cubesat or not any([args.cubesat, args.sstl100, args.point])
    run_sstl    = args.sstl100 or not any([args.cubesat, args.sstl100, args.point])

    if args.point:
        print_table('cubesat')
        print_table('sstl100')
    else:
        if run_cubesat:
            print("\nSimulating 3U CubeSat...")
            print_table('cubesat')
            plot_analysis('cubesat', '/home/claude/spacecraft_cubesat.png')
        if run_sstl:
            print("\nSimulating SSTL-100 class...")
            print_table('sstl100')
            plot_analysis('sstl100', '/home/claude/spacecraft_sstl100.png')
