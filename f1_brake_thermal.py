"""
High-performance motorsport carbon-carbon brake disc transient thermal and oxidation simulator.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
import argparse, warnings
warnings.filterwarnings('ignore')

# Physical constants
SIGMA_SB  = 5.6704e-8   # W/m²/K⁴ Stefan-Boltzmann

# Disc geometry (front disc, high-performance motorsport class)
D_OUTER   = 0.330       # m  outer diameter
D_INNER   = 0.130       # m  inner diameter (bell bore)
THICKNESS = 0.028       # m  28mm disc
RHO_CC    = 1850.0      # kg/m³ carbon-carbon density
EMISSIVITY = 0.85       # —  surface emissivity

# CC material properties (temperature-dependent)
def cp_cc(T_C):
    """C-C specific heat [J/kgK] as function of temperature in °C.
    Linear fit to published data: ~500 J/kgK at 20°C, ~1200 J/kgK at 1200°C.
    Ref: Buckley (1993), Carbon-Carbon Composites.
    """
    return np.clip(500.0 + 0.58 * T_C, 500.0, 1200.0)

def k_cc(T_C):
    """C-C through-thickness thermal conductivity [W/mK].
    Decreases with temperature due to phonon scattering.
    Ref: ~40 W/mK at 500°C, ~25 W/mK at 1000°C (through-thickness).
    """
    return max(20.0, 42.0 - 0.018 * T_C)

# Disc derived geometry
A_FACE    = np.pi/4 * (D_OUTER**2 - D_INNER**2)   # m²  one face
A_TOTAL   = 2 * A_FACE                              # m²  both faces (radiation)
VOLUME    = A_FACE * THICKNESS                      # m³
MASS      = RHO_CC * VOLUME                         # kg  ~3.3 kg

# Cooling duct parameters
D_DUCT    = 0.050       # m  duct exit diameter (50mm typical F1 duct)
A_DUCT    = np.pi/4 * D_DUCT**2   # m²  duct exit area
T_AMBIENT = 35.0        # °C  ambient/duct inlet temperature (hot race day)
RHO_AIR   = 1.15        # kg/m³  hot ambient air density

# Oxidation model
# Simplified Arrhenius: dm/dt [g/m²/s] = A_ox * exp(-E_a/(R*T_K))
# Fit to published C-C oxidation data (onset ~450°C, rapid above 750°C)
# Ref: Krenkel (2008), Ceramic Matrix Composites
A_OX      = 2.5e6       # g/(m²·s)  pre-exponential
E_A_R     = 18000.0     # K  activation energy / R
T_ONSET   = 723.15      # K  = 450°C onset temperature

# Colour palette
BG    = '#0f0f0e'; GOLD  = '#b8920a'; MOSS  = '#4a7a4b'
PAPER = '#f0ede8'; DIM   = '#8a8a7a'; RED   = '#c04040'
BLUE  = '#4080c0'; CYAN  = '#40b0c0'; GREY  = '#3a3a38'


# ══════════════════════════════════════════════════════════════════════════════
#  LAP PROFILES
# ══════════════════════════════════════════════════════════════════════════════

def lap_profile_silverstone():
    """
    Simplified Silverstone GP lap profile.
    Braking events: (t_start [s], duration [s], P_brake_per_disc [kW])
    Based on published sector time / speed data (idealised — real is confidential).
    Lap time ~1:27s. Front discs carry ~70% of braking load.
    Total energy per lap per disc: ~80-120 kJ (estimated from retardation data).
    """
    # [t_start, duration_s, P_kW_per_disc]
    events = [
        # Turn 1 (heavy braking from ~310 km/h)
        (2.0,   1.8,  850),
        # Turn 3 (medium)
        (9.5,   1.2,  550),
        # Brooklands (medium)
        (17.0,  1.0,  480),
        # Luffield (medium-light)
        (26.0,  1.2,  420),
        # Village (light)
        (35.0,  0.8,  350),
        # Loop / Aintree (light-medium)
        (41.0,  0.9,  400),
        # Vale (medium, before Club)
        (55.0,  1.3,  520),
        # Club (light-medium, high-speed)
        (66.0,  0.7,  380),
        # Abbey (light)
        (72.0,  0.7,  320),
        # Farm (medium)
        (79.0,  1.1,  470),
        # Maggotts-Becketts-Chapel (minimal, high-speed)
        # (no significant braking)
        # Stowe (medium-heavy)
        (86.0,  1.5,  600),
    ]
    lap_time = 87.0    # s (approximate)
    return events, lap_time, 'Silverstone GP'

def lap_profile_monaco():
    """
    Simplified Monaco GP lap profile.
    Extreme brake demand: 12+ heavy events per lap.
    Speed drops from 280 km/h to near-zero at hairpins.
    Lap time ~1:12s. Brakes run very hot.
    """
    events = [
        # Sainte-Devote (very heavy)
        (1.5,   2.2, 1050),
        # Massenet (medium-heavy)
        (9.0,   1.8,  820),
        # Casino Square (heavy)
        (14.5,  1.5,  780),
        # Mirabeau (very heavy)
        (19.0,  2.0,  980),
        # Portier (medium)
        (25.0,  1.3,  560),
        # Tunnel exit / Chicane (heavy)
        (29.0,  1.8,  850),
        # Nouvelle Chicane (heavy)
        (35.0,  1.6,  780),
        # Tabac (medium)
        (41.0,  1.2,  520),
        # Piscine S1 (medium-light)
        (47.0,  1.0,  460),
        # Piscine S2 (medium)
        (50.0,  1.1,  500),
        # La Rascasse (very heavy)
        (55.0,  2.2, 1020),
        # Anthony Noghes (heavy)
        (62.0,  1.6,  780),
    ]
    lap_time = 72.0    # s (approximate)
    return events, lap_time, 'Monaco GP'


# ══════════════════════════════════════════════════════════════════════════════
#  THERMAL MODEL
# ══════════════════════════════════════════════════════════════════════════════

def braking_power(t, events):
    """Return brake power [W] at time t given list of braking events."""
    P = 0.0
    for (t_start, dur, P_kW) in events:
        if t_start <= t <= t_start + dur:
            # Smooth onset/offset (10% rise/fall)
            tau = dur * 0.1
            ramp_on  = min(1.0, (t - t_start) / tau)
            ramp_off = min(1.0, (t_start + dur - t) / tau)
            P += P_kW * 1000.0 * min(ramp_on, ramp_off)
    return P


def cooling_power(T_C, m_dot_kgs):
    """
    Convective heat removal [W] from cooling duct.
    Model: turbulent jet impingement on disc face.
    Nu = 0.5 * Re^0.5 * Pr^0.4  (simplified Hilpert jet impingement)
    Ref: Martin (1977), Advances in Heat Transfer Vol 13.
    """
    if m_dot_kgs <= 0 or T_C <= T_AMBIENT:
        return 0.0
    V_jet = m_dot_kgs / (RHO_AIR * A_DUCT)
    mu_air = 1.85e-5      # Pa·s
    k_air  = 0.027        # W/mK
    Pr_air = 0.71
    Re_jet = RHO_AIR * V_jet * D_DUCT / mu_air
    Nu_jet = 0.5 * Re_jet**0.5 * Pr_air**0.4
    h_jet  = Nu_jet * k_air / D_DUCT
    # Cooling only from one face (inner face, protected from disc carrier)
    Q_cool = h_jet * A_FACE * (T_C - T_AMBIENT)
    return max(0.0, Q_cool)


def radiation_power(T_C):
    """Radiation from disc faces [W]."""
    T_K = T_C + 273.15
    T_amb_K = T_AMBIENT + 273.15
    return SIGMA_SB * EMISSIVITY * A_TOTAL * (T_K**4 - T_amb_K**4)


def oxidation_rate(T_C):
    """
    Mass loss rate [g/m²/s] due to oxidation.
    Arrhenius model, only active above T_ONSET.
    Note: pure gas-phase kinetics only — real oxidation also depends on
    oxygen diffusion through char layer (not modelled here).
    """
    T_K = T_C + 273.15
    if T_K < T_ONSET:
        return 0.0
    return A_OX * np.exp(-E_A_R / T_K)


def ode(t, y, events, m_dot_kgs):
    """
    Thermal ODE for lumped disc.
    y = [T_disc_C, mass_loss_g_m2]
    """
    T_C, m_abl = y
    T_C = max(T_AMBIENT, T_C)

    C_eff = MASS * cp_cc(T_C)   # J/K effective thermal capacitance

    Q_in   = braking_power(t, events)
    Q_cool = cooling_power(T_C, m_dot_kgs)
    Q_rad  = radiation_power(T_C)

    # Oxidation heat loss (endothermic for carbon: ~33 kJ/mol, ~2750 J/g)
    dm_dt  = oxidation_rate(T_C) * A_TOTAL   # g/s total
    Q_ox   = dm_dt * 2750.0   # W heat absorbed by oxidation reaction

    dT_dt  = (Q_in - Q_cool - Q_rad - Q_ox) / C_eff
    return [dT_dt, dm_dt]


def simulate_laps(events, lap_time, m_dot_kgs, n_laps=5, T_init_C=250.0):
    """
    Simulate n_laps and return the final quasi-steady-state lap.
    Returns: dict of time-series arrays.
    """
    T0 = T_init_C
    all_t, all_T, all_m = [], [], []

    for lap in range(n_laps):
        t_span = [0.0, lap_time]
        t_eval = np.linspace(0, lap_time, 300)
        sol = solve_ivp(ode, t_span, [T0, 0.0],
                        args=(events, m_dot_kgs),
                        method='RK45', t_eval=t_eval,
                        rtol=1e-5, atol=1e-3)
        T0 = sol.y[0, -1]
        offset = lap * lap_time
        all_t.append(sol.t + offset)
        all_T.append(sol.y[0])
        all_m.append(sol.y[1])

    return {
        't':        np.concatenate(all_t),
        'T_C':      np.concatenate(all_T),
        'm_abl':    np.concatenate(all_m),
        'lap_time': lap_time,
        'n_laps':   n_laps,
    }


def lap_stats(res, lap_idx=-1):
    """Extract stats for a given lap (default: last lap = quasi-steady)."""
    t = res['t']
    T = res['T_C']
    lt = res['lap_time']
    nl = res['n_laps']
    lap_n = nl + lap_idx if lap_idx < 0 else lap_idx
    t0 = lap_n * lt
    t1 = (lap_n + 1) * lt
    mask = (t >= t0) & (t <= t1)
    T_lap = T[mask]
    return {
        'T_max': T_lap.max(),
        'T_min': T_lap.min(),
        'T_mean': T_lap.mean(),
        'T_end': T_lap[-1] if len(T_lap) > 0 else T_lap[0],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

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


def plot_circuit(circuit_name, events, lap_time, output_path):
    """Full 6-panel brake thermal analysis for one circuit."""
    duct_flows = [0.03, 0.07, 0.12]   # kg/s
    flow_labels = ['30 g/s (narrow duct)', '70 g/s (medium)', '120 g/s (wide duct)']
    flow_colors = [GOLD, CYAN, MOSS]

    fig = plt.figure(figsize=(18, 20), facecolor=BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            hspace=0.52, wspace=0.38,
                            left=0.08, right=0.96,
                            top=0.93, bottom=0.05)

    fig.text(0.5, 0.975,
             f'F1 BRAKE DISC — TRANSIENT THERMAL ANALYSIS  |  {circuit_name.upper()}',
             ha='center', va='top', color=PAPER,
             fontsize=12, fontweight='bold', fontfamily='monospace')
    fig.text(0.5, 0.961,
             'Lumped capacitance model  ·  C-C disc  ·  Jet impingement cooling  ·  '
             'Arrhenius oxidation  ·  5-lap quasi-steady-state',
             ha='center', va='top', color=DIM, fontsize=8.5,
             fontfamily='monospace')
    
    # Panel 1: Temperature vs time (final lap)
    ax1 = fig.add_subplot(gs[0, :])   # full width
    style_ax(ax1, f'DISC TEMPERATURE vs TIME  (Final Lap  |  {circuit_name})')

    for m_dot, label, color in zip(duct_flows, flow_labels, flow_colors):
        res = simulate_laps(events, lap_time, m_dot, n_laps=5)
        t = res['t']
        T = res['T_C']
        # Show last lap
        t_last = (res['n_laps']-1) * lap_time
        mask = t >= t_last
        ax1.plot(t[mask] - t_last, T[mask], color=color, linewidth=2.0,
                 label=f'{label}')

    # Temperature window bands
    ax1.axhspan(600, 1200, alpha=0.08, color=MOSS,
                label='Normal operating range [600–1200°C, N₂ purge]')
    ax1.axhline(750, color=RED, linewidth=0.8, linestyle=':', alpha=0.5)
    ax1.axhline(1200, color=RED, linewidth=1.0, linestyle='--', alpha=0.7)
    ax1.axhline(600, color=GOLD, linewidth=1.0, linestyle='--', alpha=0.7)
    ax1.text(lap_time*0.98, 760, 'Air oxidation threshold (750°C — N₂ purge required above)',
             color=RED, fontsize=7.5, fontfamily='monospace', ha='right', alpha=0.7)
    ax1.text(lap_time*0.98, 1210, 'CC graphitisation limit (~1200°C with N₂)',
             color=RED, fontsize=7.5, fontfamily='monospace', ha='right', alpha=0.8)
    ax1.text(lap_time*0.98, 590, 'Glazing threshold (600°C)',
             color=GOLD, fontsize=7.5, fontfamily='monospace', ha='right', alpha=0.8)

    # Mark braking events
    for (t_s, dur, P_kW) in events:
        ax1.axvspan(t_s, t_s+dur, alpha=0.08, color=RED)

    ax1.set_xlabel('Lap time  [s]', fontsize=9)
    ax1.set_ylabel('Disc temperature  [°C]', fontsize=9)
    ax1.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper right')

    # Panel 2: T_max and T_min per lap (convergence)
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2, 'CONVERGENCE — T_MAX & T_MIN PER LAP')

    for m_dot, label, color in zip(duct_flows, flow_labels, flow_colors):
        res = simulate_laps(events, lap_time, m_dot, n_laps=5)
        maxs, mins = [], []
        for lap in range(5):
            t0 = lap * lap_time; t1 = t0 + lap_time
            mask = (res['t'] >= t0) & (res['t'] <= t1)
            T_lap = res['T_C'][mask]
            maxs.append(T_lap.max()); mins.append(T_lap.min())
        laps = list(range(1, 6))
        ax2.plot(laps, maxs, color=color, linewidth=2.0, marker='o', ms=5)
        ax2.plot(laps, mins, color=color, linewidth=1.2, linestyle='--',
                 marker='s', ms=4, alpha=0.6)

    ax2.axhline(750, color=RED, linewidth=0.8, linestyle=':', alpha=0.6)
    ax2.axhline(400, color=GOLD, linewidth=0.8, linestyle=':', alpha=0.5)
    ax2.set_xlabel('Lap number', fontsize=9)
    ax2.set_ylabel('Temperature  [°C]', fontsize=9)
    ax2.set_xticks(range(1, 6))

    # Panel 3: Required duct flow for window compliance
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3, 'REQUIRED DUCT FLOW — TEMPERATURE WINDOW')

    m_dots = np.linspace(0.01, 0.20, 25)
    T_maxs, T_mins = [], []
    for m_dot in m_dots:
        res = simulate_laps(events, lap_time, m_dot, n_laps=4)
        stats = lap_stats(res)
        T_maxs.append(stats['T_max'])
        T_mins.append(stats['T_min'])

    T_maxs = np.array(T_maxs); T_mins = np.array(T_mins)
    ax3.plot(m_dots*1000, T_maxs, color=RED, linewidth=2.0, label='T_max')
    ax3.plot(m_dots*1000, T_mins, color=CYAN, linewidth=2.0, label='T_min')
    ax3.fill_between(m_dots*1000, 600, 1200, alpha=0.1, color=MOSS,
                     label='Operating range [600–1200°C]')

    # Mark compliance region
    ok_mask = (T_maxs <= 1200) & (T_mins >= 600)
    if np.any(ok_mask):
        m_ok_min = m_dots[ok_mask][0] * 1000
        m_ok_max = m_dots[ok_mask][-1] * 1000
        ax3.axvspan(m_ok_min, m_ok_max, alpha=0.12, color=MOSS)
        ax3.text((m_ok_min+m_ok_max)/2, 350,
                 f'Compliant\n{m_ok_min:.0f}–{m_ok_max:.0f} g/s',
                 color=MOSS, fontsize=8, ha='center', fontfamily='monospace')

    ax3.axhline(1200, color=RED, linewidth=0.8, linestyle='--', alpha=0.7)
    ax3.axhline(750, color=RED, linewidth=0.6, linestyle=':', alpha=0.4)
    ax3.axhline(600, color=GOLD, linewidth=0.8, linestyle='--', alpha=0.5)
    ax3.set_xlabel('Duct mass flow  [g/s]', fontsize=9)
    ax3.set_ylabel('Disc temperature  [°C]', fontsize=9)
    ax3.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # Panel 4: Oxidation mass loss
    ax4 = fig.add_subplot(gs[2, 0])
    style_ax(ax4, 'OXIDATION MASS LOSS  [g/m²/lap]')

    m_dots_ox = np.linspace(0.01, 0.20, 20)
    losses = []
    for m_dot in m_dots_ox:
        res = simulate_laps(events, lap_time, m_dot, n_laps=4)
        # Oxidation in last lap
        t = res['t']; m_abl = res['m_abl']
        lt = lap_time; nl = 4
        t0 = (nl-1)*lt; t1 = nl*lt
        mask = (t >= t0) & (t <= t1)
        m_lap = m_abl[mask]
        delta_m = (m_lap[-1] - m_lap[0]) if len(m_lap) > 1 else 0.0
        losses.append(max(0.0, delta_m))

    ax4.plot(m_dots_ox*1000, losses, color=RED, linewidth=2.0)
    ax4.fill_between(m_dots_ox*1000, losses, alpha=0.2, color=RED)

    # FIA wear limit (approximate: 2mm per race = ~74mm per stint / 10 laps ≈ 7.4mm/lap)
    # Thickness loss: delta_m [g/m²] / rho [g/m³] → thickness in mm
    # rho_CC = 1850 kg/m³ = 1.85e6 g/m³
    thickness_loss = np.array(losses) / 1.85e6 * 1000   # mm/lap
    ax4r = ax4.twinx()
    ax4r.set_facecolor(BG)
    ax4r.plot(m_dots_ox*1000, thickness_loss, color=GOLD, linewidth=1.4,
              linestyle='-.', label='Thickness [mm/lap]')
    ax4r.tick_params(colors=GOLD, labelsize=9)
    ax4r.spines['right'].set_color(GREY)
    ax4r.set_ylabel('Thickness loss  [mm/lap]', fontsize=9, color=GOLD)
    ax4r.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    ax4.set_xlabel('Duct mass flow  [g/s]', fontsize=9)
    ax4.set_ylabel('Mass loss  [g/m²/lap]', fontsize=9)

    # Panel 5: Energy budget breakdown
    ax5 = fig.add_subplot(gs[2, 1])
    style_ax(ax5, 'ENERGY BUDGET  (Steady-State Lap, m_dot = 70 g/s)')

    m_dot_ref = 0.07
    res_ref = simulate_laps(events, lap_time, m_dot_ref, n_laps=5)
    t = res_ref['t']; T = res_ref['T_C']
    lt = lap_time; nl = 5
    t0 = (nl-1)*lt
    mask = t >= t0
    t_lap = t[mask]; T_lap = T[mask]
    dt = np.diff(t_lap, prepend=t_lap[0])

    E_in   = sum(braking_power(ti, events) * dti
                 for ti, dti in zip(t_lap, dt))
    E_cool = sum(cooling_power(Ti, m_dot_ref) * dti
                 for Ti, dti in zip(T_lap, dt))
    E_rad  = sum(radiation_power(Ti) * dti
                 for Ti, dti in zip(T_lap, dt))
    E_ox   = sum(oxidation_rate(Ti) * A_TOTAL * 2750.0 * dti
                 for Ti, dti in zip(T_lap, dt))
    E_stored = E_in - E_cool - E_rad - E_ox

    labels = ['Braking input', 'Duct cooling', 'Radiation', 'Oxidation', 'ΔStored']
    values = [E_in/1000, E_cool/1000, E_rad/1000, E_ox/1000, max(0,E_stored/1000)]
    colors_e = [RED, CYAN, GOLD, MOSS, BLUE]

    bars = ax5.bar(range(len(labels)), values, color=colors_e, alpha=0.8,
                   edgecolor=GREY, linewidth=0.5)
    ax5.set_xticks(range(len(labels)))
    ax5.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax5.set_ylabel('Energy per lap  [kJ]', fontsize=9)

    for bar, val in zip(bars, values):
        ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{val:.0f}', ha='center', fontsize=8, color=DIM,
                 fontfamily='monospace')

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"  Plot saved: {output_path}")


def print_table(circuit_name, events, lap_time):
    """Print summary table across duct flows."""
    print(f"\n{'='*72}")
    print(f"  {circuit_name} — Brake Thermal Summary (Quasi-steady, Lap 5)")
    print(f"{'='*72}")
    print(f"  {'m_dot[g/s]':>12}  {'T_max[°C]':>10}  {'T_min[°C]':>10}  "
          f"{'Window?':>8}  {'Loss[g/m²]':>12}")
    print(f"  {'-'*66}")

    for m_dot in [20, 30, 50, 70, 90, 120, 150]:
        res = simulate_laps(events, lap_time, m_dot/1000, n_laps=5)
        stats = lap_stats(res)
        Tmx = stats['T_max']; Tmn = stats['T_min']
        ok  = 'OK' if 600 <= Tmn and Tmx <= 1200 else ('WARN_N2' if 400 <= Tmn and Tmx <= 1400 else 'FAIL')

        # Oxidation in last lap
        t = res['t']; m_abl = res['m_abl']
        t0 = 4*lap_time; mask = t >= t0
        m_lap = m_abl[mask]
        loss = max(0.0, m_lap[-1]-m_lap[0]) if len(m_lap)>1 else 0.0

        print(f"  {m_dot:>12}  {Tmx:>10.0f}  {Tmn:>10.0f}  {ok:>8}  {loss:>12.2f}")

    print(f"{'='*72}\n")


def plot_comparison(events_s, lt_s, events_m, lt_m, output_path):
    """Side-by-side Monaco vs Silverstone comparison for m_dot = 0.07 kg/s."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    fig.suptitle('SILVERSTONE vs MONACO  —  BRAKE DISC THERMAL COMPARISON  |  m_dot = 70 g/s',
                 color=PAPER, fontsize=12, fontweight='bold',
                 fontfamily='monospace', y=0.97)
    
    m_dot = 0.07
    for ax, (ev, lt, name, color) in zip(
        axes,
        [(events_s, lt_s, 'Silverstone', CYAN),
         (events_m, lt_m, 'Monaco', GOLD)]
    ):
        style_ax(ax, f'{name}  —  T_disc vs Lap Time')
        res = simulate_laps(ev, lt, m_dot, n_laps=5)
        t = res['t']; T = res['T_C']
        t_last = 4*lt
        mask = t >= t_last
        ax.plot(t[mask]-t_last, T[mask], color=color, linewidth=2.2)

        ax.axhspan(400, 750, alpha=0.1, color=MOSS)
        ax.axhline(1200, color=RED, linewidth=0.8, linestyle='--', alpha=0.7)
        ax.axhline(750, color=RED, linewidth=0.6, linestyle=':', alpha=0.4)
        ax.axhline(600, color=GOLD, linewidth=0.8, linestyle='--', alpha=0.5)

        for (t_s, dur, P_kW) in ev:
            ax.axvspan(t_s, t_s+dur, alpha=0.10, color=RED)

        T_lap = T[mask]
        ax.text(0.05, 0.95, f'T_max = {T_lap.max():.0f}°C\nT_min = {T_lap.min():.0f}°C',
                transform=ax.transAxes, color=PAPER, fontsize=9,
                fontfamily='monospace', va='top')

        ax.set_xlabel('Lap time  [s]', fontsize=9)
        ax.set_ylabel('Temperature  [°C]', fontsize=9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"  Comparison plot saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='F1 Carbon-Carbon Brake Disc Thermal Model')
    parser.add_argument('--silverstone', action='store_true')
    parser.add_argument('--monaco',      action='store_true')
    parser.add_argument('--point',       action='store_true',
                        help='Print summary tables only (no plots)')
    args = parser.parse_args()

    run_s = args.silverstone or not any([args.silverstone, args.monaco, args.point])
    run_m = args.monaco      or not any([args.silverstone, args.monaco, args.point])

    ev_s, lt_s, name_s = lap_profile_silverstone()
    ev_m, lt_m, name_m = lap_profile_monaco()

    if args.point:
        print_table(name_s, ev_s, lt_s)
        print_table(name_m, ev_m, lt_m)
    else:
        if run_s:
            print(f"\nSimulating {name_s}...")
            print_table(name_s, ev_s, lt_s)
            plot_circuit(name_s, ev_s, lt_s, 'f1_silverstone.png')

        if run_m:
            print(f"\nSimulating {name_m}...")
            print_table(name_m, ev_m, lt_m)
            plot_circuit(name_m, ev_m, lt_m, 'f1_monaco.png')

        if run_s and run_m:
            print("\nGenerating comparison plot...")
            plot_comparison(ev_s, lt_s, ev_m, lt_m,
                           'f1_comparison.png')
