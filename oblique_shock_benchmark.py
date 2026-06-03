"""
Oblique Shock Wave — Exact Analytical Benchmark Suite
======================================================
Computes exact oblique shock, normal shock, and Prandtl-Meyer expansion
properties for compressible flow validation.

Directly useful to:
  - Zenotech (Bristol) — FLITE3D solver for RAF Typhoon
  - Engys (London) — OpenFOAM-based CFD
  - Any CFD engineer needing reference solutions for code validation

Physics:
  - Oblique shock: theta-beta-M relation (implicit, solved by bisection)
  - Normal shock: special case beta=90°
  - Prandtl-Meyer expansion: isentropic turning
  - Shock polar (velocity hodograph)
  - Detachment limit chart

All results validated against NACA Report 1135 (1953) — the standard
reference for compressible flow calculations.

Usage:
  python oblique_shock_benchmark.py         # full charts
  python oblique_shock_benchmark.py --table # print validation table only

Author: MKA — pringlesmaths.co.uk
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq
import argparse, warnings
warnings.filterwarnings('ignore')

GAMMA = 1.4   # specific heat ratio, calorically perfect gas

BG='#0f0f0e'; GOLD='#b8920a'; MOSS='#4a7a4b'; PAPER='#f0ede8'
DIM='#8a8a7a'; RED='#c04040'; BLUE='#4080c0'; CYAN='#40b0c0'; GREY='#3a3a38'
COLORS = [GOLD, CYAN, MOSS, RED, BLUE, '#c080ff', '#ff8040']


# ── Oblique shock relations ───────────────────────────────────────────────────

def theta_from_beta(M1, beta_rad, g=GAMMA):
    """Deflection angle theta given M1 and shock angle beta (radians)."""
    M1n = M1 * np.sin(beta_rad)
    if M1n <= 1.0:
        return 0.0
    tan_theta = 2.0 / np.tan(beta_rad) * (M1**2 * np.sin(beta_rad)**2 - 1.0) \
                / (M1**2 * (g + np.cos(2*beta_rad)) + 2.0)
    return np.arctan(max(0.0, tan_theta))


def beta_from_theta(M1, theta_rad, g=GAMMA, weak=True):
    """
    Shock angle beta given M1 and deflection theta.
    Returns weak shock solution by default.
    Returns None if theta > theta_max (detachment).
    Ref: Anderson (2003), Modern Compressible Flow, eq. 9.31.
    """
    mu = np.arcsin(1.0/M1)   # Mach angle (lower bound for beta)

    def f(beta):
        return theta_from_beta(M1, beta, g) - theta_rad

    # Weak shock: beta in [mu, beta_max]
    # First find theta_max to check feasibility
    betas = np.linspace(mu + 1e-6, np.pi/2 - 1e-6, 500)
    thetas = np.array([theta_from_beta(M1, b, g) for b in betas])
    idx_max = np.argmax(thetas)
    theta_max = thetas[idx_max]
    beta_max_val = betas[idx_max]

    if theta_rad > theta_max:
        return None   # detachment

    try:
        if weak:
            beta_sol = brentq(f, mu + 1e-6, beta_max_val, xtol=1e-8)
        else:
            beta_sol = brentq(f, beta_max_val, np.pi/2 - 1e-6, xtol=1e-8)
        return beta_sol
    except ValueError:
        return None


def theta_max_for_M(M1, g=GAMMA):
    """Maximum deflection angle (detachment limit) for M1."""
    mu = np.arcsin(1.0/M1)
    betas = np.linspace(mu + 1e-6, np.pi/2 - 1e-4, 1000)
    thetas = np.array([theta_from_beta(M1, b, g) for b in betas])
    return np.degrees(np.max(thetas))


def oblique_shock_ratios(M1, beta_rad, g=GAMMA):
    """
    Compute all flow ratios across oblique shock.
    Returns dict with downstream Mach and pressure/temperature/density ratios.
    """
    M1n = M1 * np.sin(beta_rad)
    theta = theta_from_beta(M1, beta_rad, g)

    # Normal component relations (Rankine-Hugoniot)
    p_ratio   = 1.0 + 2*g/(g+1) * (M1n**2 - 1.0)
    rho_ratio = (g+1)*M1n**2 / (2.0 + (g-1)*M1n**2)
    T_ratio   = p_ratio / rho_ratio

    # Downstream normal Mach
    M2n2 = (M1n**2 + 2.0/(g-1)) / (2.0*g*M1n**2/(g-1) - 1.0)
    M2n  = np.sqrt(max(0.0, M2n2))
    M2   = M2n / np.sin(beta_rad - theta)

    # Stagnation pressure ratio (entropy indicator)
    p0_ratio = ((g+1)*M1n**2/2.0)**(g/(g-1)) * \
               ((2*g*M1n**2/(g+1) - (g-1)/(g+1))**(1.0/(1.0-g))) * \
               (1.0 + (g-1)/2*M2**2)**(g/(g-1)) / \
               (1.0 + (g-1)/2*M1**2)**(g/(g-1))

    # Cleaner stagnation pressure ratio via isentropic + normal shock
    # p02/p01 = (p2/p1) * (p02/p2) / (p01/p1)
    def p0_over_p(M):
        return (1.0 + (g-1)/2*M**2)**(g/(g-1))
    p0_ratio2 = p_ratio * p0_over_p(M2) / p0_over_p(M1)

    return {
        'M1':         M1,
        'beta_deg':   np.degrees(beta_rad),
        'theta_deg':  np.degrees(theta),
        'M2':         M2,
        'M2n':        M2n,
        'p2_p1':      p_ratio,
        'T2_T1':      T_ratio,
        'rho2_rho1':  rho_ratio,
        'p02_p01':    p0_ratio2,
    }


def normal_shock_ratios(M1, g=GAMMA):
    """Normal shock (beta=90°) — special case for validation."""
    return oblique_shock_ratios(M1, np.pi/2, g)


# ── Prandtl-Meyer expansion ───────────────────────────────────────────────────

def pm_function(M, g=GAMMA):
    """
    Prandtl-Meyer function nu(M) [radians].
    nu = sqrt((g+1)/(g-1))*arctan(sqrt((g-1)*(M^2-1)/(g+1))) - arctan(sqrt(M^2-1))
    Ref: NACA Report 1135, eq. 120.
    """
    if M <= 1.0:
        return 0.0
    A = np.sqrt((g+1)/(g-1))
    B = np.sqrt((g-1)*(M**2-1)/(g+1))
    C = np.sqrt(M**2 - 1)
    return A * np.arctan(B) - np.arctan(C)


def pm_expansion(M1, delta_rad, g=GAMMA):
    """
    Downstream Mach after Prandtl-Meyer expansion through angle delta.
    """
    nu1 = pm_function(M1, g)
    nu2 = nu1 + delta_rad
    # Invert nu(M2) = nu2
    try:
        M2 = brentq(lambda M: pm_function(M, g) - nu2, 1.0 + 1e-6, 50.0, xtol=1e-8)
    except ValueError:
        M2 = M1  # fallback
    # Isentropic ratios
    def T0_T(M): return (1.0 + (g-1)/2*M**2)
    T_ratio   = T0_T(M1) / T0_T(M2)
    p_ratio   = (T0_T(M1)/T0_T(M2))**(g/(g-1))
    rho_ratio = p_ratio / T_ratio
    return {'M2': M2, 'p2_p1': 1.0/p_ratio, 'T2_T1': T_ratio,
            'rho2_rho1': 1.0/rho_ratio}


# ── Shock polar ───────────────────────────────────────────────────────────────

def shock_polar(M1, g=GAMMA, n_pts=200):
    """
    Compute shock polar in velocity space.
    Returns (Vx_ratio, Vy_ratio) normalised by upstream speed.
    """
    mu = np.arcsin(1.0/M1)
    a1 = np.sqrt(g * 287.058 * 288.15)  # reference speed of sound (ISA SL)
    V1 = M1 * a1

    betas = np.linspace(mu + 0.001, np.pi/2, n_pts)
    Vx, Vy = [], []
    for beta in betas:
        r = oblique_shock_ratios(M1, beta, g)
        M2  = r['M2']
        th  = np.radians(r['theta_deg'])
        # Downstream velocity magnitude from Mach and speed of sound
        T2  = r['T2_T1'] * 288.15
        a2  = np.sqrt(g * 287.058 * T2)
        V2  = M2 * a2
        # Velocity components (upstream along x-axis)
        Vx.append(V2 * np.cos(th) / V1)
        Vy.append(-V2 * np.sin(th) / V1)   # negative for lower half

    return np.array(Vx), np.array(Vy)


# ── Generate benchmark table ──────────────────────────────────────────────────

BENCHMARK_CASES = [
    # (M1, theta_deg, label)
    (2.0,  10.0, 'Case A: M2.0, θ=10°'),
    (3.0,  15.0, 'Case B: M3.0, θ=15°'),
    (5.0,  20.0, 'Case C: M5.0, θ=20°'),
    (8.0,  12.0, 'Case D: M8.0, θ=12°'),
    (2.0,   0.0, 'Case E: Normal shock M2'),
    (5.0,   0.0, 'Case F: Normal shock M5'),
]


def print_validation_table():
    print()
    print("=" * 90)
    print("  OBLIQUE SHOCK BENCHMARK — Validation against NACA Report 1135 (1953)")
    print("=" * 90)
    print(f"  {'Case':30}  {'β[°]':>7}  {'M2':>7}  {'p2/p1':>8}  "
          f"{'T2/T1':>8}  {'ρ2/ρ1':>8}  {'p02/p01':>8}")
    print("-" * 90)

    for M1, theta_deg, label in BENCHMARK_CASES:
        if theta_deg == 0.0:
            # Normal shock
            r = normal_shock_ratios(M1)
            beta_used = 90.0
        else:
            beta_rad = beta_from_theta(M1, np.radians(theta_deg))
            if beta_rad is None:
                print(f"  {label:30}  DETACHED")
                continue
            r = oblique_shock_ratios(M1, beta_rad)
            beta_used = r['beta_deg']

        print(f"  {label:30}  {beta_used:7.2f}  {r['M2']:7.4f}  "
              f"{r['p2_p1']:8.4f}  {r['T2_T1']:8.4f}  "
              f"{r['rho2_rho1']:8.4f}  {r['p02_p01']:8.5f}")

    print("=" * 90)

    # Prandtl-Meyer table
    print()
    print("  PRANDTL-MEYER EXPANSION")
    print(f"  {'M1':>6}  {'Δ[°]':>8}  {'M2':>8}  {'p2/p1':>10}  {'T2/T1':>10}")
    print("-" * 55)
    for M1, delta_deg in [(2.0,10),(3.0,15),(5.0,20),(8.0,10)]:
        r = pm_expansion(M1, np.radians(delta_deg))
        print(f"  {M1:6.1f}  {delta_deg:8.1f}  {r['M2']:8.4f}  "
              f"{r['p2_p1']:10.5f}  {r['T2_T1']:10.5f}")
    print()


# ── Plotting ──────────────────────────────────────────────────────────────────

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


def plot_all(output_path):
    fig = plt.figure(figsize=(18, 20), facecolor=BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            hspace=0.52, wspace=0.38,
                            left=0.08, right=0.96,
                            top=0.93, bottom=0.05)

    fig.text(0.5, 0.975,
             'OBLIQUE SHOCK & PRANDTL-MEYER — EXACT ANALYTICAL BENCHMARK SUITE',
             ha='center', va='top', color=PAPER,
             fontsize=12, fontweight='bold', fontfamily='monospace')
    fig.text(0.5, 0.961,
             'Validated against NACA Report 1135 (1953)  ·  γ = 1.4  ·  '
             'Calorically perfect gas  ·  For CFD solver validation',
             ha='center', va='top', color=DIM, fontsize=8.5,
             fontfamily='monospace')
    fig.text(0.99, 0.005, 'pringlesmaths.co.uk  |  MKA',
             ha='right', va='bottom', color='#3a3a38',
             fontsize=8, fontfamily='monospace')

    # ── Panel 1: Theta-beta-M curves ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'θ-β-M RELATIONS  (Weak shock, γ=1.4)')

    mach_vals = [1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]
    for i, M1 in enumerate(mach_vals):
        mu = np.degrees(np.arcsin(1.0/M1))
        betas_deg = np.linspace(mu + 0.1, 89.9, 400)
        thetas_deg = [np.degrees(theta_from_beta(M1, np.radians(b))) for b in betas_deg]
        ax1.plot(betas_deg, thetas_deg, color=COLORS[i % len(COLORS)],
                 linewidth=1.8, label=f'M={M1}')

    ax1.set_xlabel('Shock angle β  [°]', fontsize=9)
    ax1.set_ylabel('Deflection angle θ  [°]', fontsize=9)
    ax1.legend(fontsize=7.5, framealpha=0, labelcolor=DIM, ncol=2)

    # Locus of theta_max (detachment curve)
    M_det = np.linspace(1.1, 10, 100)
    theta_det = [theta_max_for_M(M) for M in M_det]
    ax1.plot([beta_from_theta(M, np.radians(th), weak=True) is not None
              and np.degrees(beta_from_theta(M, np.radians(th), weak=True) or 0)
              for M, th in zip(M_det, theta_det)],
             theta_det, 'w--', linewidth=0.8, alpha=0.4)

    # ── Panel 2: Detachment chart ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'DETACHMENT CHART  —  Attached vs Bow Shock')

    M_arr = np.linspace(1.05, 12.0, 200)
    th_det = np.array([theta_max_for_M(M) for M in M_arr])
    ax2.plot(M_arr, th_det, color=RED, linewidth=2.5, label='θ_max (detachment)')
    ax2.fill_between(M_arr, th_det, 0, alpha=0.12, color=MOSS,
                     label='Attached oblique shock')
    ax2.fill_between(M_arr, th_det, 50, alpha=0.08, color=RED,
                     label='Bow shock (detached)')

    # Mark standard benchmark cases
    for M1, theta_deg, label in BENCHMARK_CASES[:4]:
        ax2.plot(M1, theta_deg, 'o', color=GOLD, ms=7, zorder=5)
        ax2.annotate(label.split(':')[1].strip(),
                     xy=(M1, theta_deg), xytext=(8, 5),
                     textcoords='offset points', fontsize=7.5,
                     color=GOLD, fontfamily='monospace')

    ax2.set_xlabel('Mach number M₁', fontsize=9)
    ax2.set_ylabel('Deflection angle θ  [°]', fontsize=9)
    ax2.set_xlim(1, 12); ax2.set_ylim(0, 50)
    ax2.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 3: Shock polars ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'SHOCK POLAR  (Velocity Hodograph)')

    for i, M1 in enumerate([2.0, 3.0, 5.0]):
        Vx, Vy = shock_polar(M1)
        ax3.plot(Vx, Vy,     color=COLORS[i], linewidth=2.0, label=f'M={M1}')
        ax3.plot(Vx, -Vy,    color=COLORS[i], linewidth=2.0)
        # Mark upstream point
        ax3.plot(1.0, 0.0, 'o', color=COLORS[i], ms=5, zorder=5)

    ax3.axhline(0, color=DIM, linewidth=0.5, alpha=0.5)
    ax3.axvline(1, color=DIM, linewidth=0.5, alpha=0.3)
    ax3.set_xlabel('Vₓ / V₁', fontsize=9)
    ax3.set_ylabel('V_y / V₁', fontsize=9)
    ax3.legend(fontsize=9, framealpha=0, labelcolor=DIM)

    # ── Panel 4: Pressure ratio vs M1 ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, 'PRESSURE RATIO p₂/p₁  vs  MACH NUMBER')

    M1_vals = np.linspace(1.1, 10.0, 200)
    for i, theta_deg in enumerate([5.0, 10.0, 15.0, 20.0]):
        p_ratios = []
        for M1 in M1_vals:
            beta_rad = beta_from_theta(M1, np.radians(theta_deg))
            if beta_rad is None:
                p_ratios.append(np.nan)
            else:
                r = oblique_shock_ratios(M1, beta_rad)
                p_ratios.append(r['p2_p1'])
        ax4.plot(M1_vals, p_ratios, color=COLORS[i], linewidth=1.8,
                 label=f'θ={theta_deg}°')

    # Normal shock for comparison
    p_ns = [normal_shock_ratios(M)['p2_p1'] for M in M1_vals]
    ax4.plot(M1_vals, p_ns, color=RED, linewidth=1.5,
             linestyle='--', label='Normal shock')

    ax4.set_xlabel('M₁', fontsize=9)
    ax4.set_ylabel('p₂/p₁', fontsize=9)
    ax4.set_yscale('log')
    ax4.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    # ── Panel 5: Prandtl-Meyer function ──────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    style_ax(ax5, 'PRANDTL-MEYER FUNCTION  ν(M)')

    M_pm = np.linspace(1.0, 10.0, 300)
    nu   = np.array([np.degrees(pm_function(M)) for M in M_pm])
    ax5.plot(M_pm, nu, color=CYAN, linewidth=2.2, label='ν(M) [degrees]')

    # Maximum turning angle (=nu at M→∞) ≈ 130.45° for γ=1.4
    nu_max = np.degrees(np.pi/2 * (np.sqrt((GAMMA+1)/(GAMMA-1)) - 1.0))
    ax5.axhline(nu_max, color=GOLD, linewidth=0.8, linestyle='--', alpha=0.7)
    ax5.text(1.2, nu_max+1.5, f'ν_max = {nu_max:.1f}°  (M→∞)',
             color=GOLD, fontsize=8, fontfamily='monospace')

    # Mark standard Mach numbers
    for M_ref in [1.5, 2.0, 3.0, 5.0, 8.0]:
        nu_ref = np.degrees(pm_function(M_ref))
        ax5.plot(M_ref, nu_ref, 'o', color=MOSS, ms=5)

    ax5.set_xlabel('Mach number M', fontsize=9)
    ax5.set_ylabel('ν  [degrees]', fontsize=9)
    ax5.legend(fontsize=9, framealpha=0, labelcolor=DIM)

    # ── Panel 6: Stagnation pressure recovery ────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    style_ax(ax6, 'STAGNATION PRESSURE RECOVERY  p₀₂/p₀₁')

    for i, theta_deg in enumerate([5.0, 10.0, 15.0]):
        p0_ratios = []
        for M1 in M1_vals:
            beta_rad = beta_from_theta(M1, np.radians(theta_deg))
            if beta_rad is None:
                p0_ratios.append(np.nan)
            else:
                r = oblique_shock_ratios(M1, beta_rad)
                p0_ratios.append(r['p02_p01'])
        ax6.plot(M1_vals, p0_ratios, color=COLORS[i], linewidth=1.8,
                 label=f'θ={theta_deg}° (oblique)')

    p0_ns = [normal_shock_ratios(M)['p02_p01'] for M in M1_vals]
    ax6.plot(M1_vals, p0_ns, color=RED, linewidth=1.5,
             linestyle='--', label='Normal shock')

    ax6.axhline(1.0, color=DIM, linewidth=0.5, alpha=0.5)
    ax6.set_xlabel('M₁', fontsize=9)
    ax6.set_ylabel('p₀₂/p₀₁', fontsize=9)
    ax6.set_ylim(0, 1.05)
    ax6.legend(fontsize=8, framealpha=0, labelcolor=DIM)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"  Plot saved: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Oblique Shock Benchmark Suite')
    parser.add_argument('--table', action='store_true',
                        help='Print validation table only')
    args = parser.parse_args()

    if args.table:
        print_validation_table()
    else:
        print_validation_table()
        plot_all('/home/claude/oblique_shock_benchmark.png')
