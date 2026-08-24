"""
diurnal thermal and energy equilibrium simulator for high-altitude pseudo-satellites (HAPS)
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

def trapezoid_integrate(y, x):
    """numpy-version-independent trapezoidal integration helper"""
    return np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x))

# physical constants
SIGMA_SB    = 5.6704e-8     # W/m²/K⁴   Stefan-Boltzmann constant
R_EARTH_m   = 6.371e6       # m         Earth radius
S0_Wm2      = 1361.0        # W/m²      solar constant (top of atmosphere)
T_EARTH_K   = 255.0         # K         effective Earth radiating temperature
G0_ms2      = 9.80665       # m/s²      gravity acceleration
R_AIR       = 287.058       # J/(kg·K)  specific gas constant, air
GAMMA       = 1.4           # -         specific heat ratio, air
CP_AIR      = 1005.0        # J/(kg·K)  specific heat, air

# colour palette
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

def isa_atmosphere(altitude_m):
    """international standard atmosphere up to 80 km, returns (T_K, p_Pa, rho_kgm3, a_ms)"""
    layers = [
        (0,       11000, -0.0065, 288.15, 101325.0),
        (11000,   20000,  0.0,    216.65,  22632.1),
        (20000,   32000,  0.001,  216.65,   5474.89),
        (32000,   47000,  0.0028, 228.65,    868.019),
        (47000,   51000,  0.0,    270.65,    110.906),
        (51000,   71000, -0.0028, 270.65,     66.9389),
        (71000,   86000, -0.002,  214.65,      3.95642),
    ]
    h = np.clip(altitude_m, 0.0, 85000.0)
    T_base, p_base, lapse, h_base = None, None, None, 0.0
    for (h0, h1, lr, T0, p0) in layers:
        if h <= h1:
            T_base, p_base, lapse, h_base = T0, p0, lr, h0
            break
    if T_base is None:
        T_base, p_base, lapse, h_base = 214.65, 3.95642, 0.0, 71000.0

    dh = h - h_base
    if abs(lapse) < 1e-10:
        T_K = T_base
        p_Pa = p_base * np.exp(-G0_ms2 * dh / (R_AIR * T_base))
    else:
        T_K = T_base + lapse * dh
        p_Pa = p_base * (T_K / T_base) ** (-G0_ms2 / (lapse * R_AIR))

    rho_kgm3 = p_Pa / (R_AIR * T_K)
    a_ms = np.sqrt(GAMMA * R_AIR * T_K)
    return T_K, p_Pa, rho_kgm3, a_ms

def altitude_from_density(rho_target_kgm3):
    """altitude matching a target air density, for buoyancy tracking"""
    def f(z_m):
        return isa_atmosphere(z_m)[2] - rho_target_kgm3
    try:
        return brentq(f, 0.0, 50000.0)
    except ValueError:
        return 20000.0

def solar_flux(lat_deg, day_of_year, t_hours, altitude_km):
    """
    direct solar flux on upper horizontal surface at stratospheric altitude
    ref: Duffie & Beckman (2013), Solar Engineering of Thermal Processes
    """
    S_toa_Wm2 = S0_Wm2
    tau = 0.98 if altitude_km >= 20.0 else 0.95 + 0.0015 * altitude_km  # little air above at stratospheric alt

    delta_rad = np.radians(23.45 * np.sin(2.0 * np.pi * (day_of_year - 81.0) / 365.0))  # declination angle
    lat_rad = np.radians(lat_deg)

    hour_angle_rad = np.radians(15.0 * (t_hours - 12.0))

    cos_theta = np.sin(lat_rad) * np.sin(delta_rad) + np.cos(lat_rad) * np.cos(delta_rad) * np.cos(hour_angle_rad)
    cos_theta = np.clip(cos_theta, 0.0, 1.0)

    G_solar_Wm2 = S_toa_Wm2 * tau * cos_theta
    return G_solar_Wm2, cos_theta

def solve_t_hull(G_solar_Wm2, h_m, alpha, eps_outer, T_atm_K, h_conv_Wm2K=3.0):
    """
    steady-state hull temperature via 1D radiation and convection balance:
    alpha*G_solar + eps_outer*G_IR_earth = eps_outer*sigma*T_hull^4 + h_conv*(T_hull - T_atm)
    """
    G_IR_earth_Wm2 = SIGMA_SB * T_EARTH_K**4 * (R_EARTH_m / (R_EARTH_m + h_m))**2

    def f(T):
        q_in = alpha * G_solar_Wm2 + eps_outer * G_IR_earth_Wm2
        q_out = eps_outer * SIGMA_SB * T**4 + h_conv_Wm2K * (T - T_atm_K)
        return q_out - q_in

    try:
        T_hull_K = brentq(f, 100.0, 450.0)
    except ValueError:
        T_hull_K = T_atm_K
    return T_hull_K

def electronics_pod_thermal(P_elec_W, T_atm_K, A_pod_m2=0.5, eps_pod=0.80, P_heater_W=0.0):
    """electronics junction/pod temperature in a sealed stratospheric vessel, radiation-dominated cooling"""
    T_limit_hot_C = 85.0
    T_limit_cold_C = -20.0

    # radiation balance: P_elec + P_heater = eps_pod*sigma*A_pod*(T_pod^4 - T_atm^4)
    def f(T_pod_K):
        return P_elec_W + P_heater_W - eps_pod * SIGMA_SB * A_pod_m2 * (T_pod_K**4 - T_atm_K**4)

    try:
        T_pod_K = brentq(f, 150.0, 450.0)
    except ValueError:
        T_pod_K = T_atm_K

    T_pod_C = T_pod_K - 273.15
    status = "OK"
    if T_pod_C > T_limit_hot_C:
        status = "OVERHEATED"
    elif T_pod_C < T_limit_cold_C:
        status = "TOO COLD"

    T_target_cold_K = 253.15  # -20°C
    Q_heater_min_W = max(0.0, eps_pod * SIGMA_SB * A_pod_m2 * (T_target_cold_K**4 - T_atm_K**4) - P_elec_W)

    return T_pod_C, Q_heater_min_W, status

def simulate_diurnal_cycle(lat_deg, day_of_year, p_payload_W=150.0, p_avionics_W=100.0,
                           coating_type='white', design_alt_km=20.0):
    """full 24-hour cycle for HAPS thermal and energy systems"""
    t_hours = np.linspace(0.0, 24.0, 288)  # 5-minute intervals
    h_design_m = design_alt_km * 1e3
    T_atm_design_K, _, rho_atm_design_kgm3, _ = isa_atmosphere(h_design_m)

    if coating_type == 'white':
        alpha = 0.15
        eps_outer = 0.85
    elif coating_type == 'dark':
        alpha = 0.85
        eps_outer = 0.85
    else:  # aluminised
        alpha = 0.15
        eps_outer = 0.05

    T_hull_arr = []
    T_He_arr = []
    alt_arr = []
    P_solar_arr = []

    # 10m diameter balloon reference case, for buoyancy excursion
    V_balloon_design_m3 = 524.0
    M_balloon_dry_kg = 40.0
    T_He_design_K = T_atm_design_K  # thermal equilibrium design temperature

    for th in t_hours:
        G_sol, _ = solar_flux(lat_deg, day_of_year, th, design_alt_km)

        T_hull_K = solve_t_hull(G_sol, h_design_m, alpha, eps_outer, T_atm_design_K, h_conv_Wm2K=2.5)
        T_He_K = T_hull_K * 0.85 + T_atm_design_K * 0.15  # helium tracks hull, mixes a bit with atmosphere

        # buoyancy: volume expands with gas temp, rho_atm(z) * T_He(t) = constant
        rho_target_kgm3 = rho_atm_design_kgm3 * (T_He_design_K / T_He_K)
        alt_excursion_m = altitude_from_density(rho_target_kgm3)

        T_hull_arr.append(T_hull_K - 273.15)
        T_He_arr.append(T_He_K - 273.15)
        alt_arr.append(alt_excursion_m / 1000.0)

        A_solar_m2 = 15.0  # PHASA-35-class
        eta_solar = 0.24   # GaAs solar panel efficiency
        P_solar_W = G_sol * A_solar_m2 * eta_solar
        P_solar_arr.append(P_solar_W)

    T_hull_arr = np.array(T_hull_arr)
    T_He_arr = np.array(T_He_arr)
    alt_arr = np.array(alt_arr)
    P_solar_arr = np.array(P_solar_arr)

    P_load_W = p_payload_W + p_avionics_W
    E_harvest_Wh = trapezoid_integrate(P_solar_arr, t_hours)

    dt_hours = t_hours[1] - t_hours[0]
    E_required_Wh = P_load_W * 24.0

    # battery capacity sized by night survivability, assumes 300 Wh/kg (Li-S)
    is_daylight = P_solar_arr > P_load_W
    E_night_load_Wh = 0.0
    for i, is_day in enumerate(is_daylight):
        if not is_day:
            E_night_load_Wh += P_load_W * dt_hours

    safety_factor = 1.2
    E_battery_cap_Wh = E_night_load_Wh * safety_factor
    m_battery_kg = E_battery_cap_Wh / 300.0

    SOC_arr = []
    E_stored_Wh = E_battery_cap_Wh * 0.8  # start at 80% charge at midnight

    eta_charge = 0.92
    eta_discharge = 0.95

    for P_sol in P_solar_arr:
        P_net = P_sol - P_load_W
        if P_net > 0:
            E_stored_Wh += P_net * dt_hours * eta_charge
        else:
            E_stored_Wh += P_net * dt_hours / eta_discharge

        E_stored_Wh = np.clip(E_stored_Wh, 0.0, E_battery_cap_Wh)
        SOC_arr.append(E_stored_Wh / E_battery_cap_Wh * 100.0)

    SOC_arr = np.array(SOC_arr)
    margin = E_harvest_Wh - E_required_Wh

    T_pod_arr = []
    P_heater_arr = []
    for T_atm_C in (T_hull_arr * 0.1 + T_He_arr * 0.1 - 56.0 * 0.8):  # effective ambient temp profile
        T_atm_K = T_atm_C + 273.15
        # heaters activate if T_pod drops below -20°C
        T_p, Q_heat, _ = electronics_pod_thermal(p_payload_W, T_atm_K, A_pod_m2=0.5, eps_pod=0.8)
        T_pod_arr.append(T_p)
        P_heater_arr.append(Q_heat)

    return {
        't_hours':          t_hours,
        'T_hull_C':         T_hull_arr,
        'T_He_C':           T_He_arr,
        'altitude_km':      alt_arr,
        'P_solar_W':        P_solar_arr,
        'SOC_pct':          SOC_arr,
        'T_pod_C':          np.array(T_pod_arr),
        'P_heater_W':       np.array(P_heater_arr),
        'E_harvest_Wh':     E_harvest_Wh,
        'E_required_Wh':    E_required_Wh,
        'battery_mass_kg':  m_battery_kg,
        'margin_Wh':        margin
    }

def print_report(lat_deg=51.5, day_of_year=172):
    print()
    print("=" * 90)
    print("  STRATOSPHERIC HAPS THERMAL & ENERGY CALCULATOR — REFERENCE SCENARIO")
    print(f"  Latitude: {lat_deg} deg N (UK) | Summer Solstice (Day {day_of_year:03d})")
    print("=" * 90)

    configs = [
        ('white', 'White Coating (Solar Reflective)'),
        ('dark', 'Dark Coating (Heat Absorptive)'),
    ]

    print(f"  {'Coating':15}  {'T_hull_max':>11}  {'T_hull_min':>11}  "
          f"{'dAlt_diurnal':>13}  {'E_harvest':>10}  {'E_required':>11}  {'Margin':>9}")
    print("-" * 90)

    for ctype, label in configs:
        res = simulate_diurnal_cycle(lat_deg, day_of_year, coating_type=ctype)
        t_max = res['T_hull_C'].max()
        t_min = res['T_hull_C'].min()
        da = res['altitude_km'].max() - res['altitude_km'].min()
        print(f"  {ctype.capitalize():15}  {t_max:9.1f} deg C  {t_min:9.1f} deg C  "
              f"{da:11.2f} km  {res['E_harvest_Wh']:8.1f}Wh  "
              f"{res['E_required_Wh']:9.1f}Wh  {res['margin_Wh']:+7.1f}Wh")

    print("=" * 90)
    print("  PHASA-35 REFERENCE METRICS (White coating, June solstice, 51.5°N):")
    res_phasa = simulate_diurnal_cycle(51.5, 172, coating_type='white')
    print(f"    Battery mass required: {res_phasa['battery_mass_kg']:.2f} kg (Li-S 300 Wh/kg)")
    print(f"    Electronics Pod peak : {res_phasa['T_pod_C'].max():.1f} deg C (Limit: 85 deg C)")
    print(f"    Electronics Pod min  : {res_phasa['T_pod_C'].min():.1f} deg C (Limit: -20 deg C)")
    print(f"    Peak Heater Power req: {res_phasa['P_heater_W'].max():.1f} W")
    print("=" * 90)
    print()

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

    fig.text(0.5, 0.975, 'STRATOSPHERIC HAPS THERMAL & ENERGY CALCULATOR',
             ha='center', va='top', color=PAPER, fontsize=13, fontweight='bold',
             fontfamily='monospace')
    fig.text(0.5, 0.960,
             'Preliminary diurnal thermal excursion and solar power system sizing tool for 20 km LEO equivalent',
             ha='center', va='top', color=DIM, fontsize=8.5, fontfamily='monospace')

    res_white_s = simulate_diurnal_cycle(51.5, 172, coating_type='white')
    res_white_w = simulate_diurnal_cycle(51.5, 355, coating_type='white')
    res_dark_s  = simulate_diurnal_cycle(51.5, 172, coating_type='dark')

    t = res_white_s['t_hours']

    # panel 1: irradiance & hull temperature
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'SOLAR FLUX & HULL TEMPERATURE (51.5°N)')
    ax1.plot(t, res_white_s['T_hull_C'], color=CYAN, linewidth=2.0, label='White Hull (Summer)')
    ax1.plot(t, res_dark_s['T_hull_C'], color=RED, linewidth=2.0, label='Dark Hull (Summer)')
    ax1.plot(t, res_white_w['T_hull_C'], color=BLUE, linewidth=1.5, linestyle='--', label='White Hull (Winter)')

    ax1_r = ax1.twinx()
    ax1_r.set_facecolor(BG)
    G_sol_s = [solar_flux(51.5, 172, th, 20.0)[0] for th in t]
    ax1_r.plot(t, G_sol_s, color=GOLD, linewidth=1.0, alpha=0.3, label='Solar Irradiance')
    ax1_r.tick_params(colors=DIM, labelsize=9)
    ax1_r.set_ylabel('Solar Irradiance  [W/m²]', color=DIM, fontsize=9)
    ax1_r.spines['right'].set_color(GREY)

    ax1.set_xlabel('Time of Day  [hours]', fontsize=9)
    ax1.set_ylabel('Hull Temperature  [°C]', fontsize=9)
    ax1.set_xlim(0, 24)
    ax1.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper left')

    # panel 2: helium gas & altitude excursion
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'HELIUM TEMPERATURE & ALTITUDE EXCURSION')
    ax2.plot(t, res_white_s['altitude_km'], color=CYAN, linewidth=2.0, label='White Coating')
    ax2.plot(t, res_dark_s['altitude_km'], color=RED, linewidth=2.0, label='Dark Coating')

    ax2_r = ax2.twinx()
    ax2_r.set_facecolor(BG)
    ax2_r.plot(t, res_white_s['T_He_C'], color=GOLD, linewidth=1.5, linestyle='-.', label='Helium Temp')
    ax2_r.tick_params(colors=GOLD, labelsize=9)
    ax2_r.set_ylabel('Helium Gas Temperature  [°C]', color=GOLD, fontsize=9)
    ax2_r.spines['right'].set_color(GREY)

    ax2.set_xlabel('Time of Day  [hours]', fontsize=9)
    ax2.set_ylabel('Altitude  [km]', fontsize=9)
    ax2.set_xlim(0, 24)
    ax2.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper left')
    ax2_r.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper right')

    # panel 3: solar harvest & battery SOC
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'ENERGY SYSTEM DIURNAL PROFILE')
    ax3.plot(t, res_white_s['P_solar_W'], color=GOLD, linewidth=2.0, label='Solar Harvest [W]')
    ax3.fill_between(t, 250.0, 0.0, alpha=0.08, color=RED, label='Continuous load (250W)')

    ax3_r = ax3.twinx()
    ax3_r.set_facecolor(BG)
    ax3_r.plot(t, res_white_s['SOC_pct'], color=MOSS, linewidth=2.0, label='Battery SOC [%]')
    ax3_r.tick_params(colors=MOSS, labelsize=9)
    ax3_r.set_ylabel('Battery State of Charge  [%]', color=MOSS, fontsize=9)
    ax3_r.spines['right'].set_color(GREY)

    ax3.set_xlabel('Time of Day  [hours]', fontsize=9)
    ax3.set_ylabel('Solar Power  [W]', fontsize=9)
    ax3.set_xlim(0, 24)
    ax3.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper left')
    ax3_r.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='upper right')

    # panel 4: solar area vs latitude & season
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, 'MIN SOLAR AREA FOR ENERGY-POSITIVE MISSION')

    lats = np.linspace(0.0, 60.0, 30)
    months = np.arange(1, 13)
    MIN_AREA = np.zeros((len(months), len(lats)))

    P_load_W = 250.0  # continuous PHASA-35 load
    eta_solar = 0.24

    mid_month_days = [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]  # approx mid-month day of year

    for i, m_day in enumerate(mid_month_days):
        for j, lat in enumerate(lats):
            solar_24h = []
            for th in np.linspace(0.0, 24.0, 100):
                G_sol, _ = solar_flux(lat, m_day, th, 20.0)
                solar_24h.append(G_sol)
            E_sol_unit_Wh = trapezoid_integrate(np.array(solar_24h), np.linspace(0.0, 24.0, 100)) * eta_solar

            E_load_total = P_load_W * 24.0 / 0.85  # includes battery roundtrip efficiency
            required_area = E_load_total / E_sol_unit_Wh if E_sol_unit_Wh > 0 else np.nan
            MIN_AREA[i, j] = min(required_area, 50.0)  # capped at 50 m² for plotting

    LL, MM = np.meshgrid(lats, months)
    cf = ax4.contourf(LL, MM, MIN_AREA, levels=np.linspace(2.0, 30.0, 15), cmap='plasma')
    cs = ax4.contour(LL, MM, MIN_AREA, levels=[10.0, 15.0, 20.0, 25.0], colors='white', linewidths=0.5, alpha=0.5)
    ax4.clabel(cs, fmt='%.0f m²', fontsize=8, colors='white')

    cb = plt.colorbar(cf, ax=ax4, pad=0.01)
    cb.set_label('Req. Solar Panel Area  [m²]', color=DIM, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=DIM, labelsize=8)
    cb.outline.set_edgecolor(GREY)

    ax4.set_xlabel('Latitude  [°N]', fontsize=9)
    ax4.set_ylabel('Month of Year', fontsize=9)
    ax4.set_yticks(months)
    ax4.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    # panel 5: payload electronics thermal
    ax5 = fig.add_subplot(gs[2, 0])
    style_ax(ax5, 'PAYLOAD THERMAL BALANCE')
    powers = np.linspace(20.0, 300.0, 30)
    for i, alt in enumerate([15.0, 20.0, 25.0]):
        T_atm_K, _, _, _ = isa_atmosphere(alt * 1e3)
        T_junctions = []
        for P in powers:
            T_j, _, _ = electronics_pod_thermal(P, T_atm_K, A_pod_m2=0.5, eps_pod=0.8)
            T_junctions.append(T_j)
        ax5.plot(powers, T_junctions, color=COLORS[i], linewidth=2.0, label=f'Altitude {alt:.0f} km')

    ax5.axhline(85.0, color=RED, linewidth=1.0, linestyle='--', alpha=0.7)
    ax5.text(powers[0] + 5, 87.0, 'Hot limit: 85°C', color=RED, fontsize=8, fontfamily='monospace')
    ax5.axhline(-20.0, color=CYAN, linewidth=1.0, linestyle='--', alpha=0.7)
    ax5.text(powers[0] + 5, -17.0, 'Cold limit: -20°C', color=CYAN, fontsize=8, fontfamily='monospace')

    ax5.set_xlabel('Payload Power Dissipation  [W]', fontsize=9)
    ax5.set_ylabel('Junction Temperature  [°C]', fontsize=9)
    ax5.legend(fontsize=8, framealpha=0, labelcolor=DIM, loc='lower right')

    # panel 6: platform energy comparison
    ax6 = fig.add_subplot(gs[2, 1])
    style_ax(ax6, 'PLATFORM PERSISTENCE VS POWER CAPACITY')

    platforms = ['Solar HAPS\n(Heavy Class)', 'Lighter-than-Air\nBalloon', 'LEO Small Sat\n(100kg Class)', 'Turboprop UAV\n(MALE Class)']
    specific_energies = [300.0, 0.0, 150.0, 12000.0]  # Wh/kg fuel/battery equivalent
    endurance_days = [365.0, 30.0, 1800.0, 1.5]  # typical operational endurance

    colors_bar = [GOLD, CYAN, MOSS, RED]
    bars = ax6.bar(platforms, endurance_days, color=colors_bar, alpha=0.7, width=0.5)

    for bar, val in zip(bars, endurance_days):
        yval = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2.0, yval + 10,
                 f'{yval:.0f} days' if yval < 365 else f'{yval/365:.1f} yr',
                 ha='center', va='bottom', color=PAPER, fontsize=8, fontfamily='monospace')

    ax6.set_ylabel('Mission Endurance  [days]', fontsize=9)
    ax6.set_yscale('log')
    ax6.set_ylim(0.1, 10000.0)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"Plot saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stratospheric HAPS Thermal & Energy Calculator')
    parser.add_argument('--point', action='store_true', help='Print reference design point tables')
    args = parser.parse_args()

    if args.point:
        print_report()
    else:
        print_report()
        plot_all('haps_thermal.png')
