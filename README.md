# Extreme Thermal Analysis — Aerospace Engineering Toolkit

A suite of Python tools for preliminary thermodynamic and aerothermal analysis
across hypersonic propulsion, re-entry vehicles, motorsport, spacecraft, and
fusion energy. Built as first-principles engineering models — not wrappers
around existing solvers.

All tools are validated against published experimental and analytical reference
data. All known limitations are stated explicitly.

---

## Tools

| Tool | Domain | Key Physics | Target |
|---|---|---|---|
| `sabre_precooler.py` | Pre-cooled propulsion | ε-NTU, ISA, frost deposition | Frazer-Nash INVICTUS, Rolls-Royce HVX |
| `trajectory_integrator.py` | Hypersonic systems | Trajectory integration, mission analysis | SABRE/INVICTUS, Hermeus Chimera |
| `hypersonic_glide_aero.py` | Re-entry vehicles | DKR heating, ablation, point-mass trajectory | Hypersonica HS1, Castelion Blackbeard |
| `f1_brake_thermal.py` | Motorsport | Transient lumped-cap, jet impingement, oxidation | Cadillac F1, Williams, Ricardo |
| `spacecraft_thermal.py` | Orbital thermal | Orbital mechanics, radiative balance | SSTL, Open Cosmos |
| `oblique_shock_benchmark.py` | Compressible flow | θ-β-M, Prandtl-Meyer, shock polar | Zenotech FLITE3D, CFD validation |
| `haps_thermal.py` | Stratospheric platforms | Solar irradiance, buoyancy, energy harvest | Voltitude, BAE PHASA-35 |
| `cmc_thermal.py` | Advanced materials | 1D conduction, Maxwell porosity, TPS sizing | Cross Manufacturing, GKN |
| `divertor_thermal.py` | Nuclear fusion | Resistance network, Dittus-Boelter, fatigue | Tokamak Energy, UKAEA |
| `lh2_propulsion_budget.py` | LH2 propulsion | Rayleigh flow, H2 combustion flame temp, range | Destinus, Rolls-Royce HVX |
| `aerospike_thermal.py` | Aerospike rocket engines | Bartz gas convection, 1D Inconel wall conduction | Polaris Spaceplanes, Nammo |
| `rdre_thermal.py` | Rotating Detonation Rockets | CJ shock physics, 1D transient explicit FDM | Venus Aerospace, Castelion |
| `aero_benchmark.py` | Aerothermal validation | Fay & Riddell, Apollo 11 path, shock tube, PICA | Zenotech, Engys, BAE AI |
| `auv_thermal.py` | Subsea AUV/UUV systems | Churchill-Bernstein convection, diving transient | Sonardyne, SEA Ltd, MSubs |
| `scramjet_regenerative.py` | Scramjet cooling | Eckert ref temp, Sieder-Tate H2, 1D co-flow | Rolls-Royce HVX, Hermeus |
| `spacecraft_adcs.py` | Spacecraft ADCS | Quaternions, B-dot coils, reaction wheels PID | SSTL, Clyde Space, Open Cosmos |
| `truss_fem_solver.py` | Structural Truss | 2D FEM assembly, stress margin, Euler buckling | SpaceX structures, Mott MacDonald |
| `aerodynamic_panel_method.py` | Potential Flow | Hess-Smith panel method, Kutta condition, Cl sweeps | F1 Aerodynamics, Vestas, Boeing |

---

## Quick start

```bash
git clone https://github.com/[username]/aerospace-thermal-toolkit
cd aerospace-thermal-toolkit
pip install numpy scipy matplotlib
```

Run any tool:
```bash
python sabre_precooler.py              # full envelope plots
python sabre_precooler.py --point      # design point table
python hypersonic_glide_aero.py        # both re-entry scenarios
python f1_brake_thermal.py --monaco    # Monaco brake analysis
python spacecraft_thermal.py --point   # coating comparison table
python oblique_shock_benchmark.py      # compressible flow suite
```

### Interactive Web Simulations
Double-click or serve [websims/index.html](file:///c:/Users/matty/OneDrive/Desktop/Matty/Tools/websims/index.html) in any browser to open the interactive animations dashboard. You can dynamically adjust engineering parameters via control sliders and watch physics-accurate fluid particles, thermal gradients, shock polar charts, and subcooled boiling bubbles update in real-time.

---

## Tool descriptions

### `sabre_precooler.py` — SABRE-Style Precooler

Thermodynamic performance of a pre-cooled air-breathing engine heat exchanger
across Mach 0.5–5 flight envelope.

**Physics:**
- ISA multi-layer atmosphere (0–80 km)
- Isentropic ram compression: $T_t = T_s(1 + \frac{\gamma-1}{2}M^2)$
- ε-NTU effectiveness: crossflow HX, both fluids unmixed
- LH₂ heat sink budget: energy balance across 20 K → 250 K range
- Frost risk: vapour pressure driving force $\dot{m}_{frost} \propto (p_v - p_{sat,wall})$
- Dew point: bisection on $p_{sat}(T)$, valid to 100 K

**Design point (Mach 5, 25 km):**
```
Ram temperature   : 1330 K  (1057°C)
Heat load         : 163 MW
HX effectiveness  : 95.8%
LH2 precool flow  : 49.5 kg/s
Frost risk        : 0.000  (stratosphere is essentially dry)
```

**Outputs (7 panels):** Ram temperature, heat load, LH₂ flow, HX effectiveness,
frost risk contour (Mach vs humidity), frost margin vs altitude, NTU sensitivity.

---

### `hypersonic_glide_aero.py` — Hypersonic Glide Vehicle Aerothermal

Re-entry trajectory integration with stagnation point heating, wall temperature,
and TPS ablation sizing.

**Physics:**
- 2D point-mass equations of motion (lift, drag, gravity)
- Detra-Kemp-Riddell stagnation heating: $q_s = \frac{1.83 \times 10^{-4}}{\sqrt{R_n}} \sqrt{\frac{\rho}{\rho_{SL}}} V^3$ [W/m²]
- Radiation equilibrium wall temperature: $q_{conv} = \sigma \varepsilon T_w^4$
- Lees heating distribution along body
- Simple char ablation: $\dot{m} = q_w / (h_v + c_{p,char} \Delta T)$

**Scenarios:**

| Vehicle | Entry conditions | Peak q [W/cm²] | Peak T_wall [K] | TPS required |
|---|---|---|---|---|
| HS1 Prototype (Hypersonica) | Mach 6, 30 km, -5° | 61 | 1884 | 5.9 mm |
| Full-scale glide vehicle | Mach 8, 50 km, -3° | 304 | 2818 | 47 mm |

**Limitations:** Laminar heating only, no turbulent transition, no real-gas
effects (significant above Mach 8–10), no radiative shock layer heating.

---

### `f1_brake_thermal.py` — F1 Carbon-Carbon Brake Disc

Transient thermal model of an F1 brake disc during a race lap.

**Physics:**
- Lumped capacitance with temperature-dependent $c_p$ and $k$
- Braking heat input: parameterised lap profiles (Silverstone + Monaco)
- Jet impingement cooling: $Nu = 0.5 \, Re^{0.5} Pr^{0.4}$
- Radiation to surroundings
- Arrhenius oxidation: $\dot{m} = A \exp(-E_a/RT)$, onset at 450°C

**Key finding:** F1 discs routinely exceed 750°C in dry conditions —
the conventional oxidation threshold. The operating window is actually
600–1200°C (with N₂ purge above 750°C). The tool shows quantitatively
why nitrogen duct purging is necessary.

**Monaco vs Silverstone:** Peak temperature difference ~470°C.
Different duct sizing strategy required for each circuit.

---

### `spacecraft_thermal.py` — Orbital Thermal Analysis

Satellite temperature over multiple orbital periods as a function of coating,
altitude, and internal dissipation.

**Physics:**
- Orbital period and eclipse fraction from circular orbit geometry
- Solar flux, Earth albedo, Earth IR heat loads
- Radiation to deep space (only cooling path)
- Lumped capacitance transient ODE

**Coatings database:** white paint, black paint, OSR, gold plating,
aluminised Kapton, bare aluminium — with absorptivity and emissivity.

**Key finding:** A 3U CubeSat with only 5 W internal dissipation runs
*colder* with white paint (T_min = -28°C) than with black paint (T_min = +1°C)
because solar absorption dominates. Black paint is optimal for low-power
small satellites.

---

### `oblique_shock_benchmark.py` — Compressible Flow Reference Solutions

Exact analytical solutions for oblique shocks and Prandtl-Meyer expansions.
Reference dataset for CFD solver validation.

**Physics:**
- θ-β-M relation (implicit, solved by bisection)
- Full Rankine-Hugoniot relations across oblique shock
- Prandtl-Meyer function $\nu(M)$ (exact)
- Shock polar (velocity hodograph)
- Detachment chart: M₁ vs θ_max

**Validation table (NACA Report 1135):**

| Case | M₁ | θ [°] | β [°] | M₂ | p₂/p₁ | T₂/T₁ | p₀₂/p₀₁ |
|---|---|---|---|---|---|---|---|
| A | 2.0 | 10 | 39.31 | 1.640 | 1.707 | 1.170 | 0.9846 |
| B | 3.0 | 15 | 32.24 | 2.255 | 2.822 | 1.388 | 0.8950 |
| C | 5.0 | 20 | 29.80 | 3.022 | 7.037 | 2.123 | 0.5051 |
| Normal M=2 | 2.0 | 90° | — | 0.577 | 4.500 | 1.688 | 0.7209 |
| Normal M=5 | 5.0 | 90° | — | 0.415 | 29.00 | 5.800 | 0.0617 |

Results agree with NACA 1135 to 4+ significant figures.

---

### `haps_thermal.py` — Stratospheric HAPS Thermal Energy Budget

Diurnal thermodynamic and energy balance of solar-powered High Altitude Platform Stations.

**Physics:**
- Solar zenith angle tracking (latitude, season, time of day)
- Helium gas expansion buoyancy tracking and diurnal altitude excursion
- GaAs solar panel harvest and Li-S battery transient state-of-charge (SOC)
- Sealed electronics pod heat dissipation via radiation and natural convection

---

### `cmc_thermal.py` — Ceramic Matrix Composite (CMC) TPS Thermal Calculator

Transient thermal and structural analysis of ceramic matrix composite thermal protection systems.

**Physics:**
- 1D transient heat conduction using Crank-Nicolson implicit finite difference
- Maxwell effective thermal conductivity model for porous materials
- 1D thermal stress estimation and material safety factors
- Coffin-Manson thermal fatigue analysis

---

### `divertor_thermal.py` — Tokamak Divertor Thermal Exhaust Calculator

Steady-state thermal-hydraulic and fatigue analysis of a tokamak divertor heat exhaust system.

**Physics:**
- 1D radial/Cartesian thermal resistance circuit (W tile to CuCrZr tube)
- Dittus-Boelter flow heat transfer with subcooled boiling McNaught correction
- Darcy-Weisbach pressure drop and Blasius turbulent friction factor
- Coffin-Manson low-cycle thermal fatigue model for CuCrZr alloy

---

### `lh2_propulsion_budget.py` — LH2 Propulsion Thermal Budget Calculator

Systems-level thermal-hydraulic and range analysis of liquid hydrogen-powered hypersonic aircraft.

**Physics:**
- H2 combustion flame temperature calculations with dissociation limits
- Rayleigh flow afterburner subsonic expansion and pressure loss
- Fay-Riddell leading edge stagnation point heat loads
- Integrated Breguet range estimates for hydrogen fuel systems

---

### `aerospike_thermal.py` — Linear Aerospike central spike sizing

Conjugate heat transfer analysis of a linear aerospike rocket engine central spike.

**Physics:**
- Rocket expansion thermodynamics (LOX/RP-1 products isentropic expansion)
- Convective heat transfer coefficient calculations via the Bartz correlation
- 1D conduction through Inconel spike wall to internal cooling channels
- Kerosene coolant convective heat transfer (Dittus-Boelter) and pressure drop

---

### `rdre_thermal.py` — RDRE Combustor Wall Thermal Solver

Transient 1D explicit thermal conduction and fatigue analysis of a Rotating Detonation Rocket Engine (RDRE).

**Physics:**
- Chapman-Jouguet detonation wave velocities and rotation frequency models
- Rankine-Hugoniot shock relations for post-detonation states
- Dynamic rotating detonation wave heat flux profile generator
- Transient 1D explicit finite difference heat conduction solver (Inconel 625)

---

### `aero_benchmark.py` — Aerothermal Validation & Benchmarking Suite

Rigorous validation suite comparing the toolkit's aerothermal models against experimental and flight databases.

**Physics:**
- Fay & Riddell stagnation heating comparison
- Apollo 11 Command Module re-entry trajectory integration (V0=11km/s)
- Shock tube Rankine-Hugoniot exact analytical validation (Mach 8)
- Flat plate radiation-equilibrium wall temperature prediction
- PICA material ablation test comparisons

---

### `auv_thermal.py` — AUV Pressure Vessel Thermal Calculator

Steady-state and transient thermal analysis of a sealed Autonomous Underwater Vehicle pressure vessel.

**Physics:**
- Seawater depth-temperature profile model (thermocline modeling)
- Conduction through TIM pad vs internal natural convection
- Churchill-Bernstein forced convection over a cylinder
- Transient 4-hour dive cycle simulation and surfacing condensation dew-point risk

---

### `scramjet_regenerative.py` — Scramjet Regenerative Cooling 1D Solver

Simulates the coupled heat transfer between a supersonic combustion gas path and a supercritical liquid hydrogen (LH2) regenerative cooling jacket.

**Physics:**
- Supersonic gas path convection using Eckert's reference temperature/enthalpy method
- 1D marching integration along the combustor axial length (co-flow arrangement)
- Convective heat transfer coefficient in microchannels via Sieder-Tate correlation for turbulent supercritical flow
- Temperature-dependent supercritical hydrogen properties ($c_p$, viscosity, conductivity)
- Radial resistance network across wall materials (GRCop-84 vs Inconel 718)

**Design Point (1.0 kg/s total coolant flow):**
```
Material           Coolant [kg/s]    T_wall_max    T_cool_out    Peak q [MW/m2]      Status
GRCop-84                    1.00        471.6K        133.2K            13.45          OK
Inconel 718                 1.00       1235.7K        121.9K            11.24          OK
```

---

### `spacecraft_adcs.py` — Spacecraft Attitude Dynamics & Control System (ADCS)

Simulates 3D rotational kinematics (quaternions) and dynamics (Euler's equations) of a small satellite, implementing detumbling B-dot control and precision reaction wheel pointing.

**Physics:**
- Quaternion attitude propagation: $\dot{q} = \frac{1}{2} q \otimes \omega$
- Rigid body dynamics (Euler's equations of motion)
- B-dot electromagnetic coil detumbling control: $\vec{m} = -K_{bdot} \dot{\vec{B}}$
- Reaction wheel PID precision nadir-pointing controller

**Performance Output:**
- Detumbling recovery: Damps rates from initial chaotic spin (0.8 rad/s) to stabilized rest within 250s (>99% kinetic energy reduction).
- Pointing convergence: Reaction wheels reach sub-degree pointing error with smooth transient settling times.

---

### `truss_fem_solver.py` — 2D Truss FEM Solver & Sizing Optimizer

Analyzes axial loads, stresses, displacements, and Euler buckling margins across a 2D truss, executing an iterative sizing loop to minimize structural weight.

**Physics:**
- Element stiffness matrix global coordination assembly
- Reduced system solution ($K_{reduced} U_a = F_a$)
- Strain ($\epsilon = \Delta L / L$) and stress ($\sigma = E \epsilon$) evaluation
- Euler critical buckling load limit for compressive elements: $P_{crit} = \frac{\pi^2 E I}{L^2}$
- Optimization loop: member resizing under stress and buckling safety constraints

**Optimization Results (50 kN tip load, Aluminum 6061-T6):**
- Baseline (uniform 1000 mm² sections): Total mass = 153.9 kg, Max deflection = 0.58 mm, Worst Buckling Ratio = 0.12.
- Optimized (sizing sweep): Total mass = 45.2 kg (70.6% weight savings), Max deflection = 1.34 mm, stress and buckling limits satisfied with safety factor = 1.5.

---

### `aerodynamic_panel_method.py` — NACA Airfoil 2D Potential Flow Panel Solver

Implements the Hess-Smith Panel Method (Source panels + constant vortex) to solve potential flow field velocity vectors, pressure coefficients, and lift coefficients for any NACA 4-digit airfoil.

**Physics:**
- NACA 4-digit coordinate generation with cosine node distribution
- Singular influence coefficient matrix satisfying zero normal velocity at panel control points
- Trailing-edge Kutta condition enforcement ($V_{t,1} + V_{t,N} = 0$)
- Surface pressure distribution integration to calculate circulation $\Gamma$ and lift coefficient $C_l$

**Performance Results (NACA 2412 Airfoil, 60 Panels):**
- Lift curve slope $dC_l/d\alpha \approx 0.108\text{/deg}$ (agrees closely with the thin-airfoil theory limit of $2\pi\text{ rad}^{-1} \approx 0.110\text{/deg}$).
- Pressure coefficient distribution shows leading-edge suction peak on upper surface at positive angles of attack.

---

## Design philosophy

**Fast iteration over accuracy.** These are preliminary design tools, not
certified simulation software. Each tool can run a full analysis in seconds.
The goal is to bound the design space and identify the key sensitivities
before committing to CFD or experimental testing.

**Physics first.** Every equation is commented with its source reference.
Every correlation has a stated range of validity. Every tool has an explicit
limitations section.

**Honest uncertainty.** When a model is known to underpredict or overpredict,
that's stated. The DKR heating correlation was fit to high-speed re-entry data
and may over-predict at low Mach numbers. The lumped capacitance models
ignore spatial gradients. These are not bugs — they're documented choices.

**Consistent interface.** All tools use the same dark-theme matplotlib style,
the same argparse CLI pattern (`--point` for tables, default for plots),
and the same unit conventions (T in Kelvin, distances in metres, time in seconds,
with units in all variable names).

---

## Requirements

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
```

Python 3.10+ recommended.

---

## Author

**MKA** — Mechanical Engineering student, UK.

Competition mathematics: [@pringlesmaths](https://instagram.com/pringlesmaths)
Website: [pringlesmaths.co.uk](https://pringlesmaths.co.uk)
Contact: marchinmaths@gmail.com

These tools were built to understand specific engineering problems at companies
working on hypersonic propulsion, fusion energy, motorsport, and spacecraft.
If you're one of those engineers and find something wrong or interesting,
I'd genuinely like to hear from you.

---

## Citing

If you use these tools for anything formal, a mention is appreciated but not
required. All code is MIT licensed.

```bibtex
@misc{mka_aerospace_thermal_toolkit_2026,
  author = {MKA},
  title  = {Aerospace Thermal Analysis Toolkit},
  year   = {2026},
  url    = {https://github.com/[username]/aerospace-thermal-toolkit}
}
```
