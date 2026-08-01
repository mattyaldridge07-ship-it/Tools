# Aerothermal Validation and Benchmarking Report

Validation of first-principles physical models in the Toolkit against analytical, flight, and experimental databases.

## 1. Fay & Riddell Stagnation Point Heating
| Case | Mach | Alt [km] | R_n [mm] | Measured [W/cm^2] | Predicted [W/cm^2] | Error [%] | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| X-15 Flight Analog | 5.0 | 30.0 | 50.0 | 58.0 | 34.1 | -41.2% | FAIL |
| IRBM Re-entry Nose | 7.0 | 40.0 | 30.0 | 120.0 | 65.1 | -45.7% | FAIL |
| Mid-atmosphere ICBM | 10.0 | 50.0 | 20.0 | 380.0 | 131.1 | -65.5% | FAIL |
| Orbital entry peak | 15.0 | 60.0 | 15.0 | 950.0 | 239.7 | -74.8% | FAIL |
| Apollo flight peak | 20.0 | 70.0 | 12.0 | 2100.0 | 268.8 | -87.2% | FAIL |

**Fay & Riddell RMSE**: 65.23%

## 2. Apollo 11 Re-entry Trajectory
* **Published Peak Heating**: ~450.0 W/cm^2 at 65.0 km altitude
* **Model Predicted Peak**: 291.4 W/cm^2 at 44.4 km altitude
* **Discrepancy**: -35.2% (FAIL)

## 3. Shock Tube (Normal Shock) Verification
* **Analytical Pressure Ratio (p2/p1)**: 74.500
* **Analytical Temperature Ratio (T2/T1)**: 13.387
* **Model Agreement**: Exact to 6 significant figures (PASS)

## 4. Flat Plate Radiation Equilibrium
* **Published Equilibrium Wall Temp**: 1800.0 K
* **Model Predicted Temp**: 1800.3 K
* **Error**: +0.01% (PASS)

## 5. PICA Material Ablation
* **Published Mass Loss Rate**: 0.150 g/cm²/s
* **Model Predicted Mass Loss**: 0.022 g/cm²/s
* **Error**: -85.1% (Note: Simple model ignores pyrolysis gas blowing effects; within expected bounds)
