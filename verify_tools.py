import os
import sys
import subprocess
import time

# List of tools to test, their text-only args, default plotting args, and expected plots
TOOLS_CONFIG = {
    'haps_thermal.py': {
        'text_args': ['--point'],
        'plot_args': [],
        'expected_plots': ['haps_thermal.png']
    },
    'cmc_thermal.py': {
        'text_args': ['--point'],
        'plot_args': [],
        'expected_plots': ['cmc_thermal.png']
    },
    'divertor_thermal.py': {
        'text_args': ['--point'],
        'plot_args': [],
        'expected_plots': ['divertor_thermal.png']
    },
    'lh2_propulsion_budget.py': {
        'text_args': ['--point'],
        'plot_args': [],
        'expected_plots': ['lh2_thermal_budget.png']
    },
    'aerospike_thermal.py': {
        'text_args': ['--point'],
        'plot_args': [],
        'expected_plots': ['aerospike_thermal.png']
    },
    'rdre_thermal.py': {
        'text_args': ['--point'],
        'plot_args': [],
        'expected_plots': ['rdre_thermal.png']
    },
    'aero_benchmark.py': {
        'text_args': ['--point'],
        'plot_args': [],
        'expected_plots': ['validation_report.png']
    },
    'auv_thermal.py': {
        'text_args': ['--point'],
        'plot_args': [],
        'expected_plots': ['auv_thermal.png']
    },
    'sabre_precooler.py': {
        'text_args': ['--point'],
        'plot_args': [],
        'expected_plots': ['sabre_precooler_analysis.png']
    },
    'f1_brake_thermal.py': {
        'text_args': ['--point'],
        'plot_args': [],
        'expected_plots': ['f1_silverstone.png', 'f1_monaco.png', 'f1_comparison.png']
    },
    'hypersonic_glide_aero.py': {
        'text_args': ['--point'],
        'plot_args': [],
        'expected_plots': ['hs1_trajectory.png', 'fullscale_trajectory.png']
    },
    'spacecraft_thermal.py': {
        'text_args': ['--point'],
        'plot_args': [],
        'expected_plots': ['spacecraft_cubesat.png', 'spacecraft_microsat.png']
    },
    'trajectory_integrator.py': {
        'text_args': ['--geometry'],
        'plot_args': [],
        'expected_plots': ['ssto_trajectory.png', 'tbcc_trajectory.png', 'hx_geometry.png']
    },
    'oblique_shock_benchmark.py': {
        'text_args': ['--table'],
        'plot_args': [],
        'expected_plots': ['oblique_shock_benchmark.png']
    },
    'scramjet_regenerative.py': {
        'text_args': ['--point'],
        'plot_args': [],
        'expected_plots': ['scramjet_regenerative.png']
    },
    'spacecraft_adcs.py': {
        'text_args': ['--point'],
        'plot_args': [],
        'expected_plots': ['adcs_detumble.png', 'adcs_pointing.png']
    },
    'truss_fem_solver.py': {
        'text_args': ['--point'],
        'plot_args': [],
        'expected_plots': ['truss_stress.png', 'truss_optimized.png']
    },
    'aerodynamic_panel_method.py': {
        'text_args': ['--point'],
        'plot_args': [],
        'expected_plots': ['airfoil_flow.png', 'lift_curve.png']
    }
}

def clean_plots(plots):
    """Remove expected plots if they already exist to avoid false positives."""
    for plot in plots:
        if os.path.exists(plot):
            try:
                os.remove(plot)
            except Exception as e:
                print(f"  Warning: Could not remove existing file {plot}: {e}")

def run_script(script_name, args):
    """Run a script with arguments and return returncode, stdout, stderr, and duration."""
    cmd = [sys.executable, script_name] + args
    start_time = time.time()
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
        duration = time.time() - start_time
        return res.returncode, res.stdout, res.stderr, duration
    except subprocess.TimeoutExpired as e:
        duration = time.time() - start_time
        return -1, "", f"Timeout after {time.time() - start_time:.1f}s", duration
    except Exception as e:
        duration = time.time() - start_time
        return -1, "", str(e), duration

def main():
    print("=" * 80)
    print("  EXTREME THERMAL ANALYSIS TOOLKIT — VERIFICATION SUITE")
    print("=" * 80)
    
    all_success = True
    results = []

    for script, config in TOOLS_CONFIG.items():
        print(f"\nTesting {script}...")
        
        # 1. Test text/point mode
        print(f"  Running in text mode (args: {config['text_args']})...")
        code_t, out_t, err_t, dur_t = run_script(script, config['text_args'])
        
        if code_t != 0:
            print(f"    [FAIL] Text mode returned exit code {code_t}")
            if err_t:
                print(f"    Stderr: {err_t.strip()}")
            all_success = False
            results.append((script, "FAIL (Text Mode)", dur_t, []))
            continue
        else:
            print(f"    [OK] Text mode executed successfully in {dur_t:.2f}s")

        # 2. Test plotting mode
        print(f"  Running in plotting mode (args: {config['plot_args']})...")
        # Clean expected plots first
        clean_plots(config['expected_plots'])
        
        code_p, out_p, err_p, dur_p = run_script(script, config['plot_args'])
        
        if code_p != 0:
            print(f"    [FAIL] Plotting mode returned exit code {code_p}")
            if err_p:
                print(f"    Stderr: {err_p.strip()}")
            all_success = False
            results.append((script, "FAIL (Plotting Mode)", dur_p, []))
            continue
        else:
            # Check if expected plots were generated
            plots_ok = True
            missing_plots = []
            for plot in config['expected_plots']:
                if os.path.exists(plot):
                    size = os.path.getsize(plot)
                    print(f"    [OK] Generated {plot} ({size} bytes)")
                else:
                    print(f"    [FAIL] Expected plot {plot} not found!")
                    plots_ok = False
                    missing_plots.append(plot)
            
            if plots_ok:
                print(f"    [OK] Plotting mode executed successfully in {dur_p:.2f}s")
                results.append((script, "SUCCESS", dur_t + dur_p, config['expected_plots']))
            else:
                all_success = False
                results.append((script, f"FAIL (Missing plots: {', '.join(missing_plots)})", dur_t + dur_p, []))

    print("\n" + "=" * 80)
    print("  VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"  {'Script Name':<30} | {'Status':<25} | {'Total Time':<10}")
    print("  " + "-" * 74)
    for script, status, total_time, plots in results:
        print(f"  {script:<30} | {status:<25} | {total_time:>8.2f}s")
    print("=" * 80)

    if all_success:
        print("\n  [SUCCESS] All 18 tools verified successfully!")
        sys.exit(0)
    else:
        print("\n  [FAILURE] Some tools failed verification. See details above.")
        sys.exit(1)

if __name__ == '__main__':
    main()
