"""
Spacecraft Attitude Determination and Control System (ADCS) B-dot detumbling and reaction wheel pointing simulator.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings('ignore')

# Color Palette
BG      = '#0f0f0e'
GOLD    = '#b8920a'
MOSS    = '#4a7a4b'
RED     = '#c04040'
CYAN    = '#40b0c0'
PAPER   = '#f0ede8'
DIM     = '#8a8a7a'
GREY    = '#3a3a38'
BLUE    = '#4080c0'

# Physical & Satellite Constants
I_SAT_kgm2 = np.diag([12.0, 15.0, 8.0])  # Sat inertia matrix
I_WHEEL_kgm2 = 0.05                       # Single reaction wheel inertia
B0_TESLA = 3e-5                           # Earth magnetic field strength (30 microTesla)
ORBIT_RATE_rads = 0.0011                  # ~90 min orbit angular velocity

# Quaternion kinematics helper functions
function_q_mult = lambda q, p: np.array([
    q[0]*p[0] - q[1]*p[1] - q[2]*p[2] - q[3]*p[3],
    q[0]*p[1] + q[1]*p[0] + q[2]*p[3] - q[3]*p[2],
    q[0]*p[2] - q[1]*p[3] + q[2]*p[0] + q[3]*p[1],
    q[0]*p[3] + q[1]*p[2] - q[2]*p[1] + q[3]*p[0]
])

def q_normalize(q):
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0])

def q_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

# 1. Simulation: B-Dot Magnetic Detumbling
def run_detumble_sim(t_span=300.0, dt=0.05):
  """
  Simulates magnetic B-dot detumbling controller coils.
  The goal is to dump kinetic energy and bring initial high angular rates to zero.
  """
  n_steps = int(t_span / dt)
  t_hist = np.linspace(0, t_span, n_steps)
  
  # State variables
  omega = np.array([0.8, -0.6, 0.5])  # Initial high angular velocity (rad/s)
  q = np.array([1.0, 0.0, 0.0, 0.0])   # Initial orientation quaternion
  
  omega_hist = np.zeros((n_steps, 3))
  kinetic_energy_hist = np.zeros(n_steps)
  
  # B-dot control gain
  K_bdot = 4e4
  
  # Simple dipole orbit magnetic field model in inertial frame
  def get_B_inertial(t):
    theta = ORBIT_RATE_rads * t
    return B0_TESLA * np.array([np.cos(theta), np.sin(theta), 0.3])
  
  # Inertial to body frame conversion
  def rotate_vector(v, q):
    v_quat = np.insert(v, 0, 0.0)
    v_rot_quat = function_q_mult(function_q_mult(q, v_quat), q_conjugate(q))
    return v_rot_quat[1:]

  # Cache derivative of B
  B_prev = rotate_vector(get_B_inertial(0.0), q)
  
  I_inv = np.linalg.inv(I_SAT_kgm2)
  
  for i, t in enumerate(t_hist):
    # Earth magnetic field in Body Frame
    B_inertial = get_B_inertial(t)
    B_body = rotate_vector(B_inertial, q)
    
    # Calculate dB/dt in body frame (measured by magnetometer)
    dB_dt = (B_body - B_prev) / dt
    B_prev = B_body.copy()
    
    # Control dipole moment: m = -K_bdot * dB_dt
    m_dipole = -K_bdot * dB_dt
    
    # Restrict maximum dipole moment (coil saturation at 5.0 A.m2)
    m_dipole = np.clip(m_dipole, -5.0, 5.0)
    
    # Control torque: tau = m x B
    tau_control = np.cross(m_dipole, B_body)
    
    # Dynamic derivative: I*d_omega/dt = tau - omega x (I * omega)
    d_omega_dt = I_inv @ (tau_control - np.cross(omega, I_SAT_kgm2 @ omega))
    omega += d_omega_dt * dt
    
    # Kinematic derivative: dq/dt = 0.5 * q * omega
    omega_quat = np.insert(omega, 0, 0.0)
    dq_dt = 0.5 * function_q_mult(q, omega_quat)
    q = q_normalize(q + dq_dt * dt)
    
    # Log
    omega_hist[i] = omega
    kinetic_energy_hist[i] = 0.5 * np.dot(omega, I_SAT_kgm2 @ omega)
    
  return t_hist, omega_hist, kinetic_energy_hist

# 2. Simulation: Reaction Wheel Pointing
def run_pointing_sim(t_span=100.0, dt=0.02):
  """
  Simulates attitude control using 3 orthogonal reaction wheels.
  The goal is to align the satellite with nadir pointing.
  """
  n_steps = int(t_span / dt)
  t_hist = np.linspace(0, t_span, n_steps)
  
  # State variables
  omega = np.array([0.05, -0.04, 0.03])  # Initial small residual velocity (rad/s)
  # Initial orientation error of ~45 degrees around Z axis
  angle = 45.0 * np.pi / 180.0
  q = np.array([np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)])
  
  # Reaction wheel speeds (rad/s)
  wheel_speeds = np.array([0.0, 0.0, 0.0])
  
  # target orientation (nadir pointing)
  q_target = np.array([1.0, 0.0, 0.0, 0.0])
  
  omega_hist = np.zeros((n_steps, 3))
  q_error_hist = np.zeros((n_steps, 3))
  wheel_speeds_hist = np.zeros((n_steps, 3))
  
  # Controller gains
  Kp = 0.8
  Kd = 2.5
  Ki = 0.02
  
  integral_error = np.zeros(3)
  I_inv = np.linalg.inv(I_SAT_kgm2)
  
  for i, t in enumerate(t_hist):
    # Attitude error: q_err = q_target_inv * q
    q_err = function_q_mult(q_conjugate(q_target), q)
    # The vector part of the quaternion error represents axis-angle orientation error
    vector_err = q_err[1:]
    integral_error += vector_err * dt
    
    # PID torque
    tau_control = -Kp * vector_err - Kd * omega - Ki * integral_error
    
    # Total momentum of reaction wheels: h_w = I_wheel * wheel_speeds
    h_wheel = I_WHEEL_kgm2 * wheel_speeds
    
    # Sat gyroscopic torque
    gyro_term = np.cross(omega, I_SAT_kgm2 @ omega + h_wheel)
    
    # Satellite angular acceleration: I_sat * d_omega/dt = -tau_control - gyro_term
    d_omega_dt = I_inv @ (-tau_control - gyro_term)
    omega += d_omega_dt * dt
    
    # Reaction wheel acceleration: I_wheel * d_wheel/dt = tau_control
    d_wheel_speeds_dt = tau_control / I_WHEEL_kgm2
    wheel_speeds += d_wheel_speeds_dt * dt
    
    # Kinematic propagation
    omega_quat = np.insert(omega, 0, 0.0)
    dq_dt = 0.5 * function_q_mult(q, omega_quat)
    q = q_normalize(q + dq_dt * dt)
    
    # Log
    omega_hist[i] = omega
    q_error_hist[i] = vector_err
    wheel_speeds_hist[i] = wheel_speeds
    
  return t_hist, q_error_hist, wheel_speeds_hist

# Verification Report
def print_report():
    print()
    print("=" * 100)
    print("  SPACECRAFT 3D ATTITUDE DETERMINATION & CONTROL SYSTEM (ADCS) SOLVER -- VERIFICATION")
    print("=" * 100)
    
    print("  Running B-Dot Magnetic Detumbling Simulation...")
    t_dt, w_dt, ke_dt = run_detumble_sim(t_span=300.0)
    initial_ke = ke_dt[0]
    final_ke = ke_dt[-1]
    energy_reduction_pct = (initial_ke - final_ke) / initial_ke * 100.0
    
    print(f"    Initial Angular Rates : [{w_dt[0,0]:.3f}, {w_dt[0,1]:.3f}, {w_dt[0,2]:.3f}] rad/s")
    print(f"    Final Angular Rates   : [{w_dt[-1,0]:.3f}, {w_dt[-1,1]:.3f}, {w_dt[-1,2]:.3f}] rad/s")
    print(f"    Initial Kinetic Energy: {initial_ke:.3f} J")
    print(f"    Final Kinetic Energy  : {final_ke:.6f} J")
    print(f"    B-Dot Energy Damping  : {energy_reduction_pct:.2f}% (Target: >99.0%)")
    print("-" * 100)
    
    print("  Running 3-Axis Reaction Wheel Pointing Control Simulation...")
    t_pt, q_pt, w_pt = run_pointing_sim(t_span=100.0)
    initial_err = np.linalg.norm(q_pt[0])
    final_err = np.linalg.norm(q_pt[-1])
    pointing_accuracy_pct = (initial_err - final_err) / initial_err * 100.0
    
    print(f"    Initial Pointing Error: {initial_err:.4f}")
    print(f"    Final Pointing Error  : {final_err:.6f}")
    print(f"    Pointing Convergence  : {pointing_accuracy_pct:.2f}% (Target: >99.5%)")
    print("=" * 100)
    print()

# Plotting
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

def generate_plots():
    # 1. B-Dot Detumble Plot
    t_dt, w_dt, ke_dt = run_detumble_sim(t_span=300.0)
    
    fig1 = plt.figure(figsize=(10, 8), facecolor=BG)
    
    ax1 = fig1.add_subplot(2, 1, 1)
    style_ax(ax1, 'SATELLITE ANGULAR VELOCITY DAMPING (B-DOT CONTROL)')
    ax1.plot(t_dt, w_dt[:, 0], color=RED, label='Omega X')
    ax1.plot(t_dt, w_dt[:, 1], color=GOLD, label='Omega Y')
    ax1.plot(t_dt, w_dt[:, 2], color=CYAN, label='Omega Z')
    ax1.set_ylabel('Angular Velocity  [rad/s]', fontsize=9)
    ax1.legend(fontsize=8, framealpha=0, labelcolor=DIM)
    
    ax2 = fig1.add_subplot(2, 1, 2)
    style_ax(ax2, 'ROTATIONAL KINETIC ENERGY DECAY')
    ax2.plot(t_dt, ke_dt, color=MOSS, linewidth=2.0)
    ax2.set_xlabel('Time  [s]', fontsize=9)
    ax2.set_ylabel('Kinetic Energy  [J]', fontsize=9)
    ax2.set_yscale('log')
    
    fig1.suptitle('SPACECRAFT DETUMBLING MISSION SIMULATION (MAGNETIC B-DOT)',
                 color=PAPER, fontsize=12, fontweight='bold', fontfamily='monospace', y=0.96)
    plt.savefig('adcs_detumble.png', dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close()
    
    # 2. Pointing PID Control Plot
    t_pt, q_pt, w_pt = run_pointing_sim(t_span=100.0)
    
    fig2 = plt.figure(figsize=(10, 8), facecolor=BG)
    
    ax3 = fig2.add_subplot(2, 1, 1)
    style_ax(ax3, 'QUATERNION ATTITUDE ERROR CONVERGENCE')
    ax3.plot(t_pt, q_pt[:, 0], color=RED, label='q_err X')
    ax3.plot(t_pt, q_pt[:, 1], color=GOLD, label='q_err Y')
    ax3.plot(t_pt, q_pt[:, 2], color=CYAN, label='q_err Z')
    ax3.set_ylabel('Quaternion Error Vector Component', fontsize=9)
    ax3.legend(fontsize=8, framealpha=0, labelcolor=DIM)
    
    ax4 = fig2.add_subplot(2, 1, 2)
    style_ax(ax4, 'REACTION WHEEL SPIN VELOCITIES')
    ax4.plot(t_pt, w_pt[:, 0], color=RED, label='Wheel X')
    ax4.plot(t_pt, w_pt[:, 1], color=GOLD, label='Wheel Y')
    ax4.plot(t_pt, w_pt[:, 2], color=CYAN, label='Wheel Z')
    ax4.set_xlabel('Time  [s]', fontsize=9)
    ax4.set_ylabel('Wheel Angular Rate  [rad/s]', fontsize=9)
    ax4.legend(fontsize=8, framealpha=0, labelcolor=DIM)
    
    fig2.suptitle('3-AXIS PRECISION REACTION WHEEL NADIR-POINTING CONVERGENCE',
                 color=PAPER, fontsize=12, fontweight='bold', fontfamily='monospace', y=0.96)
    plt.savefig('adcs_pointing.png', dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close()
    
    print("Plots generated successfully:")
    print("  - adcs_detumble.png")
    print("  - adcs_pointing.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spacecraft 3D ADCS Simulator')
    parser.add_argument('--point', action='store_true', help='Print design verification summary tables')
    args = parser.parse_args()

    if args.point:
        print_report()
    else:
        print_report()
        generate_plots()
