"""
2D potential flow panel solver using the Hess-Smith method for NACA airfoils.
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

# 1. NACA Airfoil Geometry Generator
def generate_naca_4digit(camber_pct=2, position_pct=4, thickness_pct=12, n_panels=40):
    """
    Generates coordinates for a NACA 4-digit airfoil.
    Example: NACA 2412 -> camber_pct=2, position_pct=4, thickness_pct=12.
    """
    m = camber_pct / 100.0
    p = position_pct / 10.0
    t = thickness_pct / 100.0
    
    # Cosine spacing for higher density at leading and trailing edges
    beta = np.linspace(0.0, np.pi, n_panels + 1)
    x = 0.5 * (1.0 - np.cos(beta))
    
    # Mean camber line coordinates and slope
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
    if p > 0.0:
        for idx, xi in enumerate(x):
            if xi < p:
                yc[idx] = (m / (p**2)) * (2 * p * xi - xi**2)
                dyc_dx[idx] = (2 * m / (p**2)) * (p - xi)
            else:
                yc[idx] = (m / ((1.0 - p)**2)) * ((1.0 - 2 * p) + 2 * p * xi - xi**2)
                dyc_dx[idx] = (2 * m / ((1.0 - p)**2)) * (p - xi)
    
    # Thickness distribution
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * (x**2) + 0.2843 * (x**3) - 0.1015 * (x**4))
    
    # Airfoil upper and lower surfaces
    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    # Combine upper and lower coords, from trailing edge back to trailing edge
    x_nodes = np.concatenate([xl[::-1], xu[1:]])
    y_nodes = np.concatenate([yl[::-1], yu[1:]])
    
    return x_nodes, y_nodes

# 2. Hess-Smith Potential Flow Solver
def solve_panel_flow(x_nodes, y_nodes, alpha_deg, V_inf=1.0):
    """
    Solves for panel source/vortex strengths using the Hess-Smith method.
    """
    n_panels = len(x_nodes) - 1
    alpha_rad = np.radians(alpha_deg)
    
    # 1. Panel properties: control points, lengths, angles
    xc = np.zeros(n_panels)
    yc = np.zeros(n_panels)
    L = np.zeros(n_panels)
    phi = np.zeros(n_panels)  # Orientation angle relative to X-axis
    
    for i in range(n_panels):
        x1, y1 = x_nodes[i], y_nodes[i]
        x2, y2 = x_nodes[i+1], y_nodes[i+1]
        
        xc[i] = 0.5 * (x1 + x2)
        yc[i] = 0.5 * (y1 + y2)
        L[i] = np.hypot(x2 - x1, y2 - y1)
        phi[i] = np.atan2(y2 - y1, x2 - x1)
        
    # Normal and tangent unit vectors
    nx = -np.sin(phi)
    ny = np.cos(phi)
    tx = np.cos(phi)
    ty = np.sin(phi)
    
    # 2. Build Influence Coefficient Matrices
    # System equations of size (N+1) x (N+1)
    # Equations [0..N-1] enforce normal boundary velocity = 0 at N control points
    # Equation [N] enforces Kutta trailing edge condition (V_t1 + V_tN = 0)
    A = np.zeros((n_panels + 1, n_panels + 1))
    b = np.zeros(n_panels + 1)
    
    # Populate influence matrices
    for i in range(n_panels):
        # Freestream boundary contribution
        b[i] = -V_inf * (np.cos(alpha_rad) * nx[i] + np.sin(alpha_rad) * ny[i])
        
        for j in range(n_panels):
            # Local coordinates of control point i relative to panel j
            dx = xc[i] - x_nodes[j]
            dy = yc[i] - y_nodes[j]
            
            # Rotate into panel j coordinates
            x_loc = dx * np.cos(phi[j]) + dy * np.sin(phi[j])
            y_loc = -dx * np.sin(phi[j]) + dy * np.cos(phi[j])
            
            # Integral terms
            r1_sq = x_loc**2 + y_loc**2
            r2_sq = (x_loc - L[j])**2 + y_loc**2
            
            # Avoid singularity division
            if i == j:
                u_source = 0.5
                v_source = 0.0
                u_vortex = 0.0
                v_vortex = -0.5
            else:
                u_source = (0.5 / np.pi) * 0.5 * np.log(r1_sq / r2_sq)
                v_source = (0.5 / np.pi) * (np.arctan2(y_loc, x_loc - L[j]) - np.arctan2(y_loc, x_loc))
                u_vortex = -v_source
                v_vortex = u_source
                
            # Rotate velocity components back to global coordinates
            ug_s = u_source * np.cos(phi[j]) - v_source * np.sin(phi[j])
            vg_s = u_source * np.sin(phi[j]) + v_source * np.cos(phi[j])
            ug_v = u_vortex * np.cos(phi[j]) - v_vortex * np.sin(phi[j])
            vg_v = u_vortex * np.sin(phi[j]) + v_vortex * np.cos(phi[j])
            
            # Project normal vectors
            A[i, j] = ug_s * nx[i] + vg_s * ny[i]
            A[i, n_panels] += ug_v * nx[i] + vg_v * ny[i]
            
    # 3. Kutta Condition: V_t[0] + V_t[N-1] = 0
    # Let's populate the last row of A
    b[n_panels] = -V_inf * (np.cos(alpha_rad) * (tx[0] + tx[n_panels-1]) + np.sin(alpha_rad) * (ty[0] + ty[n_panels-1]))
    
    for j in range(n_panels):
        # Contributions to tangential velocities at panel 0 and panel N-1
        # Summing tangential components
        for i_idx, i in enumerate([0, n_panels-1]):
            dx = xc[i] - x_nodes[j]
            dy = yc[i] - y_nodes[j]
            x_loc = dx * np.cos(phi[j]) + dy * np.sin(phi[j])
            y_loc = -dx * np.sin(phi[j]) + dy * np.cos(phi[j])
            
            r1_sq = x_loc**2 + y_loc**2
            r2_sq = (x_loc - L[j])**2 + y_loc**2
            
            if i == j:
                u_source = 0.5
                v_source = 0.0
                u_vortex = 0.0
                v_vortex = -0.5
            else:
                u_source = (0.5 / np.pi) * 0.5 * np.log(r1_sq / r2_sq)
                v_source = (0.5 / np.pi) * (np.arctan2(y_loc, x_loc - L[j]) - np.arctan2(y_loc, x_loc))
                u_vortex = -v_source
                v_vortex = u_source
                
            ug_s = u_source * np.cos(phi[j]) - v_source * np.sin(phi[j])
            vg_s = u_source * np.sin(phi[j]) + v_source * np.cos(phi[j])
            ug_v = u_vortex * np.cos(phi[j]) - v_vortex * np.sin(phi[j])
            vg_v = u_vortex * np.sin(phi[j]) + v_vortex * np.cos(phi[j])
            
            A[n_panels, j] += ug_s * tx[i] + vg_s * ty[i]
            A[n_panels, n_panels] += ug_v * tx[i] + vg_v * ty[i]
            
    # Solve linear system
    x_sol = np.linalg.solve(A, b)
    sources = x_sol[:n_panels]
    vortex = x_sol[n_panels]
    
    # 4. Calculate local surface velocities & Pressure Coefficients (Cp)
    V_tangent = np.zeros(n_panels)
    Cp = np.zeros(n_panels)
    
    for i in range(n_panels):
        V_t_freestream = V_inf * (np.cos(alpha_rad) * tx[i] + np.sin(alpha_rad) * ty[i])
        V_t_induced = 0.0
        
        for j in range(n_panels):
            dx = xc[i] - x_nodes[j]
            dy = yc[i] - y_nodes[j]
            x_loc = dx * np.cos(phi[j]) + dy * np.sin(phi[j])
            y_loc = -dx * np.sin(phi[j]) + dy * np.cos(phi[j])
            
            r1_sq = x_loc**2 + y_loc**2
            r2_sq = (x_loc - L[j])**2 + y_loc**2
            
            if i == j:
                u_source = 0.0  # self-induced tangential source velocity is 0
                u_vortex = 0.5  # self-induced tangential vortex velocity is 0.5 * gamma
            else:
                u_source = (0.5 / np.pi) * 0.5 * np.log(r1_sq / r2_sq)
                v_source = (0.5 / np.pi) * (np.arctan2(y_loc, x_loc - L[j]) - np.arctan2(y_loc, x_loc))
                u_vortex = -v_source
                v_vortex = u_source
                
            ug = (sources[j] * u_source + vortex * u_vortex) * np.cos(phi[j]) - \
                 (sources[j] * v_source + vortex * v_vortex) * np.sin(phi[j])
            vg = (sources[j] * u_source + vortex * u_vortex) * np.sin(phi[j]) + \
                 (sources[j] * v_source + vortex * v_vortex) * np.cos(phi[j])
                 
            V_t_induced += ug * tx[i] + vg * ty[i]
            
        V_tangent[i] = V_t_freestream + V_t_induced
        Cp[i] = 1.0 - (V_tangent[i] / V_inf)**2
        
    # Lift coefficient from circulation Gamma: L = rho * V * Gamma
    # Gamma = sum(vortex * length)
    Gamma = np.sum(vortex * L)
    C_l = 2 * Gamma / (V_inf * 1.0)
    
    return xc, yc, Cp, C_l

# Verification Report
def print_report():
    print()
    print("=" * 100)
    print("  NACA 4-DIGIT AIRFOIL 2D HESS-SMITH PANEL SOLVER -- DESIGN POINT SUMMARY")
    print("=" * 100)
    print(f"  {'Airfoil':10}  {'Angle of Attack':>18}  {'Lift Coefficient (Cl)':>22}  {'Theoretical Cl (2*pi*alpha)':>28}")
    print("-" * 100)
    
    # NACA 2412
    x_nodes, y_nodes = generate_naca_4digit(2, 4, 12, n_panels=60)
    for alpha in [-4, 0, 4, 8, 12]:
        _, _, _, C_l = solve_panel_flow(x_nodes, y_nodes, alpha)
        # Theoretical lift: Cl = 2*pi*(alpha - alpha_zero)
        # For NACA 2412, zero-lift alpha is approx -2 degrees
        alpha_rad = np.radians(alpha - (-2.1))
        C_l_theory = 2.0 * np.pi * alpha_rad
        print(f"  NACA 2412   {alpha:15.1f}°  {C_l:22.4f}  {C_l_theory:28.4f}")
        
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

def compute_velocity_field(x_nodes, y_nodes, alpha_deg, grid_w=30, grid_h=20):
    """
    Computes velocity field on a grid for streamline visualization.
    """
    x_grid = np.linspace(-0.5, 1.5, grid_w)
    y_grid = np.linspace(-0.6, 0.6, grid_h)
    X, Y = np.meshgrid(x_grid, y_grid)
    U_vel = np.zeros_like(X)
    V_vel = np.zeros_like(Y)
    
    alpha_rad = np.radians(alpha_deg)
    
    # Solve system first
    n_panels = len(x_nodes) - 1
    # solve
    xc = np.zeros(n_panels)
    yc = np.zeros(n_panels)
    L = np.zeros(n_panels)
    phi = np.zeros(n_panels)
    for i in range(n_panels):
        x1, y1 = x_nodes[i], y_nodes[i]
        x2, y2 = x_nodes[i+1], y_nodes[i+1]
        xc[i] = 0.5 * (x1 + x2)
        yc[i] = 0.5 * (y1 + y2)
        L[i] = np.hypot(x2 - x1, y2 - y1)
        phi[i] = np.atan2(y2 - y1, x2 - x1)
        
    nx = -np.sin(phi)
    ny = np.cos(phi)
    tx = np.cos(phi)
    ty = np.sin(phi)
    
    A = np.zeros((n_panels + 1, n_panels + 1))
    b = np.zeros(n_panels + 1)
    
    for i in range(n_panels):
        b[i] = -1.0 * (np.cos(alpha_rad) * nx[i] + np.sin(alpha_rad) * ny[i])
        for j in range(n_panels):
            dx = xc[i] - x_nodes[j]
            dy = yc[i] - y_nodes[j]
            x_loc = dx * np.cos(phi[j]) + dy * np.sin(phi[j])
            y_loc = -dx * np.sin(phi[j]) + dy * np.cos(phi[j])
            r1_sq = x_loc**2 + y_loc**2
            r2_sq = (x_loc - L[j])**2 + y_loc**2
            if i == j:
                u_source, v_source, u_vortex, v_vortex = 0.5, 0.0, 0.0, -0.5
            else:
                u_source = (0.5 / np.pi) * 0.5 * np.log(r1_sq / r2_sq)
                v_source = (0.5 / np.pi) * (np.arctan2(y_loc, x_loc - L[j]) - np.arctan2(y_loc, x_loc))
                u_vortex = -v_source
                v_vortex = u_source
            ug_s = u_source * np.cos(phi[j]) - v_source * np.sin(phi[j])
            vg_s = u_source * np.sin(phi[j]) + v_source * np.cos(phi[j])
            ug_v = u_vortex * np.cos(phi[j]) - v_vortex * np.sin(phi[j])
            vg_v = u_vortex * np.sin(phi[j]) + v_vortex * np.cos(phi[j])
            A[i, j] = ug_s * nx[i] + vg_s * ny[i]
            A[i, n_panels] += ug_v * nx[i] + vg_v * ny[i]
            
    b[n_panels] = -1.0 * (np.cos(alpha_rad) * (tx[0] + tx[n_panels-1]) + np.sin(alpha_rad) * (ty[0] + ty[n_panels-1]))
    for j in range(n_panels):
        for i in [0, n_panels-1]:
            dx = xc[i] - x_nodes[j]
            dy = yc[i] - y_nodes[j]
            x_loc = dx * np.cos(phi[j]) + dy * np.sin(phi[j])
            y_loc = -dx * np.sin(phi[j]) + dy * np.cos(phi[j])
            r1_sq = x_loc**2 + y_loc**2
            r2_sq = (x_loc - L[j])**2 + y_loc**2
            if i == j:
                u_source, v_source, u_vortex, v_vortex = 0.5, 0.0, 0.0, -0.5
            else:
                u_source = (0.5 / np.pi) * 0.5 * np.log(r1_sq / r2_sq)
                v_source = (0.5 / np.pi) * (np.arctan2(y_loc, x_loc - L[j]) - np.arctan2(y_loc, x_loc))
                u_vortex = -v_source
                v_vortex = u_source
            ug_s = u_source * np.cos(phi[j]) - v_source * np.sin(phi[j])
            vg_s = u_source * np.sin(phi[j]) + v_source * np.cos(phi[j])
            ug_v = u_vortex * np.cos(phi[j]) - v_vortex * np.sin(phi[j])
            vg_v = u_vortex * np.sin(phi[j]) + v_vortex * np.cos(phi[j])
            A[n_panels, j] += ug_s * tx[i] + vg_s * ty[i]
            A[n_panels, n_panels] += ug_v * tx[i] + vg_v * ty[i]
            
    x_sol = np.linalg.solve(A, b)
    sources = x_sol[:n_panels]
    vortex = x_sol[n_panels]
    
    # Compute velocity field contributions at grid points
    for r in range(grid_h):
        for c in range(grid_w):
            gx, gy = X[r, c], Y[r, c]
            
            # Start with free stream
            U_vel[r, c] = np.cos(alpha_rad)
            V_vel[r, c] = np.sin(alpha_rad)
            
            # Add inductions from all panels
            for j in range(n_panels):
                dx = gx - x_nodes[j]
                dy = gy - y_nodes[j]
                
                # local panel coordinates
                x_loc = dx * np.cos(phi[j]) + dy * np.sin(phi[j])
                y_loc = -dx * np.sin(phi[j]) + dy * np.cos(phi[j])
                
                r1_sq = x_loc**2 + y_loc**2
                r2_sq = (x_loc - L[j])**2 + y_loc**2
                
                # Check inside airfoil boundary
                if r1_sq < 1e-6 or r2_sq < 1e-6:
                    continue
                    
                u_source = (0.5 / np.pi) * 0.5 * np.log(r1_sq / r2_sq)
                v_source = (0.5 / np.pi) * (np.arctan2(y_loc, x_loc - L[j]) - np.arctan2(y_loc, x_loc))
                u_vortex = -v_source
                v_vortex = u_source
                
                ug = (sources[j] * u_source + vortex * u_vortex) * np.cos(phi[j]) - \
                     (sources[j] * v_source + vortex * v_vortex) * np.sin(phi[j])
                vg = (sources[j] * u_source + vortex * u_vortex) * np.sin(phi[j]) + \
                     (sources[j] * v_source + vortex * v_vortex) * np.cos(phi[j])
                     
                U_vel[r, c] += ug
                V_vel[r, c] += vg
                
    return X, Y, U_vel, V_vel

def generate_plots():
    camber, pos, thick = 2, 4, 12
    x_nodes, y_nodes = generate_naca_4digit(camber, pos, thick, n_panels=60)
    
    alpha = 6.0
    xc, yc, Cp, _ = solve_panel_flow(x_nodes, y_nodes, alpha)
    X, Y, U, V = compute_velocity_field(x_nodes, y_nodes, alpha, grid_w=40, grid_h=30)
    
    # Plot 1: Streamline Potential Flow Field
    fig1 = plt.figure(figsize=(10, 6), facecolor=BG)
    ax1 = fig1.add_subplot(1, 1, 1)
    style_ax(ax1, f'POTENTIAL FLOW FIELD STREAMLINES (NACA {camber}{pos}{thick:02d}, Alpha = {alpha:.1f}°)')
    
    # Draw streamlines
    speed = np.sqrt(U**2 + V**2)
    ax1.streamplot(X, Y, U, V, color=speed, cmap=plt.cm.viridis, density=1.4, linewidth=1.2)
    
    # Draw airfoil solid body
    ax1.fill(x_nodes, y_nodes, color='#252523', edgecolor=GOLD, linewidth=1.5, zorder=10)
    
    ax1.set_xlim(-0.3, 1.3)
    ax1.set_ylim(-0.4, 0.4)
    ax1.set_xlabel('Chord Position  [x/c]', fontsize=9)
    ax1.set_ylabel('Vertical Height  [y/c]', fontsize=9)
    ax1.set_aspect('equal')
    
    plt.savefig('airfoil_flow.png', dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close()
    
    # Plot 2: Lift Curve & Cp Distribution
    fig2 = plt.figure(figsize=(12, 5), facecolor=BG)
    
    # Left Subplot: Cp distribution
    ax2 = fig2.add_subplot(1, 2, 1)
    style_ax(ax2, 'SURFACE PRESSURE COEFFICIENT (Cp)')
    
    # Split upper/lower panels based on index
    n_half = len(Cp) // 2
    ax2.plot(xc[:n_half], Cp[:n_half], color=CYAN, linewidth=2.0, label='Lower Surface')
    ax2.plot(xc[n_half:], Cp[n_half:], color=RED, linewidth=2.0, label='Upper Surface')
    
    ax2.invert_yaxis()  # Cp conventionally plotted negative up
    ax2.set_xlabel('Chord Position  [x/c]', fontsize=9)
    ax2.set_ylabel('Pressure Coefficient  [Cp]', fontsize=9)
    ax2.legend(fontsize=8, framealpha=0, labelcolor=DIM)
    
    # Right Subplot: Lift Curve (Cl vs Alpha)
    ax3 = fig2.add_subplot(1, 2, 2)
    style_ax(ax3, 'LIFT COEFFICIENT CURVE (Cl vs Alpha)')
    
    alphas = np.linspace(-6.0, 14.0, 15)
    Cls = []
    for a in alphas:
        _, _, _, Cl = solve_panel_flow(x_nodes, y_nodes, a)
        Cls.append(Cl)
        
    ax3.plot(alphas, Cls, color=GOLD, marker='o', label='Panel Method Solver')
    # Thin airfoil theory reference: Cl = 2*pi*(alpha - alpha_zero)
    # for NACA 2412, alpha_zero is roughly -2.1 degrees
    Cl_theory = 2.0 * np.pi * np.radians(alphas - (-2.1))
    ax3.plot(alphas, Cl_theory, color=GREY, linestyle='--', label='Thin Airfoil Theory')
    
    ax3.set_xlabel('Angle of Attack  [deg]', fontsize=9)
    ax3.set_ylabel('Lift Coefficient  [Cl]', fontsize=9)
    ax3.legend(fontsize=8, framealpha=0, labelcolor=DIM)
    
    fig2.suptitle(f'NACA {camber}{pos}{thick:02d} PRESSURE PROFILE & AERODYNAMIC LIFT PERFORMANCE',
                 color=PAPER, fontsize=11, fontweight='bold', fontfamily='monospace', y=0.96)
    
    plt.savefig('lift_curve.png', dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close()
    
    print("Plots generated successfully:")
    print("  - airfoil_flow.png")
    print("  - lift_curve.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D Hess-Smith Panel Method Solver')
    parser.add_argument('--point', action='store_true', help='Print design point verification tables')
    args = parser.parse_args()

    if args.point:
        print_report()
    else:
        print_report()
        generate_plots()
