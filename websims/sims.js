/**
 * Extreme Thermal & Mechanics Toolkit — Web Portfolio & Lab Portal
 * Physics Solvers, 60 FPS Canvas Rendering & Global Navigation
 */

// ── Color Palette Definitions ────────────────────────────────────────────────
const COLORS = {
  bg: '#0f0f0e',
  bgDark: '#050505',
  cardBg: 'rgba(26, 26, 25, 0.7)',
  border: 'rgba(58, 58, 56, 0.4)',
  text: '#f0ede8',
  textDim: '#8a8a7a',
  grey: '#3a3a38',
  
  gold: '#b8920a',
  cyan: '#40b0c0',
  red: '#c04040',
  moss: '#4a7a4b',
  blue: '#4080c0',
  orange: '#e07a22'
};

// ── App State ────────────────────────────────────────────────────────────────
let activeSection = 'lab'; // 'lab', 'catalog', 'benchmarks', 'outreach'
let activeTab = 'scramjet'; // 'scramjet', 'shock', 'brake', 'divertor', 'truss', 'airfoil'
let canvas, ctx;
let animationFrameId;

// ── Physics States ───────────────────────────────────────────────────────────
let scramjetState = {
  flowRate: 1.0,  
  mach: 2.5,
  T_cool: [],
  T_wall_hot: [],
  T_wall_cool: [],
  q_flux: [],
  fluidParticles: []
};

let shockState = {
  mach: 3.0,
  theta: 15,      
  beta: 32.24,    
  detached: false,
  particles: [],
  m2: 2.25,
  p_ratio: 2.82,
  t_ratio: 1.39
};

let brakeState = {
  flowRate: 80,   
  initialSpeed: 300, 
  speed: 0,       
  temp: 80,       
  isBraking: false,
  history: [],     
  time: 0,
  rotationAngle: 0
};

let divertorState = {
  heatFlux: 10.0, 
  velocity: 5.0,  
  nodes: 10,      
  T: [],          
  fluidParticles: [],
  bubbles: [],
  chfLimit: 16.0,
  isDNB: false
};

let trussState = {
  load: 50, // kN
  material: 'aluminum',
  E: 70e9,
  yield: 276e6,
  rho: 2700,
  nodes: [
    [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0], [3.0, 0.5]
  ],
  elements: [
    [0, 2], [1, 3], [0, 3], [1, 2], [2, 3],
    [2, 4], [3, 5], [2, 5], [3, 4], [4, 5],
    [4, 6], [5, 6]
  ],
  areas: [], // mm^2 or m^2
  displacements: [],
  stresses: [],
  forces: [],
  buckling: [],
  totalMass: 0,
  baselineMass: 0,
  isCollapsed: false
};

let airfoilState = {
  camber: 2, // %
  pos: 4,    // /10
  thick: 12, // %
  alpha: 5,  // deg
  nodesX: [],
  nodesY: [],
  xc: [],
  yc: [],
  Cp: [],
  Cl: 0,
  particles: []
};

// ── Linear Equation Solver (Gaussian Elimination) ────────────────────────────
function solveLinearSystem(A, b) {
  const n = b.length;
  // Deep copy A and b
  let A_temp = A.map(row => [...row]);
  let b_temp = [...b];
  
  for (let i = 0; i < n; i++) {
    // Find pivot
    let maxEl = Math.abs(A_temp[i][i]);
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(A_temp[k][i]) > maxEl) {
        maxEl = Math.abs(A_temp[k][i]);
        maxRow = k;
      }
    }
    
    // Swap rows
    let temp = A_temp[maxRow]; A_temp[maxRow] = A_temp[i]; A_temp[i] = temp;
    let t = b_temp[maxRow]; b_temp[maxRow] = b_temp[i]; b_temp[i] = t;
    
    if (Math.abs(A_temp[i][i]) < 1e-12) {
      return new Array(n).fill(0); // singular matrix safety
    }
    
    // Factor rows below i
    for (let k = i + 1; k < n; k++) {
      let c = -A_temp[k][i] / A_temp[i][i];
      for (let j = i; j < n; j++) {
        if (i === j) A_temp[k][j] = 0;
        else A_temp[k][j] += c * A_temp[i][j];
      }
      b_temp[k] += c * b_temp[i];
    }
  }
  
  // Back substitution
  let x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = b_temp[i] / A_temp[i][i];
    for (let k = i - 1; k >= 0; k--) {
      b_temp[k] -= A_temp[k][i] * x[i];
    }
  }
  return x;
}

function solveShockPhysics() {
  const M = shockState.mach;
  const thetaRad = (shockState.theta * Math.PI) / 180;
  const gamma = 1.4;
  
  // Implicit function f(beta) = tan(theta) - RHS
  const f = (beta) => {
    const sinB = Math.sin(beta);
    const cosB = Math.cos(beta);
    const cotB = cosB / sinB;
    const num = M * M * sinB * sinB - 1;
    const den = M * M * (gamma + Math.cos(2 * beta)) + 2;
    return Math.tan(thetaRad) - 2 * cotB * num / den;
  };
  
  let low = thetaRad + 0.0001;
  let high = Math.PI / 2;
  
  if (f(low) * f(high) > 0) {
    shockState.detached = true;
    shockState.beta = 90;
    shockState.m2 = 0.0;
    shockState.p_ratio = 1.0;
    shockState.t_ratio = 1.0;
    document.getElementById('warn-shock').style.display = 'block';
    return;
  }
  
  document.getElementById('warn-shock').style.display = 'none';
  shockState.detached = false;
  
  // Bisection loop
  let betaSol = low;
  for (let i = 0; i < 50; i++) {
    let mid = (low + high) / 2;
    let val = f(mid);
    if (Math.abs(val) < 1e-6) {
      betaSol = mid;
      break;
    }
    if (f(low) * val < 0) {
      high = mid;
    } else {
      low = mid;
      betaSol = mid;
    }
  }
  
  shockState.beta = (betaSol * 180) / Math.PI;
  
  // Solve downstream properties
  const sinB = Math.sin(betaSol);
  const cosB = Math.cos(betaSol);
  const Mn1 = M * sinB;
  
  // Static Pressure Ratio p2/p1
  shockState.p_ratio = (2 * gamma * Mn1 * Mn1 - (gamma - 1)) / (gamma + 1);
  
  // Static Temp Ratio T2/T1
  const t_num = (2 * gamma * Mn1 * Mn1 - (gamma - 1)) * ((gamma - 1) * Mn1 * Mn1 + 2);
  const t_den = (gamma + 1) * (gamma + 1) * Mn1 * Mn1;
  shockState.t_ratio = t_num / t_den;
  
  // Downstream Mach Mn2
  const Mn2_sq = ((gamma - 1) * Mn1 * Mn1 + 2) / (2 * gamma * Mn1 * Mn1 - (gamma - 1));
  shockState.m2 = Math.sqrt(Mn2_sq) / Math.sin(betaSol - thetaRad);
}

function drawPlotAxes(x, y, w, h, xLabel, yLabel) {
  // Border Grid Lines
  ctx.strokeStyle = '#222';
  ctx.lineWidth = 0.5;
  ctx.strokeRect(x, y, w, h);
  
  // Horizontal grid steps
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
  ctx.lineWidth = 0.5;
  for (let i = 1; i < 5; i++) {
    const gy = y + (i / 5) * h;
    ctx.beginPath();
    ctx.moveTo(x, gy);
    ctx.lineTo(x + w, gy);
    ctx.stroke();
  }
  
  // Title / Labels
  ctx.fillStyle = COLORS.text;
  ctx.font = '10px monospace';
  ctx.fillText(xLabel, x + w / 2 - 30, y + h + 25);
  
  // Rotated Y Label
  ctx.save();
  ctx.translate(x - 30, y + h / 2 + 30);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

function drawLegend(x, y, items) {
  items.forEach((item, idx) => {
    const py = y + idx * 14;
    ctx.fillStyle = item.color;
    ctx.fillRect(x, py, 8, 8);
    ctx.fillStyle = COLORS.text;
    ctx.font = '9px monospace';
    ctx.fillText(item.label, x + 12, py + 8);
  });
}

// ── Initialization ───────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  canvas = document.getElementById('simCanvas');
  ctx = canvas.getContext('2d');
  
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);
  
  setupGlobalNavListeners();
  setupTabListeners();
  setupControlListeners();
  initSimulationStates();
  
  tick();
});

function resizeCanvas() {
  if (canvas && canvas.parentElement) {
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width || 800;
    canvas.height = 480;
  }
}

// ── Section Nav Switcher ──────────────────────────────────────────────────────
function setupGlobalNavListeners() {
  const sections = ['lab', 'catalog', 'benchmarks'];
  sections.forEach(sec => {
    document.getElementById(`btn-nav-${sec}`).addEventListener('click', () => {
      sections.forEach(s => {
        document.getElementById(`btn-nav-${s}`).classList.remove('active');
        document.getElementById(`sec-${s}`).style.display = 'none';
      });
      document.getElementById(`btn-nav-${sec}`).classList.add('active');
      document.getElementById(`sec-${sec}`).style.display = 'block';
      activeSection = sec;
      
      if (sec === 'lab') {
        resizeCanvas();
        initSimulationStates();
      }
    });
  });
}

// ── Lab Tab Navigation ───────────────────────────────────────────────────────
function setupTabListeners() {
  const tabs = ['scramjet', 'shock', 'brake', 'divertor', 'truss', 'airfoil'];
  tabs.forEach(tab => {
    document.getElementById(`tab-${tab}`).addEventListener('click', () => {
      tabs.forEach(t => document.getElementById(`tab-${t}`).classList.remove('active'));
      document.getElementById(`tab-${tab}`).classList.add('active');
      
      tabs.forEach(t => document.getElementById(`controls-${t}`).style.display = 'none');
      document.getElementById(`controls-${tab}`).style.display = 'block';
      
      tabs.forEach(t => document.getElementById(`info-${t}-content`).style.display = 'none');
      document.getElementById(`info-${tab}-content`).style.display = 'block';
      
      activeTab = tab;
      initSimulationStates();
      updateEquationDisplay();
    });
  });
  updateEquationDisplay();
}

// ── Control Event Listeners ──────────────────────────────────────────────────
function setupControlListeners() {
  // Scramjet
  document.getElementById('slider-scram-flow').addEventListener('input', (e) => {
    scramjetState.flowRate = parseFloat(e.target.value);
    document.getElementById('val-scram-flow').innerText = scramjetState.flowRate.toFixed(1);
  });
  document.getElementById('slider-scram-mach').addEventListener('input', (e) => {
    scramjetState.mach = parseFloat(e.target.value);
    document.getElementById('val-scram-mach').innerText = scramjetState.mach.toFixed(1);
  });
  
  // Shock
  document.getElementById('slider-shock-mach').addEventListener('input', (e) => {
    shockState.mach = parseFloat(e.target.value);
    document.getElementById('val-shock-mach').innerText = shockState.mach.toFixed(1);
    solveShockPhysics();
  });
  document.getElementById('slider-shock-angle').addEventListener('input', (e) => {
    shockState.theta = parseFloat(e.target.value);
    document.getElementById('val-shock-angle').innerText = shockState.theta;
    solveShockPhysics();
  });
  
  // Brake
  document.getElementById('slider-brake-flow').addEventListener('input', (e) => {
    brakeState.flowRate = parseFloat(e.target.value);
    document.getElementById('val-brake-flow').innerText = brakeState.flowRate;
  });
  document.getElementById('slider-brake-speed').addEventListener('input', (e) => {
    brakeState.initialSpeed = parseFloat(e.target.value);
    document.getElementById('val-brake-speed').innerText = brakeState.initialSpeed;
  });
  document.getElementById('btn-brake-apply').addEventListener('click', () => {
    if (!brakeState.isBraking && brakeState.speed <= 0) {
      brakeState.speed = brakeState.initialSpeed;
      brakeState.isBraking = true;
    }
  });
  
  // Divertor
  document.getElementById('slider-divertor-flux').addEventListener('input', (e) => {
    divertorState.heatFlux = parseFloat(e.target.value);
    document.getElementById('val-divertor-flux').innerText = divertorState.heatFlux.toFixed(1);
  });
  document.getElementById('slider-divertor-vel').addEventListener('input', (e) => {
    divertorState.velocity = parseFloat(e.target.value);
    document.getElementById('val-divertor-vel').innerText = divertorState.velocity.toFixed(1);
  });

  // Truss
  document.getElementById('slider-truss-load').addEventListener('input', (e) => {
    trussState.load = parseFloat(e.target.value);
    document.getElementById('val-truss-load').innerText = trussState.load;
    solveTrussPhysics();
  });
  document.getElementById('select-truss-material').addEventListener('change', (e) => {
    trussState.material = e.target.value;
    if (trussState.material === 'aluminum') {
      trussState.E = 70e9; trussState.yield = 276e6; trussState.rho = 2700;
    } else if (trussState.material === 'steel') {
      trussState.E = 200e9; trussState.yield = 250e6; trussState.rho = 7800;
    } else {
      trussState.E = 150e9; trussState.yield = 600e6; trussState.rho = 1600;
    }
    solveTrussPhysics();
  });
  document.getElementById('btn-truss-optimize').addEventListener('click', () => {
    optimizeTrussDesign();
  });

  // Airfoil
  document.getElementById('slider-airfoil-camber').addEventListener('input', (e) => {
    airfoilState.camber = parseInt(e.target.value);
    document.getElementById('val-airfoil-camber').innerText = airfoilState.camber;
    solveAirfoilPhysics();
  });
  document.getElementById('slider-airfoil-pos').addEventListener('input', (e) => {
    airfoilState.pos = parseInt(e.target.value);
    document.getElementById('val-airfoil-pos').innerText = airfoilState.pos;
    solveAirfoilPhysics();
  });
  document.getElementById('slider-airfoil-thick').addEventListener('input', (e) => {
    airfoilState.thick = parseInt(e.target.value);
    document.getElementById('val-airfoil-thick').innerText = airfoilState.thick;
    solveAirfoilPhysics();
  });
  document.getElementById('slider-airfoil-alpha').addEventListener('input', (e) => {
    airfoilState.alpha = parseFloat(e.target.value);
    document.getElementById('val-airfoil-alpha').innerText = airfoilState.alpha.toFixed(1);
    solveAirfoilPhysics();
  });
}

// ── State Initialization ─────────────────────────────────────────────────────
function initSimulationStates() {
  if (activeTab === 'scramjet') {
    scramjetState.fluidParticles = [];
    for (let i = 0; i < 40; i++) {
      scramjetState.fluidParticles.push({
        x: Math.random() * canvas.width * 0.55,
        channel: Math.floor(Math.random() * 2), 
        speed: 1 + Math.random() * 2
      });
    }
  } else if (activeTab === 'shock') {
    solveShockPhysics();
    shockState.particles = [];
    for (let i = 0; i < 60; i++) {
      shockState.particles.push({
        x: Math.random() * canvas.width * 0.55,
        y: Math.random() * canvas.height * 0.75,
        speed: 4
      });
    }
  } else if (activeTab === 'brake') {
    brakeState.speed = 0;
    brakeState.temp = 80;
    brakeState.isBraking = false;
    brakeState.history = Array(150).fill(80);
    brakeState.time = 0;
  } else if (activeTab === 'divertor') {
    divertorState.T = Array(divertorState.nodes).fill(100.0);
    divertorState.fluidParticles = [];
    divertorState.bubbles = [];
    for (let i = 0; i < 30; i++) {
      divertorState.fluidParticles.push({
        x: Math.random() * 60 + canvas.width * 0.42,
        y: Math.random() * canvas.height * 0.75,
        speed: 2 + Math.random() * 2
      });
    }
  } else if (activeTab === 'truss') {
    // 12 elements baseline areas = 1000 mm²
    trussState.areas = new Array(trussState.elements.length).fill(0.001);
    solveTrussPhysics();
  } else if (activeTab === 'airfoil') {
    solveAirfoilPhysics();
    airfoilState.particles = [];
    for (let i = 0; i < 80; i++) {
      airfoilState.particles.push({
        x: Math.random() * canvas.width * 0.55,
        y: Math.random() * canvas.height * 0.75
      });
    }
  }
}

// ── 2D Truss Solver Algorithms ───────────────────────────────────────────────
function solveTrussPhysics() {
  const n_nodes = trussState.nodes.length;
  const n_elements = trussState.elements.length;
  const n_dof = 2 * n_nodes;
  
  let K_global = Array(n_dof).fill(0).map(() => Array(n_dof).fill(0));
  let F_global = new Array(n_dof).fill(0);
  
  // Apply external loads (50 kN tip downward at Node 6)
  F_global[2 * 6 + 1] = -trussState.load * 1000.0;
  
  // Element properties cached
  let lengths = [];
  let cosines = [];
  let sines = [];
  
  for (let idx = 0; idx < n_elements; idx++) {
    const [n1, n2] = trussState.elements[idx];
    const [x1, y1] = trussState.nodes[n1];
    const [x2, y2] = trussState.nodes[n2];
    
    const L = Math.hypot(x2 - x1, y2 - y1);
    const c = (x2 - x1) / L;
    const s = (y2 - y1) / L;
    
    lengths.push(L);
    cosines.push(c);
    sines.push(s);
    
    const k_local = (trussState.areas[idx] * trussState.E / L);
    const k_mat = [
      [ c*c,  c*s, -c*c, -c*s],
      [ c*s,  s*s, -c*s, -s*s],
      [-c*c, -c*s,  c*c,  c*s],
      [-c*s, -s*s,  c*s,  s*s]
    ];
    
    const dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1];
    for (let r = 0; r < 4; r++) {
      for (let cl = 0; cl < 4; cl++) {
        K_global[dofs[r]][dofs[cl]] += k_local * k_mat[r][cl];
      }
    }
  }
  
  // Boundary constraints (Node 0 & 1 pinned)
  let active_dofs = [];
  for (let i = 0; i < n_dof; i++) {
    if (i !== 0 && i !== 1 && i !== 2 && i !== 3) {
      active_dofs.push(i);
    }
  }
  
  // Partition matrices
  const size_a = active_dofs.length;
  let K_aa = Array(size_a).fill(0).map(() => Array(size_a).fill(0));
  let F_a = new Array(size_a).fill(0);
  
  for (let r = 0; r < size_a; r++) {
    F_a[r] = F_global[active_dofs[r]];
    for (let c = 0; c < size_a; c++) {
      K_aa[r][c] = K_global[active_dofs[r]][active_dofs[c]];
    }
  }
  
  const U_a = solveLinearSystem(K_aa, F_a);
  
  trussState.displacements = new Array(n_dof).fill(0);
  for (let idx = 0; idx < size_a; idx++) {
    trussState.displacements[active_dofs[idx]] = U_a[idx];
  }
  
  // Stresses and Buckling margins
  trussState.stresses = [];
  trussState.forces = [];
  trussState.buckling = [];
  trussState.isCollapsed = false;
  
  for (let idx = 0; idx < n_elements; idx++) {
    const [n1, n2] = trussState.elements[idx];
    const L = lengths[idx];
    const c = cosines[idx];
    const s = sines[idx];
    
    const u1 = trussState.displacements[2*n1];
    const v1 = trussState.displacements[2*n1+1];
    const u2 = trussState.displacements[2*n2];
    const v2 = trussState.displacements[2*n2+1];
    
    const dL = (u2 - u1) * c + (v2 - v1) * s;
    const strain = dL / L;
    const stress = trussState.E * strain;
    const force = stress * trussState.areas[idx];
    
    trussState.stresses.push(stress);
    trussState.forces.push(force);
    
    if (force < 0) { // Compression member
      const I_moment = 0.05 * Math.pow(trussState.areas[idx], 2);
      const P_crit = (Math.PI * Math.PI * trussState.E * I_moment) / (L * L);
      const b_ratio = Math.abs(force) / P_crit;
      trussState.buckling.push(b_ratio);
      if (b_ratio > 1.0) trussState.isCollapsed = true;
    } else {
      trussState.buckling.push(0.0);
    }
    
    if (Math.abs(stress) > trussState.yield) {
      trussState.isCollapsed = true;
    }
  }
  
  // Compute mass
  let totalMass = 0;
  for (let i = 0; i < n_elements; i++) {
    totalMass += trussState.areas[i] * lengths[i] * trussState.rho;
  }
  trussState.totalMass = totalMass;
  
  document.getElementById('warn-truss').style.display = trussState.isCollapsed ? 'block' : 'none';
}

function optimizeTrussDesign() {
  const n_elements = trussState.elements.length;
  const allowable_stress = trussState.yield / 1.5;
  
  for (let iter = 0; iter < 30; iter++) {
    solveTrussPhysics();
    
    let new_areas = [];
    for (let idx = 0; idx < n_elements; idx++) {
      const stress_demand = Math.abs(trussState.stresses[idx]) / allowable_stress;
      let buckling_demand = 0.0;
      if (trussState.forces[idx] < 0) {
        buckling_demand = Math.sqrt(trussState.buckling[idx] * 1.5);
      }
      
      const scale = Math.max(0.1, Math.max(stress_demand, buckling_demand));
      let A_new = trussState.areas[idx] * (0.5 + 0.5 * scale);
      A_new = Math.max(1e-6, A_new); // minimum area limit (1 mm²)
      new_areas.push(A_new);
    }
    
    trussState.areas = new_areas;
  }
  solveTrussPhysics();
}

// ── NACA Airfoil Flow Solver (Hess-Smith) ────────────────────────────────────
function solveAirfoilPhysics() {
  const camber_pct = airfoilState.camber;
  const position_pct = airfoilState.pos;
  const thickness_pct = airfoilState.thick;
  const n_panels = 30;
  
  const m = camber_pct / 100.0;
  const p = position_pct / 10.0;
  const t = thickness_pct / 100.0;
  
  // Cosine node distribution
  let beta = [];
  for (let i = 0; i <= n_panels; i++) {
    beta.push((i * Math.PI) / n_panels);
  }
  const x = beta.map(b => 0.5 * (1.0 - Math.cos(b)));
  
  let yc = new Array(x.length).fill(0);
  let dyc_dx = new Array(x.length).fill(0);
  
  if (p > 0.0) {
    x.forEach((xi, idx) => {
      if (xi < p) {
        yc[idx] = (m / (p*p)) * (2 * p * xi - xi*xi);
        dyc_dx[idx] = (2 * m / (p*p)) * (p - xi);
      } else {
        yc[idx] = (m / Math.pow(1.0 - p, 2)) * ((1.0 - 2 * p) + 2 * p * xi - xi*xi);
        dyc_dx[idx] = (2 * m / Math.pow(1.0 - p, 2)) * (p - xi);
      }
    });
  }
  
  const yt = x.map(xi => 5.0 * t * (0.2969 * Math.sqrt(xi) - 0.1260 * xi - 0.3516 * xi*xi + 0.2843 * Math.pow(xi, 3) - 0.1015 * Math.pow(xi, 4)));
  
  let xu = [], yu = [], xl = [], yl = [];
  x.forEach((xi, idx) => {
    const theta = Math.atan(dyc_dx[idx]);
    xu.push(xi - yt[idx] * Math.sin(theta));
    yu.push(yc[idx] + yt[idx] * Math.cos(theta));
    xl.push(xi + yt[idx] * Math.sin(theta));
    yl.push(yc[idx] - yt[idx] * Math.cos(theta));
  });
  
  // Combine surfaces
  airfoilState.nodesX = [...xl.reverse(), ...xu.slice(1)];
  airfoilState.nodesY = [...yl.reverse(), ...yu.slice(1)];
  
  const N = airfoilState.nodesX.length - 1; // actual panels
  const alpha_rad = (airfoilState.alpha * Math.PI) / 180;
  
  // Panel midpoint geometry
  let xc = [], yc_mid = [], L = [], phi = [];
  for (let i = 0; i < N; i++) {
    const x1 = airfoilState.nodesX[i], y1 = airfoilState.nodesY[i];
    const x2 = airfoilState.nodesX[i+1], y2 = airfoilState.nodesY[i+1];
    xc.push(0.5 * (x1 + x2));
    yc_mid.push(0.5 * (y1 + y2));
    L.push(Math.hypot(x2 - x1, y2 - y1));
    phi.push(Math.atan2(y2 - y1, x2 - x1));
  }
  
  airfoilState.xc = xc;
  airfoilState.yc = yc_mid;
  airfoilState.L = L;
  airfoilState.phi = phi;
  
  let nx = phi.map(p => -Math.sin(p));
  let ny = phi.map(p => Math.cos(p));
  let tx = phi.map(p => Math.cos(p));
  let ty = phi.map(p => Math.sin(p));
  
  // Influence coefficient system
  let A = Array(N + 1).fill(0).map(() => Array(N + 1).fill(0));
  let b = new Array(N + 1).fill(0);
  
  for (let i = 0; i < N; i++) {
    b[i] = -1.0 * (Math.cos(alpha_rad) * nx[i] + Math.sin(alpha_rad) * ny[i]);
    
    for (let j = 0; j < N; j++) {
      const dx = xc[i] - airfoilState.nodesX[j];
      const dy = yc_mid[i] - airfoilState.nodesY[j];
      
      const x_loc = dx * Math.cos(phi[j]) + dy * Math.sin(phi[j]);
      const y_loc = -dx * Math.sin(phi[j]) + dy * Math.cos(phi[j]);
      
      const r1_sq = x_loc*x_loc + y_loc*y_loc;
      const r2_sq = Math.pow(x_loc - L[j], 2) + y_loc*y_loc;
      
      let u_s, v_s, u_v, v_v;
      if (i === j) {
        u_s = 0.5; v_s = 0.0; u_v = 0.0; v_v = -0.5;
      } else {
        u_s = (0.5 / Math.PI) * 0.5 * Math.log(r1_sq / r2_sq);
        v_s = (0.5 / Math.PI) * (Math.atan2(y_loc, x_loc - L[j]) - Math.atan2(y_loc, x_loc));
        u_v = -v_s;
        v_v = u_s;
      }
      
      const ug_s = u_s * Math.cos(phi[j]) - v_s * Math.sin(phi[j]);
      const vg_s = u_s * Math.sin(phi[j]) + v_s * Math.cos(phi[j]);
      const ug_v = u_v * Math.cos(phi[j]) - v_v * Math.sin(phi[j]);
      const vg_v = u_v * Math.sin(phi[j]) + v_v * Math.cos(phi[j]);
      
      A[i][j] = ug_s * nx[i] + vg_s * ny[i];
      A[i][N] += ug_v * nx[i] + vg_v * ny[i];
    }
  }
  
  // Kutta Condition at last row
  b[N] = -1.0 * (Math.cos(alpha_rad) * (tx[0] + tx[N-1]) + Math.sin(alpha_rad) * (ty[0] + ty[N-1]));
  for (let j = 0; j < N; j++) {
    for (let i_idx of [0, N-1]) {
      const dx = xc[i_idx] - airfoilState.nodesX[j];
      const dy = yc_mid[i_idx] - airfoilState.nodesY[j];
      
      const x_loc = dx * Math.cos(phi[j]) + dy * Math.sin(phi[j]);
      const y_loc = -dx * Math.sin(phi[j]) + dy * Math.cos(phi[j]);
      
      const r1_sq = x_loc*x_loc + y_loc*y_loc;
      const r2_sq = Math.pow(x_loc - L[j], 2) + y_loc*y_loc;
      
      let u_s, v_s, u_v, v_v;
      if (i_idx === j) {
        u_s = 0.5; v_s = 0.0; u_v = 0.0; v_v = -0.5;
      } else {
        u_s = (0.5 / Math.PI) * 0.5 * Math.log(r1_sq / r2_sq);
        v_s = (0.5 / Math.PI) * (Math.atan2(y_loc, x_loc - L[j]) - Math.atan2(y_loc, x_loc));
        u_v = -v_s;
        v_v = u_s;
      }
      
      const ug_s = u_s * Math.cos(phi[j]) - v_s * Math.sin(phi[j]);
      const vg_s = u_s * Math.sin(phi[j]) + v_s * Math.cos(phi[j]);
      const ug_v = u_v * Math.cos(phi[j]) - v_v * Math.sin(phi[j]);
      const vg_v = u_v * Math.sin(phi[j]) + v_v * Math.cos(phi[j]);
      
      A[N][j] += ug_s * tx[i_idx] + vg_s * ty[i_idx];
      A[N][N] += ug_v * tx[i_idx] + vg_v * ty[i_idx];
    }
  }
  
  const sol = solveLinearSystem(A, b);
  const sources = sol.slice(0, N);
  const vortex = sol[N];
  airfoilState.sources = sources;
  airfoilState.vortex = vortex;
  
  // Calculate velocities and Pressure Coefficient Cp
  airfoilState.Cp = [];
  for (let i = 0; i < N; i++) {
    const V_t_free = 1.0 * (Math.cos(alpha_rad) * tx[i] + Math.sin(alpha_rad) * ty[i]);
    let V_t_ind = 0.0;
    
    for (let j = 0; j < N; j++) {
      const dx = xc[i] - airfoilState.nodesX[j];
      const dy = yc_mid[i] - airfoilState.nodesY[j];
      
      const x_loc = dx * Math.cos(phi[j]) + dy * Math.sin(phi[j]);
      const y_loc = -dx * Math.sin(phi[j]) + dy * Math.cos(phi[j]);
      
      const r1_sq = x_loc*x_loc + y_loc*y_loc;
      const r2_sq = Math.pow(x_loc - L[j], 2) + y_loc*y_loc;
      
      let u_s, v_s, u_v, v_v;
      if (i === j) {
        u_s = 0.0; u_v = 0.5; v_s = 0.0; v_v = 0.0;
      } else {
        u_s = (0.5 / Math.PI) * 0.5 * Math.log(r1_sq / r2_sq);
        v_s = (0.5 / Math.PI) * (Math.atan2(y_loc, x_loc - L[j]) - Math.atan2(y_loc, x_loc));
        u_v = -v_s;
        v_v = u_s;
      }
      
      const ug = (sources[j] * u_s + vortex * u_v) * Math.cos(phi[j]) - (sources[j] * v_s + vortex * v_v) * Math.sin(phi[j]);
      const vg = (sources[j] * u_s + vortex * u_v) * Math.sin(phi[j]) + (sources[j] * v_s + vortex * v_v) * Math.cos(phi[j]);
      
      V_t_ind += ug * tx[i] + vg * ty[i];
    }
    
    const V_local = V_t_free + V_t_ind;
    airfoilState.Cp.push(1.0 - V_local*V_local);
  }
  
  // Circulation Gamma = sum(vortex * L)
  let Gamma = 0;
  for (let i = 0; i < N; i++) Gamma += vortex * L[i];
  airfoilState.Cl = 2.0 * Gamma;
}

// Compute velocity at arbitrary coordinates (x,y) around airfoil
function getPotentialFlowVelocity(x, y) {
  const alpha_rad = (airfoilState.alpha * Math.PI) / 180;
  let u = Math.cos(alpha_rad);
  let v = Math.sin(alpha_rad);
  
  if (!airfoilState.sources) {
    solveAirfoilPhysics();
  }
  
  const N = airfoilState.xc.length;
  const nodesX = airfoilState.nodesX;
  const nodesY = airfoilState.nodesY;
  const phi = airfoilState.phi;
  const L = airfoilState.L;
  const sources = airfoilState.sources;
  const vortex = airfoilState.vortex;
  
  for (let j = 0; j < N; j++) {
    const dx = x - nodesX[j];
    const dy = y - nodesY[j];
    
    // Rotate into panel coordinates
    const cosPhi = Math.cos(phi[j]);
    const sinPhi = Math.sin(phi[j]);
    const x_loc = dx * cosPhi + dy * sinPhi;
    const y_loc = -dx * sinPhi + dy * cosPhi;
    
    const r1_sq = x_loc * x_loc + y_loc * y_loc;
    const r2_sq = (x_loc - L[j]) * (x_loc - L[j]) + y_loc * y_loc;
    
    if (r1_sq < 1e-4 || r2_sq < 1e-4) continue;
    
    const u_source = (0.5 / Math.PI) * 0.5 * Math.log(r1_sq / r2_sq);
    const v_source = (0.5 / Math.PI) * (Math.atan2(y_loc, x_loc - L[j]) - Math.atan2(y_loc, x_loc));
    const u_vortex = -v_source;
    const v_vortex = u_source;
    
    const ug = (sources[j] * u_source + vortex * u_vortex) * cosPhi - (sources[j] * v_source + vortex * v_vortex) * sinPhi;
    const vg = (sources[j] * u_source + vortex * u_vortex) * sinPhi + (sources[j] * v_source + vortex * v_vortex) * cosPhi;
    
    u += ug;
    v += vg;
  }
  
  return {u, v};
}

// ── Physics Numerical Loops (60Hz Ticks) ─────────────────────────────────────
function updatePhysics() {
  if (activeTab === 'scramjet') {
    const flows = scramjetState.flowRate;
    const Mach = scramjetState.mach;
    
    const T_rec = 2200 + Mach * 220; 
    const h_gas = 350 + Mach * 90;   
    const h_cool = 4200 * Math.pow(flows / 1.0, 0.8);
    
    const R_g = 1.0 / h_gas;
    const R_w = 0.0015 / 320.0;
    const R_c = 1.0 / h_cool;
    const R_tot = R_g + R_w + R_c;
    
    let T_c_curr = 40.0;
    scramjetState.T_cool = [];
    scramjetState.T_wall_hot = [];
    scramjetState.T_wall_cool = [];
    scramjetState.q_flux = [];
    
    const dx = 1.0 / 10;
    for (let i = 0; i < 10; i++) {
      const q = (T_rec - T_c_curr) / R_tot;
      const T_wh = T_rec - q * R_g;
      const T_wc = T_wh - q * R_w;
      
      const dT_cool = (q * 0.02 * dx) / (flows * 14200.0);
      
      scramjetState.T_cool.push(T_c_curr);
      scramjetState.T_wall_hot.push(T_wh);
      scramjetState.T_wall_cool.push(T_wc);
      scramjetState.q_flux.push(q / 1e6); 
      
      T_c_curr += dT_cool;
    }
    
    const maxT = Math.max(...scramjetState.T_wall_hot);
    document.getElementById('warn-scramjet').style.display = maxT > 1000 ? 'block' : 'none';
    
    scramjetState.fluidParticles.forEach(p => {
      p.x += p.speed * (flows * 1.5 + 0.5);
      if (p.x > canvas.width * 0.55) p.x = 0;
    });
    
  } else if (activeTab === 'shock') {
    const M1 = shockState.mach;
    const thetaRad = (shockState.theta * Math.PI) / 180;
    const betaRad = (shockState.beta * Math.PI) / 180;
    
    shockState.particles.forEach(p => {
      const wX = canvas.width * 0.22;
      const wY = canvas.height * 0.5;
      
      let crossed = false;
      let angleToCross = 0;
      
      if (shockState.detached) {
        const dist = p.x - wX;
        const bY = wY - Math.sqrt(Math.max(0, 16000 + 400 * dist));
        const bY_bottom = wY + Math.sqrt(Math.max(0, 16000 + 400 * dist));
        
        if (p.x > wX - 50 && (p.y < bY || p.y > bY_bottom || p.x > wX)) {
          crossed = true;
          const dx = p.x - wX;
          const dy = p.y - wY;
          angleToCross = Math.atan2(dy, dx);
        }
      } else {
        const dx = p.x - wX;
        const shockY_top = wY - dx * Math.tan(betaRad);
        const shockY_bottom = wY + dx * Math.tan(betaRad);
        
        if (p.x > wX) {
          if (p.y < wY && p.y >= shockY_top) {
            crossed = true;
            angleToCross = -thetaRad;
          } else if (p.y > wY && p.y <= shockY_bottom) {
            crossed = true;
            angleToCross = thetaRad;
          }
        }
      }
      
      if (crossed) {
        const vRatio = Math.max(0.3, shockState.m2 / M1);
        const speed = p.speed * vRatio;
        p.x += speed * Math.cos(angleToCross);
        p.y += speed * Math.sin(angleToCross);
      } else {
        p.x += p.speed;
      }
      
      if (p.x > canvas.width * 0.55 || p.y < 0 || p.y > canvas.height * 0.75) {
        p.x = 0;
        p.y = Math.random() * canvas.height * 0.75;
      }
    });
    
  } else if (activeTab === 'brake') {
    const flow = brakeState.flowRate; 
    const dt = 1 / 60; 
    
    const h_conv = 15 + Math.pow(flow, 0.6) * Math.pow(brakeState.speed / 100, 0.5) * 5;
    const q_conv = h_conv * 0.09 * (brakeState.temp - 25); 
    
    const T_K = brakeState.temp + 273.15;
    const q_rad = 0.09 * 0.85 * 5.6704e-8 * (Math.pow(T_K, 4) - Math.pow(298.15, 4));
    
    let q_in = 0;
    
    if (brakeState.isBraking && brakeState.speed > 0) {
      const decel = 30; 
      brakeState.speed -= decel * dt;
      if (brakeState.speed < 0) {
        brakeState.speed = 0;
        brakeState.isBraking = false;
      }
      
      const massCar = 800; 
      const v_mps = (brakeState.speed / 3.6);
      const decel_mps2 = (decel / 3.6);
      q_in = 0.95 * (massCar * v_mps * decel_mps2) / 4; 
    }
    
    const dT = ((q_in - q_conv - q_rad) / (1.6 * 1500)) * dt;
    brakeState.temp += dT;
    
    if (brakeState.speed > 0) {
      brakeState.rotationAngle += (brakeState.speed * 0.1) * dt;
    }
    
    document.getElementById('warn-brake-hot').style.display = brakeState.temp > 750 ? 'block' : 'none';
    document.getElementById('warn-brake-cold').style.display = brakeState.temp < 300 ? 'block' : 'none';
    
    brakeState.time += dt;
    if (Math.floor(brakeState.time * 60) % 2 === 0) {
      brakeState.history.push(brakeState.temp);
      if (brakeState.history.length > 150) {
        brakeState.history.shift();
      }
    }
    
  } else if (activeTab === 'divertor') {
    const subSteps = 5;
    const dt = (1 / 60) / subSteps;
    const dx = 0.0025; 
    
    const k_grid = [130, 130, 130, 130, 130, 380, 380, 320, 320, 320];
    const rho_grid = [19300, 19300, 19300, 19300, 19300, 8960, 8960, 8900, 8900, 8900];
    const cp_grid = [130, 130, 130, 130, 130, 385, 385, 390, 390, 390];
    
    divertorState.chfLimit = 10.0 + 1.2 * divertorState.velocity;
    divertorState.isDNB = divertorState.heatFlux > divertorState.chfLimit;
    
    document.getElementById('warn-divertor').style.display = divertorState.isDNB ? 'block' : 'none';
    
    let h_water = 800 + Math.pow(divertorState.velocity, 0.8) * 3500;
    if (divertorState.isDNB) {
      h_water *= 0.08; 
    }
    
    for (let step = 0; step < subSteps; step++) {
      let T_new = [...divertorState.T];
      
      const q_in = divertorState.heatFlux * 1e6; 
      const alpha_0 = k_grid[0] / (rho_grid[0] * cp_grid[0]);
      T_new[0] = divertorState.T[0] + (alpha_0 * dt / (dx * dx)) * (2 * divertorState.T[1] - 2 * divertorState.T[0] + (2 * q_in * dx / k_grid[0]));
      
      for (let i = 1; i < divertorState.nodes - 1; i++) {
        const k_mid = (k_grid[i] + k_grid[i+1]) / 2;
        const alpha = k_mid / (rho_grid[i] * cp_grid[i]);
        T_new[i] = divertorState.T[i] + (alpha * dt / (dx * dx)) * (divertorState.T[i+1] - 2 * divertorState.T[i] + divertorState.T[i-1]);
      }
      
      const T_water = 60.0; 
      const alpha_9 = k_grid[9] / (rho_grid[9] * cp_grid[9]);
      const Bi = h_water * dx / k_grid[9];
      T_new[9] = divertorState.T[9] + (alpha_9 * dt / (dx * dx)) * (2 * divertorState.T[8] - 2 * divertorState.T[9] - 2 * Bi * (divertorState.T[9] - T_water));
      
      divertorState.T = T_new;
    }
    
    divertorState.fluidParticles.forEach(p => {
      p.y += p.speed * (divertorState.velocity * 0.5 + 0.5);
      if (p.y > canvas.height * 0.75) p.y = 0;
    });
    
    if (divertorState.T[9] > 100.0) {
      const spawnChance = divertorState.isDNB ? 0.4 : 0.05 * (divertorState.T[9] - 100);
      if (Math.random() < spawnChance && divertorState.bubbles.length < 80) {
        divertorState.bubbles.push({
          x: canvas.width * 0.42,
          y: Math.random() * canvas.height * 0.75,
          size: 1 + Math.random() * 2,
          speedY: divertorState.velocity * 1.5,
          speedX: divertorState.isDNB ? (Math.random() - 0.5) * 2 : Math.random() * 2 
        });
      }
    }
    
    divertorState.bubbles.forEach((b, idx) => {
      b.y += b.speedY;
      b.x += b.speedX;
      b.size += divertorState.isDNB ? 0.3 : 0.1; 
      if (b.y > canvas.height * 0.75 || b.size > 20) {
        divertorState.bubbles.splice(idx, 1);
      }
    });
  } else if (activeTab === 'truss') {
    // Truss FEM is steady state, solved on slider inputs (solveTrussPhysics is called directly)
  } else if (activeTab === 'airfoil') {
    // Trace airfoil potential flow field streamlines
    airfoilState.particles.forEach(p => {
      // Map canvas pixel coords to airfoil aerodynamic space
      const flowWidth = canvas.width * 0.55;
      const originX = flowWidth * 0.22;
      const sizeChord = flowWidth * 0.56;
      const originY = canvas.height * 0.38;
      
      const aeroX = (p.x - originX) / sizeChord;
      const aeroY = -(p.y - originY) / sizeChord;
      
      const vel = getPotentialFlowVelocity(aeroX, aeroY);
      
      // Update particle coordinates (Y is inverted in canvas space)
      p.x += vel.u * 3.5;
      p.y -= vel.v * 3.5;
      
      // Wrap particles if they cross boundaries, go offscreen, or stagnate
      if (p.x > flowWidth || p.y < 0 || p.y > canvas.height * 0.75 || Math.abs(vel.u) < 0.05) {
        p.x = 0;
        p.y = Math.random() * canvas.height * 0.75;
      }
    });
  }
}

// ── Rendering Loop ───────────────────────────────────────────────────────────
function tick() {
  if (activeSection === 'lab') {
    updatePhysics();
    draw();
  }
  animationFrameId = requestAnimationFrame(tick);
}

// ── Drawing Coordination ─────────────────────────────────────────────────────
function draw() {
  ctx.fillStyle = COLORS.bgDark;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  const divideX = canvas.width * 0.58;
  ctx.strokeStyle = COLORS.border;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(divideX, 0);
  ctx.lineTo(divideX, canvas.height);
  ctx.stroke();
  
  if (activeTab === 'scramjet') {
    drawScramjetSim(divideX);
    drawScramjetPlot(divideX);
  } else if (activeTab === 'shock') {
    drawShockSim(divideX);
    drawShockPlot(divideX);
  } else if (activeTab === 'brake') {
    drawBrakeSim(divideX);
    drawBrakePlot(divideX);
  } else if (activeTab === 'divertor') {
    drawDivertorSim(divideX);
    drawDivertorPlot(divideX);
  } else if (activeTab === 'truss') {
    drawTrussSim(divideX);
    drawTrussPlot(divideX);
  } else if (activeTab === 'airfoil') {
    drawAirfoilSim(divideX);
    drawAirfoilPlot(divideX);
  }
}

// ── Render: Scramjet Combustor ───────────────────────────────────────────────
function drawScramjetSim(width) {
  const cY = canvas.height * 0.38;
  const h = 75; 
  
  ctx.fillStyle = 'rgba(224, 122, 34, 0.08)';
  ctx.fillRect(0, cY - h, width, h * 2);
  
  ctx.strokeStyle = COLORS.red;
  ctx.lineWidth = 2;
  ctx.beginPath();
  const time = Date.now() * 0.005;
  for (let x = 0; x < width; x += 10) {
    const yOff = Math.sin(x * 0.05 - time) * 12 + Math.cos(x * 0.02 - time * 0.5) * 8;
    if (x === 0) ctx.moveTo(x, cY + yOff);
    else ctx.lineTo(x, cY + yOff);
  }
  ctx.stroke();
  
  ctx.strokeStyle = COLORS.gold;
  ctx.beginPath();
  for (let x = 0; x < width; x += 10) {
    const yOff = Math.sin(x * 0.04 + time) * 10 - Math.cos(x * 0.03 + time * 0.8) * 6;
    if (x === 0) ctx.moveTo(x, cY - 20 + yOff);
    else ctx.lineTo(x, cY - 20 + yOff);
  }
  ctx.stroke();
  
  const jacketH = 30;
  const topJacketY = cY - h - jacketH - 10;
  const bottomJacketY = cY + h + 10;
  
  ctx.fillStyle = 'rgba(64, 176, 192, 0.06)';
  ctx.fillRect(0, topJacketY, width, jacketH);
  ctx.fillRect(0, bottomJacketY, width, jacketH);
  
  ctx.strokeStyle = COLORS.border;
  ctx.lineWidth = 1;
  ctx.strokeRect(0, topJacketY, width, jacketH);
  ctx.strokeRect(0, bottomJacketY, width, jacketH);
  
  scramjetState.fluidParticles.forEach(p => {
    const y = p.channel === 0 ? topJacketY + jacketH/2 : bottomJacketY + jacketH/2;
    const normX = p.x / width;
    const tempIndex = Math.min(9, Math.floor(normX * 10));
    const localT = scramjetState.T_cool[tempIndex] || 40.0;
    
    const r = Math.min(255, Math.floor((localT - 40) * 1.5));
    const g = Math.min(200, Math.floor((localT - 40) * 0.8 + 100));
    const b = Math.max(100, 255 - Math.floor((localT - 40) * 1.2));
    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
    
    ctx.beginPath();
    ctx.arc(p.x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  });
  
  const wallH = 10;
  const topWallY = cY - h - 10;
  const bottomWallY = cY + h;
  
  for (let i = 0; i < 10; i++) {
    const xStart = (width / 10) * i;
    const w = width / 10;
    const T_hot = scramjetState.T_wall_hot[i] || 400.0;
    const hotColor = `hsl(${Math.max(0, 240 - (T_hot - 300) * 0.35)}, 80%, 45%)`;
    
    ctx.fillStyle = hotColor;
    ctx.fillRect(xStart, topWallY, w, wallH);
    ctx.fillRect(xStart, bottomWallY, w, wallH);
  }
  
  ctx.fillStyle = COLORS.text;
  ctx.font = '10px monospace';
  ctx.fillText("SUPERSONIC COMBUSTION GAS PATH", 15, cY + 5);
  ctx.fillStyle = COLORS.cyan;
  ctx.fillText("LH2 COOLING JACKETS (CO-FLOW)", 15, topJacketY - 6);
}

function drawScramjetPlot(divideX) {
  const plotX = divideX + 40;
  const plotY = 60;
  const plotW = canvas.width - plotX - 30;
  const plotH = canvas.height - 120;
  
  drawPlotAxes(plotX, plotY, plotW, plotH, "Axial Position [m]", "Temperature [K]");
  
  const ptsCool = [];
  const ptsWall = [];
  
  for (let i = 0; i < 10; i++) {
    const nx = i / 9;
    const px = plotX + nx * plotW;
    const pyC = plotY + plotH - ((scramjetState.T_cool[i] - 40) / 1100) * plotH;
    const pyW = plotY + plotH - ((scramjetState.T_wall_hot[i] - 40) / 1100) * plotH;
    ptsCool.push({x: px, y: pyC});
    ptsWall.push({x: px, y: pyW});
  }
  
  ctx.strokeStyle = COLORS.red;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ptsWall.forEach((p, idx) => {
    if (idx === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.stroke();
  
  ctx.strokeStyle = COLORS.cyan;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ptsCool.forEach((p, idx) => {
    if (idx === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.stroke();
  
  const safeY = plotY + plotH - ((1000 - 40) / 1100) * plotH;
  ctx.strokeStyle = 'rgba(192, 64, 64, 0.7)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(plotX, safeY);
  ctx.lineTo(plotX + plotW, safeY);
  ctx.stroke();
  ctx.setLineDash([]);
  
  ctx.fillStyle = COLORS.red;
  ctx.font = '8px monospace';
  ctx.fillText("GRCop-84 Safe Limit (1000 K)", plotX + 10, safeY - 4);
  
  drawLegend(plotX + 15, plotY + 15, [
    { label: "Wall Hot Side Temp", color: COLORS.red },
    { label: "Cryogenic LH2 Temp", color: COLORS.cyan }
  ]);
  
  ctx.fillStyle = COLORS.textDim;
  ctx.font = '9px monospace';
  ctx.fillText("0.0", plotX - 8, plotY + plotH + 12);
  ctx.fillText("1.0", plotX + plotW - 10, plotY + plotH + 12);
  ctx.fillText("40K", plotX - 25, plotY + plotH);
  ctx.fillText("1100K", plotX - 35, plotY + 8);
}

// ── Render: Oblique Shock Wave ───────────────────────────────────────────────
function drawShockSim(width) {
  const cY = canvas.height * 0.38;
  const wX = width * 0.4; 
  const wY = cY;
  
  const thetaRad = (shockState.theta * Math.PI) / 180;
  const wedgeLength = width * 0.38;
  const topX = wX + wedgeLength;
  const topY = wY - wedgeLength * Math.tan(thetaRad);
  const bottomY = wY + wedgeLength * Math.tan(thetaRad);
  
  ctx.fillStyle = '#222';
  ctx.beginPath();
  ctx.moveTo(wX, wY);
  ctx.lineTo(topX, topY);
  ctx.lineTo(topX, bottomY);
  ctx.closePath();
  ctx.fill();
  
  ctx.strokeStyle = COLORS.grey;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  
  if (shockState.detached) {
    ctx.strokeStyle = COLORS.red;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let x = wX - 80; x < width * 0.7; x += 5) {
      const dx = x - wX;
      const yOff = Math.sqrt(Math.max(0, 16000 + 400 * dx));
      if (x === wX - 80) ctx.moveTo(x, wY - yOff);
      else ctx.lineTo(x, wY - yOff);
    }
    for (let x = wX - 80; x < width * 0.7; x += 5) {
      const dx = x - wX;
      const yOff = Math.sqrt(Math.max(0, 16000 + 400 * dx));
      if (x === wX - 80) ctx.moveTo(x, wY + yOff);
      else ctx.lineTo(x, wY + yOff);
    }
    ctx.stroke();
  } else {
    const betaRad = (shockState.beta * Math.PI) / 180;
    const lineL = width * 0.5;
    ctx.strokeStyle = COLORS.gold;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(wX, wY);
    ctx.lineTo(wX + lineL, wY - lineL * Math.tan(betaRad));
    ctx.moveTo(wX, wY);
    ctx.lineTo(wX + lineL, wY + lineL * Math.tan(betaRad));
    ctx.stroke();
  }
  
  shockState.particles.forEach(p => {
    ctx.fillStyle = shockState.detached ? COLORS.red : COLORS.cyan;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
    ctx.fill();
  });
  
  ctx.fillStyle = COLORS.text;
  ctx.font = '10px monospace';
  ctx.fillText(`M1: ${shockState.mach.toFixed(1)}`, 15, 20);
  ctx.fillText(`Wedge Deflection: ${shockState.theta}°`, 15, 34);
  if (!shockState.detached) {
    ctx.fillText(`Shock Angle: ${shockState.beta.toFixed(1)}°`, 15, 48);
  } else {
    ctx.fillStyle = COLORS.red;
    ctx.fillText("Shock Detached", 15, 48);
  }
}

function drawShockPlot(divideX) {
  const plotX = divideX + 40;
  const plotY = 60;
  const plotW = canvas.width - plotX - 30;
  const plotH = canvas.height - 120;
  
  drawPlotAxes(plotX, plotY, plotW, plotH, "Wedge Angle [deg]", "Shock Wave Angle [deg]");
  
  const M = shockState.mach;
  const gamma = 1.4;
  
  const f_eqn = (thetaDeg, betaRad) => {
    const thetaRad = (thetaDeg * Math.PI) / 180;
    const sinB = Math.sin(betaRad);
    const cosB = Math.cos(betaRad);
    const cotB = cosB / sinB;
    const num = M * M * sinB * sinB - 1;
    const den = M * M * (gamma + Math.cos(2 * betaRad)) + 2;
    return Math.tan(thetaRad) - 2 * cotB * num / den;
  };
  
  const ptsCurve = [];
  
  for (let th = 0; th <= 40; th += 1) {
    let low = (th * Math.PI) / 180 + 0.0001;
    let high = Math.PI / 2;
    
    if (f_eqn(th, low) * f_eqn(th, high) <= 0) {
      let sol = low;
      for (let iter = 0; iter < 25; iter++) {
        let mid = (low + high) / 2;
        let val = f_eqn(th, mid);
        if (f_eqn(th, low) * val < 0) high = mid;
        else { low = mid; sol = mid; }
      }
      const py = plotY + plotH - ((sol * 180 / Math.PI) / 90) * plotH;
      const px = plotX + (th / 40) * plotW;
      ptsCurve.push({x: px, y: py});
    }
  }
  
  ctx.strokeStyle = COLORS.cyan;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ptsCurve.forEach((p, idx) => {
    if (idx === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.stroke();
  
  if (!shockState.detached) {
    const px = plotX + (shockState.theta / 40) * plotW;
    const py = plotY + plotH - (shockState.beta / 90) * plotH;
    ctx.fillStyle = COLORS.gold;
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = COLORS.text;
    ctx.stroke();
  }
  
  ctx.fillStyle = COLORS.text;
  ctx.font = '9px monospace';
  ctx.fillText(`Downstream Mach M2 : ${shockState.detached ? "N/A" : shockState.m2.toFixed(2)}`, plotX + 15, plotY + 15);
  ctx.fillText(`Static Pressure p2/p1: ${shockState.detached ? "N/A" : shockState.p_ratio.toFixed(2)}`, plotX + 15, plotY + 28);
  ctx.fillText(`Static Temp T2/T1    : ${shockState.detached ? "N/A" : shockState.t_ratio.toFixed(2)}`, plotX + 15, plotY + 41);
  
  ctx.fillStyle = COLORS.textDim;
  ctx.fillText("0°", plotX - 5, plotY + plotH + 12);
  ctx.fillText("40°", plotX + plotW - 10, plotY + plotH + 12);
  ctx.fillText("0°", plotX - 15, plotY + plotH);
  ctx.fillText("90°", plotX - 20, plotY + 8);
}

// ── Render: F1 Brake Disc ───────────────────────────────────────────────────
function drawBrakeSim(width) {
  const cX = width / 2;
  const cY = canvas.height * 0.38;
  const outerR = 90;
  const innerR = 40;
  
  ctx.save();
  ctx.translate(cX, cY);
  ctx.rotate(brakeState.rotationAngle);
  
  ctx.fillStyle = '#1c1c1a';
  ctx.beginPath();
  ctx.arc(0, 0, outerR, 0, Math.PI*2);
  ctx.arc(0, 0, innerR, 0, Math.PI*2, true); 
  ctx.fill();
  
  if (brakeState.temp > 100) {
    const heatAlpha = Math.min(0.85, (brakeState.temp - 100) / 800);
    const r = Math.min(255, Math.floor((brakeState.temp - 100) * 0.4 + 100));
    const g = Math.min(180, Math.max(0, Math.floor((brakeState.temp - 400) * 0.3)));
    const b = Math.min(60, Math.max(0, Math.floor((brakeState.temp - 600) * 0.1)));
    
    let grad = ctx.createRadialGradient(0, 0, innerR, 0, 0, outerR);
    grad.addColorStop(0, 'rgba(0,0,0,0)');
    grad.addColorStop(0.3, `rgba(${r},${g},${b},${heatAlpha})`);
    grad.addColorStop(1, `rgba(${r - 50},${g},${b},${heatAlpha * 0.7})`);
    
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(0, 0, outerR, 0, Math.PI*2);
    ctx.arc(0, 0, innerR, 0, Math.PI*2, true);
    ctx.fill();
  }
  
  ctx.strokeStyle = '#050505';
  ctx.lineWidth = 1.5;
  for (let i = 0; i < 24; i++) {
    const angle = (i * Math.PI) / 12;
    ctx.beginPath();
    ctx.moveTo(Math.cos(angle) * innerR, Math.sin(angle) * innerR);
    ctx.lineTo(Math.cos(angle) * outerR, Math.sin(angle) * outerR);
    ctx.stroke();
  }
  
  ctx.restore();
  
  ctx.fillStyle = COLORS.grey;
  ctx.strokeStyle = COLORS.border;
  ctx.lineWidth = 1.5;
  
  ctx.save();
  ctx.translate(cX, cY);
  ctx.beginPath();
  ctx.arc(0, 0, outerR + 10, -Math.PI/6, -Math.PI/2, true);
  ctx.lineTo(Math.cos(-Math.PI/2)*(innerR + 10), Math.sin(-Math.PI/2)*(innerR + 10));
  ctx.arc(0, 0, innerR + 10, -Math.PI/2, -Math.PI/6);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();
  
  ctx.fillStyle = 'rgba(64, 176, 192, 0.5)';
  const time = Date.now() * 0.005;
  for (let i = 0; i < 15; i++) {
    const xOff = (i * 12 + time * 30) % 150;
    const px = cX - outerR - 60 + xOff;
    const py = cY - 20 + Math.sin(i * 1.3 + time) * 6;
    if (px < cX - 10) {
      ctx.beginPath();
      ctx.arc(px, py, 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  
  ctx.fillStyle = COLORS.text;
  ctx.font = '10px monospace';
  ctx.fillText("CARBON-CARBON DISC ASSEMBLY", cX - 70, cY + outerR + 25);
  ctx.fillStyle = COLORS.cyan;
  ctx.fillText("COOLING INLET DUCT", cX - outerR - 120, cY - 35);
}

function drawBrakePlot(divideX) {
  const plotX = divideX + 40;
  const plotY = 60;
  const plotW = canvas.width - plotX - 30;
  const plotH = canvas.height - 120;
  
  drawPlotAxes(plotX, plotY, plotW, plotH, "Time History", "Disc Temperature [°C]");
  
  const pts = [];
  const len = brakeState.history.length;
  for (let i = 0; i < len; i++) {
    const px = plotX + (i / 149) * plotW;
    const val = brakeState.history[i];
    const py = plotY + plotH - ((val - 20) / 1000) * plotH;
    pts.push({x: px, y: py});
  }
  
  ctx.strokeStyle = COLORS.gold;
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  pts.forEach((p, idx) => {
    if (idx === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.stroke();
  
  const glazeY = plotY + plotH - ((300 - 20) / 1000) * plotH;
  ctx.strokeStyle = 'rgba(64, 176, 192, 0.4)';
  ctx.setLineDash([2, 2]);
  ctx.beginPath();
  ctx.moveTo(plotX, glazeY);
  ctx.lineTo(plotX + plotW, glazeY);
  ctx.stroke();
  
  const oxidY = plotY + plotH - ((750 - 20) / 1000) * plotH;
  ctx.strokeStyle = 'rgba(192, 64, 64, 0.5)';
  ctx.beginPath();
  ctx.moveTo(plotX, oxidY);
  ctx.lineTo(plotX + plotW, oxidY);
  ctx.stroke();
  ctx.setLineDash([]);
  
  ctx.fillStyle = COLORS.cyan;
  ctx.font = '8px monospace';
  ctx.fillText("Glazing Limit (300°C)", plotX + 5, glazeY + 10);
  ctx.fillStyle = COLORS.red;
  ctx.fillText("Oxidation Limit (750°C)", plotX + 5, oxidY - 4);
  
  ctx.fillStyle = COLORS.text;
  ctx.font = '10px monospace';
  ctx.fillText(`Current Temp : ${brakeState.temp.toFixed(1)} °C`, plotX + 15, plotY + 15);
  ctx.fillText(`Vehicle Speed: ${brakeState.speed.toFixed(0)} km/h`, plotX + 15, plotY + 28);
  
  ctx.fillStyle = COLORS.textDim;
  ctx.fillText("Past 5s", plotX, plotY + plotH + 12);
  ctx.fillText("Now", plotX + plotW - 20, plotY + plotH + 12);
  ctx.fillText("20°C", plotX - 30, plotY + plotH);
  ctx.fillText("1020°C", plotX - 45, plotY + 8);
}

// ── Render: Tokamak Divertor ─────────────────────────────────────────────────
function drawDivertorSim(width) {
  const blockW = width * 0.48;
  const startX = 20;
  
  const x_W = 0.62 * blockW;
  const x_Cu = 0.72 * blockW;
  const x_Tube = 0.88 * blockW;
  
  let plasmaGrad = ctx.createLinearGradient(0, 0, startX, 0);
  plasmaGrad.addColorStop(0, 'rgba(128, 0, 255, 0.4)');
  plasmaGrad.addColorStop(1, 'rgba(128, 0, 255, 0)');
  ctx.fillStyle = plasmaGrad;
  ctx.fillRect(0, 0, startX, canvas.height * 0.75);
  
  ctx.strokeStyle = 'rgba(255, 128, 255, 0.6)';
  ctx.lineWidth = 1;
  if (Math.random() < 0.1) {
    ctx.beginPath();
    ctx.moveTo(0, Math.random() * canvas.height * 0.75);
    ctx.lineTo(startX, Math.random() * canvas.height * 0.75);
    ctx.stroke();
  }
  
  ctx.fillStyle = '#222';
  ctx.fillRect(startX, 10, x_W - startX, canvas.height * 0.75 - 20);
  ctx.strokeStyle = COLORS.grey;
  ctx.strokeRect(startX, 10, x_W - startX, canvas.height * 0.75 - 20);
  
  ctx.fillStyle = '#8f563b';
  ctx.fillRect(x_W, 10, x_Cu - x_W, canvas.height * 0.75 - 20);
  ctx.strokeRect(x_W, 10, x_Cu - x_W, canvas.height * 0.75 - 20);
  
  ctx.fillStyle = '#b77355';
  ctx.fillRect(x_Cu, 10, x_Tube - x_Cu, canvas.height * 0.75 - 20);
  ctx.strokeRect(x_Cu, 10, x_Tube - x_Cu, canvas.height * 0.75 - 20);
  
  ctx.fillStyle = 'rgba(64, 128, 192, 0.1)';
  ctx.fillRect(x_Tube, 10, blockW - x_Tube, canvas.height * 0.75 - 20);
  ctx.strokeRect(x_Tube, 10, blockW - x_Tube, canvas.height * 0.75 - 20);
  
  divertorState.fluidParticles.forEach(p => {
    ctx.fillStyle = COLORS.cyan;
    ctx.beginPath();
    const px = x_Tube + (p.x % (blockW - x_Tube));
    ctx.arc(px, p.y, 2, 0, Math.PI * 2);
    ctx.fill();
  });
  
  ctx.fillStyle = divertorState.isDNB ? 'rgba(140, 140, 140, 0.7)' : 'rgba(255,255,255,0.7)';
  divertorState.bubbles.forEach(b => {
    ctx.beginPath();
    const px = x_Tube + b.x % (blockW - x_Tube);
    ctx.arc(px, b.y, b.size, 0, Math.PI * 2);
    ctx.fill();
  });
  
  if (divertorState.isDNB) {
    ctx.fillStyle = 'rgba(100, 100, 100, 0.6)';
    ctx.fillRect(x_Tube, 10, 8, canvas.height * 0.75 - 20);
  }
  
  ctx.strokeStyle = COLORS.red;
  ctx.lineWidth = 2;
  const tFlux = Date.now() * 0.01;
  for (let i = 0; i < 4; i++) {
    const y = 50 + i * 80;
    const arrowX = startX + 20 + (tFlux % 40);
    ctx.beginPath();
    ctx.moveTo(arrowX, y);
    ctx.lineTo(arrowX + 25, y);
    ctx.lineTo(arrowX + 20, y - 4);
    ctx.moveTo(arrowX + 25, y);
    ctx.lineTo(arrowX + 20, y + 4);
    ctx.stroke();
  }
  
  ctx.fillStyle = COLORS.text;
  ctx.font = '8px monospace';
  ctx.fillText("PLASMA HF", startX - 18, 28);
  ctx.fillText("TUNGSTEN SHIELD", startX + 10, 24);
  ctx.fillText("Cu INTER", x_W + 2, 24);
  ctx.fillText("CuCrZr", x_Cu + 2, 24);
  ctx.fillStyle = COLORS.cyan;
  ctx.fillText("WATER", x_Tube + 15, 24);
}

function drawDivertorPlot(divideX) {
  const plotX = divideX + 40;
  const plotY = 60;
  const plotW = canvas.width - plotX - 30;
  const plotH = canvas.height - 120;
  
  drawPlotAxes(plotX, plotY, plotW, plotH, "Conduction Thickness [x]", "Temperature [°C]");
  
  const pts = [];
  for (let i = 0; i < divertorState.nodes; i++) {
    const nx = i / (divertorState.nodes - 1);
    const px = plotX + nx * plotW;
    const val = divertorState.T[i];
    const py = plotY + plotH - ((val - 60) / 840) * plotH;
    pts.push({x: px, y: py});
  }
  
  ctx.strokeStyle = COLORS.gold;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  pts.forEach((p, idx) => {
    if (idx === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.stroke();
  
  ctx.fillStyle = COLORS.cyan;
  pts.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fill();
  });
  
  ctx.fillStyle = COLORS.text;
  ctx.font = '9px monospace';
  ctx.fillText(`Plasma Heat Flux  : ${divertorState.heatFlux.toFixed(1)} MW/m²`, plotX + 15, plotY + 15);
  ctx.fillText(`CHF Critical Limit: ${divertorState.chfLimit.toFixed(1)} MW/m²`, plotX + 15, plotY + 28);
  ctx.fillText(`Peak Tile Temp    : ${divertorState.T[0].toFixed(0)} °C`, plotX + 15, plotY + 41);
  ctx.fillText(`Water-Wall Interface: ${divertorState.T[9].toFixed(0)} °C`, plotX + 15, plotY + 54);
  
  ctx.fillStyle = COLORS.textDim;
  ctx.fillText("0.0 (Plasma)", plotX, plotY + plotH + 12);
  ctx.fillText("25mm (Water)", plotX + plotW - 35, plotY + plotH + 12);
  ctx.fillText("60°C", plotX - 30, plotY + plotH);
  ctx.fillText("900°C", plotX - 35, plotY + 8);
}

// ── Render: 2D Truss Solver ──────────────────────────────────────────────────
function drawTrussSim(width) {
  // Map truss coords to canvas space: Node 0 is at bottom left, Node 6 at center right
  // Node 0: (0.0, 0.0) -> px: 40, py: 320
  // Node 1: (0.0, 1.0) -> px: 40, py: 120 (Y inverted in canvas)
  const scaleX = 140; // pixels per meter
  const scaleY = 180;
  const originX = 60;
  const originY = 320;
  
  // Draw pinned supports (triangles) at Node 0 and Node 1
  ctx.fillStyle = COLORS.gold;
  for (let nIdx of [0, 1]) {
    const px = originX + trussState.nodes[nIdx][0] * scaleX;
    const py = originY - trussState.nodes[nIdx][1] * scaleY;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px - 12, py - 8);
    ctx.lineTo(px - 12, py + 8);
    ctx.closePath();
    ctx.fill();
  }
  
  // Compute scaled deformed coordinates
  const amp = 300.0; // amplify displacements
  let def_px = [];
  let def_py = [];
  
  for (let i = 0; i < trussState.nodes.length; i++) {
    const dx = trussState.displacements[2*i] * amp;
    const dy = trussState.displacements[2*i+1] * amp;
    
    def_px.push(originX + (trussState.nodes[i][0] + dx) * scaleX);
    def_py.push(originY - (trussState.nodes[i][1] + dy) * scaleY);
  }
  
  // Render Elements
  for (let i = 0; i < trussState.elements.length; i++) {
    const [n1, n2] = trussState.elements[i];
    const stress = trussState.stresses[i] || 0.0;
    
    // Draw baseline references (dotted lines)
    ctx.strokeStyle = '#222';
    ctx.lineWidth = 0.8;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(originX + trussState.nodes[n1][0] * scaleX, originY - trussState.nodes[n1][1] * scaleY);
    ctx.lineTo(originX + trussState.nodes[n2][0] * scaleX, originY - trussState.nodes[n2][1] * scaleY);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Map stress to color (Red: Tension, Blue: Compression)
    const limit = trussState.yield / 1.5;
    let colorVal = Math.min(1.0, Math.max(-1.0, stress / limit));
    let color;
    if (colorVal > 0) {
      color = `rgba(192, 64, 64, ${0.2 + 0.8 * colorVal})`; // tension red
    } else {
      color = `rgba(64, 128, 192, ${0.2 + 0.8 * Math.abs(colorVal)})`; // compression blue
    }
    
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.0 + (trussState.areas[i] / 0.001) * 3.5;
    ctx.beginPath();
    ctx.moveTo(def_px[n1], def_py[n1]);
    ctx.lineTo(def_px[n2], def_py[n2]);
    ctx.stroke();
  }
  
  // Render Joints
  ctx.fillStyle = COLORS.text;
  for (let i = 0; i < trussState.nodes.length; i++) {
    ctx.beginPath();
    ctx.arc(def_px[i], def_py[i], 4, 0, Math.PI * 2);
    ctx.fill();
  }
  
  // Tip downward force arrow at Node 6
  ctx.strokeStyle = COLORS.red;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(def_px[6], def_py[6] - 40);
  ctx.lineTo(def_px[6], def_py[6] - 5);
  ctx.lineTo(def_px[6] - 5, def_py[6] - 12);
  ctx.moveTo(def_px[6], def_py[6] - 5);
  ctx.lineTo(def_px[6] + 5, def_py[6] - 12);
  ctx.stroke();
  
  ctx.fillStyle = COLORS.text;
  ctx.font = '10px monospace';
  ctx.fillText("Pinned Boundaries", 5, 80);
  ctx.fillStyle = COLORS.red;
  ctx.fillText(`TIP LOAD: ${trussState.load} kN`, def_px[6] - 40, def_py[6] - 50);
}

function drawTrussPlot(divideX) {
  const plotX = divideX + 40;
  const plotY = 60;
  const plotW = canvas.width - plotX - 30;
  const plotH = canvas.height - 120;
  
  drawPlotAxes(plotX, plotY, plotW, plotH, "Truss Members [1-12]", "Relative Member Stress [%]");
  
  // Draw bar chart of member stress levels vs yield stress limit
  const n_elements = trussState.elements.length;
  const limit = trussState.yield / 1.5;
  const barW = (plotW / n_elements) * 0.6;
  const step = plotW / n_elements;
  
  for (let i = 0; i < n_elements; i++) {
    const stress = trussState.stresses[i] || 0.0;
    const ratio = Math.abs(stress) / limit;
    
    const bx = plotX + i * step + step * 0.2;
    const bh = Math.min(1.0, ratio) * plotH;
    const by = plotY + plotH - bh;
    
    ctx.fillStyle = stress > 0 ? COLORS.red : COLORS.blue;
    ctx.fillRect(bx, by, barW, bh);
    
    // Draw bar outline
    ctx.strokeStyle = COLORS.border;
    ctx.lineWidth = 0.5;
    ctx.strokeRect(bx, by, barW, bh);
  }
  
  // Limit line (100% capacity)
  ctx.strokeStyle = 'rgba(192,64,64,0.6)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(plotX, plotY);
  ctx.lineTo(plotX + plotW, plotY);
  ctx.stroke();
  ctx.setLineDash([]);
  
  ctx.fillStyle = COLORS.red;
  ctx.font = '8px monospace';
  ctx.fillText("Allowable Yield Stress Limit (SF=1.5)", plotX + 10, plotY - 4);
  
  // Statistics display
  ctx.fillStyle = COLORS.text;
  ctx.font = '9px monospace';
  ctx.fillText(`Total Structural Mass: ${trussState.totalMass.toFixed(1)} kg`, plotX + 15, plotY + 15);
  ctx.fillText(`Max Joint Deflection : ${Math.max(...trussState.displacements.map(Math.abs))*1000.0.toFixed(2)} mm`, plotX + 15, plotY + 28);
  ctx.fillText(`Yield Safety Margin  : ${trussState.isCollapsed ? "EXCEEDED" : "SAFE"}`, plotX + 15, plotY + 41);
  
  ctx.fillStyle = COLORS.textDim;
  ctx.fillText("1", plotX + step*0.2, plotY + plotH + 12);
  ctx.fillText("12", plotX + 11*step + step*0.2, plotY + plotH + 12);
  ctx.fillText("0%", plotX - 20, plotY + plotH);
  ctx.fillText("100%", plotX - 30, plotY + 8);
}

// ── Render: NACA Airfoil ─────────────────────────────────────────────────────
function drawAirfoilSim(width) {
  const flowWidth = width;
  
  // Render streamline tracer particles
  ctx.fillStyle = COLORS.cyan;
  airfoilState.particles.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 1.5, 0, Math.PI * 2);
    ctx.fill();
  });
  
  // Render airfoil solid body shape
  // Nodes mapping: x_aero = 0.0 to 1.0 -> px: flowWidth*0.25 to flowWidth*0.75
  const originX = flowWidth * 0.22;
  const sizeChord = flowWidth * 0.56;
  const originY = canvas.height * 0.38;
  
  ctx.fillStyle = '#222';
  ctx.strokeStyle = COLORS.gold;
  ctx.lineWidth = 2.0;
  
  ctx.beginPath();
  const N = airfoilState.nodesX.length;
  for (let i = 0; i < N; i++) {
    const px = originX + airfoilState.nodesX[i] * sizeChord;
    const py = originY - airfoilState.nodesY[i] * sizeChord;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  
  // Vector arrow of alpha AoA
  const alphaRad = (airfoilState.alpha * Math.PI) / 180;
  ctx.strokeStyle = COLORS.gold;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(originX - 60, originY);
  ctx.lineTo(originX - 10, originY - 50 * Math.sin(alphaRad));
  // arrow head
  const arrowX = originX - 10;
  const arrowY = originY - 50 * Math.sin(alphaRad);
  ctx.lineTo(arrowX - 8, arrowY + 2);
  ctx.moveTo(arrowX, arrowY);
  ctx.lineTo(arrowX - 4, arrowY + 8);
  ctx.stroke();
  
  ctx.fillStyle = COLORS.text;
  ctx.font = '10px monospace';
  ctx.fillText("NACA " + airfoilState.camber + airfoilState.pos + (airfoilState.thick < 10 ? '0' : '') + airfoilState.thick + " PROFILE", originX + 20, originY + 80);
}

function drawAirfoilPlot(divideX) {
  const plotX = divideX + 40;
  const plotY = 60;
  const plotW = canvas.width - plotX - 30;
  const plotH = canvas.height - 120;
  
  drawPlotAxes(plotX, plotY, plotW, plotH, "Chord Coordinate [x/c]", "Pressure Coefficient [Cp]");
  
  // Plot the Cp distribution (conventionally inverted y-axis, negative values up)
  const N = airfoilState.Cp.length;
  const half = N / 2;
  const ptsLower = [];
  const ptsUpper = [];
  
  for (let i = 0; i < N; i++) {
    const nx = airfoilState.xc[i];
    const px = plotX + nx * plotW;
    const val = airfoilState.Cp[i];
    // Map Cp range from -2.0 to 1.0
    const py = plotY + ((val - 1.0) / -3.0) * plotH;
    
    if (i < half) ptsLower.push({x: px, y: py});
    else ptsUpper.push({x: px, y: py});
  }
  
  // Render Upper surface Cp (negative values peak up)
  ctx.strokeStyle = COLORS.red;
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ptsUpper.forEach((p, idx) => {
    if (idx === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.stroke();
  
  // Render Lower surface Cp
  ctx.strokeStyle = COLORS.cyan;
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ptsLower.forEach((p, idx) => {
    if (idx === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.stroke();
  
  // Current values
  ctx.fillStyle = COLORS.text;
  ctx.font = '10px monospace';
  ctx.fillText(`Lift Coefficient Cl: ${airfoilState.Cl.toFixed(3)}`, plotX + 15, plotY + 15);
  ctx.fillText(`Angle of Attack AoA: ${airfoilState.alpha.toFixed(1)}°`, plotX + 15, plotY + 28);
  
  drawLegend(plotX + 15, plotY + 45, [
    { label: "Upper Surface (Suction)", color: COLORS.red },
    { label: "Lower Surface (Pressure)", color: COLORS.cyan }
  ]);
  
  ctx.fillStyle = COLORS.textDim;
  ctx.fillText("0.0 (LE)", plotX, plotY + plotH + 12);
  ctx.fillText("1.0 (TE)", plotX + plotW - 35, plotY + plotH + 12);
  ctx.fillText("1.0 (Cp)", plotX - 42, plotY + plotH);
  ctx.fillText("-2.0 (Cp)", plotX - 48, plotY + 8);
}

// ── Global Equation Display Helper ──────────────────────────────────────────
function updateEquationDisplay() {
  const container = document.getElementById('equation-display');
  if (activeTab === 'scramjet') {
    container.innerHTML = `
      <div style="font-family: var(--font-mono); font-size: 0.8rem; line-height: 1.6;">
        <p style="color: var(--accent-gold); font-weight: bold; margin-bottom: 0.5rem;">Supersonic Convection:</p>
        <code>q_flux = h_g * (T_rec - T_wall_hot)</code><br>
        <code>T_rec = T_static * (1 + r * ((&gamma;-1)/2) * M^2)</code>
        <p style="color: var(--accent-cyan); font-weight: bold; margin: 0.8rem 0 0.5rem 0;">Coolant Channels (Sieder-Tate):</p>
        <code>Nu = 0.023 * Re^0.8 * Pr^0.4</code><br>
        <code>h_c = Nu * k_fluid / D_hydraulic</code>
        <p style="color: var(--text-primary); font-family: var(--font-sans); margin-top: 1rem; font-size: 0.8rem;">
          GRCop-84 thermal conductivity <code>k = 320 W/mK</code> maintains walls below 1000 K. Inconel 718 at <code>19 W/mK</code> overheats rapidly.
        </p>
      </div>
    `;
  } else if (activeTab === 'shock') {
    container.innerHTML = `
      <div style="font-family: var(--font-mono); font-size: 0.8rem; line-height: 1.6;">
        <p style="color: var(--accent-gold); font-weight: bold; margin-bottom: 0.5rem;">&theta;-&beta;-M Relation:</p>
        <code>tan(&theta;) = 2*cot(&beta;) * [ (M₁² sin²&beta; - 1) / (M₁²(&gamma; + cos 2&beta;) + 2) ]</code>
        <p style="color: var(--accent-cyan); font-weight: bold; margin: 0.8rem 0 0.5rem 0;">Shock Relations (&gamma; = 1.4):</p>
        <code>M_n1 = M₁ * sin(&beta;)</code><br>
        <code>p₂/p₁ = (2&gamma; M_n1² - (&gamma;-1)) / (&gamma;+1)</code><br>
        <code>T₂/T₁ = [ (2&gamma; M_n1² - (&gamma;-1))((&gamma;-1)M_n1² + 2) ] / [ (&gamma;+1)² M_n1² ]</code>
      </div>
    `;
  } else if (activeTab === 'brake') {
    container.innerHTML = `
      <div style="font-family: var(--font-mono); font-size: 0.8rem; line-height: 1.6;">
        <p style="color: var(--accent-gold); font-weight: bold; margin-bottom: 0.5rem;">Energy Integration:</p>
        <code>dT/dt = (Q_brake - Q_conv - Q_rad) / (m_disc * c_p)</code>
        <p style="color: var(--accent-cyan); font-weight: bold; margin: 0.8rem 0 0.5rem 0;">Arrhenius Mass Loss Rate:</p>
        <code>dM/dt = A * exp(-E_a / (R * T_K))</code>
        <p style="color: var(--text-primary); font-family: var(--font-sans); margin-top: 1rem; font-size: 0.8rem;">
          Discs operate best in <code>300°C -- 700°C</code>. Below 300°C leads to glazing. Above 750°C triggers rapid oxidation in atmospheric air.
        </p>
      </div>
    `;
  } else if (activeTab === 'divertor') {
    container.innerHTML = `
      <div style="font-family: var(--font-mono); font-size: 0.8rem; line-height: 1.6;">
        <p style="color: var(--accent-gold); font-weight: bold; margin-bottom: 0.5rem;">1D Transient Conduction FDM:</p>
        <code>&rho; c_p (&part;T/&part;t) = k (&part;²T/&part;x²)</code><br>
        <code>T_new[i] = T[i] + Fo * (T[i+1] - 2T[i] + T[i-1])</code>
        <p style="color: var(--accent-cyan); font-weight: bold; margin: 0.8rem 0 0.5rem 0;">Cooling Water Heat Transfer:</p>
        <code>q_water = h_w * (T_wall - T_water)</code><br>
        <code>CHF Limit &approx; 10 + 1.2 * V_water [MW/m²]</code>
      </div>
    `;
  } else if (activeTab === 'truss') {
    container.innerHTML = `
      <div style="font-family: var(--font-mono); font-size: 0.8rem; line-height: 1.6;">
        <p style="color: var(--accent-gold); font-weight: bold; margin-bottom: 0.5rem;">FEM Stiffness Formulation:</p>
        <code>[K_local] = (A * E / L) * [[c², cs, -c², -cs], [cs, s², -cs, -s²], [-c², -cs, c², cs], [-cs, -s², cs, s²]]</code>
        <p style="color: var(--accent-cyan); font-weight: bold; margin: 0.8rem 0 0.5rem 0;">System Solution:</p>
        <code>[K_reduced] * {U_reduced} = {F_reduced}</code><br>
        <code>&sigma;_axial = E * (dL / L)</code>
        <p style="color: var(--text-primary); font-family: var(--font-sans); margin-top: 1rem; font-size: 0.8rem;">
          Tension elements are sized against yield stress. Compression elements are sized against Euler buckling: <code>P_crit = pi² * E * I / L²</code>.
        </p>
      </div>
    `;
  } else if (activeTab === 'airfoil') {
    container.innerHTML = `
      <div style="font-family: var(--font-mono); font-size: 0.8rem; line-height: 1.6;">
        <p style="color: var(--accent-gold); font-weight: bold; margin-bottom: 0.5rem;">Hess-Smith boundary equations:</p>
        <code>Sum_j [ A_ij * q_j ] + B_iN * &gamma; = -V_inf * n_i</code>
        <p style="color: var(--accent-cyan); font-weight: bold; margin: 0.8rem 0 0.5rem 0;">Kutta Condition (Trailing Edge):</p>
        <code>V_t1 + V_tN = 0</code><br>
        <code>C_p = 1 - (V / V_inf)²</code>
      </div>
    `;
  }
}

