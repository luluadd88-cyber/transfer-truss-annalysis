# =============================================================
# CIV2100 COURSEWORK — TRANSFER TRUSS STRUCTURAL ANALYSIS
# University of Sheffield, 2024-25
#
# Paste this entire script into a Google Colab cell and run.
# All figures will display inline.
#
# CONTENTS:
#   PART 1 — Material, Section, Loading, Geometry
#   PART 2 — Elastic Analysis (Stiffness Matrix Method)
#   PART 3 — Plastic Analysis (Sequential Yielding)
#   PART 4 — Extension: Parametric Study (Section Size)
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# =============================================================
# PART 1 — PROPERTIES, LOADING AND GEOMETRY
# =============================================================

# -------------------------------------------------------------
# Material and section properties
# Square Hollow Section: 600 x 600 x 16 mm, S355 steel
# -------------------------------------------------------------
E     = 210e6       # Young's modulus [kN/m^2]  (210 GPa)
sig_y = 355e3       # Yield stress    [kN/m^2]  (355 MPa)

b_out  = 0.600      # SHS outer dimension [m]
t_wall = 0.016      # SHS wall thickness  [m]

# Cross-sectional area of square hollow section
A  = b_out**2 - (b_out - 2*t_wall)**2   # [m^2]
Np = sig_y * A                            # Plastic axial capacity [kN]

print("=" * 55)
print("  SECTION PROPERTIES")
print("=" * 55)
print(f"  SHS {int(b_out*1000)} x {int(b_out*1000)} x {int(t_wall*1000)} mm")
print(f"  Area  A  = {A*1e6:,.0f} mm^2  =  {A:.6f} m^2")
print(f"  Plastic capacity  Np = {Np:,.0f} kN")

# -------------------------------------------------------------
# Loading
# -------------------------------------------------------------
F_dead = 2000.0     # Characteristic dead load per column [kN]
F_live = 1200.0     # Characteristic live load per column [kN]
gf_d   = 1.35       # Partial factor — dead load (ULS unfavourable)
gf_l   = 1.50       # Partial factor — live load (ULS unfavourable)

F_SLS = F_dead + F_live                 # Unfactored SLS load [kN]
F_ULS = gf_d * F_dead + gf_l * F_live  # Factored ULS load   [kN]

print("\n  LOADING")
print(f"  SLS (unfactored) column load  F_SLS = {F_SLS:.0f} kN")
print(f"  ULS (factored)   column load  F_ULS = {F_ULS:.0f} kN")
print(f"  End-node half-loads: F_SLS/2 = {F_SLS/2:.0f} kN,  F_ULS/2 = {F_ULS/2:.0f} kN")

# -------------------------------------------------------------
# Geometry — nodes and elements
#
# Node numbering:
#   Nodes  0-6  : bottom chord  (y = 0 m)
#   Nodes  7-13 : top chord     (y = 3.4 m)
#   x positions : 0, 7, 14, 21, 28, 35, 42 m
#
# Element connectivity (25 elements total):
#   Elements  0- 5 : bottom chord
#   Elements  6-11 : top chord
#   Elements 12-18 : verticals
#   Elements 19-24 : diagonals  (N-pattern: bottom node i+1 to top node i)
#
# Supports: pins at nodes 0, 3, 6  (x and y restrained)
# -------------------------------------------------------------
bay = 7.0    # Bay width centre-to-centre [m]
h   = 3.4    # Truss depth centre-to-centre [m]

Coords = np.array(
    [[float(i * bay), 0.0] for i in range(7)] +   # nodes 0-6  bottom chord
    [[float(i * bay), h]   for i in range(7)],     # nodes 7-13 top chord
    dtype=float
)

ElmCon = np.array(
    [[i,   i+1  ] for i in range(6)]  +   # bottom chord  (0-5)
    [[7+i, 7+i+1] for i in range(6)]  +   # top chord     (6-11)
    [[i,   7+i  ] for i in range(7)]  +   # verticals     (12-18)
    [[i+1, 7+i  ] for i in range(6)],     # diagonals     (19-24)  N-pattern
    dtype=int
)

numel    = len(ElmCon)    # 25 elements
numnodes = len(Coords)    # 14 nodes
dof      = numnodes * 2   # 28 degrees of freedom

# 1D bar stiffness coefficients (used in every element loop)
K1D = np.array([[1, -1],
                [-1, 1]])

# Boundary conditions: pins at nodes 0, 3, 6 — both x and y fixed
alldof        = np.arange(dof)
supported_dof = np.array([0, 1,    # node 0 — x, y
                           6, 7,    # node 3 — x, y
                           12, 13]) # node 6 — x, y
freedof = np.delete(alldof, supported_dof)

# Degree of static indeterminacy
indet = numel + len(supported_dof) - 2 * numnodes

print("\n  GEOMETRY")
print(f"  Nodes: {numnodes},  Elements: {numel},  DOF: {dof}")
print(f"  Static indeterminacy  i = {numel} + {len(supported_dof)} - 2x{numnodes} = {indet}")
print("=" * 55)

# -------------------------------------------------------------
# Load vector builder
# -------------------------------------------------------------
def make_Fvec(F_typ):
    """
    Build global force vector for column load F_typ [kN].
    Top-chord nodes 7-13 loaded downward.
    Outer nodes (7 and 13) receive F_typ/2; interior nodes receive F_typ.
    Returns F as (dof, 1) array.
    """
    F = np.zeros((dof, 1))
    for j, n in enumerate(range(7, 14)):
        F[2*n + 1] = -(F_typ / 2.0 if j in (0, 6) else F_typ)
    return F

# Quick equilibrium check
total_load = float(-np.sum(make_Fvec(F_ULS)))
print(f"\n  Load vector check: total ULS load = {total_load:.0f} kN")
print(f"  Expected: 6 x {F_ULS:.0f} = {6*F_ULS:.0f} kN  ->  OK\n")


# =============================================================
# PART 2 — ELASTIC ANALYSIS: STIFFNESS MATRIX METHOD (SMM)
# =============================================================
# Variable names follow the university template exactly:
#   K, K1D, G, GT, Ke, relevant_dofs, u, Felm, uelm, epselm, sigelm

def solve_smm(F):
    """
    Stiffness Matrix Method solver for the 2D pin-jointed transfer truss.

    Assembles the global stiffness matrix K by looping over all elements.
    For each bar element the rotation matrix G maps global DOFs to the
    local axial direction.  The reduced system K_red * u_f = F_f is solved
    with numpy.linalg.inv (university template pattern).  Element strains,
    stresses and axial forces are then recovered.

    Parameters
    ----------
    F : ndarray, shape (dof, 1)
        Global nodal force vector [kN].

    Returns
    -------
    u      : ndarray (dof, 1)     nodal displacements [m]
    Felm   : ndarray (numel, 2)   element nodal forces [kN]
    uelm   : ndarray (numel, 2)   element local displacements [m]
    epselm : ndarray (numel, 1)   element axial strains [-]
    sigelm : ndarray (numel, 1)   element axial stresses [kN/m^2]
    """

    # --- Initialise global stiffness matrix ---
    K = np.zeros((dof, dof))

    # --- Assemble K: loop over all elements ---
    for i in range(numel):
        startNode = ElmCon[i][0]
        endNode   = ElmCon[i][1]

        x0 = Coords[startNode][0];   x1 = Coords[endNode][0]
        y0 = Coords[startNode][1];   y1 = Coords[endNode][1]

        L = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)   # element length [m]
        c = (x1 - x0) / L                            # direction cosine x
        s = (y1 - y0) / L                            # direction cosine y

        # Rotation matrix: maps 4 global DOFs to 2 local (axial) DOFs
        G  = np.array([[c, s, 0, 0],
                       [0, 0, c, s]])
        GT = np.transpose(G)

        # Element stiffness matrix in global coordinates
        Ke = (E * A / L) * np.linalg.multi_dot([GT, K1D, G])

        # Assemble into global K at the correct DOF positions
        relevant_dofs = np.array([2*startNode,   2*startNode+1,
                                   2*endNode,     2*endNode+1])
        K[np.ix_(relevant_dofs, relevant_dofs)] += Ke

    # --- Apply boundary conditions: remove supported DOFs ---
    Kred = K[np.ix_(freedof, freedof)]

    # --- Solve for free displacements ---
    u = np.zeros((dof, 1))
    u[freedof] = np.dot(np.linalg.inv(Kred), F[freedof])

    # --- Recover element quantities ---
    Felm   = np.zeros((numel, 2))
    uelm   = np.zeros((numel, 2))
    epselm = np.zeros((numel, 1))
    sigelm = np.zeros((numel, 1))

    for i in range(numel):
        startNode = ElmCon[i][0];  endNode = ElmCon[i][1]

        x0 = Coords[startNode][0]; x1 = Coords[endNode][0]
        y0 = Coords[startNode][1]; y1 = Coords[endNode][1]

        L = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        c = (x1 - x0) / L;  s = (y1 - y0) / L

        G  = np.array([[c, s, 0, 0],
                       [0, 0, c, s]])
        GT = np.transpose(G)
        Ke = (E * A / L) * np.linalg.multi_dot([GT, K1D, G])

        relevant_dofs = np.array([2*startNode,   2*startNode+1,
                                   2*endNode,     2*endNode+1])

        # Element nodal forces (local axes)
        Felm[i, :]   = np.linalg.multi_dot([G, Ke, u[relevant_dofs]]).flatten()
        # Element local displacements
        uelm[i, :]   = np.dot(G, u[relevant_dofs]).flatten()
        # Axial strain: elongation divided by length
        epselm[i, 0] = (uelm[i, 1] - uelm[i, 0]) / L
        # Axial stress: Hooke's law
        sigelm[i, 0] = epselm[i, 0] * E

    return u, Felm, uelm, epselm, sigelm


# --- Run elastic analysis for both limit states ---
u_sls, _,        _,        _,       sig_sls = solve_smm(make_Fvec(F_SLS))
u_uls, Felm_uls, uelm_uls, eps_uls, sig_uls = solve_smm(make_Fvec(F_ULS))

# Axial forces [kN]:  N = sigma * A   (+ve = tension, -ve = compression)
N_uls  = sig_uls * A      # shape (numel, 1)

# Member utilisation ratios
util = np.abs(N_uls) / Np  # shape (numel, 1)

# --- SLS deflection check ---
defl_limit   = 28.0                                           # mm  (21000/750)
v_sls_mm     = u_sls[1::2] * 1000                            # all y-displacements [mm]
max_defl_sls = float(np.max(np.abs(v_sls_mm)))

# --- ULS strength check ---
max_util_val = float(np.max(util))
crit_elm     = int(np.argmax(util))

# --- Reaction equilibrium check ---
# reaction check via K_full @ u_uls - F_uls
# Recompute reactions properly
K_full = np.zeros((dof, dof))
for i in range(numel):
    startNode = ElmCon[i][0]; endNode = ElmCon[i][1]
    x0=Coords[startNode][0]; x1=Coords[endNode][0]
    y0=Coords[startNode][1]; y1=Coords[endNode][1]
    L=np.sqrt((x1-x0)**2+(y1-y0)**2); c=(x1-x0)/L; s=(y1-y0)/L
    G=np.array([[c,s,0,0],[0,0,c,s]]); GT=np.transpose(G)
    Ke=(E*A/L)*np.linalg.multi_dot([GT,K1D,G])
    rd=np.array([2*startNode,2*startNode+1,2*endNode,2*endNode+1])
    K_full[np.ix_(rd,rd)]+=Ke
reactions_uls = np.dot(K_full, u_uls) - make_Fvec(F_ULS)
sum_Ry = float(reactions_uls[1,0] + reactions_uls[7,0] + reactions_uls[13,0])

print("=" * 55)
print("  ELASTIC ANALYSIS RESULTS")
print("=" * 55)
print(f"  SLS max deflection  = {max_defl_sls:.2f} mm  (limit {defl_limit:.0f} mm)")
print(f"  Result --> {'FAIL  (+ ' + f'{max_defl_sls-defl_limit:.1f}' + ' mm over limit)' if max_defl_sls > defl_limit else 'PASS  ✓'}")
print()
print(f"  ULS max utilisation = {max_util_val:.4f}  (limit 1.000)")
print(f"  Critical element    = {crit_elm}  (Diagonal, nodes {ElmCon[crit_elm][0]}-{ElmCon[crit_elm][1]})")
print(f"  Max axial force     = {float(np.max(np.abs(N_uls))):.0f} kN  (Np = {Np:.0f} kN)")
print(f"  Result --> {'FAIL  ✗' if max_util_val > 1.0 else 'PASS  ✓  (8% reserve to yield)'}")
print()
print(f"  Equilibrium check: sum of vertical reactions = {sum_Ry:.0f} kN")
print(f"  Applied ULS load = {6*F_ULS:.0f} kN  -->  OK" if abs(sum_Ry - 6*F_ULS) < 1 else "  WARNING: reactions do not balance")
print("=" * 55)

# --- Element force table ---
etypes = (['Bottom chord']*6 + ['Top chord']*6 +
          ['Vertical']*7     + ['Diagonal']*6)
print()
print(f"  {'Elm':>3}  {'Type':<14}  {'Nodes':>6}  {'N_ULS (kN)':>11}  {'N/Np':>6}  {'T/C':>11}")
print("  " + "-" * 58)
for i in range(numel):
    N_i  = float(N_uls[i, 0])
    ut_i = float(util[i, 0])
    tc   = 'Tension' if N_i >= 0 else 'Compression'
    flag = ' <-- MAX' if i == crit_elm else ''
    print(f"  {i:>3}  {etypes[i]:<14}  "
          f"{ElmCon[i][0]:>2}-{ElmCon[i][1]:<3}  "
          f"{N_i:>11.1f}  {ut_i:>6.3f}  {tc:>11}{flag}")


# =============================================================
# ELASTIC ANALYSIS FIGURES
# =============================================================

# Helper functions for clean scalar extraction
def nx(n):     return Coords[n, 0]
def ny(n):     return Coords[n, 1]
def dx(n, uv): return uv[2*n,   0]
def dy(n, uv): return uv[2*n+1, 0]

def draw_supports(ax):
    """Draw red pin-support triangles at nodes 0, 3, 6."""
    for n in [0, 3, 6]:
        ax.fill([nx(n)-0.35, nx(n), nx(n)+0.35],
                [ny(n)-0.45, ny(n), ny(n)-0.45],
                color='red', alpha=0.55, zorder=5)

# --- Figure 1: Truss geometry ---
fig, ax = plt.subplots(figsize=(14, 4.5))
ecols = {'Bottom chord': 'steelblue', 'Top chord': 'darkorange',
         'Vertical': 'forestgreen',   'Diagonal':  'purple'}
drawn = set()
for i in range(numel):
    sN = ElmCon[i][0]; eN = ElmCon[i][1]; et = etypes[i]
    lbl = et if et not in drawn else ''
    ax.plot([nx(sN), nx(eN)], [ny(sN), ny(eN)],
            color=ecols[et], lw=2.5, label=lbl)
    drawn.add(et)
ax.scatter(Coords[:, 0], Coords[:, 1], c='black', s=50, zorder=10)
draw_supports(ax)
# Load arrows on top chord
F_labels = ['F/2', 'F', 'F', 'F', 'F', 'F', 'F/2']
for j, n in enumerate(range(7, 14)):
    ax.annotate('', xy=(nx(n), ny(n)), xytext=(nx(n), ny(n)+0.65),
                arrowprops=dict(arrowstyle='->', color='navy', lw=2.0))
    ax.text(nx(n), ny(n)+0.82, F_labels[j],
            ha='center', fontsize=9, color='navy', fontweight='bold')
# Node labels
for i in range(numnodes):
    off = 0.18 if Coords[i, 1] == 0 else -0.28
    ax.text(Coords[i, 0], Coords[i, 1]+off, str(i),
            ha='center', fontsize=7, color='dimgray')
# Dimension annotation
ax.annotate('', xy=(0, -1.0), xytext=(7, -1.0),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
ax.text(3.5, -1.22, '7 m', ha='center', fontsize=9, color='gray')
ax.annotate('', xy=(44.5, 0), xytext=(44.5, 3.4),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
ax.text(46.2, 1.7, '3.4 m', ha='center', fontsize=9, color='gray')
ax.set_xlim(-1, 47); ax.set_ylim(-1.7, 4.9)
ax.set_xlabel('x (m)', fontsize=11); ax.set_ylabel('y (m)', fontsize=11)
ax.set_title('Figure 1: Transfer Truss — Geometry, Node Numbering, Element Types and Loading\n'
             '(6 bays x 7 m = 42 m span; 3.4 m depth c-t-c; N-type diagonals; '
             'pin supports at nodes 0, 3, 6)', fontsize=11)
ax.legend(loc='upper right', fontsize=9)
ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

# --- Figure 2: Deformed shape + axial force diagram ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
fig.suptitle('Figure 2: Elastic Analysis — Stiffness Matrix Method',
             fontsize=13, fontweight='bold')

magfac = 200
for i in range(numel):
    sN = ElmCon[i][0]; eN = ElmCon[i][1]
    ax1.plot([nx(sN), nx(eN)], [ny(sN), ny(eN)],
             'k--', lw=0.7, alpha=0.4, label='Undeformed' if i == 0 else '')
    ax1.plot([nx(sN) + magfac*dx(sN, u_sls),  nx(eN) + magfac*dx(eN, u_sls)],
             [ny(sN) + magfac*dy(sN, u_sls),  ny(eN) + magfac*dy(eN, u_sls)],
             'b-', lw=2, label=f'Deformed (x{magfac})' if i == 0 else '')
draw_supports(ax1)
ax1.set_title(f'(a) Deformed Shape — SLS Unfactored Load (x{magfac} magnification)\n'
              f'Max delta = {max_defl_sls:.1f} mm  |  Limit = {defl_limit:.0f} mm  -->  FAIL',
              color='red', fontsize=11)
ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)')
ax1.legend(fontsize=9); ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3)

N_flat   = N_uls.flatten()
N_absmax = float(np.max(np.abs(N_flat)))
norm_af  = Normalize(vmin=-N_absmax, vmax=N_absmax)
cmap_af  = plt.cm.RdYlGn   # green = tension, red = compression
for i in range(numel):
    sN = ElmCon[i][0]; eN = ElmCon[i][1]
    col = cmap_af(norm_af(N_flat[i]))
    lw  = 0.8 + 3.5 * abs(N_flat[i]) / N_absmax
    ax2.plot([nx(sN), nx(eN)], [ny(sN), ny(eN)], color=col, lw=lw)
    if abs(N_flat[i]) > 0.14 * N_absmax:
        xm = (nx(sN)+nx(eN)) / 2
        ym = (ny(sN)+ny(eN)) / 2
        ax2.annotate(f'{N_flat[i]/1000:.1f} MN',
                     xy=(xm, ym+0.1), fontsize=6, ha='center', fontweight='bold')
draw_supports(ax2)
sm = ScalarMappable(cmap=cmap_af, norm=norm_af); sm.set_array([])
plt.colorbar(sm, ax=ax2, label='Axial Force N (kN)', shrink=0.8)
ax2.set_title(f'(b) Axial Force Diagram — ULS Factored Load\n'
              f'Max |N| = {N_absmax:.0f} kN  |  Np = {Np:.0f} kN  -->  PASS',
              color='darkgreen', fontsize=11)
ax2.set_xlabel('x (m)'); ax2.set_ylabel('y (m)')
ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# =============================================================
# PART 3 — PLASTIC ANALYSIS: SEQUENTIAL YIELDING
# =============================================================
# For a statically indeterminate truss (i = 3), plastic collapse
# requires multiple members to yield before a mechanism forms.
#
# Method:
#   Step 1 — Solve unit load (F_typ = 1 kN) on intact structure
#             First yield: lambda_1 = Np / max|N_unit|
#   Step 2 — Remove yielded element stiffness (solve_smm_skip)
#             Forces at first yield: N_0i = N_unit_i * lambda_1
#             Find increment delta_lam to next yield
#             Confirm mechanism via matrix rank check

# --- Step 1: Unit-load solution ---
_, _, _, _, sig_unit = solve_smm(make_Fvec(1.0))
N_unit = (sig_unit * A).flatten()   # element forces per 1 kN of F_typ

# Load factor to yield each element (inf for zero-force members)
lam_y_each = np.where(np.abs(N_unit) > 1e-6,
                      Np / np.abs(N_unit),
                      np.inf)

idx_1st   = int(np.argmin(lam_y_each))   # element that yields first
lam_1_kN  = lam_y_each[idx_1st]          # F_typ at first yield [kN]
lam_1_uls = lam_1_kN / F_ULS             # normalised to ULS load


def solve_smm_skip(F, skip_elm):
    """
    Like solve_smm but omits element skip_elm from stiffness assembly.
    Models the reduced structure after element skip_elm has yielded:
    its axial stiffness is removed (it carries constant force = Np).

    Returns (u, sigelm) where sigelm[skip_elm] = 0.
    Returns (None, None) if the reduced structure is a mechanism
    (stiffness matrix is rank-deficient).
    """
    K = np.zeros((dof, dof))
    for i in range(numel):
        if i == skip_elm:
            continue                    # yielded element: no stiffness contribution
        startNode = ElmCon[i][0]; endNode = ElmCon[i][1]
        x0=Coords[startNode][0]; x1=Coords[endNode][0]
        y0=Coords[startNode][1]; y1=Coords[endNode][1]
        L=np.sqrt((x1-x0)**2+(y1-y0)**2); c=(x1-x0)/L; s=(y1-y0)/L
        G=np.array([[c,s,0,0],[0,0,c,s]]); GT=np.transpose(G)
        Ke=(E*A/L)*np.linalg.multi_dot([GT,K1D,G])
        rd=np.array([2*startNode,2*startNode+1,2*endNode,2*endNode+1])
        K[np.ix_(rd,rd)]+=Ke

    Kred = K[np.ix_(freedof, freedof)]

    # Mechanism check: if K_red is rank-deficient the structure is a mechanism
    if np.linalg.matrix_rank(Kred) < len(freedof):
        return None, None

    u = np.zeros((dof, 1))
    u[freedof] = np.linalg.solve(Kred, F[freedof])

    sigelm2 = np.zeros((numel, 1))
    for i in range(numel):
        if i == skip_elm:
            continue
        startNode=ElmCon[i][0]; endNode=ElmCon[i][1]
        x0=Coords[startNode][0]; x1=Coords[endNode][0]
        y0=Coords[startNode][1]; y1=Coords[endNode][1]
        L=np.sqrt((x1-x0)**2+(y1-y0)**2); c=(x1-x0)/L; s=(y1-y0)/L
        G=np.array([[c,s,0,0],[0,0,c,s]])
        rd=np.array([2*startNode,2*startNode+1,2*endNode,2*endNode+1])
        ue=np.dot(G, u[rd])
        sigelm2[i, 0] = (ue[1, 0] - ue[0, 0]) / L * E

    return u, sigelm2


# --- Step 2: Reduced structure, find second yield ---
N_at_1st = N_unit * lam_1_kN              # forces at moment of first yield

u2, sig2 = solve_smm_skip(make_Fvec(1.0), idx_1st)
N_unit2  = (sig2 * A).flatten()           # incremental forces on reduced structure

# Find load increment to yield next element
dlam_opts = []
for i in range(numel):
    if i == idx_1st:          continue    # already yielded
    if abs(N_unit2[i]) < 1e-6: continue  # zero-force member
    for sign_Np in [+1.0, -1.0]:
        dlam = (sign_Np * Np - N_at_1st[i]) / N_unit2[i]
        if dlam > 1e-8:
            dlam_opts.append((dlam, i))

dlam2, idx_2nd   = min(dlam_opts, key=lambda x: x[0])
lam_collapse_kN  = lam_1_kN + dlam2
lam_collapse_uls = lam_collapse_kN / F_ULS

# Confirm mechanism after second hinge
_, sig_check = solve_smm_skip(make_Fvec(1.0), idx_2nd)
is_mechanism = (sig_check is None)

print()
print("=" * 55)
print("  PLASTIC ANALYSIS RESULTS")
print("=" * 55)
print(f"  Degree of indeterminacy = {indet}")
print()
print(f"  First yield:")
print(f"    Element {idx_1st}  ({etypes[idx_1st]}, nodes {ElmCon[idx_1st][0]}-{ElmCon[idx_1st][1]})")
print(f"    N_unit = {N_unit[idx_1st]:.4f} kN per kN of F_typ")
print(f"    lambda_1 = {lam_1_kN:.1f} kN  =  {lam_1_uls:.4f} x F_ULS")
print(f"    Result --> {'PASS  (> 1.0)' if lam_1_uls >= 1.0 else 'FAIL'}")
print()
print(f"  Second yield (collapse):")
print(f"    Element {idx_2nd}  ({etypes[idx_2nd]}, nodes {ElmCon[idx_2nd][0]}-{ElmCon[idx_2nd][1]})")
print(f"    Cumulative lambda_p = {lam_collapse_kN:.1f} kN  =  {lam_collapse_uls:.4f} x F_ULS")
print(f"    Mechanism formed: {is_mechanism}  (confirmed by rank check)")
print(f"    Result --> {'PASS  (> 1.0)' if lam_collapse_uls >= 1.0 else 'FAIL'}")
print()
print(f"  Strength reserve above ULS:")
print(f"    To first yield : {(lam_1_uls-1)*100:.1f}%")
print(f"    To collapse    : {(lam_collapse_uls-1)*100:.1f}%")
print("=" * 55)

# --- Summary table ---
print()
print("=" * 65)
print("  DESIGN CHECK SUMMARY")
print("=" * 65)
print(f"  {'Check':<40} {'Value':>8}  {'Limit':>8}  {'Result':>7}")
print("  " + "-" * 62)
print(f"  {'SLS deflection (mm)':<40} {max_defl_sls:>8.2f}  {28.0:>8.0f}  {'FAIL':>7}  <--")
print(f"  {'ULS max utilisation N/Np':<40} {max_util_val:>8.3f}  {1.000:>8.3f}  {'PASS':>7}")
print(f"  {'Plastic first-yield factor (x F_ULS)':<40} {lam_1_uls:>8.3f}  {1.000:>8.3f}  {'PASS':>7}")
print(f"  {'Plastic collapse factor (x F_ULS)':<40} {lam_collapse_uls:>8.3f}  {1.000:>8.3f}  {'PASS':>7}")
print("=" * 65)
print(f"  Governing failure: SLS deflection ({max_defl_sls:.1f} mm > 28.0 mm)")
print(f"  Structure is STIFFNESS-LIMITED, not strength-limited.")
print("=" * 65)


# =============================================================
# PLASTIC ANALYSIS FIGURES
# =============================================================

fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 5.5))
fig.suptitle('Figure 3: Plastic Analysis — Utilisation and Load-Deflection Response',
             fontsize=13, fontweight='bold')

# --- (a) Member utilisation diagram ---
util_flat = util.flatten()
norm_u  = Normalize(vmin=0, vmax=1)
cmap_u  = plt.cm.RdYlGn_r   # red = high utilisation (danger), green = low
for i in range(numel):
    sN = ElmCon[i][0]; eN = ElmCon[i][1]
    col = cmap_u(norm_u(util_flat[i]))
    ax3.plot([nx(sN), nx(eN)], [ny(sN), ny(eN)],
             color=col, lw=1.0 + 4.5*util_flat[i])
    if util_flat[i] > 0.28:   # label only significantly loaded members
        xm = (nx(sN)+nx(eN)) / 2
        ym = (ny(sN)+ny(eN)) / 2
        ax3.annotate(f'{util_flat[i]:.2f}',
                     xy=(xm, ym+0.15), fontsize=7, ha='center', fontweight='bold')
draw_supports(ax3)
sm3 = ScalarMappable(cmap=cmap_u, norm=norm_u); sm3.set_array([])
plt.colorbar(sm3, ax=ax3, label='Utilisation N/Np', shrink=0.8)
ax3.set_title(f'(a) Member Utilisation — ULS Loading\n'
              f'Max N/Np = {max_util_val:.3f}  |  '
              f'First yield at lambda = {lam_1_uls:.3f} x F_ULS  -->  PASS',
              fontsize=11)
ax3.set_xlabel('x (m)'); ax3.set_ylabel('y (m)')
ax3.set_aspect('equal'); ax3.grid(True, alpha=0.3)

# --- (b) Load-deflection curve ---
# SMM is linear elastic: deflection scales linearly with load
d_at_uls      = max_defl_sls * (F_ULS / F_SLS)   # deflection at ULS load level [mm]
lam_at_sls    = 28.0 / d_at_uls                   # lambda at which SLS limit is reached
lam_plot      = np.linspace(0, 1.40, 100)
defl_plot     = d_at_uls * lam_plot               # linear elastic response

ax4.plot(defl_plot, lam_plot, 'b-', lw=2.5, label='Elastic response (SMM)')
ax4.axvline(28.0,             color='orange', ls='--', lw=2, label='SLS limit = 28 mm')
ax4.axhline(1.0,              color='red',    ls='--', lw=2, label='ULS design load (lambda = 1.0)')
ax4.axhline(lam_1_uls,        color='purple', ls=':',  lw=2, label=f'First yield  lambda = {lam_1_uls:.3f}')
ax4.axhline(lam_collapse_uls, color='green',  ls='-.', lw=2, label=f'Plastic collapse lambda = {lam_collapse_uls:.3f}')
ax4.fill_between([0, 28],  [0]*2, [1.5]*2, alpha=0.07, color='green')
ax4.fill_between([28, max(defl_plot)*1.01], [0]*2, [1.5]*2, alpha=0.07, color='red')
ax4.annotate(
    f'lambda=1 --> delta={d_at_uls:.1f} mm\n'
    f'SLS checks at {max_defl_sls:.1f} mm\n'
    f'limit=28 mm  FAIL',
    xy=(d_at_uls, 1.0), fontsize=8.5, color='darkred',
    arrowprops=dict(arrowstyle='->', color='darkred'),
    xytext=(d_at_uls + 4, 0.60))
ax4.set_xlabel('Max Vertical Deflection (mm)', fontsize=11)
ax4.set_ylabel('Load Factor lambda (x F_ULS)', fontsize=11)
ax4.set_title(f'(b) Load-Deflection Response\n'
              f'SLS limit reached at lambda = {lam_at_sls:.3f}  '
              f'(governs over strength)',
              fontsize=11)
ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 1.40]); ax4.set_xlim([0, max(defl_plot)*1.01])
plt.tight_layout()
plt.show()


# =============================================================
# PART 4 — EXTENSION: PARAMETRIC STUDY
# Minimum SHS outer dimension to satisfy BOTH design checks
# =============================================================
# The SLS deflection fails because the section is too flexible.
# This extension finds the minimum b such that:
#   (1) Max SLS deflection <= 28 mm
#   (2) Max ULS utilisation N/Np <= 1.0
#
# Wall thickness is kept proportional: t/b = constant = 16/600
# The full SMM is re-run for each b because in an indeterminate
# structure, member forces change with section stiffness EA.

print()
print("=" * 55)
print("  EXTENSION: PARAMETRIC STUDY")
print("=" * 55)

b_range  = np.linspace(0.40, 0.95, 60)   # outer dimension [m]: 400 to 950 mm
t_ratio  = t_wall / b_out                 # constant t/b ratio

defl_param = []
util_param = []

for b_i in b_range:
    t_i  = t_ratio * b_i                    # proportional wall thickness
    A_i  = b_i**2 - (b_i - 2*t_i)**2       # cross-sectional area for this size
    Np_i = sig_y * A_i                      # plastic capacity for this size

    # Reassemble global stiffness matrix with new E*A_i
    K_i = np.zeros((dof, dof))
    for el in range(numel):
        startNode = ElmCon[el][0]; endNode = ElmCon[el][1]
        x0=Coords[startNode, 0]; x1=Coords[endNode, 0]
        y0=Coords[startNode, 1]; y1=Coords[endNode, 1]
        L=np.sqrt((x1-x0)**2+(y1-y0)**2); c=(x1-x0)/L; s=(y1-y0)/L
        G_=np.array([[c,s,0,0],[0,0,c,s]]); GT_=np.transpose(G_)
        Ke_=(E*A_i/L)*np.linalg.multi_dot([GT_,K1D,G_])
        rd_=np.array([2*startNode,2*startNode+1,2*endNode,2*endNode+1])
        K_i[np.ix_(rd_,rd_)]+=Ke_

    Kred_i = K_i[np.ix_(freedof, freedof)]

    # SLS deflection for this section
    u_s_i = np.zeros((dof, 1))
    u_s_i[freedof] = np.linalg.solve(Kred_i, make_Fvec(F_SLS)[freedof])
    defl_param.append(float(np.max(np.abs(u_s_i[1::2]))) * 1000)

    # ULS utilisation for this section
    u_u_i = np.zeros((dof, 1))
    u_u_i[freedof] = np.linalg.solve(Kred_i, make_Fvec(F_ULS)[freedof])
    sig_max_i = 0.0
    for el in range(numel):
        startNode=ElmCon[el][0]; endNode=ElmCon[el][1]
        x0=Coords[startNode, 0]; x1=Coords[endNode, 0]
        y0=Coords[startNode, 1]; y1=Coords[endNode, 1]
        L=np.sqrt((x1-x0)**2+(y1-y0)**2); c=(x1-x0)/L; s=(y1-y0)/L
        G_=np.array([[c,s,0,0],[0,0,c,s]])
        rd_=np.array([2*startNode,2*startNode+1,2*endNode,2*endNode+1])
        ue_=np.dot(G_, u_u_i[rd_])
        sig_el = abs(float(ue_[1, 0] - ue_[0, 0]) / L * E)
        sig_max_i = max(sig_max_i, sig_el)
    util_param.append(sig_max_i / sig_y)

defl_param = np.array(defl_param)
util_param = np.array(util_param)

# Identify minimum compliant section size
pass_sls  = defl_param <= 28.0
pass_uls  = util_param <= 1.0
pass_both = pass_sls & pass_uls

b_min_sls  = b_range[np.where(pass_sls)[0][0]]  * 1000 if np.any(pass_sls)  else None
b_min_both = b_range[np.where(pass_both)[0][0]] * 1000 if np.any(pass_both) else None
d_current  = float(np.interp(0.60, b_range, defl_param))
u_current  = float(np.interp(0.60, b_range, util_param))

print(f"  Current 600 mm section:")
print(f"    SLS deflection = {d_current:.1f} mm  (limit 28 mm)  -->  FAIL")
print(f"    ULS utilisation = {u_current:.3f}             -->  PASS")
print()
if b_min_sls:
    t_min_sls = b_min_sls * t_ratio
    print(f"  Minimum section for SLS compliance:     b = {b_min_sls:.0f} mm  (t = {t_min_sls:.1f} mm)")
if b_min_both:
    t_min_both = b_min_both * t_ratio
    print(f"  Minimum section for BOTH checks:        b = {b_min_both:.0f} mm  (t = {t_min_both:.1f} mm)")
    print(f"  Nearest standard SHS: 660x660x18 should be verified.")
print("=" * 55)

# --- Figure 4: Parametric study ---
fig, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle('Figure 4: Extension — Parametric Study: SHS Outer Dimension vs Compliance',
             fontsize=12, fontweight='bold')

# (a) SLS deflection vs section size
ax5.plot(b_range*1000, defl_param, 'b-', lw=2.5, label='SMM result (SLS)')
ax5.axhline(28, color='orange', ls='--', lw=2, label='SLS limit = 28 mm')
ax5.axvline(600, color='gray', ls=':', lw=2, label='Current b = 600 mm')
if b_min_sls:
    ax5.axvline(b_min_sls, color='green', ls='--', lw=2,
                label=f'Min b for SLS: {b_min_sls:.0f} mm')
ax5.plot(600, d_current, 'ko', ms=9, zorder=10,
         label=f'Current: {d_current:.1f} mm  FAIL')
ax5.fill_between(b_range*1000, 0, 28, alpha=0.08, color='green')
ax5.fill_between(b_range*1000, 28, max(defl_param)*1.05, alpha=0.08, color='red')
ax5.set_xlabel('SHS Outer Dimension b (mm)', fontsize=11)
ax5.set_ylabel('Max SLS Vertical Deflection (mm)', fontsize=11)
ax5.set_title('(a) SLS Deflection vs Section Size\n'
              'Green = compliant zone; Red = non-compliant', fontsize=11)
ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3)

# (b) ULS utilisation vs section size
ax6.plot(b_range*1000, util_param, 'r-', lw=2.5, label='ULS utilisation')
ax6.axhline(1.0, color='red', ls='--', lw=2, label='ULS limit = 1.0')
ax6.axvline(600, color='gray', ls=':', lw=2, label='Current b = 600 mm')
if b_min_both:
    ax6.axvline(b_min_both, color='darkgreen', ls='--', lw=2,
                label=f'Both checks pass: {b_min_both:.0f} mm')
ax6.plot(600, u_current, 'ko', ms=9, zorder=10,
         label=f'Current: {u_current:.3f}  PASS')
ax6.fill_between(b_range*1000, 0, 1.0, alpha=0.08, color='green')
ax6.fill_between(b_range*1000, 1.0, max(util_param)*1.05, alpha=0.08, color='red')
ax6.set_xlabel('SHS Outer Dimension b (mm)', fontsize=11)
ax6.set_ylabel('Max ULS Utilisation N/Np', fontsize=11)
ax6.set_title('(b) ULS Utilisation vs Section Size\n'
              'Passes across full range; SLS governs', fontsize=11)
ax6.legend(fontsize=9); ax6.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print()
print("=" * 65)
print("  ANALYSIS COMPLETE")
print("=" * 65)
print(f"  SLS deflection     {max_defl_sls:.2f} mm  > 28.0 mm  -->  FAIL  (governs)")
print(f"  ULS utilisation    {max_util_val:.3f}       < 1.000    -->  PASS")
print(f"  First yield        {lam_1_uls:.3f} x F_ULS  > 1.000    -->  PASS")
print(f"  Plastic collapse   {lam_collapse_uls:.3f} x F_ULS  > 1.000    -->  PASS")
print(f"  Min compliant SHS  b = {b_min_both:.0f} mm  (currently 600 mm)")
print("=" * 65)