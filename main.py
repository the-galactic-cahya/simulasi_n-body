"""
Simulasi Gravitasi N-body -- Sistem Bintang-Planet (3 Benda)
============================================================
Satuan:
  Panjang : AU  (Astronomical Unit, 1 AU = 1.496e11 m)
  Massa   : M_sun (massa Matahari, 1 M_sun = 1.989e30 kg)
  Waktu   : tahun (yr, 1 yr = 3.156e7 s)

Dengan satuan ini: G = 4 * pi^2  (AU^3 / (M_sun * yr^2))
Bukti: T_bumi = 2*pi*sqrt(a^3/(G*M)) = 2*pi*sqrt(1/(4*pi^2)) = 1 yr -- benar

Sistem:
  Benda 0 : Bintang   (analog Matahari),  M = 1.0       M_sun
  Benda 1 : Planet 1  (analog Jupiter),   M = 9.5479e-4 M_sun, a = 5.2036 AU
  Benda 2 : Planet 2  (analog Saturnus),  M = 2.8588e-4 M_sun, a = 9.5371 AU

Integrasi : Leapfrog KDK (Kick-Drift-Kick) -- simplektik, konservatif, orde-2

Referensi :
  Murray & Dermott, Solar System Dynamics, Cambridge Univ. Press (1999)
  Yoshida H., Phys. Lett. A 150, 262 (1990)
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

# ============================================================
#  KONSTANTA & SATUAN
# ============================================================
G = 4.0 * np.pi**2   # AU^3 M_sun^-1 yr^-2  (eksakta dalam satuan astronomi)

# ============================================================
#  PARAMETER SIMULASI
# ============================================================
DT        = 0.001     # timestep [yr]  = ~0.365 hari
STEPS     = 120_000   # 120 tahun  (~10x periode Jupiter 11.86 yr)
SOFTENING = 1e-4      # softening [AU] -- kecil, hampir nol untuk sistem planet
TAIL_LEN  = 600       # panjang ekor orbit [step]
SKIP_INIT = 8         # kecepatan awal animasi [step/frame]

# ============================================================
#  KONDISI AWAL -- Matahari-Jupiter-Saturnus
# ============================================================
M_star = 1.0          # M_sun
M_p1   = 9.5479e-4    # Jupiter:  1/1047.56 M_sun
M_p2   = 2.8588e-4    # Saturnus: 1/3498.5  M_sun
masses = np.array([M_star, M_p1, M_p2])

a1 = 5.2036           # AU  (Jupiter)
a2 = 9.5371           # AU  (Saturnus)

# Kecepatan orbit melingkar Keplerian v = sqrt(G*M_star/a)
v1 = np.sqrt(G * M_star / a1)
v2 = np.sqrt(G * M_star / a2)

# Periode Keplerian (verifikasi)
T1 = 2.0 * np.pi * np.sqrt(a1**3 / (G * M_star))   # ~11.86 yr
T2 = 2.0 * np.pi * np.sqrt(a2**3 / (G * M_star))   # ~29.46 yr

# Posisi awal (bidang xy, inklinasi = 0)
# Planet 1 di kanan, Planet 2 di kiri (konfigurasi oposisi)
positions = np.array([
    [ 0.0, 0.0],
    [ a1,  0.0],
    [-a2,  0.0],
], dtype=float)

velocities = np.array([
    [0.0,  0.0],
    [0.0,  v1 ],
    [0.0, -v2 ],
], dtype=float)

# Koreksi ke center-of-mass frame:
# geser posisi & kecepatan agar pusat massa diam di origin
r_com = np.sum(masses[:, np.newaxis] * positions,  axis=0) / masses.sum()
v_com = np.sum(masses[:, np.newaxis] * velocities, axis=0) / masses.sum()
positions  -= r_com
velocities -= v_com

# ============================================================
#  FUNGSI FISIKA
# ============================================================

def acceleration(pos):
    """
    Percepatan gravitasi semua benda (vectorized, softened).
    pos : (N, 2)  ->  return (N, 2)
    """
    disp  = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]   # (N,N,2)
    dist2 = np.sum(disp**2, axis=2) + SOFTENING**2          # (N,N)
    np.fill_diagonal(dist2, np.inf)
    inv_d3   = dist2**(-1.5)
    weighted = disp * inv_d3[:, :, np.newaxis] * masses[np.newaxis, :, np.newaxis]
    return G * np.sum(weighted, axis=1)


def calculate_energy(pos, vel):
    """
    Energi kinetik, potensial, dan total.
    Menggunakan softening yang sama dengan integrasi agar konsisten.
    Satuan: M_sun AU^2 yr^-2
    """
    ke = 0.5 * np.sum(masses[:, np.newaxis] * vel**2)
    pe = 0.0
    N  = len(masses)
    for i in range(N):
        for j in range(i + 1, N):
            r   = np.sqrt(np.sum((pos[j] - pos[i])**2) + SOFTENING**2)
            pe -= G * masses[i] * masses[j] / r
    return ke, pe, ke + pe


def angular_momentum_z(pos, vel):
    """Momentum angular total komponen z: L_z = sum m_i (x_i vy_i - y_i vx_i)."""
    return float(np.sum(
        masses * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0])
    ))


def linear_momentum(vel):
    """Momentum linear total p = sum m_i v_i. Harus konstan = 0."""
    return np.sum(masses[:, np.newaxis] * vel, axis=0)

# ============================================================
#  INTEGRASI LEAPFROG  (KDK -- Kick-Drift-Kick)
# ============================================================
N_bodies = len(masses)

print("=" * 60)
print("  SIMULASI N-BODY GRAVITASI -- Sistem Bintang-Planet")
print("=" * 60)
print(f"  G            = {G:.6f}  AU^3 M_sun^-1 yr^-2")
print(f"  M_bintang    = {M_star:.4f}  M_sun")
print(f"  M_planet1    = {M_p1:.6f}  M_sun  (analog Jupiter)")
print(f"  M_planet2    = {M_p2:.6f}  M_sun  (analog Saturnus)")
print(f"  a1           = {a1:.4f}  AU    T1 = {T1:.4f} yr")
print(f"  a2           = {a2:.4f}  AU    T2 = {T2:.4f} yr")
print(f"  DT           = {DT:.4f}  yr  = {DT*365.25:.2f} hari")
print(f"  STEPS        = {STEPS:,}  ({STEPS*DT:.0f} yr total)")
print("=" * 60)
print("Menghitung lintasan...", end=" ", flush=True)

history        = np.zeros((STEPS + 1, N_bodies, 2))
vel_history    = np.zeros((STEPS + 1, N_bodies, 2))
energy_history = np.zeros((STEPS + 1, 3))   # KE, PE, E_total
Lz_history     = np.zeros(STEPS + 1)

history[0]        = positions.copy()
vel_history[0]    = velocities.copy()
energy_history[0] = calculate_energy(positions, velocities)
Lz_history[0]     = angular_momentum_z(positions, velocities)

acc = acceleration(positions)
for step in range(1, STEPS + 1):
    v_half     = velocities + 0.5 * acc * DT    # (1) half-kick
    positions  = positions  + v_half * DT        # (2) drift
    new_acc    = acceleration(positions)
    velocities = v_half + 0.5 * new_acc * DT    # (3) half-kick
    acc        = new_acc
    history[step]        = positions
    vel_history[step]    = velocities
    energy_history[step] = calculate_energy(positions, velocities)
    Lz_history[step]     = angular_momentum_z(positions, velocities)

print("selesai.")

time_history   = np.arange(STEPS + 1) * DT
initial_energy = energy_history[0, 2]
initial_Lz     = Lz_history[0]

# Galat konservasi relatif
rel_energy_err = np.where(
    abs(initial_energy) > np.finfo(float).eps,
    np.abs((energy_history[:, 2] - initial_energy) / initial_energy),
    np.nan
)
rel_Lz_err = np.abs((Lz_history - initial_Lz) / (abs(initial_Lz) + 1e-30))

p_final = linear_momentum(vel_history[-1])
print(f"  KE awal                : {energy_history[0,0]:.8f}  M_sun AU^2 yr^-2")
print(f"  PE awal                : {energy_history[0,1]:.8f}  M_sun AU^2 yr^-2")
print(f"  E total awal           : {initial_energy:.8f}  M_sun AU^2 yr^-2")
print(f"  E total akhir          : {energy_history[-1,2]:.8f}  M_sun AU^2 yr^-2")
print(f"  Galat E rel. maks.     : {np.nanmax(rel_energy_err):.3e}")
print(f"  Galat Lz rel. maks.    : {np.nanmax(rel_Lz_err):.3e}")
print(f"  Momentum linear akhir  : px={p_final[0]:.3e}  py={p_final[1]:.3e}  M_sun AU yr^-1")

# ============================================================
#  SIMPAN CSV  (header dengan satuan fisik)
# ============================================================
csv_path = "data_n_body.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "time_yr",
        "x0_AU", "y0_AU", "vx0_AU_yr", "vy0_AU_yr",
        "x1_AU", "y1_AU", "vx1_AU_yr", "vy1_AU_yr",
        "x2_AU", "y2_AU", "vx2_AU_yr", "vy2_AU_yr",
        "KE_Msun_AU2_yr2", "PE_Msun_AU2_yr2", "E_total_Msun_AU2_yr2",
        "rel_energy_error", "Lz_Msun_AU2_yr", "rel_Lz_error",
    ])
    for i in range(STEPS + 1):
        row = [time_history[i]]
        for b in range(N_bodies):
            row += [history[i,b,0], history[i,b,1],
                    vel_history[i,b,0], vel_history[i,b,1]]
        row += [energy_history[i,0], energy_history[i,1], energy_history[i,2],
                rel_energy_err[i], Lz_history[i], rel_Lz_err[i]]
        writer.writerow(row)
print(f"  Data tersimpan         : {csv_path}")

# ============================================================
#  SETUP FIGURE
# ============================================================
BG      = "#080812"
C_GRID  = "#ffffff"
C_SPINE = "#1e1e3a"
C_TEXT  = "#aaaacc"
C_DIM   = "#4a4a6a"
C_KE    = "#4FC3F7"
C_PE    = "#FF7043"
C_TOT   = "#FFD700"
C_ERR   = "#B0BEC5"

plt.style.use("dark_background")
fig = plt.figure(figsize=(14, 8.5), facecolor=BG)
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(
    3, 2,
    figure=fig,
    left=0.06, right=0.975,
    top=0.915, bottom=0.20,
    wspace=0.32, hspace=0.60,
    width_ratios=[1.55, 1],
    height_ratios=[1, 1, 1],
)

ax_anim = fig.add_subplot(gs[:, 0])
ax_ke   = fig.add_subplot(gs[0, 1])
ax_pe   = fig.add_subplot(gs[1, 1])
ax_err  = fig.add_subplot(gs[2, 1])

for ax in (ax_anim, ax_ke, ax_pe, ax_err):
    ax.set_facecolor(BG)
    ax.grid(True, color=C_GRID, alpha=0.055, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_edgecolor(C_SPINE)
    ax.tick_params(colors=C_DIM, labelsize=7.5)

# ── Area plot animasi ────────────────────────────────────────
pad  = 1.5
xlim = (history[:, :, 0].min() - pad, history[:, :, 0].max() + pad)
ylim = (history[:, :, 1].min() - pad, history[:, :, 1].max() + pad)
ax_anim.set_xlim(xlim)
ax_anim.set_ylim(ylim)
ax_anim.set_aspect("equal")
ax_anim.set_xlabel("x  [AU]", fontsize=10, color=C_TEXT, labelpad=6)
ax_anim.set_ylabel("y  [AU]", fontsize=10, color=C_TEXT)
ax_anim.tick_params(colors=C_DIM, labelsize=8.5)
for sp in ax_anim.spines.values():
    sp.set_edgecolor(C_SPINE)

# Lingkaran orbit Keplerian referensi (ideal, tanpa gangguan)
theta_ref = np.linspace(0, 2 * np.pi, 500)
for a_ref in [a1, a2]:
    ax_anim.plot(
        a_ref * np.cos(theta_ref), a_ref * np.sin(theta_ref),
        color="#2a2a50", lw=0.8, ls="--", zorder=1
    )

title_anim = ax_anim.set_title(
    "Simulasi Gravitasi N-body  |  t = 0.000 yr",
    fontsize=11.5, fontweight="bold", color="white", pad=7
)

# ── Konfigurasi benda ────────────────────────────────────────
CONFIGS = [
    {
        "color": "#FFF176", "label": f"Bintang  (1.000 M\u2609)",
        "size": 280, "marker": "*", "lw": 1.2, "zg": 6,
        "glow": "#FFF9C4",
    },
    {
        "color": "#4FC3F7",
        "label": f"Jupiter  ({M_p1:.4e} M\u2609, a={a1} AU)",
        "size": 80, "marker": "o", "lw": 1.5, "zg": 5,
        "glow": "#BBDEFB",
    },
    {
        "color": "#FF7043",
        "label": f"Saturnus ({M_p2:.4e} M\u2609, a={a2} AU)",
        "size": 60, "marker": "o", "lw": 1.5, "zg": 5,
        "glow": "#FFCCBC",
    },
]

# Ekor orbit fading (LineCollection)
tail_collections = []
for cfg in CONFIGS:
    lc = LineCollection([], linewidths=cfg["lw"], capstyle="round", zorder=3)
    ax_anim.add_collection(lc)
    tail_collections.append(lc)

# Titik: lapisan glow + inti
dots_glow, dots_main = [], []
for cfg in CONFIGS:
    g = ax_anim.scatter([], [], s=cfg["size"] * 2.8, color=cfg["glow"],
                        marker=cfg["marker"], alpha=0.18, linewidths=0,
                        zorder=cfg["zg"] - 1)
    m = ax_anim.scatter([], [], s=cfg["size"], color=cfg["color"],
                        marker=cfg["marker"], edgecolors="#ffffff",
                        linewidths=0.6, zorder=cfg["zg"])
    dots_glow.append(g)
    dots_main.append(m)

# Label kecepatan
speed_texts = []
for cfg in CONFIGS:
    t = ax_anim.text(
        0, 0, "", fontsize=7.5, color=cfg["color"], zorder=7,
        va="bottom", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=BG,
                  edgecolor=cfg["color"], alpha=0.6, linewidth=0.5)
    )
    speed_texts.append(t)

# Legend
for cfg in CONFIGS:
    ax_anim.plot([], [], color=cfg["color"], lw=1.8, label=cfg["label"])
ax_anim.legend(
    fontsize=8, framealpha=0.30, facecolor="#0d0d20",
    edgecolor=C_SPINE, labelcolor="white", loc="upper right"
)

# Info box fisika live (kiri bawah)
info_box = ax_anim.text(
    0.015, 0.015, "", transform=ax_anim.transAxes, fontsize=7.5,
    color=C_TEXT, va="bottom", family="monospace",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#0a0a1e",
              edgecolor=C_SPINE, alpha=0.88)
)

# ── Panel energi kanan ───────────────────────────────────────
ax_ke.set_title("Energi Kinetik",        fontsize=8.5, color=C_TEXT, pad=3)
ax_pe.set_title("Energi Potensial",      fontsize=8.5, color=C_TEXT, pad=3)
ax_err.set_title("Galat Energi Relatif", fontsize=8.5, color=C_TEXT, pad=3)

ax_ke.set_ylabel("KE  [M$_\\odot$ AU$^2$ yr$^{-2}$]", fontsize=7, color=C_DIM)
ax_pe.set_ylabel("PE  [M$_\\odot$ AU$^2$ yr$^{-2}$]", fontsize=7, color=C_DIM)
ax_err.set_ylabel("|$\\Delta$E/E$_0$|",               fontsize=7, color=C_DIM)
ax_err.set_xlabel("Waktu  [yr]",                       fontsize=8, color=C_DIM)
ax_err.set_yscale("log")

THIN = 600
t_t   = time_history[::THIN]

# Latar redup (data penuh, referensi)
ax_ke.plot( t_t, energy_history[::THIN, 0], color=C_KE,  lw=0.6, alpha=0.20)
ax_ke.plot( t_t, energy_history[::THIN, 2], color=C_TOT, lw=0.6, alpha=0.12, ls="--")
ax_pe.plot( t_t, energy_history[::THIN, 1], color=C_PE,  lw=0.6, alpha=0.20)
ax_err.plot(t_t, rel_energy_err[::THIN],    color=C_ERR, lw=0.6, alpha=0.20)

ax_ke.legend(
    [plt.Line2D([0],[0], color=C_TOT, ls="--", lw=1.2)], ["E total"],
    fontsize=7, framealpha=0.2, facecolor="#0a0a1e",
    edgecolor=C_SPINE, labelcolor="white"
)

# Garis live (bergerak seiring animasi)
live_ke,  = ax_ke.plot( [], [], color=C_KE,  lw=1.6, zorder=4)
live_pe,  = ax_pe.plot( [], [], color=C_PE,  lw=1.6, zorder=4)
live_tot, = ax_ke.plot( [], [], color=C_TOT, lw=1.2, ls="--", alpha=0.75, zorder=3)
live_err, = ax_err.plot([], [], color=C_ERR, lw=1.6, zorder=4)

# Garis vertikal posisi waktu saat ini
vline_ke  = ax_ke.axvline( 0, color="white", lw=0.8, alpha=0.4, zorder=5)
vline_pe  = ax_pe.axvline( 0, color="white", lw=0.8, alpha=0.4, zorder=5)
vline_err = ax_err.axvline(0, color="white", lw=0.8, alpha=0.4, zorder=5)

# Batas sumbu panel energi (dari data penuh, tidak berubah)
T_max = time_history[-1]
_ep   = 0.06

v0 = min(energy_history[:,0].min(), energy_history[:,2].min())
v1 = max(energy_history[:,0].max(), energy_history[:,2].max())
sp = max(v1 - v0, 1e-12)
ax_ke.set_xlim(0, T_max)
ax_ke.set_ylim(v0 - _ep*sp, v1 + _ep*sp)

v0 = energy_history[:,1].min(); v1 = energy_history[:,1].max()
sp = max(v1 - v0, 1e-12)
ax_pe.set_xlim(0, T_max)
ax_pe.set_ylim(v0 - _ep*sp, v1 + _ep*sp)

_ep2 = rel_energy_err[rel_energy_err > 0]
ax_err.set_xlim(0, T_max)
if len(_ep2):
    ax_err.set_ylim(_ep2.min() * 0.3, rel_energy_err.max() * 8)
else:
    ax_err.set_ylim(1e-16, 1e-6)

# ── Judul figur ──────────────────────────────────────────────
fig.text(
    0.5, 0.972,
    (f"Simulasi N-body Gravitasi -- Leapfrog KDK  |  "
     f"G = 4\u03c0\u00b2 AU\u00b3 M\u2609\u207b\u00b9 yr\u207b\u00b2  |  "
     f"DT = {DT} yr  |  {STEPS:,} langkah  |  Durasi = {STEPS*DT:.0f} yr"),
    ha="center", va="top", fontsize=9.5, color="#ccccee",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#0d0d20",
              edgecolor=C_SPINE, alpha=0.90)
)

# ============================================================
#  WIDGET
# ============================================================
state = {"skip": SKIP_INIT, "paused": False, "frame": 0}
_BC  = "#0d0d20"
_BCH = "#1a1a38"

ax_sl = fig.add_axes([0.06, 0.115, 0.46, 0.025])
ax_sl.set_facecolor(_BC)
slider = Slider(ax=ax_sl, label="Kecepatan", valmin=1, valmax=80,
                valinit=SKIP_INIT, valstep=1, color="#2244aa")
slider.label.set_color(C_TEXT); slider.label.set_fontsize(8.5)
slider.valtext.set_color("#FFD700"); slider.valtext.set_fontsize(8.5)

ax_bp = fig.add_axes([0.555, 0.095, 0.115, 0.038])
btn_pause = Button(ax_bp, "Pause", color=_BC, hovercolor=_BCH)
btn_pause.label.set_color("#FFD700"); btn_pause.label.set_fontsize(9.5)

ax_br = fig.add_axes([0.680, 0.095, 0.115, 0.038])
btn_reset = Button(ax_br, "Reset", color=_BC, hovercolor=_BCH)
btn_reset.label.set_color("#FF7043"); btn_reset.label.set_fontsize(9.5)

ax_bs = fig.add_axes([0.805, 0.095, 0.155, 0.038])
btn_save = Button(ax_bs, "Simpan PNG", color=_BC, hovercolor=_BCH)
btn_save.label.set_color("#81C784"); btn_save.label.set_fontsize(9.5)

fig.text(0.06, 0.088, "step/frame", fontsize=7.5, color=C_DIM, va="center")


def on_slider(val):
    state["skip"] = int(slider.val)
    fig.canvas.draw_idle()


def on_pause(event):
    state["paused"] = not state["paused"]
    if state["paused"]:
        btn_pause.label.set_text("Resume")
        btn_pause.label.set_color("#81C784")
    else:
        btn_pause.label.set_text("Pause")
        btn_pause.label.set_color("#FFD700")
    fig.canvas.draw_idle()


def on_reset(event):
    state["frame"] = 0
    state["paused"] = False
    btn_pause.label.set_text("Pause")
    btn_pause.label.set_color("#FFD700")
    fig.canvas.draw_idle()


_save_ctr = [0]


def on_save(event):
    _save_ctr[0] += 1
    idx  = min(state["frame"] * state["skip"], STEPS)
    name = f"frame_{idx*DT:.2f}yr_{_save_ctr[0]:03d}.png"
    fig.savefig(name, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"  Frame disimpan: {name}")


slider.on_changed(on_slider)
btn_pause.on_clicked(on_pause)
btn_reset.on_clicked(on_reset)
btn_save.on_clicked(on_save)


# ============================================================
#  HELPER -- ekor orbit fading
# ============================================================

def make_fading_segments(xs, ys, color_hex):
    """Segmen LineCollection dengan opacity makin tebal ke ujung terbaru."""
    n = len(xs)
    if n < 2:
        return [], []
    segs   = [[(xs[k], ys[k]), (xs[k+1], ys[k+1])] for k in range(n - 1)]
    alphas = np.linspace(0.02, 0.88, n - 1) ** 1.15
    r, g, b, _ = mcolors.to_rgba(color_hex)
    colors = [(r, g, b, float(a)) for a in alphas]
    return segs, colors


# ============================================================
#  FUNGSI ANIMASI
# ============================================================

def update(frame_num):
    if not state["paused"]:
        state["frame"] += 1
        if state["frame"] * state["skip"] > STEPS:
            state["frame"] = 0

    skip = state["skip"]
    idx  = min(state["frame"] * skip, STEPS)

    # Ekor orbit fading
    tail_start = max(0, idx - TAIL_LEN * skip)
    tail_idx   = np.arange(tail_start, idx + 1, max(1, skip))

    for i, (lc, cfg) in enumerate(zip(tail_collections, CONFIGS)):
        xs = history[tail_idx, i, 0]
        ys = history[tail_idx, i, 1]
        segs, colors = make_fading_segments(xs, ys, cfg["color"])
        if segs:
            lc.set_segments(segs)
            lc.set_colors(colors)
        else:
            lc.set_segments([])

    # Posisi benda (glow + inti)
    for i in range(N_bodies):
        px, py = history[idx, i]
        dots_glow[i].set_offsets([[px, py]])
        dots_main[i].set_offsets([[px, py]])

    # Label kecepatan
    off = 0.045 * (xlim[1] - xlim[0])
    for i, (txt, cfg) in enumerate(zip(speed_texts, CONFIGS)):
        px, py = history[idx, i]
        vx, vy = vel_history[idx, i]
        spd    = np.hypot(vx, vy)
        txt.set_position((px + off, py + off))
        txt.set_text(f"v={spd:.3f} AU/yr")

    # Judul
    t = idx * DT
    title_anim.set_text(
        f"Simulasi Gravitasi N-body  |  t = {t:.3f} yr  "
        f"({t/T1:.1f} T_J)"
    )

    # Info box fisika live
    KE, PE, E = energy_history[idx]
    Lz  = Lz_history[idx]
    dEr = rel_energy_err[idx]
    d01 = np.linalg.norm(history[idx, 1] - history[idx, 0])
    d02 = np.linalg.norm(history[idx, 2] - history[idx, 0])
    d12 = np.linalg.norm(history[idx, 2] - history[idx, 1])
    pct = 100.0 * idx / STEPS
    info_box.set_text(
        f"t       : {t:>9.3f} yr   ({pct:.0f}%)\n"
        f"KE      : {KE:>+12.6f}  M\u2609 AU\u00b2 yr\u207b\u00b2\n"
        f"PE      : {PE:>+12.6f}  M\u2609 AU\u00b2 yr\u207b\u00b2\n"
        f"E_total : {E:>+12.6f}  M\u2609 AU\u00b2 yr\u207b\u00b2\n"
        f"Lz      : {Lz:>+12.6f}  M\u2609 AU\u00b2 yr\u207b\u00b9\n"
        f"|dE/E0| : {dEr:>12.3e}\n"
        f"d(0-1)  : {d01:>9.4f}  AU\n"
        f"d(0-2)  : {d02:>9.4f}  AU\n"
        f"d(1-2)  : {d12:>9.4f}  AU"
    )

    # Panel energi live
    end_sl = min(idx + 1, STEPS + 1)
    sl     = slice(0, end_sl, THIN)
    t_sl   = time_history[sl]

    live_ke.set_data( t_sl, energy_history[sl, 0])
    live_pe.set_data( t_sl, energy_history[sl, 1])
    live_tot.set_data(t_sl, energy_history[sl, 2])
    live_err.set_data(t_sl, rel_energy_err[sl])

    for vl in (vline_ke, vline_pe, vline_err):
        vl.set_xdata([t, t])

    return (tail_collections + dots_glow + dots_main + speed_texts +
            [title_anim, info_box,
             live_ke, live_pe, live_tot, live_err,
             vline_ke, vline_pe, vline_err])


anim = animation.FuncAnimation(
    fig, update, interval=20, blit=True, cache_frame_data=False
)

print("\nAnimasi berjalan.")
print("  Slider     : atur kecepatan (step/frame)")
print("  Pause      : tahan / lanjutkan")
print("  Reset      : ulang dari t = 0")
print("  Simpan PNG : screenshot frame saat ini")
plt.show()

# ============================================================
#  GRAFIK ENERGI FINAL (setelah animasi ditutup)
# ============================================================
fig_e, axes_e = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig_e.patch.set_facecolor(BG)
fig_e.suptitle(
    "Analisis Konservasi Energi & Momentum Angular\n"
    f"Simulasi N-body Leapfrog KDK -- G=4\u03c0\u00b2, DT={DT} yr, "
    f"{STEPS:,} langkah, {STEPS*DT:.0f} yr",
    fontsize=11, color="white", fontweight="bold", y=0.99
)

plot_specs = [
    (
        "Energi Kinetik & Total",
        "E  [M$_{\\odot}$ AU$^2$ yr$^{-2}$]",
        [
            (energy_history[:, 0], C_KE,  "KE",      1.0, "-"),
            (energy_history[:, 2], C_TOT, "E total", 1.5, "--"),
        ],
        False,
    ),
    (
        "Energi Potensial",
        "PE  [M$_{\\odot}$ AU$^2$ yr$^{-2}$]",
        [(energy_history[:, 1], C_PE, "PE", 1.0, "-")],
        False,
    ),
    (
        "Galat Energi Relatif  |$\\Delta$E/E$_0$|",
        "|$\\Delta$E/E$_0$|",
        [(rel_energy_err, C_ERR, "|dE/E0|", 1.0, "-")],
        True,
    ),
]

for ax_e, (title_e, ylabel_e, curves, logy) in zip(axes_e, plot_specs):
    ax_e.set_facecolor(BG)
    ax_e.grid(True, alpha=0.15, color=C_GRID)
    ax_e.tick_params(colors=C_DIM, labelsize=8.5)
    ax_e.set_ylabel(ylabel_e, fontsize=9, color=C_DIM)
    ax_e.set_title(title_e,   fontsize=10, color=C_TEXT, pad=4)
    for sp in ax_e.spines.values():
        sp.set_color(C_SPINE)
    if logy:
        ax_e.set_yscale("log")
    for data, color, label, lw, ls in curves:
        ax_e.plot(time_history, data, color=color, lw=lw, ls=ls, label=label)
    ax_e.legend(fontsize=8.5, framealpha=0.2, facecolor="#0a0a1e",
                edgecolor=C_SPINE, labelcolor="white")

axes_e[-1].set_xlabel("Waktu  [yr]", fontsize=9.5, color=C_DIM)

# Anotasi galat maksimum
max_err = np.nanmax(rel_energy_err)
idx_max = np.nanargmax(rel_energy_err)
axes_e[2].annotate(
    f"maks = {max_err:.2e}",
    xy=(time_history[idx_max], max_err),
    xytext=(time_history[idx_max] + 5, max_err * 5),
    fontsize=8.5, color=C_ERR,
    arrowprops=dict(arrowstyle="->", color=C_ERR, lw=0.9),
)

# Garis vertikal periode orbital Jupiter & Saturnus
for ax_e in axes_e:
    for k, (T, col) in enumerate([(T1, C_KE), (T2, C_PE)]):
        n_periods = int(STEPS * DT / T)
        for m in range(1, n_periods + 1):
            ax_e.axvline(m * T, color=col, lw=0.35, alpha=0.18, ls=":", zorder=0)

fig_e.tight_layout(rect=[0, 0, 1, 0.965])
fig_e.savefig("energi_n_body.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.show()

print("\nAnalisis selesai.")
print(f"  E total awal     : {initial_energy:.10f}  M_sun AU^2 yr^-2")
print(f"  E total akhir    : {energy_history[-1,2]:.10f}  M_sun AU^2 yr^-2")
print(f"  Galat E maks.    : {np.nanmax(rel_energy_err):.3e}")
print(f"  Galat Lz maks.   : {np.nanmax(rel_Lz_err):.3e}")
print("  Grafik disimpan  : energi_n_body.png")
print(f"  Data disimpan    : {csv_path}")
