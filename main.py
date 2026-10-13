import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import os

# ============================================================
#  PARAMETER SIMULASI
# ============================================================
G         = 1.0       # konstanta gravitasi (satuan simulasi)
DT        = 0.001     # timestep leapfrog
STEPS     = 40000     # jumlah langkah integrasi
SOFTENING = 1e-3      # softening length (hindari singularitas)
TAIL_LEN  = 500       # panjang ekor orbit (dalam step)
SKIP_INIT = 5         # kecepatan awal animasi (step/frame)

# ============================================================
#  KONDISI AWAL  — 1 bintang + 2 planet orbit melingkar
# ============================================================
masses = np.array([1.0, 0.001, 0.001])

R1, R2 = 1.0, 1.6
V1 = np.sqrt(G * masses[0] / R1)
V2 = np.sqrt(G * masses[0] / R2)

positions = np.array([
    [0.0, 0.0],
    [R1,  0.0],
    [-R2, 0.0],
], dtype=float)

velocities = np.array([
    [0.0,  0.0],
    [0.0,  V1 ],
    [0.0, -V2 ],
], dtype=float)

# ============================================================
#  FUNGSI FISIKA
# ============================================================

def acceleration(pos):
    """Percepatan gravitasi N-body (leapfrog-compatible, vectorized)."""
    disp  = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    dist2 = np.sum(disp**2, axis=2) + SOFTENING**2
    np.fill_diagonal(dist2, np.inf)
    inv_d3   = dist2**(-1.5)
    weighted = disp * inv_d3[:, :, np.newaxis] * masses[np.newaxis, :, np.newaxis]
    return G * np.sum(weighted, axis=1)


def calculate_energy(pos, vel):
    """Energi kinetik, potensial, dan total sistem."""
    kinetic   = 0.5 * np.sum(masses[:, np.newaxis] * vel**2)
    potential = 0.0
    for i in range(len(masses)):
        for j in range(i + 1, len(masses)):
            d = np.sqrt(np.sum((pos[j] - pos[i])**2) + SOFTENING**2)
            potential -= G * masses[i] * masses[j] / d
    return kinetic, potential, kinetic + potential


def angular_momentum(pos, vel):
    """Momentum angular total sistem (komponen z)."""
    Lz = np.sum(masses * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]))
    return Lz

# ============================================================
#  INTEGRASI LEAPFROG
# ============================================================
print("Menghitung lintasan dan energi...", end=" ", flush=True)

history        = np.zeros((STEPS + 1, len(masses), 2))
vel_history    = np.zeros((STEPS + 1, len(masses), 2))
energy_history = np.zeros((STEPS + 1, 3))   # KE, PE, E_total
Lz_history     = np.zeros(STEPS + 1)

history[0]        = positions.copy()
vel_history[0]    = velocities.copy()
energy_history[0] = calculate_energy(positions, velocities)
Lz_history[0]     = angular_momentum(positions, velocities)

acc = acceleration(positions)
for step in range(1, STEPS + 1):
    v_half      = velocities + 0.5 * acc * DT
    positions   = positions  + v_half * DT
    new_acc     = acceleration(positions)
    velocities  = v_half + 0.5 * new_acc * DT
    acc         = new_acc
    history[step]        = positions
    vel_history[step]    = velocities
    energy_history[step] = calculate_energy(positions, velocities)
    Lz_history[step]     = angular_momentum(positions, velocities)

print("selesai.")

time_history   = np.arange(STEPS + 1) * DT
initial_energy = energy_history[0, 2]
initial_Lz     = Lz_history[0]

if abs(initial_energy) > np.finfo(float).eps:
    relative_energy_error = np.abs(
        (energy_history[:, 2] - initial_energy) / initial_energy
    )
else:
    relative_energy_error = np.full(STEPS + 1, np.nan)

relative_Lz_error = np.abs((Lz_history - initial_Lz) / (abs(initial_Lz) + 1e-30))

# Cetak ringkasan ke terminal
print(f"  KE awal            : {energy_history[0, 0]:.10f}")
print(f"  PE awal            : {energy_history[0, 1]:.10f}")
print(f"  E total awal       : {initial_energy:.10f}")
print(f"  E total akhir      : {energy_history[-1, 2]:.10f}")
print(f"  Galat E maks.      : {np.nanmax(relative_energy_error):.3e}")
print(f"  Galat Lz maks.     : {np.nanmax(relative_Lz_error):.3e}")

# Simpan CSV
with open("data_energi_n_body.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "time", "kinetic_energy", "potential_energy",
        "total_energy", "relative_energy_error",
        "angular_momentum_z", "relative_Lz_error"
    ])
    for i in range(STEPS + 1):
        writer.writerow([
            time_history[i],
            energy_history[i, 0], energy_history[i, 1], energy_history[i, 2],
            relative_energy_error[i],
            Lz_history[i], relative_Lz_error[i],
        ])
print("  Data tersimpan     : data_energi_n_body.csv")

# ============================================================
#  SETUP FIGURE — animasi + panel energi live
# ============================================================
BG      = "#080812"
C_GRID  = "#ffffff"
C_SPINE = "#2a2a4a"
C_TEXT  = "#aaaacc"
C_DIM   = "#555577"

plt.style.use("dark_background")

fig = plt.figure(figsize=(13, 8), facecolor=BG)
fig.patch.set_facecolor(BG)

# Layout: animasi kiri (lebar), panel energi kanan (lebih sempit)
gs = gridspec.GridSpec(
    3, 2,
    figure=fig,
    left=0.06, right=0.97,
    top=0.93,  bottom=0.13,
    wspace=0.32, hspace=0.55,
    width_ratios=[1.55, 1],
    height_ratios=[1, 1, 1],
)

ax_anim   = fig.add_subplot(gs[:, 0])          # animasi — seluruh kolom kiri
ax_ke     = fig.add_subplot(gs[0, 1])          # KE live
ax_pe     = fig.add_subplot(gs[1, 1])          # PE live
ax_err    = fig.add_subplot(gs[2, 1])          # galat energi live

for ax in (ax_anim, ax_ke, ax_pe, ax_err):
    ax.set_facecolor(BG)
    ax.grid(True, color=C_GRID, alpha=0.06, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_edgecolor(C_SPINE)
    ax.tick_params(colors=C_DIM, labelsize=7.5)

# ── Sumbu animasi ────────────────────────────────────────────
pad  = 0.30
xlim = (history[:, :, 0].min() - pad, history[:, :, 0].max() + pad)
ylim = (history[:, :, 1].min() - pad, history[:, :, 1].max() + pad)
ax_anim.set_xlim(xlim)
ax_anim.set_ylim(ylim)
ax_anim.set_aspect("equal")
ax_anim.set_xlabel("x  (sat. sim.)", fontsize=10, color=C_TEXT)
ax_anim.set_ylabel("y  (sat. sim.)", fontsize=10, color=C_TEXT)
ax_anim.tick_params(colors=C_DIM, labelsize=8)
for sp in ax_anim.spines.values():
    sp.set_edgecolor(C_SPINE)

title_anim = ax_anim.set_title(
    "Simulasi Gravitasi N-body  |  t = 0.000",
    fontsize=12, fontweight="bold", color="white", pad=8
)

# ── Konfigurasi benda ────────────────────────────────────────
CONFIGS = [
    {"color": "#FFD700", "label": "Bintang",  "size": 260, "marker": "*", "lw": 2.0, "zg": 6},
    {"color": "#4FC3F7", "label": "Planet 1", "size":  70, "marker": "o", "lw": 1.4, "zg": 5},
    {"color": "#FF7043", "label": "Planet 2", "size":  70, "marker": "o", "lw": 1.4, "zg": 5},
]

# Ekor orbit — LineCollection agar bisa dibuat fading opacity
tail_collections = []
for cfg in CONFIGS:
    lc = LineCollection([], linewidths=cfg["lw"], zorder=3)
    ax_anim.add_collection(lc)
    tail_collections.append(lc)

# Titik benda
dots = []
for cfg in CONFIGS:
    sc = ax_anim.scatter(
        [], [], s=cfg["size"], color=cfg["color"],
        marker=cfg["marker"], zorder=cfg["zg"],
        edgecolors="white", linewidths=0.7
    )
    dots.append(sc)

# Label kecepatan di samping benda
speed_texts = []
for cfg in CONFIGS:
    t = ax_anim.text(
        0, 0, "", fontsize=7.5, color=cfg["color"], zorder=7,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.2", facecolor=BG,
                  edgecolor=cfg["color"], alpha=0.55, linewidth=0.6)
    )
    speed_texts.append(t)

# Legend
for i, cfg in enumerate(CONFIGS):
    ax_anim.plot([], [], color=cfg["color"], lw=1.6,
                 label=f"{cfg['label']}  (m = {masses[i]:.3f})")
ax_anim.legend(
    fontsize=8.5, framealpha=0.25, facecolor="#10102a",
    edgecolor=C_SPINE, labelcolor="white", loc="upper right"
)

# Info box kiri bawah (fisika live)
info_box = ax_anim.text(
    0.02, 0.02, "", transform=ax_anim.transAxes, fontsize=7.8,
    color=C_TEXT, va="bottom", family="monospace",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#10102a",
              edgecolor=C_SPINE, alpha=0.80)
)

# ── Panel energi — inisialisasi plot kosong ──────────────────
THIN = 500   # thinning: plot setiap n step agar tidak berat
t_thin  = time_history[::THIN]

# Palet warna panel
C_KE  = "#4FC3F7"
C_PE  = "#FF7043"
C_TOT = "#FFD700"
C_ERR = "#B0BEC5"

for ax, ylabel, col, ttl in [
    (ax_ke,  "E_k",          C_KE,  "Energi Kinetik"),
    (ax_pe,  "E_p",          C_PE,  "Energi Potensial"),
    (ax_err, "|dE/E0|",      C_ERR, "Galat Energi Relatif"),
]:
    ax.set_ylabel(ylabel, fontsize=8, color=C_DIM)
    ax.set_title(ttl, fontsize=8.5, color=C_TEXT, pad=4)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

ax_err.set_xlabel("Waktu  (sat. sim.)", fontsize=8, color=C_DIM)
ax_err.set_yscale("log")

# Plot latar (seluruh data, redup) — referensi
ax_ke.plot( t_thin, energy_history[::THIN, 0], color=C_KE,  lw=0.6, alpha=0.25)
ax_pe.plot( t_thin, energy_history[::THIN, 1], color=C_PE,  lw=0.6, alpha=0.25)
ax_ke.plot( t_thin, energy_history[::THIN, 2], color=C_TOT, lw=0.6, alpha=0.15,
            label="E total", linestyle="--")
ax_err.plot(t_thin, relative_energy_error[::THIN], color=C_ERR, lw=0.6, alpha=0.25)

ax_ke.legend(fontsize=7, framealpha=0.2, facecolor="#10102a",
             edgecolor=C_SPINE, labelcolor="white")

# Garis live (bergerak seiring animasi)
live_ke,  = ax_ke.plot( [], [], color=C_KE,  lw=1.6, zorder=4)
live_pe,  = ax_pe.plot( [], [], color=C_PE,  lw=1.6, zorder=4)
live_tot, = ax_ke.plot( [], [], color=C_TOT, lw=1.2, zorder=3, linestyle="--", alpha=0.7)
live_err, = ax_err.plot([], [], color=C_ERR, lw=1.6, zorder=4)

# Marker posisi waktu saat ini (garis vertikal)
vline_ke  = ax_ke.axvline( 0, color="white", lw=0.7, alpha=0.4, zorder=5)
vline_pe  = ax_pe.axvline( 0, color="white", lw=0.7, alpha=0.4, zorder=5)
vline_err = ax_err.axvline(0, color="white", lw=0.7, alpha=0.4, zorder=5)

# Batas sumbu panel energi (tetap, dari data penuh)
_pad_e = 0.05
for ax_e, col_idx in [(ax_ke, 0), (ax_pe, 1)]:
    vmin = energy_history[:, col_idx].min()
    vmax = energy_history[:, col_idx].max()
    span = vmax - vmin if vmax != vmin else 1e-9
    ax_e.set_xlim(0, time_history[-1])
    ax_e.set_ylim(vmin - _pad_e * span, vmax + _pad_e * span)

ax_ke.set_xlim(0, time_history[-1])
_etot_min = energy_history[:, 2].min()
_etot_max = energy_history[:, 2].max()
_span_tot = max(_etot_max - _etot_min, 1e-9)
# Extend ylim agar E_total (dashed) juga masuk
ax_ke.set_ylim(
    min(energy_history[:, 0].min(), _etot_min) - _pad_e * max(energy_history[:, 0].max() - energy_history[:, 0].min(), _span_tot),
    max(energy_history[:, 0].max(), _etot_max) + _pad_e * max(energy_history[:, 0].max() - energy_history[:, 0].min(), _span_tot),
)

_err_pos = relative_energy_error[relative_energy_error > 0]
ax_err.set_xlim(0, time_history[-1])
if len(_err_pos):
    ax_err.set_ylim(_err_pos.min() * 0.5, relative_energy_error.max() * 5)
else:
    ax_err.set_ylim(1e-16, 1e-6)

# ── Judul figure ─────────────────────────────────────────────
fig.text(
    0.5, 0.975,
    "Simulasi Gravitasi N-body  |  Metode Leapfrog  |  G = 1,  DT = 0.001,  N = 40 000 langkah",
    ha="center", va="top", fontsize=10.5, color="white",
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#10102a",
              edgecolor=C_SPINE, alpha=0.85)
)

# ============================================================
#  WIDGET — slider + tombol
# ============================================================
state = {"skip": SKIP_INIT, "paused": False, "frame": 0}

_btn_clr  = "#10102a"
_btn_hclr = "#1e1e40"

ax_slider = fig.add_axes([0.06, 0.065, 0.46, 0.025])
ax_slider.set_facecolor(_btn_clr)
slider = Slider(
    ax=ax_slider, label="Kecepatan", valmin=1, valmax=50,
    valinit=SKIP_INIT, valstep=1, color="#3355aa"
)
slider.label.set_color(C_TEXT); slider.label.set_fontsize(8.5)
slider.valtext.set_color("#FFD700"); slider.valtext.set_fontsize(8.5)

ax_btn_pause = fig.add_axes([0.56, 0.047, 0.12, 0.038])
btn_pause = Button(ax_btn_pause, "Pause", color=_btn_clr, hovercolor=_btn_hclr)
btn_pause.label.set_color("#FFD700"); btn_pause.label.set_fontsize(9.5)

ax_btn_reset = fig.add_axes([0.70, 0.047, 0.12, 0.038])
btn_reset = Button(ax_btn_reset, "Reset", color=_btn_clr, hovercolor=_btn_hclr)
btn_reset.label.set_color("#FF7043"); btn_reset.label.set_fontsize(9.5)

ax_btn_save = fig.add_axes([0.84, 0.047, 0.12, 0.038])
btn_save = Button(ax_btn_save, "Simpan PNG", color=_btn_clr, hovercolor=_btn_hclr)
btn_save.label.set_color("#69F0AE"); btn_save.label.set_fontsize(9.5)

fig.text(0.06, 0.040, "step/frame", fontsize=7.5, color=C_DIM, va="center")


def on_slider(val):
    state["skip"] = int(slider.val)
    fig.canvas.draw_idle()


def on_pause(event):
    state["paused"] = not state["paused"]
    if state["paused"]:
        btn_pause.label.set_text("Resume")
        btn_pause.label.set_color("#69F0AE")
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


_save_counter = [0]

def on_save(event):
    _save_counter[0] += 1
    skip  = state["skip"]
    idx   = min(state["frame"] * skip, STEPS)
    fname = f"frame_t{idx * DT:.3f}_{_save_counter[0]:03d}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"  Frame disimpan: {fname}")


slider.on_changed(on_slider)
btn_pause.on_clicked(on_pause)
btn_reset.on_clicked(on_reset)
btn_save.on_clicked(on_save)


# ============================================================
#  HELPER — buat segmen fading untuk LineCollection
# ============================================================

def _make_fading_segments(xs, ys):
    """
    Kembalikan (segments, colors) untuk LineCollection.
    Ujung ekor transparan, ujung terbaru opak.
    """
    n = len(xs)
    if n < 2:
        return [], []
    segments = [[(xs[k], ys[k]), (xs[k+1], ys[k+1])] for k in range(n - 1)]
    alphas   = np.linspace(0.03, 0.85, n - 1)
    return segments, alphas


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

    # ── Ekor orbit fading ────────────────────────────────────
    tail_start = max(0, idx - TAIL_LEN * skip)
    tail_idx   = np.arange(tail_start, idx + 1, max(1, skip))

    for i, (lc, cfg) in enumerate(zip(tail_collections, CONFIGS)):
        xs = history[tail_idx, i, 0]
        ys = history[tail_idx, i, 1]
        segs, alphas = _make_fading_segments(xs, ys)
        if segs:
            r, g, b, _ = mcolors.to_rgba(cfg["color"])
            colors = [(r, g, b, a) for a in alphas]
            lc.set_segments(segs)
            lc.set_colors(colors)
        else:
            lc.set_segments([])

    # ── Posisi benda ─────────────────────────────────────────
    for i, (sc, cfg) in enumerate(zip(dots, CONFIGS)):
        px, py = history[idx, i]
        sc.set_offsets([[px, py]])

    # ── Label kecepatan ──────────────────────────────────────
    offset = 0.055 * (xlim[1] - xlim[0])
    for i, (txt, cfg) in enumerate(zip(speed_texts, CONFIGS)):
        px, py = history[idx, i]
        vx, vy = vel_history[idx, i]
        speed  = np.hypot(vx, vy)
        txt.set_position((px + offset, py + offset))
        txt.set_text(f"v = {speed:.3f}")

    # ── Judul ────────────────────────────────────────────────
    t = idx * DT
    title_anim.set_text(f"Simulasi Gravitasi N-body  |  t = {t:.3f}")

    # ── Info box fisika ───────────────────────────────────────
    KE, PE, E_tot = energy_history[idx]
    Lz            = Lz_history[idx]
    d12 = np.linalg.norm(history[idx, 1] - history[idx, 0])
    d13 = np.linalg.norm(history[idx, 2] - history[idx, 0])
    d23 = np.linalg.norm(history[idx, 2] - history[idx, 1])
    pct = 100.0 * idx / STEPS
    info_box.set_text(
        f"Step : {idx:>6d} / {STEPS}  ({pct:.0f}%)\n"
        f"t    : {t:.4f}\n"
        f"KE   : {KE:+.6f}\n"
        f"PE   : {PE:+.6f}\n"
        f"E    : {E_tot:+.6f}\n"
        f"Lz   : {Lz:+.6f}\n"
        f"d(0-1): {d12:.4f}\n"
        f"d(0-2): {d13:.4f}\n"
        f"d(1-2): {d23:.4f}"
    )

    # ── Panel energi live ────────────────────────────────────
    # Thinning agar tidak terlalu banyak titik
    step_thin = max(1, THIN)
    end_t     = min(idx + 1, STEPS + 1)
    sl        = slice(0, end_t, step_thin)

    t_sl  = time_history[sl]
    ke_sl = energy_history[sl, 0]
    pe_sl = energy_history[sl, 1]
    et_sl = energy_history[sl, 2]
    er_sl = relative_energy_error[sl]

    live_ke.set_data( t_sl, ke_sl)
    live_pe.set_data( t_sl, pe_sl)
    live_tot.set_data(t_sl, et_sl)
    live_err.set_data(t_sl, er_sl)

    for vl in (vline_ke, vline_pe, vline_err):
        vl.set_xdata([t, t])

    return (tail_collections + dots + speed_texts +
            [title_anim, info_box,
             live_ke, live_pe, live_tot, live_err,
             vline_ke, vline_pe, vline_err])


anim = animation.FuncAnimation(
    fig, update, interval=20, blit=True, cache_frame_data=False
)

print("\nAnimasi berjalan.")
print("  Slider  : atur kecepatan (step/frame)")
print("  Pause   : tahan / lanjutkan animasi")
print("  Reset   : mulai ulang dari awal")
print("  Simpan PNG : screenshot frame saat ini")
plt.show()

# ============================================================
#  GRAFIK ENERGI FINAL (setelah animasi ditutup)
# ============================================================
fig_e, axes_e = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
fig_e.patch.set_facecolor(BG)
fig_e.suptitle(
    "Analisis Energi & Momentum Angular  |  Simulasi N-body Leapfrog",
    fontsize=12, color="white", fontweight="bold", y=0.98
)

labels_e = [
    ("Energi Kinetik & Total",   "Energi",       [
        (energy_history[:, 0], C_KE,  "KE",      1.0),
        (energy_history[:, 2], C_TOT, "E total", 1.2),
    ]),
    ("Energi Potensial",         "Energi",       [
        (energy_history[:, 1], C_PE,  "PE",      1.0),
    ]),
    ("Galat Energi Relatif",     "|dE/E0|",      [
        (relative_energy_error, C_ERR, "|dE/E0|", 1.0),
    ]),
]

for ax_e, (title_e, ylabel_e, curves) in zip(axes_e, labels_e):
    ax_e.set_facecolor(BG)
    ax_e.grid(True, alpha=0.15)
    ax_e.tick_params(colors=C_DIM, labelsize=8)
    ax_e.set_ylabel(ylabel_e, fontsize=9, color=C_DIM)
    ax_e.set_title(title_e, fontsize=10, color=C_TEXT, pad=4)
    for sp in ax_e.spines.values():
        sp.set_color(C_SPINE)
    for data, color, label, lw in curves:
        ax_e.plot(time_history, data, color=color, lw=lw, label=label)
    ax_e.legend(fontsize=8.5, framealpha=0.2, facecolor="#10102a",
                edgecolor=C_SPINE, labelcolor="white")

axes_e[2].set_yscale("log")
axes_e[2].set_xlabel("Waktu simulasi", fontsize=9, color=C_DIM)

# Anotasi galat maksimum
max_err = np.nanmax(relative_energy_error)
idx_max = np.nanargmax(relative_energy_error)
axes_e[2].annotate(
    f"maks = {max_err:.2e}",
    xy=(time_history[idx_max], max_err),
    xytext=(time_history[idx_max] + 2, max_err * 3),
    fontsize=8, color=C_ERR,
    arrowprops=dict(arrowstyle="->", color=C_ERR, lw=0.8),
)

fig_e.tight_layout(rect=[0, 0, 1, 0.97])
fig_e.savefig("energi_n_body.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.show()

print("\nAnalisis energi selesai.")
print(f"  E total awal   : {initial_energy:.10f}")
print(f"  E total akhir  : {energy_history[-1, 2]:.10f}")
print(f"  Galat E maks.  : {np.nanmax(relative_energy_error):.3e}")
print(f"  Galat Lz maks. : {np.nanmax(relative_Lz_error):.3e}")
print("  Grafik disimpan: energi_n_body.png")
