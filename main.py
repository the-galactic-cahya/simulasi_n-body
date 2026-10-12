import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

# Parameter simulasi
G = 1.0
DT = 0.001
STEPS = 40000
SOFTENING = 1e-3
TAIL_LEN = 400
SKIP_INIT = 5

# Kondisi awal
masses = np.array([1.0, 0.001, 0.001])
R1, R2 = 1.0, 1.6
V1 = np.sqrt(G * masses[0] / R1)
V2 = np.sqrt(G * masses[0] / R2)

positions = np.array([
    [0.0, 0.0],
    [R1, 0.0],
    [-R2, 0.0],
], dtype=float)

velocities = np.array([
    [0.0, 0.0],
    [0.0, V1],
    [0.0, -V2],
], dtype=float)


def acceleration(pos):
    """Menghitung percepatan gravitasi untuk setiap benda."""
    disp = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    dist2 = np.sum(disp**2, axis=2) + SOFTENING**2
    np.fill_diagonal(dist2, np.inf)
    inv_d3 = dist2**(-1.5)
    weighted = (
        disp * inv_d3[:, :, np.newaxis]
        * masses[np.newaxis, :, np.newaxis]
    )
    return G * np.sum(weighted, axis=1)


def calculate_energy(pos, vel):
    """Menghitung energi kinetik, potensial, dan total."""
    kinetic = 0.5 * np.sum(masses[:, np.newaxis] * vel**2)
    potential = 0.0
    for i in range(len(masses)):
        for j in range(i + 1, len(masses)):
            distance = np.sqrt(
                np.sum((pos[j] - pos[i])**2) + SOFTENING**2
            )
            potential -= G * masses[i] * masses[j] / distance
    return kinetic, potential, kinetic + potential


# Integrasi leapfrog dan pencatatan energi
print("Menghitung lintasan dan energi... ", end="", flush=True)
history = np.zeros((STEPS + 1, len(masses), 2))
vel_history = np.zeros((STEPS + 1, len(masses), 2))
energy_history = np.zeros((STEPS + 1, 3))
history[0] = positions
vel_history[0] = velocities
energy_history[0] = calculate_energy(positions, velocities)

acc = acceleration(positions)
for step in range(1, STEPS + 1):
    v_half = velocities + 0.5 * acc * DT
    positions = positions + v_half * DT
    new_acc = acceleration(positions)
    velocities = v_half + 0.5 * new_acc * DT
    acc = new_acc
    history[step] = positions
    vel_history[step] = velocities
    energy_history[step] = calculate_energy(positions, velocities)
print("selesai.")

time_history = np.arange(STEPS + 1) * DT
initial_energy = energy_history[0, 2]
if abs(initial_energy) > np.finfo(float).eps:
    relative_energy_error = np.abs(
        (energy_history[:, 2] - initial_energy) / initial_energy
    )
else:
    relative_energy_error = np.full(STEPS + 1, np.nan)

print(f"Energi kinetik awal   : {energy_history[0, 0]:.10f}")
print(f"Energi potensial awal : {energy_history[0, 1]:.10f}")
print(f"Energi total awal     : {initial_energy:.10f}")
print(f"Energi total akhir    : {energy_history[-1, 2]:.10f}")
print(
    "Galat energi relatif maksimum: "
    f"{np.nanmax(relative_energy_error):.6e}"
)

with open("data_energi_n_body.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "time", "kinetic_energy", "potential_energy",
        "total_energy", "relative_energy_error"
    ])
    for i in range(STEPS + 1):
        writer.writerow([
            time_history[i],
            energy_history[i, 0],
            energy_history[i, 1],
            energy_history[i, 2],
            relative_energy_error[i],
        ])
print("Data energi tersimpan sebagai data_energi_n_body.csv")

# Pengaturan animasi
state = {"skip": SKIP_INIT, "paused": False, "frame": 0}
BG = "#0d0d1a"
plt.style.use("dark_background")
fig = plt.figure(figsize=(8, 9.2))
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0.08, 0.22, 0.86, 0.72])
ax.set_facecolor(BG)

CONFIGS = [
    {"color": "#FFD700", "label": "Bintang", "size": 220, "marker": "*", "lw": 2.0},
    {"color": "#00BFFF", "label": "Planet 1", "size": 60, "marker": "o", "lw": 1.2},
    {"color": "#FF6B6B", "label": "Planet 2", "size": 60, "marker": "o", "lw": 1.2},
]
pad = 0.25
xlim = (history[:, :, 0].min() - pad, history[:, :, 0].max() + pad)
ylim = (history[:, :, 1].min() - pad, history[:, :, 1].max() + pad)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect("equal")
ax.grid(True, color="white", alpha=0.07, linewidth=0.5)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor("#333355")
ax.set_xlabel("Posisi x", fontsize=11, color="#aaaacc")
ax.set_ylabel("Posisi y", fontsize=11, color="#aaaacc")
ax.tick_params(colors="#aaaacc", labelsize=9)
title = ax.set_title(
    "Simulasi Gravitasi N-body | t = 0.000",
    fontsize=13, fontweight="bold", color="white", pad=10
)

tails, dots, speed_texts = [], [], []
for cfg in CONFIGS:
    line, = ax.plot([], [], color=cfg["color"], lw=cfg["lw"], alpha=0.6, zorder=3)
    tails.append(line)
    scat = ax.scatter(
        [], [], s=cfg["size"], color=cfg["color"], marker=cfg["marker"],
        zorder=5, edgecolors="white", linewidths=0.8
    )
    dots.append(scat)
    txt = ax.text(
        0, 0, "", fontsize=7.5, color=cfg["color"], zorder=6,
        va="bottom", ha="left",
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor=BG,
            edgecolor=cfg["color"], alpha=0.55, linewidth=0.6
        )
    )
    speed_texts.append(txt)

for i, cfg in enumerate(CONFIGS):
    ax.plot(
        [], [], color=cfg["color"], lw=1.8,
        label=f"{cfg['label']} (M={masses[i]:.3f})"
    )
ax.legend(
    fontsize=9, framealpha=0.25, facecolor="#1a1a2e",
    edgecolor="#444466", labelcolor="white", loc="upper right"
)
info_box = ax.text(
    0.02, 0.03, "", transform=ax.transAxes, fontsize=8,
    color="#8888aa", va="bottom",
    bbox=dict(
        boxstyle="round,pad=0.4", facecolor="#1a1a2e",
        edgecolor="#333355", alpha=0.7
    )
)

ax_slider = fig.add_axes([0.15, 0.11, 0.70, 0.03])
ax_slider.set_facecolor("#1a1a2e")
slider = Slider(
    ax=ax_slider, label="Kecepatan", valmin=1, valmax=50,
    valinit=SKIP_INIT, valstep=1, color="#4466aa"
)
slider.label.set_color("#aaaacc")
slider.label.set_fontsize(9)
slider.valtext.set_visible(False)
speed_label = fig.text(
    0.86, 0.125, f"{SKIP_INIT}× step/frame",
    fontsize=8, color="#FFD700", ha="left", va="center"
)

ax_btn = fig.add_axes([0.42, 0.04, 0.16, 0.045])
btn_pause = Button(ax_btn, "⏸ Pause", color="#1a1a2e", hovercolor="#2a2a4e")
btn_pause.label.set_color("#FFD700")
btn_pause.label.set_fontsize(10)
ax_btn_reset = fig.add_axes([0.62, 0.04, 0.16, 0.045])
btn_reset = Button(ax_btn_reset, "↺ Reset", color="#1a1a2e", hovercolor="#2a2a4e")
btn_reset.label.set_color("#FF6B6B")
btn_reset.label.set_fontsize(10)


def on_slider(val):
    state["skip"] = int(slider.val)
    speed_label.set_text(f"{int(slider.val)}× step/frame")
    fig.canvas.draw_idle()


def on_pause(event):
    state["paused"] = not state["paused"]
    if state["paused"]:
        btn_pause.label.set_text("▶ Resume")
        btn_pause.label.set_color("#00FF88")
    else:
        btn_pause.label.set_text("⏸ Pause")
        btn_pause.label.set_color("#FFD700")
    fig.canvas.draw_idle()


def on_reset(event):
    state["frame"] = 0
    state["paused"] = False
    btn_pause.label.set_text("⏸ Pause")
    btn_pause.label.set_color("#FFD700")
    fig.canvas.draw_idle()


slider.on_changed(on_slider)
btn_pause.on_clicked(on_pause)
btn_reset.on_clicked(on_reset)


def draw_frame(frame_num):
    """Render posisi benda pada frame saat ini."""
    skip = state["skip"]
    idx = min(state["frame"] * skip, STEPS)
    tail_start = max(0, idx - TAIL_LEN * skip)
    tail_idx = np.arange(tail_start, idx + 1, max(1, skip))

    for i in range(len(masses)):
        tails[i].set_data(
            history[tail_idx, i, 0], history[tail_idx, i, 1]
        )
        px, py = history[idx, i]
        dots[i].set_offsets([[px, py]])
        vx, vy = vel_history[idx, i]
        speed = np.hypot(vx, vy)
        offset = 0.06 * (xlim[1] - xlim[0])
        speed_texts[i].set_position((px + offset, py + offset))
        speed_texts[i].set_text(f"v = {speed:.3f}")

    t = idx * DT
    title.set_text(f"Simulasi Gravitasi N-body | t = {t:.3f}")
    pct = 100 * idx / STEPS
    total_frames = max(1, STEPS // max(1, skip))
    lap = state["frame"] // (total_frames + 1) + 1
    info_box.set_text(
        f"Step {idx}/{STEPS} ({pct:.0f}%) | putaran animasi ke-{lap}\n"
        f"Kecepatan: {skip}× step/frame"
    )
    return tails + dots + speed_texts + [title, info_box]


def update(frame_num):
    if not state["paused"]:
        state["frame"] += 1
        if state["frame"] * state["skip"] > STEPS:
            state["frame"] = 0
    return draw_frame(frame_num)


anim = animation.FuncAnimation(
    fig, update, interval=20, blit=True, cache_frame_data=False
)
fig.suptitle("", fontsize=1)
print("Animasi berjalan. Gunakan slider, Pause/Resume, dan Reset.")
plt.show()

# Grafik energi tampil setelah jendela animasi ditutup
fig_energy, (ax_energy, ax_error) = plt.subplots(
    2, 1, figsize=(10, 7), sharex=True
)
fig_energy.patch.set_facecolor(BG)
for axis in (ax_energy, ax_error):
    axis.set_facecolor(BG)
    axis.grid(True, alpha=0.25)
    axis.tick_params(colors="#ccccdd")
    axis.xaxis.label.set_color("#ccccdd")
    axis.yaxis.label.set_color("#ccccdd")
    axis.title.set_color("white")
    for spine in axis.spines.values():
        spine.set_color("#555577")

ax_energy.plot(time_history, energy_history[:, 0], label="Energi kinetik", linewidth=1.0)
ax_energy.plot(time_history, energy_history[:, 1], label="Energi potensial", linewidth=1.0)
ax_energy.plot(time_history, energy_history[:, 2], label="Energi total", linewidth=1.5)
ax_energy.set_ylabel("Energi")
ax_energy.set_title("Energi Sistem N-body terhadap Waktu")
ax_energy.legend()

ax_error.plot(time_history, relative_energy_error, linewidth=1.0, label="Galat energi relatif")
ax_error.set_xlabel("Waktu simulasi")
ax_error.set_ylabel("|ΔE / E₀|")
ax_error.set_title("Galat Energi Relatif")
ax_error.legend()

fig_energy.tight_layout()
fig_energy.savefig("energi_n_body.png", dpi=200, bbox_inches="tight")
plt.show()

print("\nAnalisis energi selesai.")
print(f"Energi total awal  : {initial_energy:.10f}")
print(f"Energi total akhir : {energy_history[-1, 2]:.10f}")
print(f"Galat energi relatif maksimum: {np.nanmax(relative_energy_error):.6e}")
print("Grafik energi tersimpan sebagai energi_n_body.png")
