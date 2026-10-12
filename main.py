import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

# ── Parameter simulasi ────────────────────────────────────────────────────────
G         = 1.0
DT        = 0.001
STEPS     = 40000    # t_max=40 → Planet1 ~6 putaran, Planet2 ~3 putaran
SOFTENING = 1e-3
TAIL_LEN  = 400      # panjang ekor lintasan (dalam frame)
SKIP_INIT = 5        # step fisika per frame (kecepatan awal animasi)

# ── Kondisi awal ──────────────────────────────────────────────────────────────
masses = np.array([1.0, 0.001, 0.001])

# Planet 1: orbit lingkaran r=1.0, v_sirkular = sqrt(G*M/r) = 1.0
# Planet 2: orbit lingkaran r=1.6, v_sirkular = sqrt(G*M/r) ≈ 0.7906
# Keduanya dimulai di sumbu-x berlawanan agar tidak bertabrakan
R1 = 1.0
R2 = 1.6
V1 = np.sqrt(G * masses[0] / R1)   # ~1.000
V2 = np.sqrt(G * masses[0] / R2)   # ~0.791

positions = np.array([
    [ 0.0,  0.0],   # Bintang di pusat
    [ R1,   0.0],   # Planet 1 di kanan
    [-R2,   0.0],   # Planet 2 di kiri (radius berbeda)
], dtype=float)

velocities = np.array([
    [0.0,  0.0],    # Bintang diam
    [0.0,  V1],     # Planet 1 bergerak ke atas
    [0.0, -V2],     # Planet 2 bergerak ke bawah
], dtype=float)

# ── Integrasi fisika (leapfrog) ───────────────────────────────────────────────
def acceleration(pos):
    disp  = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    dist2 = np.sum(disp**2, axis=2) + SOFTENING**2
    np.fill_diagonal(dist2, np.inf)
    inv_d3   = dist2**(-1.5)
    weighted = disp * inv_d3[:, :, np.newaxis] * masses[np.newaxis, :, np.newaxis]
    return G * np.sum(weighted, axis=1)

print("Menghitung lintasan... ", end="", flush=True)
history     = np.zeros((STEPS + 1, len(masses), 2))
vel_history = np.zeros((STEPS + 1, len(masses), 2))
history[0]     = positions
vel_history[0] = velocities

acc = acceleration(positions)
for step in range(1, STEPS + 1):
    v_half     = velocities + 0.5 * acc * DT
    positions  = positions  + v_half * DT
    new_acc    = acceleration(positions)
    velocities = v_half + 0.5 * new_acc * DT
    acc        = new_acc
    history[step]     = positions
    vel_history[step] = velocities
print("selesai.")

# ── State animasi ─────────────────────────────────────────────────────────────
state = {
    "skip":    SKIP_INIT,   # step fisika per frame (dikontrol slider)
    "paused":  False,
    "frame":   0,           # frame saat ini
}

# ── Setup figure ──────────────────────────────────────────────────────────────
BG = "#0d0d1a"
plt.style.use("dark_background")

fig = plt.figure(figsize=(8, 9.2))
fig.patch.set_facecolor(BG)

# Area utama simulasi
ax = fig.add_axes([0.08, 0.22, 0.86, 0.72])   # [left, bottom, width, height]
ax.set_facecolor(BG)

CONFIGS = [
    {"color": "#FFD700", "label": "Bintang",  "size": 220, "marker": "*", "lw": 2.0},
    {"color": "#00BFFF", "label": "Planet 1", "size": 60,  "marker": "o", "lw": 1.2},
    {"color": "#FF6B6B", "label": "Planet 2", "size": 60,  "marker": "o", "lw": 1.2},
]

pad  = 0.25
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

title = ax.set_title("Simulasi Gravitasi N-body  |  t = 0.000",
                     fontsize=13, fontweight="bold", color="white", pad=10)

# ── Objek animasi ─────────────────────────────────────────────────────────────
tails, dots, speed_texts = [], [], []

for i, cfg in enumerate(CONFIGS):
    line, = ax.plot([], [], color=cfg["color"], lw=cfg["lw"], alpha=0.6, zorder=3)
    tails.append(line)

    scat = ax.scatter([], [], s=cfg["size"], color=cfg["color"],
                      marker=cfg["marker"], zorder=5,
                      edgecolors="white", linewidths=0.8)
    dots.append(scat)

    txt = ax.text(0, 0, "", fontsize=7.5, color=cfg["color"], zorder=6,
                  va="bottom", ha="left",
                  bbox=dict(boxstyle="round,pad=0.2", facecolor=BG,
                            edgecolor=cfg["color"], alpha=0.55, linewidth=0.6))
    speed_texts.append(txt)

for i, cfg in enumerate(CONFIGS):
    ax.plot([], [], color=cfg["color"], lw=1.8,
            label=f"{cfg['label']}  (M={masses[i]:.3f})")
ax.legend(fontsize=9, framealpha=0.25, facecolor="#1a1a2e",
          edgecolor="#444466", labelcolor="white", loc="upper right")

info_box = ax.text(
    0.02, 0.03, "", transform=ax.transAxes,
    fontsize=8, color="#8888aa", va="bottom",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e",
              edgecolor="#333355", alpha=0.7)
)

# ── Widget: Slider kecepatan ──────────────────────────────────────────────────
ax_slider = fig.add_axes([0.15, 0.11, 0.70, 0.03])
ax_slider.set_facecolor("#1a1a2e")

slider = Slider(
    ax=ax_slider,
    label="Kecepatan",
    valmin=1,
    valmax=50,
    valinit=SKIP_INIT,
    valstep=1,
    color="#4466aa",
)
slider.label.set_color("#aaaacc")
slider.label.set_fontsize(9)

# Sembunyikan valtext bawaan (penyebab tabrakan), ganti dengan teks kustom di atas slider
slider.valtext.set_visible(False)
speed_label = fig.text(
    0.86, 0.125, f"{SKIP_INIT}× step/frame",
    fontsize=8, color="#FFD700", ha="left", va="center"
)

# ── Widget: Tombol Pause / Resume ─────────────────────────────────────────────
ax_btn = fig.add_axes([0.42, 0.04, 0.16, 0.045])
btn_pause = Button(ax_btn, "⏸  Pause",
                   color="#1a1a2e", hovercolor="#2a2a4e")
btn_pause.label.set_color("#FFD700")
btn_pause.label.set_fontsize(10)

# ── Widget: Tombol Reset ──────────────────────────────────────────────────────
ax_btn_reset = fig.add_axes([0.62, 0.04, 0.16, 0.045])
btn_reset = Button(ax_btn_reset, "↺  Reset",
                   color="#1a1a2e", hovercolor="#2a2a4e")
btn_reset.label.set_color("#FF6B6B")
btn_reset.label.set_fontsize(10)

# ── Callback widget ───────────────────────────────────────────────────────────
def on_slider(val):
    state["skip"] = int(slider.val)
    speed_label.set_text(f"{int(slider.val)}× step/frame")
    fig.canvas.draw_idle()

def on_pause(event):
    state["paused"] = not state["paused"]
    if state["paused"]:
        btn_pause.label.set_text("▶  Resume")
        btn_pause.label.set_color("#00FF88")
    else:
        btn_pause.label.set_text("⏸  Pause")
        btn_pause.label.set_color("#FFD700")
    fig.canvas.draw_idle()

def on_reset(event):
    state["frame"]  = 0
    state["paused"] = False
    btn_pause.label.set_text("⏸  Pause")
    btn_pause.label.set_color("#FFD700")
    fig.canvas.draw_idle()

slider.on_changed(on_slider)
btn_pause.on_clicked(on_pause)
btn_reset.on_clicked(on_reset)

# ── Fungsi animasi ────────────────────────────────────────────────────────────
def draw_frame(frame_num):
    """Render state["frame"] ke canvas."""
    skip = state["skip"]
    idx  = min(state["frame"] * skip, STEPS)

    tail_start = max(0, idx - TAIL_LEN * skip)

    for i in range(len(masses)):
        # Ekor: ambil setiap `skip` titik agar panjang ekor konsisten
        tail_idx = np.arange(tail_start, idx + 1, max(1, skip))
        tails[i].set_data(history[tail_idx, i, 0], history[tail_idx, i, 1])

        px, py = history[idx, i, 0], history[idx, i, 1]
        dots[i].set_offsets([[px, py]])

        vx, vy = vel_history[idx, i, 0], vel_history[idx, i, 1]
        speed  = np.hypot(vx, vy)

        offset = 0.06 * (xlim[1] - xlim[0])
        speed_texts[i].set_position((px + offset, py + offset))
        speed_texts[i].set_text(f"v = {speed:.3f}")

    t = idx * DT
    title.set_text(f"Simulasi Gravitasi N-body  |  t = {t:.3f}")

    total_frames = STEPS // max(1, skip)
    pct = 100 * (state["frame"] % (total_frames + 1)) / max(1, total_frames)
    lap  = state["frame"] // (total_frames + 1) + 1
    info_box.set_text(
        f"Step {idx}/{STEPS}  ({pct:.0f}%)  putaran ke-{lap}\n"
        f"Kecepatan: {skip}× step/frame"
    )

    return tails + dots + speed_texts + [title, info_box]


def update(frame_num):
    if not state["paused"]:
        skip = state["skip"]
        state["frame"] += 1
        # Loop kembali ke awal setelah seluruh simulasi selesai
        if state["frame"] * skip > STEPS:
            state["frame"] = 0
    return draw_frame(frame_num)


anim = animation.FuncAnimation(
    fig, update,
    interval=20,       # ms — refresh rate tetap 50 fps, kecepatan lewat slider
    blit=True,
    cache_frame_data=False,
)

# Judul window
fig.suptitle("", fontsize=1)   # dummy agar layout tidak geser
plt.subplots_adjust(bottom=0)  # diatur manual lewat add_axes

print("Animasi berjalan. Gunakan slider untuk mengatur kecepatan.")
plt.show()
