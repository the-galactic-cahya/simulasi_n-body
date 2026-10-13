"""
Analisis Resonansi Orbital -- Sistem Jupiter-Saturnus
======================================================
Membaca data dari data_n_body.csv yang dihasilkan main.py,
lalu menganalisis resonansi 5:2 (near-resonance) antara
Jupiter dan Saturnus.

Konsep:
  - Mean motion  n = 2*pi / T  [rad/yr]
  - Mean longitude  lambda = n*t + lambda_0
  - Resonant angle  phi = p*lambda_J - q*lambda_S + (q-p)*omega
    Untuk resonansi 5:2: phi = 2*lambda_J - 5*lambda_S + 3*omega_J
    Kalau phi berosilasi (librate) -> resonansi terkunci
    Kalau phi berputar terus (circulate) -> near-resonance (kondisi Jup-Sat)

Referensi:
  Murray & Dermott, Solar System Dynamics, Cambridge (1999), Bab 8
  Michtchenko & Ferraz-Mello, AJ 122, 474 (2001)
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import lombscargle
import os

# ============================================================
#  BACA DATA CSV
# ============================================================
csv_path = "data_n_body.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"File '{csv_path}' tidak ditemukan.\n"
        "Jalankan main.py terlebih dahulu untuk menghasilkan data."
    )

print(f"Membaca {csv_path}...", end=" ", flush=True)
data = []
with open(csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({k: float(v) for k, v in row.items()})
print(f"selesai ({len(data):,} baris).")

# Ekstrak kolom
t   = np.array([d["time_yr"]  for d in data])
x0  = np.array([d["x0_AU"]   for d in data])
y0  = np.array([d["y0_AU"]   for d in data])
x1  = np.array([d["x1_AU"]   for d in data])
y1  = np.array([d["y1_AU"]   for d in data])
x2  = np.array([d["x2_AU"]   for d in data])
y2  = np.array([d["y2_AU"]   for d in data])

# ============================================================
#  HITUNG BESARAN ORBITAL
# ============================================================

def orbital_elements(x, y, vx, vy, x_star, y_star, vx_star, vy_star,
                     M_star, M_planet, G=4*np.pi**2):
    """
    Hitung semi-major axis dan eksentrisitas dari posisi & kecepatan
    relatif terhadap bintang (two-body approximation).

    Return: (a, e, theta) -- a [AU], e [-], theta [rad] (true longitude)
    """
    # posisi & kecepatan relatif
    dx = x - x_star;   dy = y - y_star
    dvx = vx - vx_star; dvy = vy - vy_star

    mu = G * (M_star + M_planet)
    r  = np.sqrt(dx**2 + dy**2)
    v2 = dvx**2 + dvy**2

    # energi spesifik
    eps = 0.5 * v2 - mu / r

    # semi-major axis dari hukum vis-viva: eps = -mu/(2a)
    a = -mu / (2.0 * eps)

    # momentum sudut spesifik (komponen z)
    h = dx * dvy - dy * dvx

    # vektor eksentrisitas (Laplace-Runge-Lenz)
    ex = (v2 * dx - (dx*dvx + dy*dvy) * dvx) / mu - dx / r
    ey = (v2 * dy - (dx*dvx + dy*dvy) * dvy) / mu - dy / r
    e  = np.sqrt(ex**2 + ey**2)

    # argumen perihelion omega = arctan2(ey, ex)
    omega = np.arctan2(ey, ex)

    # true longitude = arctan2(y_rel, x_rel)  (approx mean longitude untuk e kecil)
    theta = np.arctan2(dy, dx)

    return a, e, omega, theta


# Kecepatan dari diferensiasi numerik (data CSV tidak menyimpan vx1/vy1 planet
# dalam kolom yang mudah diakses -- kita baca ulang)
# Sebenarnya data_n_body.csv MENYIMPAN kecepatan. Baca ulang.
print("Menghitung elemen orbital...", end=" ", flush=True)

vx0 = np.array([d["vx0_AU_yr"] for d in data])
vy0 = np.array([d["vy0_AU_yr"] for d in data])
vx1 = np.array([d["vx1_AU_yr"] for d in data])
vy1 = np.array([d["vy1_AU_yr"] for d in data])
vx2 = np.array([d["vx2_AU_yr"] for d in data])
vy2 = np.array([d["vy2_AU_yr"] for d in data])

M_star = 1.0
M_J    = 9.5479e-4
M_S    = 2.8588e-4

N = len(t)
a1_arr  = np.zeros(N);  e1_arr  = np.zeros(N)
a2_arr  = np.zeros(N);  e2_arr  = np.zeros(N)
om1_arr = np.zeros(N);  th1_arr = np.zeros(N)
om2_arr = np.zeros(N);  th2_arr = np.zeros(N)

for i in range(N):
    a1_arr[i], e1_arr[i], om1_arr[i], th1_arr[i] = orbital_elements(
        x1[i], y1[i], vx1[i], vy1[i],
        x0[i], y0[i], vx0[i], vy0[i],
        M_star, M_J
    )
    a2_arr[i], e2_arr[i], om2_arr[i], th2_arr[i] = orbital_elements(
        x2[i], y2[i], vx2[i], vy2[i],
        x0[i], y0[i], vx0[i], vy0[i],
        M_star, M_S
    )

print("selesai.")

# ============================================================
#  MEAN MOTION  (dari semi-major axis: n = sqrt(G*M/a^3))
# ============================================================
G = 4.0 * np.pi**2
n1 = np.sqrt(G * (M_star + M_J) / a1_arr**3)   # [rad/yr]
n2 = np.sqrt(G * (M_star + M_S) / a2_arr**3)   # [rad/yr]

# Rasio mean motion (harus mendekati 5/2 = 2.5 untuk resonansi 5:2)
# Catatan: n1 > n2 karena Jupiter lebih dalam
ratio = n1 / n2

# Periode dari mean motion
T1_arr = 2.0 * np.pi / n1
T2_arr = 2.0 * np.pi / n2

print(f"  Mean motion Jupiter rata2  : {n1.mean():.6f} rad/yr  "
      f"(T = {T1_arr.mean():.4f} yr)")
print(f"  Mean motion Saturnus rata2 : {n2.mean():.6f} rad/yr  "
      f"(T = {T2_arr.mean():.4f} yr)")
print(f"  Rasio n_J/n_S              : {ratio.mean():.6f}  "
      f"(5/2 = {5/2:.6f}, selisih = {abs(ratio.mean()-2.5)/2.5*100:.3f}%)")

# ============================================================
#  MEAN LONGITUDE
# ============================================================
# lambda = theta + omega + M  (untuk orbit hampir melingkar, lambda ~ theta)
# Kita pakai true longitude theta sebagai aproksimasi pertama,
# lalu buat lambda berkesinambungan (unwrap)
lambda1 = np.unwrap(th1_arr)   # [rad], berkesinambungan
lambda2 = np.unwrap(th2_arr)

# ============================================================
#  SUDUT RESONANSI  phi = 2*lambda_J - 5*lambda_S + 3*omega_J
#  (resonansi 5:2, p=2, q=5)
# ============================================================
phi_raw = 2.0 * lambda1 - 5.0 * lambda2 + 3.0 * om1_arr
# Lipat ke [-pi, pi] agar mudah divisualisasikan
phi = (phi_raw + np.pi) % (2 * np.pi) - np.pi

# ============================================================
#  POWER SPECTRUM -- frekuensi orbital (Lomb-Scargle)
# ============================================================
# Gunakan posisi radial r1 dan r2 sebagai sinyal periodik
r1 = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
r2 = np.sqrt((x2 - x0)**2 + (y2 - y0)**2)

# Subsample agar cepat (setiap 10 langkah)
ss   = 10
t_ss = t[::ss]
r1_ss = r1[::ss] - r1[::ss].mean()
r2_ss = r2[::ss] - r2[::ss].mean()

# Frekuensi yang dicari: 0.01 sampai 2 siklus/tahun
f_min, f_max = 0.01, 2.0
freqs = np.linspace(f_min, f_max, 5000)
ang_freqs = 2 * np.pi * freqs

pgram1 = lombscargle(t_ss, r1_ss, ang_freqs, normalize=True)
pgram2 = lombscargle(t_ss, r2_ss, ang_freqs, normalize=True)

# Frekuensi dominan
f1_peak = freqs[np.argmax(pgram1)]
f2_peak = freqs[np.argmax(pgram2)]
print(f"  Frekuensi dominan Jupiter  : {f1_peak:.5f} siklus/yr  "
      f"(T = {1/f1_peak:.4f} yr)")
print(f"  Frekuensi dominan Saturnus : {f2_peak:.5f} siklus/yr  "
      f"(T = {1/f2_peak:.4f} yr)")
print(f"  Rasio frekuensi            : {f1_peak/f2_peak:.5f}  "
      f"(ideal 5/2 = 2.50000)")

# ============================================================
#  PLOT
# ============================================================
BG     = "#080812"
C_GRID = "#ffffff"
C_SPIN = "#1e1e3a"
C_TEXT = "#aaaacc"
C_DIM  = "#4a4a6a"
C_J    = "#4FC3F7"    # Jupiter
C_S    = "#FF7043"    # Saturnus
C_PHI  = "#FFD700"    # sudut resonansi
C_R    = "#81C784"    # rasio

plt.style.use("dark_background")
fig = plt.figure(figsize=(14, 10), facecolor=BG)
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    left=0.07, right=0.97,
    top=0.91,  bottom=0.08,
    wspace=0.38, hspace=0.55,
)

ax_a    = fig.add_subplot(gs[0, 0])   # semi-major axis vs waktu
ax_e    = fig.add_subplot(gs[1, 0])   # eksentrisitas vs waktu
ax_T    = fig.add_subplot(gs[2, 0])   # periode vs waktu
ax_rat  = fig.add_subplot(gs[0, 1])   # rasio mean motion vs waktu
ax_phi  = fig.add_subplot(gs[1, 1])   # sudut resonansi vs waktu
ax_hist = fig.add_subplot(gs[2, 1])   # histogram sudut resonansi
ax_ps1  = fig.add_subplot(gs[0, 2])   # power spectrum Jupiter
ax_ps2  = fig.add_subplot(gs[1, 2])   # power spectrum Saturnus
ax_xy   = fig.add_subplot(gs[2, 2])   # diagram x-y orbit

axes_all = [ax_a, ax_e, ax_T, ax_rat, ax_phi,
            ax_hist, ax_ps1, ax_ps2, ax_xy]
for ax in axes_all:
    ax.set_facecolor(BG)
    ax.grid(True, color=C_GRID, alpha=0.06, linewidth=0.4)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_edgecolor(C_SPIN)
    ax.tick_params(colors=C_DIM, labelsize=7.5)

# ── Plot 1: semi-major axis ───────────────────────────────────
ax_a.plot(t, a1_arr, color=C_J, lw=0.8, label="Jupiter")
ax_a.plot(t, a2_arr, color=C_S, lw=0.8, label="Saturnus")
ax_a.axhline(5.2036, color=C_J, lw=0.5, ls="--", alpha=0.4)
ax_a.axhline(9.5371, color=C_S, lw=0.5, ls="--", alpha=0.4)
ax_a.set_title("Semi-major Axis",   fontsize=9, color=C_TEXT, pad=3)
ax_a.set_ylabel("a  [AU]",          fontsize=8, color=C_DIM)
ax_a.set_xlabel("Waktu  [yr]",      fontsize=8, color=C_DIM)
ax_a.legend(fontsize=7.5, framealpha=0.2, facecolor="#0a0a1e",
            edgecolor=C_SPIN, labelcolor="white")

# ── Plot 2: eksentrisitas ─────────────────────────────────────
ax_e.plot(t, e1_arr, color=C_J, lw=0.8, label="Jupiter")
ax_e.plot(t, e2_arr, color=C_S, lw=0.8, label="Saturnus")
# Nilai referensi eksentrisitas nyata
ax_e.axhline(0.0489, color=C_J, lw=0.5, ls="--", alpha=0.5,
             label="e_J nyata = 0.049")
ax_e.axhline(0.0565, color=C_S, lw=0.5, ls=":",  alpha=0.5,
             label="e_S nyata = 0.057")
ax_e.set_title("Eksentrisitas Orbital", fontsize=9, color=C_TEXT, pad=3)
ax_e.set_ylabel("e  [-]",               fontsize=8, color=C_DIM)
ax_e.set_xlabel("Waktu  [yr]",          fontsize=8, color=C_DIM)
ax_e.legend(fontsize=6.5, framealpha=0.2, facecolor="#0a0a1e",
            edgecolor=C_SPIN, labelcolor="white", ncol=2)

# ── Plot 3: periode orbital ───────────────────────────────────
ax_T.plot(t, T1_arr, color=C_J, lw=0.8, label="Jupiter")
ax_T.plot(t, T2_arr, color=C_S, lw=0.8, label="Saturnus")
ax_T.axhline(11.862, color=C_J, lw=0.5, ls="--", alpha=0.5,
             label="T_J nyata = 11.86 yr")
ax_T.axhline(29.457, color=C_S, lw=0.5, ls=":",  alpha=0.5,
             label="T_S nyata = 29.46 yr")
ax_T.set_title("Periode Orbital",   fontsize=9, color=C_TEXT, pad=3)
ax_T.set_ylabel("T  [yr]",          fontsize=8, color=C_DIM)
ax_T.set_xlabel("Waktu  [yr]",      fontsize=8, color=C_DIM)
ax_T.legend(fontsize=6.5, framealpha=0.2, facecolor="#0a0a1e",
            edgecolor=C_SPIN, labelcolor="white", ncol=2)

# ── Plot 4: rasio mean motion ─────────────────────────────────
ax_rat.plot(t, ratio, color=C_R, lw=0.8, label="n_J / n_S")
ax_rat.axhline(2.5, color="white", lw=1.0, ls="--", alpha=0.6,
               label="5/2 = 2.500 (resonansi tepat)")
ax_rat.axhline(ratio.mean(), color=C_R, lw=0.6, ls=":",
               label=f"rata2 = {ratio.mean():.5f}")
ax_rat.set_title("Rasio Mean Motion  n_J / n_S", fontsize=9, color=C_TEXT, pad=3)
ax_rat.set_ylabel("n_J / n_S  [-]",              fontsize=8, color=C_DIM)
ax_rat.set_xlabel("Waktu  [yr]",                 fontsize=8, color=C_DIM)
ax_rat.legend(fontsize=7, framealpha=0.2, facecolor="#0a0a1e",
              edgecolor=C_SPIN, labelcolor="white")

# ── Plot 5: sudut resonansi ───────────────────────────────────
# Scatter kecil agar pola bisa terlihat
ss_phi = max(1, len(t) // 3000)
ax_phi.scatter(t[::ss_phi], phi[::ss_phi],
               s=0.4, color=C_PHI, alpha=0.5, linewidths=0)
ax_phi.axhline(0,    color="white", lw=0.5, ls="--", alpha=0.3)
ax_phi.axhline( np.pi, color="white", lw=0.4, ls=":", alpha=0.2)
ax_phi.axhline(-np.pi, color="white", lw=0.4, ls=":", alpha=0.2)
ax_phi.set_title(r"Sudut Resonansi  $\phi = 2\lambda_J - 5\lambda_S + 3\omega_J$",
                 fontsize=9, color=C_TEXT, pad=3)
ax_phi.set_ylabel(r"$\phi$  [rad]",  fontsize=8, color=C_DIM)
ax_phi.set_xlabel("Waktu  [yr]",     fontsize=8, color=C_DIM)
ax_phi.set_ylim(-np.pi - 0.3, np.pi + 0.3)
ax_phi.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax_phi.set_yticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"],
                        fontsize=7, color=C_DIM)

# Anotasi: circulate = near-resonance, librate = resonance lock
circ_frac = np.sum(np.abs(np.diff(phi)) > np.pi) / len(phi)
status = "SIRKULASI (near-resonance)" if circ_frac > 0.01 else "LIBRASI (resonansi terkunci)"
ax_phi.text(0.02, 0.95, status, transform=ax_phi.transAxes,
            fontsize=7.5, color=C_PHI, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a1e",
                      edgecolor=C_PHI, alpha=0.7))

# ── Plot 6: histogram sudut resonansi ────────────────────────
counts, bin_edges = np.histogram(phi, bins=72, range=(-np.pi, np.pi))
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
ax_hist.bar(bin_centers, counts, width=bin_edges[1]-bin_edges[0],
            color=C_PHI, alpha=0.75, edgecolor="none")
ax_hist.set_title(r"Distribusi $\phi$  (histogram)",
                  fontsize=9, color=C_TEXT, pad=3)
ax_hist.set_xlabel(r"$\phi$  [rad]", fontsize=8, color=C_DIM)
ax_hist.set_ylabel("Jumlah",         fontsize=8, color=C_DIM)
ax_hist.set_xlim(-np.pi, np.pi)
ax_hist.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax_hist.set_xticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"],
                         fontsize=7, color=C_DIM)
# Kalau distribusi seragam -> sirkulasi; kalau ada puncak -> librasi
ax_hist.axhline(len(phi) / 72, color="white", lw=0.7, ls="--",
                alpha=0.4, label="distribusi seragam")
ax_hist.legend(fontsize=7, framealpha=0.2, facecolor="#0a0a1e",
               edgecolor=C_SPIN, labelcolor="white")

# ── Plot 7 & 8: power spectrum ────────────────────────────────
for ax_ps, pgram, f_peak, label, color in [
    (ax_ps1, pgram1, f1_peak, "Jupiter",  C_J),
    (ax_ps2, pgram2, f2_peak, "Saturnus", C_S),
]:
    ax_ps.plot(freqs, pgram, color=color, lw=0.8)
    ax_ps.axvline(f_peak, color="white", lw=0.8, ls="--", alpha=0.6,
                  label=f"f = {f_peak:.4f} /yr\nT = {1/f_peak:.3f} yr")
    ax_ps.set_title(f"Power Spectrum -- {label}", fontsize=9, color=C_TEXT, pad=3)
    ax_ps.set_xlabel("Frekuensi  [siklus/yr]", fontsize=8, color=C_DIM)
    ax_ps.set_ylabel("Normalized Power",        fontsize=8, color=C_DIM)
    ax_ps.legend(fontsize=7.5, framealpha=0.2, facecolor="#0a0a1e",
                 edgecolor=C_SPIN, labelcolor="white")

# ── Plot 9: orbit xy ──────────────────────────────────────────
ss_xy = max(1, len(t) // 4000)
ax_xy.scatter(x0[::ss_xy], y0[::ss_xy], s=0.3, color="#FFF176", alpha=0.5)
ax_xy.scatter(x1[::ss_xy], y1[::ss_xy], s=0.3, color=C_J, alpha=0.4,
              label="Jupiter")
ax_xy.scatter(x2[::ss_xy], y2[::ss_xy], s=0.3, color=C_S, alpha=0.4,
              label="Saturnus")
ax_xy.set_aspect("equal")
ax_xy.set_title("Orbit (proyeksi xy)", fontsize=9, color=C_TEXT, pad=3)
ax_xy.set_xlabel("x  [AU]",            fontsize=8, color=C_DIM)
ax_xy.set_ylabel("y  [AU]",            fontsize=8, color=C_DIM)
ax_xy.legend(fontsize=7.5, framealpha=0.2, facecolor="#0a0a1e",
             edgecolor=C_SPIN, labelcolor="white", markerscale=5)

# ── Judul figur ───────────────────────────────────────────────
fig.text(
    0.5, 0.970,
    "Analisis Resonansi Orbital 5:2  --  Sistem Jupiter-Saturnus  "
    f"(simulasi {t[-1]:.0f} yr)",
    ha="center", va="top", fontsize=11, color="white", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d0d20",
              edgecolor=C_SPIN, alpha=0.90)
)

# ── Simpan & tampilkan ────────────────────────────────────────
out_path = "analisis_resonansi.png"
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"\nGrafik disimpan: {out_path}")

# Ringkasan statistik ke terminal
print("\n" + "=" * 58)
print("  RINGKASAN ANALISIS RESONANSI 5:2")
print("=" * 58)
print(f"  a_Jupiter  : {a1_arr.mean():.4f} +/- {a1_arr.std():.6f}  AU")
print(f"  a_Saturnus : {a2_arr.mean():.4f} +/- {a2_arr.std():.6f}  AU")
print(f"  e_Jupiter  : {e1_arr.mean():.5f} +/- {e1_arr.std():.6f}")
print(f"  e_Saturnus : {e2_arr.mean():.5f} +/- {e2_arr.std():.6f}")
print(f"  T_Jupiter  : {T1_arr.mean():.4f} +/- {T1_arr.std():.5f}  yr  (nyata: 11.862 yr)")
print(f"  T_Saturnus : {T2_arr.mean():.4f} +/- {T2_arr.std():.5f}  yr  (nyata: 29.457 yr)")
print(f"  n_J/n_S    : {ratio.mean():.6f} +/- {ratio.std():.8f}  (5/2 = 2.500000)")
print(f"  Deviasi dari 5:2 : {abs(ratio.mean()-2.5)/2.5*100:.4f}%")
print(f"  f_J (Lomb-Scargle): {f1_peak:.5f} siklus/yr  (T = {1/f1_peak:.4f} yr)")
print(f"  f_S (Lomb-Scargle): {f2_peak:.5f} siklus/yr  (T = {1/f2_peak:.4f} yr)")
print(f"  Status phi         : {status}")
print("=" * 58)

plt.show()
