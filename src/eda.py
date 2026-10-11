"""Task 2: exploratory plots on the raw (unstandardised) data."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prepare import load_clean

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

df = load_clean()
no = df[df["fault"] == 0]
yes = df[df["fault"] == 1]

# 2.1 currents vs row number, first 2000 rows, fault shaded
first = df.iloc[:2000]
x = np.arange(len(first))
f = first["fault"].to_numpy()
fig, ax = plt.subplots(figsize=(12, 4))
for col in ["Ia", "Ib", "Ic"]:
    ax.plot(x, first[col], label=col, lw=0.8)
starts = np.where((f == 1) & (np.r_[0, f[:-1]] == 0))[0]
ends = np.where((f == 1) & (np.r_[f[1:], 0] == 0))[0]
for s, e in zip(starts, ends):
    ax.axvspan(s, e + 1, color="red", alpha=0.2)
ax.set_xlabel("Row number")
ax.set_ylabel("Current (A)")
ax.set_title("Line currents, first 2000 rows (red = fault)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "plot1_currents_vs_row.png", dpi=150)

# 2.2 scatter Ia vs Ib
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(no["Ia"], no["Ib"], s=4, alpha=0.4, c="tab:blue", label="no fault")
ax.scatter(yes["Ia"], yes["Ib"], s=4, alpha=0.4, c="tab:red", label="fault")
ax.set_xlabel("Ia (A)")
ax.set_ylabel("Ib (A)")
ax.set_title("Ia vs Ib")
ax.legend(markerscale=3)
fig.tight_layout()
fig.savefig(FIG / "plot2_scatter_ia_ib.png", dpi=150)

# 2.3 histogram of Ia
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(df["Ia"].min(), df["Ia"].max(), 80)
ax.hist(no["Ia"], bins=bins, alpha=0.6, color="tab:blue", label="no fault")
ax.hist(yes["Ia"], bins=bins, alpha=0.6, color="tab:red", label="fault")
ax.set_xlabel("Ia (A)")
ax.set_ylabel("Count")
ax.set_title("Histogram of Ia")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "plot3_hist_ia.png", dpi=150)

print("Saved 3 figures to", FIG)
