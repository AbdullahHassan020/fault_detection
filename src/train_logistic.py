"""Task 3: logistic regression by hand on feature sets R and Q."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prepare import prepare
from logistic import add_intercept, cost, fit_logistic, predict
from metrics import accuracy

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
MOD = ROOT / "results" / "models"
for folder in (FIG, TAB, MOD):
    folder.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.01, 0.1, 1, 10]
N_ITERS = 2000

alpha_rows, final_rows = [], []
for fs in ["R", "Q"]:
    d = prepare(fs)
    Xtr, ytr = add_intercept(d["X_train"]), d["y_train"]
    Xva, yva = add_intercept(d["X_val"]), d["y_val"]

    # sanity check: with theta = 0 every h is 0.5, so J must be log 2
    J0 = cost(Xtr, ytr, np.zeros(Xtr.shape[1]))
    assert np.isclose(J0, np.log(2))
    print(f"\nSet {fs}: J at theta = 0 is {J0:.4f}")

    # 3.4 run every alpha, keep the largest one whose cost falls at every iteration
    fig, ax = plt.subplots(figsize=(7, 4.5))
    chosen = None
    for alpha in ALPHAS:
        theta, J = fit_logistic(Xtr, ytr, alpha, N_ITERS)
        falls = bool(np.all(np.diff(J) < 0))
        alpha_rows.append({"set": fs, "alpha": alpha, "final_J": J[-1], "decreases_every_iter": falls})
        ax.plot(J, label=f"alpha = {alpha}")
        if falls:
            chosen = alpha
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("J(theta)")
    ax.set_title(f"Cost against iteration, Set {fs}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / f"cost_curves_set_{fs}.png", dpi=150)
    plt.close(fig)

    # 3.5 final model
    theta, J = fit_logistic(Xtr, ytr, chosen, N_ITERS)
    tr_acc = accuracy(ytr, predict(Xtr, theta))
    va_acc = accuracy(yva, predict(Xva, theta))
    final_rows.append({"set": fs, "alpha": chosen, "final_J": round(J[-1], 4),
                       "train_acc": round(tr_acc, 4), "val_acc": round(va_acc, 4)})
    print(f"chosen alpha = {chosen}, train acc = {tr_acc:.4f}, val acc = {va_acc:.4f}")
    print("theta =", np.round(theta, 4))
    np.savez(MOD / f"logistic_theta_{fs}.npz", theta=theta, mean=d["mean"], std=d["std"])

pd.DataFrame(alpha_rows).to_csv(TAB / "logistic_alpha_search.csv", index=False)
final = pd.DataFrame(final_rows)
final.to_csv(TAB / "logistic_final.csv", index=False)
print("\n", final.to_string(index=False))
