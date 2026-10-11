"""Task 4: linear and RBF SVMs on Set R, chosen on the validation set."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from prepare import prepare
from metrics import accuracy

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
MOD = ROOT / "results" / "models"
for folder in (FIG, TAB, MOD):
    folder.mkdir(parents=True, exist_ok=True)

d = prepare("R")
Xtr, ytr, Xva, yva = d["X_train"], d["y_train"], d["X_val"], d["y_val"]

Cs = [0.1, 1, 10, 100]
gammas = [0.1, 1]

# every setting to try: (kernel, C, gamma)
settings = [("linear", C, None) for C in Cs]
settings += [("rbf", C, g) for C in Cs for g in gammas]

rows, models = [], []
for kernel, C, g in settings:
    if kernel == "linear":
        model = SVC(kernel="linear", C=C)
    else:
        model = SVC(kernel="rbf", C=C, gamma=g)
    model.fit(Xtr, ytr)
    rows.append({"kernel": kernel, "C": C, "gamma": "-" if g is None else g,
                 "val_acc": round(accuracy(yva, model.predict(Xva)), 4),
                 "n_support": int(model.n_support_.sum())})
    models.append(model)

table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv(TAB / "svm_results.csv", index=False)

# best setting per kernel (the first one wins a tie), saved with joblib
for kernel in ["linear", "rbf"]:
    best = table[table["kernel"] == kernel]["val_acc"].idxmax()
    print(f"\nBest {kernel}: C = {table.loc[best, 'C']}, gamma = {table.loc[best, 'gamma']}, "
          f"val acc = {table.loc[best, 'val_acc']}")
    joblib.dump(models[best], MOD / f"svm_{kernel}.joblib")
# the app and test script need the training mean and std to scale new rows
np.savez(MOD / "svm_scaler_R.npz", mean=d["mean"], std=d["std"])

# figure: validation accuracy against C
fig, ax = plt.subplots(figsize=(7, 4.5))
lin = table[table["kernel"] == "linear"]
ax.plot(lin["C"], lin["val_acc"], "o-", label="linear")
for g in gammas:
    sub = table[(table["kernel"] == "rbf") & (table["gamma"] == g)]
    ax.plot(sub["C"], sub["val_acc"], "o-", label=f"rbf, gamma = {g}")
ax.set_xscale("log")
ax.set_xlabel("C")
ax.set_ylabel("Validation accuracy")
ax.set_title("Validation accuracy against C")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "svm_val_accuracy_vs_C.png", dpi=150)
