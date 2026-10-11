"""Task 5: evaluate the four final models once on the test rows."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from prepare import prepare
from logistic import add_intercept, predict
from metrics import accuracy, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "results" / "models"
TAB = ROOT / "results" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

dR, dQ = prepare("R"), prepare("Q")
y_test = dR["y_test"]            # same test rows for both feature sets

theta_R = np.load(MOD / "logistic_theta_R.npz")["theta"]
theta_Q = np.load(MOD / "logistic_theta_Q.npz")["theta"]
svm_linear = joblib.load(MOD / "svm_linear.joblib")
svm_rbf = joblib.load(MOD / "svm_rbf.joblib")

predictions = {
    "Logistic regression, Set R": predict(add_intercept(dR["X_test"]), theta_R),
    "Logistic regression, Set Q": predict(add_intercept(dQ["X_test"]), theta_Q),
    "Linear SVM": svm_linear.predict(dR["X_test"]),
    "RBF SVM": svm_rbf.predict(dR["X_test"]),
}

rows = []
for name, y_pred in predictions.items():
    cm = confusion_matrix(y_test, y_pred)
    assert cm.sum() == len(y_test)           # every test row is counted once
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{name}: accuracy = {accuracy(y_test, y_pred):.4f}, missed faults = {fn}")
    print(cm)
    rows.append({"model": name, "test_acc": round(accuracy(y_test, y_pred), 4),
                 "TN": tn, "FP": fp, "FN": fn, "TP": tp})

table = pd.DataFrame(rows)
print("\n", table.to_string(index=False))
table.to_csv(TAB / "test_results.csv", index=False)
