"""Task 7: fault detector front end.

Run from the repository root:  streamlit run app/app.py
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from logistic import add_intercept, hypothesis, predict  # noqa: E402

MODELS = ROOT / "results" / "models"


@st.cache_resource
def load_models():
    q = np.load(MODELS / "logistic_theta_Q.npz")
    s = np.load(MODELS / "svm_scaler_R.npz")
    svm = joblib.load(MODELS / "svm_rbf.joblib")
    return q["theta"], q["mean"], q["std"], svm, s["mean"], s["std"]


theta_q, mean_q, std_q, svm, mean_r, std_r = load_models()

EXAMPLES = {
    "Healthy row": dict(Ia=-27.854, Ib=63.278, Ic=-35.424, Va=-0.589, Vb=0.170, Vc=0.419),
    "Fault row": dict(Ia=522.673, Ib=-7.088, Ic=41.159, Va=0.269, Vb=-0.582, Vc=0.313),
}
NAMES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

st.set_page_config(page_title="Line fault detector", layout="centered")
st.title("Transmission line fault detector")
st.write("Enter one instant of the three line currents (A) and three line voltages "
         "(per unit). Both saved models predict fault or no fault.")

# example buttons write into the input boxes
c1, c2, _ = st.columns([1, 1, 2])
for col, label in zip((c1, c2), EXAMPLES):
    if col.button(f"Load: {label}"):
        for k, v in EXAMPLES[label].items():
            st.session_state[k] = float(v)

for k in NAMES:
    st.session_state.setdefault(k, float(EXAMPLES["Healthy row"][k]))

left, right = st.columns(2)
with left:
    st.subheader("Currents (A)")
    for k in ["Ia", "Ib", "Ic"]:
        st.number_input(k, key=k, format="%.3f")
with right:
    st.subheader("Voltages (per unit)")
    for k in ["Va", "Vb", "Vc"]:
        st.number_input(k, key=k, format="%.3f")

x = np.array([[st.session_state[k] for k in NAMES]])

# Set Q logistic regression: raw + squares, standardised, intercept added
xq = np.hstack([x, x ** 2])
xq = add_intercept((xq - mean_q) / std_q)
prob = float(hypothesis(xq, theta_q)[0])
pred_lr = int(predict(xq, theta_q)[0])

# RBF SVM: Set R, standardised
xr = (x - mean_r) / std_r
pred_svm = int(svm.predict(xr)[0])
score = float(svm.decision_function(xr)[0])

st.divider()
a, b = st.columns(2)
for col, title, pred, detail in [
    (a, "Logistic regression (Set Q)", pred_lr, f"P(fault) = {prob:.3f}"),
    (b, "RBF SVM", pred_svm, f"Decision value = {score:+.3f}"),
]:
    with col:
        st.subheader(title)
        if pred == 1:
            st.error("FAULT")
        else:
            st.success("NO FAULT")
        st.caption(detail)

if pred_lr != pred_svm:
    st.warning("The two models disagree on this input.")
