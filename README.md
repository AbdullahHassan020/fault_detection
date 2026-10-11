# Transmission line fault detection

Classify one instant of three line currents (Ia, Ib, Ic) and three line voltages
(Va, Vb, Vc) as fault (1) or no fault (0). Logistic regression is written by hand
in NumPy. The SVMs use `sklearn.svm.SVC` only. A Streamlit page shows two saved
models side by side. All tables, figures and written answers are in
`results/analysis.md`.

## Setup

```
pip install -r requirements.txt
```

Put `detect_dataset.csv` from Kaggle in `data/`. Run every command from the
repository root.

## Run everything, in this order

| Step | Command | What it does |
|---|---|---|
| Check loader | `python src/load_data.py` | prints shape, columns, first 5 rows |
| Task 1 | `python src/prepare.py` | cleaning report and split summary |
| Task 2 | `python src/eda.py` | three exploratory plots |
| Task 3 | `python src/train_logistic.py` | alpha search, cost plots, accuracy, theta |
| Task 4 | `python src/train_svm.py` | SVM table, accuracy against C plot, saved SVMs |
| Task 5 | `python src/test_models.py` | test accuracy, confusion matrices, missed faults |
| Task 7 | `streamlit run app/app.py` | web page at http://localhost:8501 |

Tasks 3 and 4 must run before Task 5 and the app, because they save the models.
Plots go to `results/figures/`, CSV tables to `results/tables/`, models and theta
to `results/models/`.

## Front end

Type the six values or press one of the two example buttons. The page shows the
Set Q logistic regression and the RBF SVM side by side, with the probability of
fault for the first and the decision value for the second. It warns when the two
models disagree.

![App screenshot](results/figures/app_screenshot.png)

## Layout

```
data/        detect_dataset.csv
src/         load_data.py, prepare.py, eda.py, logistic.py, metrics.py,
             train_logistic.py, train_svm.py, test_models.py
results/     tables/, figures/, models/, analysis.md
app/         app.py
```
