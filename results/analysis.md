# Results and analysis

Dataset: `detect_dataset.csv`, a simulated three-phase line. Every number below comes from the scripts in `src/`. Run them in the order given in the README to reproduce it.

## Task 1: data preparation

**Table 1: data preparation**

| Item | Value |
|---|---|
| Raw rows / raw columns | 12001 / 9 |
| Columns removed | `Unnamed: 7`, `Unnamed: 8` |
| Missing values after cleaning | 0 in every column |
| Fault rows / no-fault rows | 5496 / 6505 |
| Train / validation / test rows | 7200 / 2400 / 2401 |
| Fault fraction in train / validation / test | 0.4607 / 0.4512 / 0.4565 |

The two removed columns come from trailing commas in the CSV and contain no values at all. The label `Output (S)` was renamed to `fault` and kept out of the features. The six measurement columns are numeric. All features are standardised with the training mean and standard deviation, and the same numbers are used for validation and test.

## Task 2: exploratory plots

![Currents against row number](figures/plot1_currents_vs_row.png)

After a short start-up transient in the first few rows, all three currents are plain sine waves with peaks near 63 A until row 504. The first fault runs from row 504 to row 1632. During it Ia swings up to about 880 A, while Ib and Ic are affected much less (up to about 224 A and 132 A). After the fault clears, all three currents go back to normal.

![Ia against Ib](figures/plot2_scatter_ia_ib.png)

Almost all healthy rows (99.97%) have both |Ia| and |Ib| under 110 A, so they form a small cluster around the origin. Fault rows draw large loops out to about 880 A, but some of the fault traces cross the middle and run through the healthy cluster.

![Histogram of Ia](figures/plot3_hist_ia.png)

Healthy Ia stays between -170 A and 104 A, while fault Ia covers the whole range from -884 A to 886 A. 57.5% of the fault rows have |Ia| above 400 A, which no healthy row comes close to, but 23.7% of the fault rows have |Ia| below 100 A and sit on top of the healthy peak.

**Could one straight line in the (Ia, Ib) plane separate the two classes?** No. A fault makes the current much larger, but the current is still a sine wave, so a faulted phase passes through zero twice per cycle. At those instants Ia is close to zero and the point lands in the same small region as normal operation. The healthy rows sit in the middle of the plane and the fault rows lie on every side of them and also run through them. A straight line cuts the plane into two half-planes and cannot keep a central cluster on one side while fault points surround it. What tells a fault apart is the size of the current, not its signed value at one instant.

## Task 3: logistic regression by hand

With θ = 0 the cost is 0.6931 (log 2) for both feature sets, as it should be. We ran 2000 iterations for α in {0.01, 0.1, 1, 10}.

![Cost, Set R](figures/cost_curves_set_R.png)
![Cost, Set Q](figures/cost_curves_set_Q.png)

For Set R, α = 0.01, 0.1 and 1 lower J at every iteration, while α = 10 overshoots and J jumps up and down, so we kept α = 1. For Set Q all four values lower J at every iteration, so we kept α = 10, the largest in the grid. J for Set Q is still falling slowly at iteration 2000.

**Table 2: logistic regression**

| Feature set | Chosen α | Final J(θ) | Validation accuracy |
|---|---|---|---|
| Set R | 1 | 0.6772 | 0.7308 |
| Set Q | 10 | 0.0534 | 0.9783 |

Training accuracy is 0.7258 for Set R and 0.9788 for Set Q.

## Task 4: support vector machines (Set R)

**Table 3: SVM validation results**

| Kernel | C | γ | Validation accuracy | Support vectors |
|---|---|---|---|---|
| linear | 0.1 | - | 0.7258 | 5849 |
| linear | 1 | - | 0.7258 | 5836 |
| linear | 10 | - | 0.7258 | 5835 |
| linear | 100 | - | 0.7258 | 5836 |
| rbf | 0.1 | 0.1 | 0.9500 | 1347 |
| rbf | 0.1 | 1 | 0.9804 | 848 |
| rbf | 1 | 0.1 | 0.9750 | 723 |
| rbf | 1 | 1 | 0.9917 | 482 |
| rbf | 10 | 0.1 | 0.9921 | 381 |
| rbf | 10 | 1 | 0.9962 | 342 |
| rbf | 100 | 0.1 | 0.9954 | 174 |
| rbf | 100 | 1 | 0.9967 | 266 |

![Validation accuracy against C](figures/svm_val_accuracy_vs_C.png)

We kept the linear SVM with C = 0.1 (all four values tie, so the first one stays) and the RBF SVM with C = 100 and γ = 1. Both are saved in `results/models/`.

## Task 5: test results

The test rows were used once, after all choices were made. The test set has 2401 rows, 1096 of them faults, and every confusion matrix adds up to 2401.

**Table 4: test results**

| Model | Accuracy | TN | False alarms | Missed faults | TP |
|---|---|---|---|---|---|
| Logistic regression, Set R | 0.7264 | 1305 | 0 | 657 | 439 |
| Logistic regression, Set Q | 0.9788 | 1299 | 6 | 45 | 1051 |
| Linear SVM (C=0.1) | 0.7359 | 1305 | 0 | 634 | 462 |
| RBF SVM (C=100, γ=1) | 0.9958 | 1301 | 4 | 6 | 1090 |

**Table 5: learned θ (Set Q, standardised features)**

| Weight | Feature | Value |
|---|---|---|
| θ0 | intercept | 24.9209 |
| θ1 | Ia | 2.3273 |
| θ2 | Ib | -5.3581 |
| θ3 | Ic | -1.7365 |
| θ4 | Va | -0.5336 |
| θ5 | Vb | 0.4167 |
| θ6 | Vc | 0.0986 |
| θ7 | Ia² | 18.8912 |
| θ8 | Ib² | 4.7543 |
| θ9 | Ic² | 13.9569 |
| θ10 | Va² | -6.0276 |
| θ11 | Vb² | -6.4643 |
| θ12 | Vc² | -6.1407 |

## Task 6: analysis

**6.1 Why do logistic regression (Set R) and the linear SVM score almost the same?**
Both models draw one straight boundary through the same six standardised features. Only the loss is different (log loss for logistic regression, hinge loss for the SVM). When no straight line fits the data well, both end up with a similarly poor line. Validation accuracy is 0.7308 and 0.7258, test accuracy is 0.7264 and 0.7359, and both miss several hundred faults (657 and 634). So the limit comes from the straight boundary and not from the training method. It tells us the two classes are not linearly separable in the raw features: healthy rows sit in the middle of the plane and fault rows lie around them, as plot 2 showed. Both models still beat the 54.4% you get on the test set by always answering "no fault", because a line can cut off the fault rows that lie on one side.

**6.2 Signs of the Set Q weights.**
The weights on the I² terms are positive (Ia² 18.89, Ib² 4.75, Ic² 13.96). The weights on the V² terms are negative (Va² -6.03, Vb² -6.46, Vc² -6.14). During a fault the current rises sharply: the mean of Ia² is about 2,057 for healthy rows and 308,250 for fault rows, roughly 150 times more. A large I² points to a fault, so its weight is positive. The voltage does the opposite. It sags during a fault, and the mean of Va² drops from 0.187 to 0.040. A small V² points to a fault, so its weight is negative. The three V² weights are almost equal, which fits a sag that shows on all phases. The three I² weights differ more, but the squared currents are correlated with each other, so we should not read too much into the split between them.

**6.3 Support vectors, linear against RBF.**
The best linear SVM has 5849 support vectors, which is 81% of the 7200 training rows. The best RBF SVM has 266, which is 3.7%. A very large number of support vectors means most points lie inside the margin or on the wrong side of it. The model cannot find a boundary with a clear gap between the classes, so nearly every point takes part in defining it. This is another sign that the data is not linearly separable. With few support vectors, a handful of points define a clean boundary.

**6.4 What happens as C increases?**
For the RBF SVM, accuracy goes up and the number of support vectors goes down. With γ = 1, accuracy goes 0.9804, 0.9917, 0.9962, 0.9967 and support vectors go 848, 482, 342, 266. With γ = 0.1, accuracy goes 0.9500, 0.9750, 0.9921, 0.9954 and support vectors go 1347, 723, 381, 174. C sets the penalty for points that break the margin. A small C tolerates many violations, so the margin is wide and soft, many points end up inside it (many support vectors) and the boundary is smooth but less accurate. A large C punishes violations hard, so the boundary bends to fit more training points and fewer points stay inside the margin. The gain gets small at high C (0.9962 to 0.9967), and a very large C can overfit, which is why the validation set decides where to stop. For the linear SVM nothing changes with C (0.7258 and about 5840 support vectors everywhere), because no value of C can remove the violations that a straight line cannot avoid.

**6.5 Is the test accuracy optimistic or pessimistic?**
Optimistic. The file holds only 5 blocks of consecutive fault rows (1004 to 1134 rows each), and the split is random by row. As a result 84.5% of the test rows have a direct neighbour (the row just before or after) in the training set, and neighbouring samples are almost identical. A model can score well just by recognising a sample it has practically seen, which is not the same as detecting a new fault. A real relay meets faults it was never trained on, at a different place, with a different fault resistance and a different moment of inception. A fairer test would split by event, for example train on 3 fault blocks and test on the other 2. On new events the accuracy is probably lower than 0.9958.

**6.6 Which model would we deploy?**
The RBF SVM. It misses 6 of the 1096 test faults (0.55%) and gives 4 false alarms in 1305 healthy rows (0.31%). Logistic regression on Set Q misses 45 faults (4.1%) with 6 false alarms. The two linear models on raw features never raise a false alarm, but they miss 657 and 634 faults (60% and 58%), which is not acceptable because a missed fault is worse than an unneeded trip. All 6 faults missed by the RBF SVM have every current below 64 A, so at that instant they look like normal operation. A real relay would not act on a single sample. It could trip only when several consecutive samples are predicted as fault, which would catch isolated misses like these at the price of a very small delay. Before deployment the model should also be tested with a split by event (see 6.5), because the numbers above are optimistic.

## Main question

A straight-line boundary on raw instantaneous values does not work: both linear models reach only about 73% and miss more than half of the faults. Both fixes work. Changing the features is the simplest one, since adding the six squares to the same logistic regression lifts test accuracy to 97.9%. Changing the classifier gives the best result, because the RBF SVM reaches 99.6% on the raw features with only 6 missed faults.

## Front end

![App screenshot](figures/app_screenshot.png)