# Amazon Consumer Household Size Prediction

**What you buy says more than who you are.**

[![Live Dashboard](https://img.shields.io/badge/dashboard-live-e07a1f)](https://summerkirbs.github.io/Amazon-Consumer-Household-Size-Prediction/)
[![R](https://img.shields.io/badge/R-analysis-276DC3?logo=r&logoColor=white)](https://www.r-project.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A survey records how many people live in each respondent's household. A purchase log records
1.85 million line items for those same accounts over six and a half years. This project asks
whether purchase behavior predicts household size better than demographics alone.

Short answer: yes, but weakly. Children's products carry most of the signal, spending per line
carries none, and three different learners finish within one fold's worth of noise of each
other. The interesting result is not which model won. It is that almost every plausible
hypothesis about household consumption turns out to be flat.

The public dashboard is a static site built from the R code in this repository. No
respondent-level data is published.

## The finding

| Signal | Correlation with household size | Verdict |
|---|---|---|
| Children's products | **0.228** | Real, and by far the strongest |
| Total spend | 0.104 | Weak |
| Purchase lines per day | 0.103 | Weak |
| Total items | 0.098 | Weak |
| Shared consumables | 0.035 | Flat |
| Average line value | 0.005 | Flat |
| Food and grocery | −0.003 | Flat |

Mean children's-product lines run 3.6 → 5.9 → 10.6 → 15.5 across household sizes 1 to 4. Food
and grocery averages wander between 20.6 and 22.6 with no order to them. Shared consumables,
the paper towels and detergent a larger household should burn through faster, barely move.
Average spend per line sits near $26 for every household size, and the average basket is flat
at about $53.

Demographics do even less. Race group means span 0.15 of a person. State means span less than
one person once groups with fewer than 20 respondents are set aside. The only outliers are tiny
groups: one respondent outside the US and nine in Hawaii.

## Model results

Ten-fold cross-validation on the 3,027 labeled rows, the same folds for every learner:

| Learner | Mean RMSE | SD | Best fold | Worst fold |
|---|---|---|---|---|
| Random forest | 1.066 | 0.034 | 1.031 | 1.133 |
| XGBoost | 1.074 | 0.035 | 1.019 | 1.143 |
| Linear model | 1.079 | 0.027 | 1.043 | 1.129 |

RMSE is in household-size units, so a value near 1.07 means the model is off by about one
person. The spread between the best and worst learner is 0.013, which is smaller than the
typical fold-to-fold standard deviation of 0.032. That is a tie, and reporting a winner
would be reading noise.

The submitted version of this project reported XGBoost winning by a wide margin. That margin
does not reproduce when all three learners are put through the same pipeline and scored on the
same folds.

## Data

| File | Rows | Columns | Contents |
|---|---|---|---|
| `survey_train_test.csv` | 5,027 | 25 | Respondent survey with demographics and self-reported household size |
| `amazon-purchases.csv` | 1,850,717 | 8 | Line-item purchase log, January 2018 to August 2024 |

The target takes values 1, 2, 3 and 4, where 4 means "4 or more". The survey caps the answer,
so a household of four and a household of seven are indistinguishable.

Two thousand of the 5,027 survey rows are flagged `test` and have **no household size**. They
are a prediction set, not a holdout. Every model score in this repository is cross-validated on
the remaining 3,027 labeled rows, because a score computed on unlabeled rows would not be a
score.

Neither source file is redistributed. Put both CSV files in `DATA/` to rebuild.

## Repository layout

```
.
├── index.html               # the dashboard
├── model.md                 # the full analysis, as an executable document
├── R/render.R               # knits model.md and regenerates every figure
├── assets/
│   ├── css/style.css        # dashboard styles
│   └── img/                 # figures written by the knit
├── outputs/                 # tables and predictions the site reads
└── DATA/                    # source CSVs (not committed)
```

`model.md` is the single source of truth for the analysis. The dashboard is a presentation
layer; every number on it traces back to a table in `outputs/`, a figure in `assets/img/`, or
the row counts printed by `model.md`.

## Reproduce

```r
# 1. Put survey_train_test.csv and amazon-purchases.csv in DATA/
# 2. From the repository root:
Rscript R/render.R
```

That knits `model.md` to `model.html`, writes the figures into `assets/img/`, and refreshes the
tables in `outputs/`. A fixed seed keeps the cross-validated table stable between runs.

Requirements: R 4.4 or newer, plus `tidyverse`, `mlr3verse`, `ranger`, `xgboost`, `knitr` and
`rmarkdown`.

## What this rebuild changes

The analysis was submitted for STATS 101C in Fall 2025. Revision work focused on defects that
changed the conclusion, not on cosmetics:

| Issue | Submitted | Here |
|---|---|---|
| Test rows | Scored as a holdout | Recognized as unlabeled; ten-fold CV only |
| `purchase_freq` | Described as "items per day" | Renamed `purchase_lines_per_day`, which is what the code computes |
| Time anchor | `Sys.Date()`, so results drifted daily | Fixed at the last order date, 2024-08-15 |
| Kids' items | Counted pet toys | Pet categories excluded |
| Food items | Counted pet food, storage containers, blenders | Exact category names only |
| Shared consumables | Counted skin cleansers and toothbrush heads | Paper and cleaning products only |
| Categoricals | `as.numeric(factor(x))`, inventing an order | Real factors; rare states pooled into "Other" |
| ICE plots | One line per respondent, unreadable | 80 sampled curves plus a bold partial-dependence average |
| Benchmark chart | Labeled MSE, plotted RMSE | Labeled RMSE, with fold spread |
| Predictions | Rounded for two learners, raw for XGBoost | One rounding and clipping rule for all |

## Limitations

- **No holdout score.** The test rows are unlabeled, so every estimate is cross-validated on the
  same rows the models were fitted to.
- **The target is censored at 4.** Part of the residual error is the ceiling, not the model.
- **Category bins are keyword rules.** They are documented and reproducible, but they are not
  Amazon's taxonomy and they will miss products.
- **XGBoost was tuned on this data.** Its hyperparameters came from an earlier search on the
  same rows, so its score is mildly optimistic relative to the untuned baselines.
- **Three learners.** A LightGBM number from the original report is not reproduced because it
  came from a different pipeline and would not be comparable.
- **Observational.** Buying children's products correlates with household size. It does not
  establish causation in either direction.

## Authors

Angela Huang, Euleena Trinh, Luna Takahashi, Catherine Wang.

## License

Apache 2.0. See [LICENSE](LICENSE).
