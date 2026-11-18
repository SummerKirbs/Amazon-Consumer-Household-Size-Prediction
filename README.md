# Amazon Consumer Household Size Prediction

**What you buy says more than who you are.**

[![Live Dashboard](https://img.shields.io/badge/dashboard-live-e07a1f)](https://summerkirbs.github.io/Amazon-Consumer-Household-Size-Prediction/)
[![R](https://img.shields.io/badge/R-analysis-276DC3?logo=r&logoColor=white)](https://www.r-project.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

An Amazon survey records how many people live in each respondent's household. A purchase log
records 1.85 million line items for those same accounts over six and a half years. This project
asks whether purchase behavior predicts household size better than demographics alone.

Short answer: yes, but weakly. Children's products carry most of the signal, spending per line
carries none, and three different learners finish within one fold's worth of noise of each
other. The interesting result is not which model won. It is that almost every plausible
hypothesis about household consumption turns out to be flat.

## Results

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

### Model scores

The 3,027 labeled rows are split once, stratified on the target: 2,424 to fit and compare, 603
held back. The learners are compared by ten-fold cross-validation inside the 80%, then the best
learner is refit on all 3,027 labeled rows to produce the final predictions.

| Learner | Mean RMSE | SD | Best fold | Worst fold |
|---|---|---|---|---|
| Random forest | 1.063 | 0.017 | 1.046 | 1.095 |
| XGBoost | 1.065 | 0.020 | 1.033 | 1.101 |
| Linear model | 1.079 | 0.020 | 1.058 | 1.115 |

RMSE is in household-size units, so a value near 1.07 means the model is off by about one
person. The spread between the best and worst learner is 0.016, which is smaller than the
typical fold-to-fold standard deviation of 0.019. That is a tie, and reporting a winner would
be reading noise.

The held-out 20% agrees on the leader and refuses to order the rest:

| Learner | Holdout RMSE |
|---|---|
| Random forest | 1.083 |
| Linear model | 1.085 |
| XGBoost | 1.091 |

## Data

The data is Open E-commerce 1.0, published by Berke et al. in *Scientific Data* (2024). It
contains the Amazon purchase histories of 5,027 U.S. survey respondents who agreed to share
them, exported from Amazon's order-history reports and linked to their survey answers by a
randomly generated respondent ID. Collection ran from November 2022 to March 2023 and covers
purchases back to January 2018.

> Berke, A., Calacci, D., Mahari, R., Yabe, T., Larson, K., & Pentland, S. (2024). Open
> e-commerce 1.0, five years of crowdsourced U.S. Amazon purchase histories with user
> demographics. *Scientific Data*, 11, 491. <https://doi.org/10.1038/s41597-024-03329-6>

| File | Rows | Columns | Contents |
|---|---|---|---|
| `survey_train_test.csv` | 5,027 | 25 | Respondent survey with demographics and self-reported household size |
| `amazon-purchases.csv` | 1,850,717 | 8 | Line-item purchase log, January 2018 to August 2024 |

Each purchase row is a line item rather than an order. It has an order date but no order ID, so
an order can only be reconstructed by grouping on the date. The target is the survey question
"how many people do you consider to be in your household", which is not the same as the number
of people sharing the Amazon account.

The target takes values 1, 2, 3 and 4, where 4 means "4 or more". The survey caps the answer,
so a household of four and a household of seven are indistinguishable.

Two thousand of the 5,027 survey rows are flagged `test` and have **no household size**. They
are a prediction set, not a holdout, so no score is computed on them. Evaluation happens
inside the 3,027 labeled rows.

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

## Reproduce

```r
# 1. Put survey_train_test.csv and amazon-purchases.csv in DATA/
# 2. From the repository root:
Rscript R/render.R
```

That knits `model.md` to `model.html`, writes the figures into `assets/img/`, and refreshes the
tables in `outputs/`. A fixed seed keeps the split and the cross-validated table stable
between runs.

Requirements: R 4.4 or newer, plus `tidyverse`, `mlr3verse`, `ranger`, `xgboost`, `knitr` and
`rmarkdown`.

## Limitations

- **One holdout, one number.** The 603 held-out rows give a single estimate with a standard
  error of roughly 0.03 RMSE. It can confirm the cross-validated picture but cannot reorder
  learners that differ by 0.016.
- **The target is censored at 4.** Part of the residual error is the ceiling, not the model.
- **Category bins are keyword rules.** They are documented and reproducible, but they are not
  Amazon's taxonomy and they will miss products.
- **XGBoost was tuned on this data.** Its hyperparameters came from a grid search on the same
  rows, so its score is mildly optimistic relative to the untuned baselines.
- **Gift cards are included.** The log has 41,665 gift-card lines from 3,189 respondents, about
  3% of total spend. They are transfers rather than household consumption, and the source paper
  flags them as a known quirk.
- **The sample is not the U.S. population.** It skews female, younger and more educated than
  the census, and under-represents Black, Other and multiracial respondents. The model
  describes these volunteers, not Amazon shoppers generally.
- **Observational.** Buying children's products correlates with household size. It does not
  establish causation in either direction.

## License

Apache 2.0. See [LICENSE](LICENSE).
