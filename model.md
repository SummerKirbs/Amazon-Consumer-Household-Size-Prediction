---
title: "Amazon Consumer Household Size Prediction"
subtitle: "STATS 101C final project"
output:
  html_document:
    toc: true
    toc_float: true
    code_folding: show
    df_print: paged
  pdf_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(
  echo       = TRUE,
  comment    = "#>",
  message    = FALSE,
  warning    = FALSE,
  fig.width  = 9.6,
  fig.height = 4.8,
  dpi        = 200,
  out.width  = "100%",
  fig.path   = "assets/img/"
)

# Cross-validation folds and the forest are both random. One seed at the top means the
# numbers in this document do not move between runs.
set.seed(101)
```

## What this asks

The survey asks each respondent how many people live in their household. The purchase
log records what that same account bought, when, and for how much.

The question: **can purchase behavior predict household size better than demographics
alone?**

The answer is short. Children's products carry most of the signal. Purchase volume carries a
little. Demographics carry almost none. And the three models we compare land within 0.016 RMSE
of each other, which is less than the fold-to-fold noise. That is a tie, not a winner.

## Data

The data is Open E-commerce 1.0 (Berke et al., 2024): Amazon purchase histories from 5,027
U.S. survey respondents who chose to share them, linked to their survey answers by a randomly
generated respondent ID. The dataset is published on Harvard Dataverse and described in
*Scientific Data*: <https://doi.org/10.1038/s41597-024-03329-6>.

Two details from the source paper matter here. First, each purchase row is a line item, not an
order: it carries an order date, but there is no order ID, so "order" can only be reconstructed
by grouping on the date. Second, the target is the survey question "how many people do you
consider to be in your household", which is not the same as the number of people sharing the
Amazon account.

The extract runs from January 2018 to August 2024. Collection itself took place between
November 2022 and March 2023, when Amazon retired the order-history export page the study
depended on.

```{r data-load}
# The two source files are not committed. Put them in DATA/ before knitting.
data_dir <- "DATA"

survey  <- read.csv(file.path(data_dir, "survey_train_test.csv"))
purchases <- read.csv(file.path(data_dir, "amazon-purchases.csv"))

purchases$Order.Date <- as.Date(purchases$Order.Date)

dim(survey)
dim(purchases)
```

```{r data-shape}
# One row per respondent, one row per purchase line.
nrow(survey)
nrow(purchases)

# The flag marks the unlabeled prediction set, not a scored holdout.
table(survey$test)

# The target only takes four values.
table(survey$Q.amazon.use.hh.size.num, useNA = "ifany")
```

Two facts drive every choice that follows.

1. **The 2,000 `test` rows have no household size.** They are a prediction set, not a
   holdout, so scoring them would not measure anything. Evaluation happens inside the 3,027
   labeled rows: a stratified 80/20 split, then ten-fold cross-validation on the 80%.
2. **The target is capped at 4.** The survey groups "4 or more" into one bucket, so this is
   an ordered four-class problem squeezed into a regression. A model cannot distinguish a
   household of 4 from a household of 7.

```{r package-setup}
library(tidyverse)
library(mlr3verse)
library(ranger)
library(xgboost)
```

## Feature engineering

Purchase lines are rolled up to the respondent level. Every feature below is a summary of
that person's whole history, not a single order.

```{r feature-build}
# Anchor every time-based feature to one fixed date.
# Using Sys.Date() made the pipeline drift a few days every time it ran, which moved
# purchase rates for no substantive reason. This is the last order date in the extract.
SNAPSHOT_DATE <- as.Date("2024-08-15")
stopifnot(SNAPSHOT_DATE == max(purchases$Order.Date, na.rm = TRUE))

purchase_features <- purchases |>
  mutate(line_value = Purchase.Price.Per.Unit * Quantity) |>
  group_by(Survey.ResponseID) |>
  summarize(
    first_purchase = min(Order.Date, na.rm = TRUE),
    days_active     = as.numeric(SNAPSHOT_DATE - first_purchase),

    # Lines per day over the account's observed life. This counts order lines, not items,
    # so the name says "lines" to match what the code actually computes.
    # The earlier name was purchase_freq, which the report read as items per day.
    order_lines = n(),
    purchase_lines_per_day = order_lines / pmax(days_active, 1),

    total_spend      = sum(line_value, na.rm = TRUE),
    avg_line_value   = mean(line_value, na.rm = TRUE),
    total_items      = sum(Quantity, na.rm = TRUE),
    avg_line_quantity = mean(Quantity, na.rm = TRUE),

    # Category bins. These are hand-written keyword matches, so they are approximations,
    # not Amazon's own taxonomy. Each one is scoped so that obvious false friends stay out.
    # Children: a bare "TOY" match would also catch pet toys, so pets are excluded.
    children_items = sum(
      grepl("BABY|CHILD|TOY", Category, ignore.case = TRUE) &
        !grepl("PET", Category, ignore.case = TRUE)
    ),
    # Food: exact category names only. A substring match also catches food storage
    # containers, blenders and spatulas, and PET_FOOD, which says nothing about household
    # size.
    food_items = sum(grepl(
      "^(FOOD|GROCERY|SNACK_FOOD_BAR|MEAT|POULTRY|VEGETABLE|FRUIT|BEVERAGE|COFFEE|TEA|DRINK|CANDY|BAKED|DAIRY|SEAFOOD|GRAIN|SPICE|CONDIMENT|NUTRITIONAL_SUPPLEMENT)$",
      Category, ignore.case = TRUE
    )),
    # Shared consumables: paper goods and cleaning products a household burns through.
    # Matching on "CLEANSER" or "TOOTH" would also pull in skin cleansers, toothbrush
    # heads and toilet seats, so the list names the specific categories instead.
    multiuse_items = sum(grepl(
      "TOILET_PAPER|PAPER_TOWEL|LAUNDRY_DETERGENT|DISHWASHER_DETERGENT|CLEANING_AGENT|SURFACE_CLEANING_WIPE|FACIAL_TISSUE|TOOTH_CLEANING_AGENT|TOOTHBRUSH|TOOTH_WHITENER|DETERGENT",
      Category, ignore.case = TRUE
    )),

    .groups = "drop"
  ) |>
  select(-first_purchase, -days_active)

survey_features <- survey |>
  select(Survey.ResponseID, Q.amazon.use.hh.size.num, Q.demos.gender, Q.demos.state, test)

dataset <- survey_features |>
  left_join(purchase_features, by = "Survey.ResponseID")

# Every respondent in the survey has purchase history, so the join is clean.
sum(is.na(dataset$total_spend))
```

### Encoding

Categorical fields stay categorical. Encoding them as integers invents an ordering: it tells
the model that two states alphabetically far apart are numerically far apart, and that one
gender is twice another. That is not a defensible encoding for a linear model.

One real problem shows up once state is a factor: a state with a single respondent lands in
exactly one cross-validation fold, so the other nine folds contain an unseen level and `lm()`
errors out. Numeric codes hide that failure instead of fixing it. The fix is to pool rare
states.

```{r encoding}
# Keep states with at least 20 labeled respondents. Everything else becomes "Other".
state_keep <- dataset |>
  filter(test == FALSE) |>
  count(Q.demos.state) |>
  filter(n >= 20) |>
  pull(Q.demos.state)

state_levels  <- c(state_keep, "Other")
gender_levels <- sort(unique(dataset$Q.demos.gender))

# Apply the same levels to train and prediction rows so the design matrix lines up.
encode_factors <- function(d) {
  d |>
    mutate(
      Q.demos.state  = factor(if_else(Q.demos.state %in% state_keep, Q.demos.state, "Other"),
                              levels = state_levels),
      Q.demos.gender = factor(Q.demos.gender, levels = gender_levels)
    )
}

training <- dataset |> filter(test == FALSE) |> select(-test) |> encode_factors()
testing  <- dataset |> filter(test == TRUE)  |> select(-test) |> encode_factors()

# How much did pooling shrink the state field?
length(unique(dataset$Q.demos.state))
length(state_levels)
```

## Exploratory data analysis

### What actually separates small households from large ones

```{r eda-household-counts}
# Class balance first. The four groups are close enough that no single class dominates.
household_counts <- training |>
  count(household = Q.amazon.use.hh.size.num)

household_counts

ggplot(household_counts, aes(x = factor(household), y = n)) +
  geom_col(fill = "#0b2545", width = 0.68) +
  geom_text(aes(label = scales::comma(n)), vjust = -0.5, size = 3.4, colour = "#0b2545") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    title = "Labeled respondents by household size",
    subtitle = "3,027 labeled rows; the groups are balanced enough for cross-validation",
    x = "Household size", y = "Respondents"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank())
```

```{r eda-children-items}
# The headline relationship. Children's products climb steeply with household size.
by_household <- training |>
  group_by(household = Q.amazon.use.hh.size.num) |>
  summarize(
    n                = n(),
    children_items   = mean(children_items),
    food_items       = mean(food_items),
    multiuse_items   = mean(multiuse_items),
    total_items      = mean(total_items),
    total_spend      = mean(total_spend),
    avg_line_value   = mean(avg_line_value),
    lines_per_day    = mean(purchase_lines_per_day),
    .groups = "drop"
  )

by_household

ggplot(by_household, aes(x = factor(household), y = children_items)) +
  geom_col(fill = "#e07a1f", width = 0.68) +
  geom_text(aes(label = round(children_items, 1)), vjust = -0.5, size = 3.6, colour = "#7a3d06") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    title = "Children's products rise steeply with household size",
    subtitle = "Mean number of baby, child and toy lines per respondent",
    x = "Household size", y = "Children's items (mean)"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank())
```

```{r eda-purchase-volume}
# Volume moves too, but far more gently. This is the "more people, more stuff" effect.
ggplot(by_household, aes(x = factor(household), y = total_items)) +
  geom_col(fill = "#2b6f8f", width = 0.68) +
  geom_text(aes(label = round(total_items)), vjust = -0.5, size = 3.6, colour = "#123646") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    title = "Total items purchased rises gently",
    subtitle = "Mean item count per respondent, 2018-2024",
    x = "Household size", y = "Items purchased (mean)"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank())
```

```{r eda-category-groups}
# Three keyword bins on one shared axis, each indexed to its own mean at household size 1.
# The earlier version gave every facet its own y-scale, which made the two flat bins look
# as steep as the children's bin. Indexing removes that illusion.
category_index <- by_household |>
  select(household, children_items, food_items, multiuse_items) |>
  pivot_longer(-household, names_to = "group", values_to = "mean_items") |>
  group_by(group) |>
  mutate(index = 100 * mean_items / mean_items[household == 1]) |>
  ungroup() |>
  mutate(group = recode(group,
    children_items = "Children's products",
    food_items     = "Food and grocery",
    multiuse_items = "Shared consumables"
  ))

category_index

ggplot(category_index, aes(x = household, y = index, colour = group)) +
  geom_hline(yintercept = 100, linetype = "dashed", colour = "#9aa7b4") +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2.6) +
  # Label only the last point: labelling all four collides where the lines are close.
  geom_text(data = filter(category_index, household == max(household)),
            aes(label = sprintf("%.0f%%", index)), vjust = -0.9, size = 3.6, show.legend = FALSE) +
  scale_colour_manual(values = c(
    "Children's products" = "#e07a1f",
    "Food and grocery"    = "#3f8f7a",
    "Shared consumables"  = "#8d9aa8"
  )) +
  scale_x_continuous(breaks = 1:4) +
  scale_y_continuous(limits = c(80, 490), labels = function(x) paste0(x, "%")) +
  labs(
    title = "Against a household of one, only children's products keep climbing",
    subtitle = "Each category indexed to its own mean at household size 1; dashed line is no change",
    x = "Household size", y = "Change vs. household of one", colour = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    legend.position = "top",
    legend.justification = "left"
  )
```

```{r eda-spend-flat}
# Spending does not separate the groups. This is the flat line the report described,
# but it is average *line* value, not average basket. Grouping lines by order date, the
# average basket is about $53, roughly twice this, and just as flat.
ggplot(by_household, aes(x = factor(household), y = avg_line_value)) +
  geom_col(fill = "#c8b6a6", width = 0.68) +
  geom_text(aes(label = scales::dollar(round(avg_line_value, 2))), vjust = -0.5, size = 3.6, colour = "#5c4a3d") +
  coord_cartesian(ylim = c(0, 30)) +
  labs(
    title = "Average spend per purchase line is flat",
    subtitle = "Around $26 per line regardless of household size",
    x = "Household size", y = "Line value (USD, mean)"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank())
```

```{r eda-rate}
# Frequency helps a little. Larger households order more often, not more per line.
ggplot(by_household, aes(x = factor(household), y = lines_per_day)) +
  geom_col(fill = "#4a6fa5", width = 0.68) +
  geom_text(aes(label = sprintf("%.3f", lines_per_day)), vjust = -0.5, size = 3.6, colour = "#22375c") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    title = "Larger households order slightly more often",
    subtitle = "Purchase lines per day since the first order",
    x = "Household size", y = "Lines per day (mean)"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank())
```

### Demographics

```{r eda-race}
# Race groups, with multi-race answers collapsed into one category.
# This reads the survey table directly because the race column is not a model feature,
# and because the 2,000 blank race values are exactly the unlabeled prediction rows.
race_labeled <- survey |>
  filter(test == FALSE) |>
  mutate(race = if_else(str_detect(Q.demos.race, ","), "Multiracial", Q.demos.race))

race_summary <- race_labeled |>
  group_by(race) |>
  summarize(
    n              = n(),
    mean_household = mean(Q.amazon.use.hh.size.num),
    se             = sd(Q.amazon.use.hh.size.num) / sqrt(n()),
    .groups = "drop"
  ) |>
  # Same 20-respondent floor used for states. Below it the interval is wider than the
  # whole range of the chart, so those groups get dropped rather than drawn as a spike.
  filter(n >= 20) |>
  arrange(mean_household)

race_summary

# Dot plot with 95% intervals. Bars would imply the differences are bigger than they are.
ggplot(race_summary, aes(x = mean_household, y = reorder(race, mean_household))) +
  geom_vline(xintercept = mean(race_labeled$Q.amazon.use.hh.size.num),
             linetype = "dashed", colour = "#9aa7b4") +
  geom_errorbar(aes(xmin = mean_household - 1.96 * se, xmax = mean_household + 1.96 * se),
                orientation = "y", width = 0, colour = "#9aa7b4") +
  geom_point(size = 3, colour = "#0b2545") +
  geom_text(aes(x = mean_household + 1.96 * se, label = paste0("n=", scales::comma(n))),
            hjust = -0.2, size = 3, colour = "#5c6773") +
  scale_x_continuous(limits = c(2.1, 3.05), breaks = seq(2.1, 2.9, by = 0.2)) +
  labs(
    title = "Race groups sit within a narrow band",
    subtitle = "Mean household size with 95% intervals; groups with 20+ labeled respondents",
    x = "Mean household size", y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_blank())
```

```{r eda-state, fig.height = 8}
# A bar chart of all 52 levels puts every label on top of its neighbour and lets the two
# smallest groups (one non-US respondent, nine Hawaii respondents) set the axis for everyone.
# A dot plot of the well-sampled states tells the same story and stays readable.
# A taller panel and larger base text keep 30-plus state names legible at page width.
state_summary <- dataset |>
  filter(test == FALSE) |>
  group_by(Q.demos.state) |>
  summarize(
    n              = n(),
    mean_household = mean(Q.amazon.use.hh.size.num),
    .groups = "drop"
  ) |>
  filter(n >= 20) |>
  arrange(mean_household)

state_summary

ggplot(state_summary, aes(x = mean_household, y = reorder(Q.demos.state, mean_household))) +
  geom_vline(xintercept = mean(training$Q.amazon.use.hh.size.num),
             linetype = "dashed", colour = "#9aa7b4") +
  geom_segment(aes(x = 2.0, xend = mean_household,
                   y = reorder(Q.demos.state, mean_household),
                   yend = reorder(Q.demos.state, mean_household)),
               colour = "#e7dccd", linewidth = 0.6) +
  geom_point(size = 2.4, colour = "#0b2545") +
  scale_x_continuous(limits = c(1.98, 2.7), breaks = seq(2.0, 2.7, by = 0.1)) +
  labs(
    title = "State averages span less than one person",
    subtitle = "States with at least 20 labeled respondents; dashed line is the overall mean",
    x = "Mean household size", y = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.y = element_blank())
```

```{r eda-correlation}
# Simple correlations with the target, ordered. This is the whole story in one column.
numeric_features <- c(
  "children_items", "purchase_lines_per_day", "total_spend", "total_items",
  "multiuse_items", "avg_line_value", "avg_line_quantity", "food_items"
)

cor_table <- tibble(
  feature = numeric_features,
  correlation = map_dbl(numeric_features,
                        ~ cor(training[[.x]], training$Q.amazon.use.hh.size.num,
                              use = "complete.obs"))
) |>
  arrange(correlation) |>
  mutate(label_x = correlation + if_else(correlation > 0, 0.006, -0.006))

cor_table

ggplot(cor_table, aes(x = correlation, y = reorder(feature, correlation),
                      fill = correlation > 0.05)) +
  geom_vline(xintercept = 0, colour = "#9aa7b4") +
  geom_col(width = 0.62, show.legend = FALSE) +
  scale_fill_manual(values = c(`TRUE` = "#e07a1f", `FALSE` = "#c3ccd6")) +
  geom_text(aes(x = label_x, label = sprintf("%.3f", correlation)),
            hjust = 0.5, size = 3.2) +
  scale_x_continuous(limits = c(-0.03, 0.27)) +
  labs(
    title = "Children's products dominate every other purchase signal",
    subtitle = "Pearson correlation with household size, labeled rows only",
    x = "Correlation", y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.y = element_blank())
```

## Models

Three learners, one pipeline, one resampling scheme. Keeping them on the same footing is the
only way the comparison means anything.

### Train and holdout split

The labeled rows are split once, stratified on the target, before any model is fitted.

```{r split}
# 80/20 split of the 3,027 labeled rows. Stratifying on the target keeps the class mix
# identical on both sides, which matters because size 3 is the smallest class.
# The holdout is scored once, at the end, and never used to choose a model.
set.seed(101)

labeled <- training |> mutate(.row = row_number())

holdout_rows <- labeled |>
  group_by(Q.amazon.use.hh.size.num) |>
  slice_sample(prop = 0.2) |>
  pull(.row)

analysis_train <- labeled[-holdout_rows, ] |> select(-.row)
analysis_hold  <- labeled[holdout_rows, ]  |> select(-.row)

nrow(analysis_train)
nrow(analysis_hold)

# Close enough to the full-data proportions that the split is not reshaping the problem.
table(analysis_train$Q.amazon.use.hh.size.num)
table(analysis_hold$Q.amazon.use.hh.size.num)
```

```{r task-build}
# Cross-validation runs on the 80% only.
task <- as_task_regr(
  analysis_train |> select(-Survey.ResponseID),
  target = "Q.amazon.use.hh.size.num",
  id = "household_size"
)

# XGBoost hyperparameters from a grid search over this training data.
# They are reused as-is so every learner below is scored on identical folds, but tuning on
# the same rows makes the XGBoost score mildly optimistic. The other two use defaults.
xgb_params <- list(
  eta              = 0.09926108,
  max_depth        = 3,
  min_child_weight = 9,
  nrounds          = 1571,
  subsample        = 0.9756695,
  colsample_bytree = 0.8403539,
  gamma            = 3.487319,
  nthread          = 1,
  verbose          = 0
)

lrn_lm  <- lrn("regr.lm", id = "linear model")
lrn_rf  <- as_learner(po("imputemedian") %>>% lrn("regr.ranger", importance = "impurity"))
lrn_xgb <- as_learner(
  po("imputemedian") %>>% po("encode", method = "one-hot") %>>%
    lrn(
      "regr.xgboost",
      eta              = xgb_params$eta,
      max_depth        = xgb_params$max_depth,
      min_child_weight = xgb_params$min_child_weight,
      nrounds          = xgb_params$nrounds,
      subsample        = xgb_params$subsample,
      colsample_bytree = xgb_params$colsample_bytree,
      gamma            = xgb_params$gamma,
      nthread          = xgb_params$nthread,
      verbose          = xgb_params$verbose
    )
)

lrn_rf$id  <- "random forest"
lrn_xgb$id <- "xgboost"
```

```{r model-lm}
# Same specification the benchmark learner uses: every feature, nothing hand-picked.
# Fitted on the 80% so its coefficients describe the model that gets cross-validated.
linear_model <- lm(
  Q.amazon.use.hh.size.num ~ .,
  data = analysis_train |> select(-Survey.ResponseID)
)

summary(linear_model)
```

The linear model is reported for interpretation, not for prediction. Its coefficients are
associations with a four-level ordinal target, and the factor dummies for thirty-odd states are
mostly noise. Nothing here is causal.

```{r model-fit}
lrn_rf$train(task)
lrn_xgb$train(task)
```

## Interpretation

```{r model-importance}
# XGBoost gain importance. Gain rewards splits that actually reduce error, so it is a
# better ranking than raw split counts.
importance <- lrn_xgb$importance()
importance

importance_df <- tibble(
  feature    = names(importance),
  importance = as.numeric(importance)
) |>
  slice_max(importance, n = 12, with_ties = FALSE) |>
  arrange(importance)

importance_df

ggplot(importance_df, aes(x = importance, y = reorder(feature, importance))) +
  geom_col(fill = "#0b2545", width = 0.65) +
  labs(
    title = "Children's products are the top XGBoost feature",
    subtitle = "Gain importance, top 12 features",
    x = "Gain", y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.y = element_blank())
```

### Partial dependence and ICE

An ICE plot draws one curve per respondent. With 3,027 rows that is 3,027 overlapping lines,
which collapses into a solid block of colour. Two changes make it readable:

- draw a sample of 80 curves, faint, so the spread is visible but the page is not;
- draw the partial-dependence average on top in bold.

```{r ice-helper}
# Build ICE curves and a PDP by hand.
# iml::FeatureEffect plots every row, so this computes the same quantity with an
# explicit sample size and returns a tidy frame we control.
ice_pdp <- function(learner, data, feature, grid, n_ice = 80, seed = 1) {
  set.seed(seed)
  sample_rows <- data[sample(nrow(data), min(n_ice, nrow(data))), , drop = FALSE]

  curves <- map(grid, function(value) {
    d <- sample_rows
    d[[feature]] <- value
    learner$predict_newdata(d)$response
  })

  ice_matrix <- do.call(cbind, curves)

  list(
    ice = tibble(
      id    = rep(seq_len(nrow(ice_matrix)), times = length(grid)),
      value = rep(grid, each = nrow(ice_matrix)),
      yhat  = as.vector(ice_matrix)
    ),
    pdp = tibble(value = grid, yhat = colMeans(ice_matrix))
  )
}

# Quantile grid so the curves spend their resolution where the data actually is.
quantile_grid <- function(x, n = 20, upper = 0.95) {
  unique(quantile(x, probs = seq(0, upper, length.out = n), na.rm = TRUE))
}
```

```{r model-pdp-children}
children_effect <- ice_pdp(
  lrn_xgb,
  training |> select(-Survey.ResponseID, -Q.amazon.use.hh.size.num),
  "children_items",
  grid = quantile_grid(training$children_items)
)

ggplot() +
  geom_line(data = children_effect$ice,
            aes(x = value, y = yhat, group = id),
            colour = "#a0410a", alpha = 0.42, linewidth = 0.28) +
  geom_line(data = children_effect$pdp,
            aes(x = value, y = yhat),
            colour = "#0b2545", linewidth = 1.3) +
  labs(
    title = "More children's products means a larger predicted household",
    subtitle = "80 sampled ICE curves in orange; bold line is the partial-dependence average",
    x = "Children's items purchased", y = "Predicted household size"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank())
```

```{r model-pdp-spend}
spend_effect <- ice_pdp(
  lrn_xgb,
  training |> select(-Survey.ResponseID, -Q.amazon.use.hh.size.num),
  "total_spend",
  grid = quantile_grid(training$total_spend)
)

# Spend is heavily skewed, so the x-axis is log-scaled. Without that, every curve piles up
# at the left edge and the shape is invisible.
ggplot() +
  geom_line(data = spend_effect$ice,
            aes(x = value, y = yhat, group = id),
            colour = "#24607c", alpha = 0.30, linewidth = 0.28) +
  geom_line(data = spend_effect$pdp,
            aes(x = value, y = yhat),
            colour = "#0b2545", linewidth = 1.3) +
  scale_x_log10(labels = scales::dollar) +
  labs(
    title = "Total spend barely moves the prediction",
    subtitle = "80 sampled ICE curves in blue; bold line is the partial-dependence average",
    x = "Total spend over the observed period (log scale)", y = "Predicted household size"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank())
```

The children's curve climbs and then levels off around 20 items. The spend curve is flat
through the range where almost every respondent sits, then sags at the far right, where a
handful of very large spenders pull the average down. Spending more does not imply a bigger
household; below the tail it mostly reflects how long the account has been active.

## Benchmarking

Ten-fold cross-validation inside the 80% training rows. The holdout is not involved.

```{r benchmark}
resampling <- rsmp("cv", folds = 10)

design <- benchmark_grid(
  tasks       = list(task),
  learners    = list(lrn_lm, lrn_rf, lrn_xgb),
  resamplings = list(resampling)
)

bmr <- benchmark(design)

# Mean RMSE per learner.
bmr$aggregate(msr("regr.rmse"))
```

```{r benchmark-summary}
# Fold-level scores, so the spread is visible instead of hidden behind a mean.
fold_scores <- bmr$score(msr("regr.rmse"))

benchmark_summary <- fold_scores |>
  group_by(learner_id) |>
  summarize(
    rmse_mean = mean(regr.rmse),
    rmse_sd   = sd(regr.rmse),
    rmse_min  = min(regr.rmse),
    rmse_max  = max(regr.rmse),
    .groups = "drop"
  ) |>
  arrange(rmse_mean) |>
  # Fix the draw order once so both charts below agree on it.
  mutate(learner = factor(learner_id, levels = learner_id))

fold_scores <- fold_scores |>
  mutate(learner = factor(learner_id, levels = benchmark_summary$learner_id))

benchmark_summary
```

```{r benchmark-rmse}
# Dot plot with fold spread. Overlapping ranges mean the ranking is not real.
# The ten individual folds sit on top of the min-max bar so the spread is a distribution,
# not just two endpoints.
ggplot(benchmark_summary, aes(x = rmse_mean, y = learner)) +
  geom_segment(aes(x = rmse_min, xend = rmse_max, yend = learner),
               colour = "#e7dccd", linewidth = 4.5, lineend = "round") +
  geom_point(data = fold_scores, aes(x = regr.rmse),
             colour = "#0b2545", alpha = 0.45, size = 2.2,
             position = position_jitter(height = 0.13, width = 0, seed = 1)) +
  geom_point(size = 4.2, colour = "#e07a1f") +
  geom_text(aes(label = sprintf("%.3f", rmse_mean)), vjust = -2, size = 3.6, colour = "#0b2545") +
  scale_x_continuous(breaks = seq(1.02, 1.14, by = 0.02)) +
  labs(
    title = "Ten-fold cross-validated RMSE",
    subtitle = "Orange dots are means, navy dots are the ten folds, bars span the full range",
    x = "RMSE (household size units)", y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.y = element_blank())
```

```{r benchmark-folds}
# Same numbers, fold by fold. Colours are assigned best to worst so the three lines stay
# distinguishable without relying on ggplot's default palette.
fold_colours <- setNames(
  c("#e07a1f", "#2b6f8f", "#0b2545")[seq_len(nrow(benchmark_summary))],
  benchmark_summary$learner_id
)

ggplot(fold_scores, aes(x = factor(iteration), y = regr.rmse, colour = learner_id)) +
  geom_line(aes(group = learner_id), linewidth = 0.7, alpha = 0.7) +
  geom_point(size = 1.9) +
  scale_colour_manual(values = fold_colours) +
  labs(
    title = "Every fold puts the three models in the same band",
    subtitle = "Fold-to-fold noise is larger than the gap between learners",
    x = "Cross-validation fold", y = "RMSE", colour = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "top"
  )
```

```{r benchmark-read}
# How big is the gap relative to the noise?
best  <- benchmark_summary |> slice_min(rmse_mean, n = 1)
worst <- benchmark_summary |> slice_max(rmse_mean, n = 1)

tibble(
  spread_rmse   = worst$rmse_mean - best$rmse_mean,
  typical_sd    = mean(benchmark_summary$rmse_sd),
  spread_in_sd  = (worst$rmse_mean - best$rmse_mean) / mean(benchmark_summary$rmse_sd)
)
```

The gap between the best and worst learner is 0.016, about four-fifths of the typical
fold-to-fold standard deviation of 0.019. That is a tie. Reporting a winner here would be
reading noise.

### Holdout check

```{r holdout}
# One look at the 20% that was held back. Model selection already happened on the folds
# above, so nothing here can influence which learner gets used.
rmse <- function(pred, obs) sqrt(mean((pred - obs)^2))

holdout_x <- analysis_hold |> select(-Survey.ResponseID, -Q.amazon.use.hh.size.num)
holdout_y <- analysis_hold$Q.amazon.use.hh.size.num

holdout_scores <- tibble(
  learner = c("linear model", "random forest", "xgboost"),
  rmse = c(
    rmse(predict(linear_model, newdata = analysis_hold), holdout_y),
    rmse(lrn_rf$predict_newdata(holdout_x)$response, holdout_y),
    rmse(lrn_xgb$predict_newdata(holdout_x)$response, holdout_y)
  )
) |>
  arrange(rmse)

holdout_scores
```

A 603-row holdout gives one number per learner, and the standard error of an RMSE estimate
that size is around 0.03. That is wide enough that a holdout cannot reorder three learners
which differ by 0.016. It can only confirm that the cross-validated picture is not an
artifact of the folds, which is what it is for.

The forest leads here too, and the other two swap places relative to the folds. That swap is
the point: at this sample size the ordering below the leader is not stable.

## Predictions

```{r predictions}
# One rounding rule for all three models.
# The target only ever takes 1, 2, 3 or 4, so a prediction of 0 or 5 is not a legal answer,
# and leaving decimals in the file would not match the submission format either.
finalize <- function(pred, lo = 1, hi = 4) {
  pmin(pmax(round(pred), lo), hi)
}

testing_x <- testing |> select(-Survey.ResponseID, -Q.amazon.use.hh.size.num)

# The best cross-validated learner wins the final fit.
best_id <- best$learner_id

# Refit that learner on every labeled row before predicting the 2,000 unlabeled ones.
# The holdout score describes the 80% model, but discarding 603 labeled rows at prediction
# time would only make the answers worse.
task_full <- as_task_regr(
  training |> select(-Survey.ResponseID),
  target = "Q.amazon.use.hh.size.num",
  id = "household_size_full"
)

final_model <- list(
  "linear model"  = lrn_lm,
  "random forest" = lrn_rf,
  "xgboost"       = lrn_xgb
)[[best_id]]$clone(deep = TRUE)

final_model$train(task_full)

pred_final <- finalize(final_model$predict_newdata(testing_x)$response)

final_predictions <- tibble(
  Survey.ResponseID        = testing$Survey.ResponseID,
  Q.amazon.use.hh.size.num = pred_final
)

table(final_predictions$Q.amazon.use.hh.size.num)
```

Almost every prediction lands on 2 or 3, with a single 1 and a dozen 4s. That is what
squared-error loss does: it pulls predictions toward the mean, so rare extreme households are
never the best guess. For an RMSE target that shrinkage is correct. If the goal were exact
class matches, an ordinal classifier would be the better tool.

```{r write-outputs}
# Predictions and the summary tables the site reads from.
dir.create("outputs", showWarnings = FALSE)

write.csv(final_predictions, "outputs/hh_size_predictions_final.csv", row.names = FALSE)
write.csv(benchmark_summary,  "outputs/model_benchmark.csv",         row.names = FALSE)
write.csv(holdout_scores,     "outputs/holdout_scores.csv",           row.names = FALSE)
write.csv(by_household,       "outputs/eda_by_household.csv",        row.names = FALSE)
write.csv(cor_table,          "outputs/feature_correlations.csv",    row.names = FALSE)
write.csv(race_summary,       "outputs/eda_race.csv",                row.names = FALSE)
write.csv(state_summary,      "outputs/eda_state.csv",               row.names = FALSE)

# Where the results tables landed.
list.files("outputs")
```

```{r session}
sessionInfo()
```

## Limitations

- **One holdout, one number.** The 603 held-out rows give a single estimate with a standard
  error of roughly 0.03 RMSE. It confirms the cross-validated ranking; it cannot refine it.
- **Target is censored at 4.** Households of 4 and 7 look identical to the model.
- **Category bins are hand-written keyword rules.** They are reproducible and documented,
  but they are not Amazon's taxonomy and they will miss products. The three bins are
  approximate, and the food and consumables bins carry almost no signal.
- **XGBoost was tuned on this data.** Its hyperparameters come from a grid search over the
  same rows, so its score is mildly optimistic relative to the LM and RF defaults.
- **Observational data.** Purchase behavior correlates with household size. It does not
  establish that buying children's products causes a bigger household, or the reverse.
- **Gift cards are included.** The log contains 41,665 gift-card lines from 3,189 respondents,
  about 3% of total spend. They are transfers rather than household consumption, and the source
  paper flags them as a known quirk. Excluding them is a reasonable robustness check that this
  analysis does not run.
- **The sample is not the U.S. population.** Berke et al. report that it skews female, younger
  and more educated than the census, and under-represents Black, Other and multiracial
  respondents. The model describes these 5,027 volunteers, not Amazon shoppers generally.
