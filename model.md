---
title: "Amazon Consumer Household Size Prediction Model"
output:
  pdf_document: default
  html_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Data And Packages

```{r}
survey <- read.csv("survey_train_test.csv")
amazon_purchase <- read.csv("amazon-purchases.csv")

library(mlr3verse)
library(mlr3tuning)
library(ggplot2)
library(tidyverse)
library(ranger)
library(xgboost)
library(iml)
```

## Feature Engineering

```{r}
# Keep the survey fields needed for modeling.
survey_features <- survey |>
  select(Survey.ResponseID, Q.amazon.use.hh.size.num, Q.demos.gender, Q.demos.state, test)

# Summarize each respondent's purchase history.
purchase_features <- amazon_purchase |>
  mutate(Order.Date = as.Date(Order.Date)) |>
  group_by(Survey.ResponseID) |>
  summarize(
    first_purchase = min(Order.Date, na.rm = TRUE),
    since_first_purchase = as.numeric(Sys.Date() - first_purchase),
    purchase_freq = ifelse(n() > 1, n() / pmax(since_first_purchase, 1), 0),
    total_spend = sum(Purchase.Price.Per.Unit * Quantity, na.rm = TRUE),
    avg_purchase_value = mean(Purchase.Price.Per.Unit * Quantity, na.rm = TRUE),
    total_items = sum(Quantity, na.rm = TRUE),
    avg_purchase_quantity = mean(Quantity, na.rm = TRUE),
    children_items = sum(grepl("BABY|CHILD|TOY", Category, ignore.case = TRUE)),
    food_items = sum(grepl("FOOD|GROCERY|MEAT|POULTRY", Category, ignore.case = TRUE)),
    multiuse_items = sum(grepl("CLEANING|DETERGENT|TISSUE|TOILET|TOOTH|TOWEL", Category, ignore.case = TRUE)),
    .groups = "drop"
  ) |>
  select(-first_purchase, -since_first_purchase)

# Join survey and purchase history into one modeling table.
dataset <- survey_features |>
  left_join(purchase_features, by = "Survey.ResponseID")

clean_dataset <- dataset |>
  mutate(across(
    -c(Survey.ResponseID, test),
    ~ if (is.character(.x)) as.numeric(factor(.x)) else as.numeric(.x)
  ))

# Split the cleaned data while keeping the original test flag intact.
training <- clean_dataset |>
  filter(test == FALSE) |>
  select(-test)

testing <- clean_dataset |>
  filter(test == TRUE) |>
  select(-test)

# Quick state-level summaries for exploration.
demographic_patterns <- survey |>
  filter(test == FALSE) |>
  group_by(Q.demos.state) |>
  summarize(
    prop_early_adult = sum(Q.demos.age %in% c("18 - 24 years", "25 - 34 years"), na.rm = TRUE) / n(),
    prop_mid_adult = sum(Q.demos.age %in% c("35 - 44 years", "45 - 54 years"), na.rm = TRUE) / n(),
    prop_high_income = sum(grepl("100,000|150,000|200,000", Q.demos.income), na.rm = TRUE) / n(),
    prop_mid_income = sum(grepl("50,000|75,000", Q.demos.income), na.rm = TRUE) / n(),
    prop_low_income = sum(grepl("25,000", Q.demos.income), na.rm = TRUE) / n(),
    prop_na_income = sum(grepl("prefer not to say", Q.demos.income, ignore.case = TRUE), na.rm = TRUE) / n(),
    prop_college = sum(grepl("Degree", Q.demos.education, ignore.case = TRUE), na.rm = TRUE) / n(),
    prop_na_college = sum(grepl("prefer not to say", Q.demos.education, ignore.case = TRUE), na.rm = TRUE) / n(),
    .groups = "drop"
  )

behavioral_patterns <- survey |>
  filter(test == FALSE) |>
  group_by(Q.demos.state) |>
  summarize(
    substance_rate = sum(Q.substance.use.cigarettes == "Yes" | Q.substance.use.marijuana == "Yes" | Q.substance.use.alcohol == "Yes") / n(),
    health_issue_rate = sum(Q.personal.diabetes == "Yes" | Q.personal.wheelchair == "Yes") / n(),
    .groups = "drop"
  )
```

## Exploratory Data Analysis

```{r}
survey_test <- survey |> filter(test == TRUE)
survey_train <- survey |> filter(test == FALSE)

purchase_per_user <- amazon_purchase |>
  mutate(Total.Purchase = Purchase.Price.Per.Unit * Quantity) |>
  group_by(Survey.ResponseID) |>
  summarize(
    Total.Purchase.per.User = sum(Total.Purchase),
    Avg.Purchase.per.User = mean(Total.Purchase),
    Purchase.Times = n(),
    .groups = "drop"
  )

amazon_eda <- purchase_per_user |>
  inner_join(survey_train, by = "Survey.ResponseID")

# State-level household size.
amazon_eda_state <- amazon_eda |>
  filter(!is.na(Total.Purchase.per.User), !is.na(Q.demos.state)) |>
  group_by(Q.demos.state) |>
  summarize(
    total_purchase_amount = mean(Total.Purchase.per.User),
    avg_purchase_amount = mean(Avg.Purchase.per.User),
    purchase_times = mean(Purchase.Times),
    count = n(),
    avg_household_size = mean(Q.amazon.use.hh.size.num, na.rm = TRUE),
    .groups = "drop"
  )

ggplot(amazon_eda_state, aes(x = as.factor(Q.demos.state), y = avg_household_size)) +
  geom_col(fill = "skyblue3") +
  labs(title = "Average Household Size by State", x = "State", y = "Avg Household Size") +
  theme_minimal()

# Race-level household size.
amazon_eda_race <- amazon_eda |>
  filter(!is.na(Total.Purchase.per.User), !is.na(Q.demos.race)) |>
  separate_rows(Q.demos.race, sep = ",") |>
  group_by(Survey.ResponseID) |>
  mutate(count = n()) |>
  ungroup() |>
  mutate(race = ifelse(count != 1, "Multi", Q.demos.race)) |>
  group_by(race) |>
  summarize(
    total_purchase_amount = mean(Total.Purchase.per.User),
    avg_purchase_amount = mean(Avg.Purchase.per.User),
    purchase_times = mean(Purchase.Times),
    count = n(),
    avg_household_size = mean(Q.amazon.use.hh.size.num, na.rm = TRUE),
    .groups = "drop"
  )

ggplot(amazon_eda_race, aes(x = as.factor(race), y = avg_household_size)) +
  geom_col(fill = "skyblue3") +
  labs(title = "Average Household Size by Race", x = "Race", y = "Avg Household Size") +
  theme_minimal()

# Household size versus spending.
amazon_eda_purchase <- amazon_eda |>
  filter(!is.na(Total.Purchase.per.User), !is.na(Q.amazon.use.hh.size.num)) |>
  group_by(Q.amazon.use.hh.size.num) |>
  summarize(
    total_purchase_amount = mean(Total.Purchase.per.User),
    avg_purchase_amount = mean(Avg.Purchase.per.User),
    purchase_times = mean(Purchase.Times),
    count = n(),
    .groups = "drop"
  )

ggplot(amazon_eda_purchase, aes(x = as.factor(Q.amazon.use.hh.size.num), y = avg_purchase_amount)) +
  geom_col(fill = "skyblue3") +
  labs(title = "Average Purchase Amount by Household Size", x = "Household Size", y = "Avg Purchase Amount ($)") +
  theme_minimal()

ggplot(amazon_eda_purchase, aes(x = as.factor(Q.amazon.use.hh.size.num), y = purchase_times)) +
  geom_col(fill = "skyblue3") +
  labs(title = "Average Number of Items Purchased by Household Size", x = "Household Size", y = "Avg Number of Items Purchased") +
  theme_minimal()

# Use the purchase summary from the feature-engineering step.
purchase_hist_eda <- survey_train |>
  inner_join(purchase_features, by = "Survey.ResponseID")

purchase_hist_eda_grouped <- purchase_hist_eda |>
  group_by(Q.amazon.use.hh.size.num) |>
  summarize(
    avg_purchase_freq = mean(purchase_freq, na.rm = TRUE),
    avg_children_items = mean(children_items, na.rm = TRUE),
    avg_food_items = mean(food_items, na.rm = TRUE),
    avg_multiuse_items = mean(multiuse_items, na.rm = TRUE),
    .groups = "drop"
  )

ggplot(purchase_hist_eda_grouped, aes(x = as.factor(Q.amazon.use.hh.size.num), y = avg_purchase_freq)) +
  geom_col(fill = "skyblue3") +
  labs(title = "Purchase Frequency and Household Size", x = "Household Size", y = "Purchase Frequency") +
  theme_minimal()

ggplot(purchase_hist_eda_grouped, aes(x = as.factor(Q.amazon.use.hh.size.num), y = avg_children_items)) +
  geom_col(fill = "skyblue3") +
  labs(title = "Children Items by Household Size", x = "Household Size", y = "Avg Number of Purchases") +
  theme_minimal()

ggplot(purchase_hist_eda_grouped, aes(x = as.factor(Q.amazon.use.hh.size.num), y = avg_food_items)) +
  geom_col(fill = "skyblue3") +
  labs(title = "Food Items by Household Size", x = "Household Size", y = "Avg Number of Purchases") +
  theme_minimal()

ggplot(purchase_hist_eda_grouped, aes(x = as.factor(Q.amazon.use.hh.size.num), y = avg_multiuse_items)) +
  geom_col(fill = "skyblue3") +
  labs(title = "Multiuse Items by Household Size", x = "Household Size", y = "Avg Number of Purchases") +
  theme_minimal()

amazon_eda_2 <- amazon_eda |>
  filter(!is.na(Total.Purchase.per.User), !is.na(Q.demos.age)) |>
  group_by(Q.demos.age) |>
  summarize(
    total_purchase_amount = mean(Total.Purchase.per.User),
    avg_purchase_amount = mean(Avg.Purchase.per.User),
    purchase_times = mean(Purchase.Times),
    .groups = "drop"
  )

ggplot(amazon_eda_2, aes(x = as.factor(Q.demos.age), y = purchase_times)) +
  geom_col(fill = "skyblue3") +
  labs(title = "Average Number of Purchases by Age Group", x = "Age", y = "Avg Number of Purchases") +
  theme_minimal()
```

## Linear Model

```{r}
linear_model <- lm(
  Q.amazon.use.hh.size.num ~ Q.demos.gender + Q.demos.state + purchase_freq + total_spend + avg_purchase_value +
    total_items + avg_purchase_quantity + children_items + food_items + multiuse_items,
  data = training
)

summary(linear_model)

lm_predictions <- predict(linear_model, newdata = testing) |>
  round() |>
  pmax(1)

hh_size_lm <- data.frame(
  Survey.ResponseID = testing$Survey.ResponseID,
  Q.amazon.use.hh.size.num = lm_predictions
)

write.csv(hh_size_lm, "hh_size_predictions2.csv", row.names = FALSE)
```

## Random Forest Model

```{r}
training_range_rf <- range(training$Q.amazon.use.hh.size.num, na.rm = TRUE)
task_rf <- as_task_regr(training |> select(-Survey.ResponseID), target = "Q.amazon.use.hh.size.num")

learner_rf <- lrn("regr.ranger", importance = "impurity")
learner_rf$train(task_rf)

importance_scores <- learner_rf$model$model$variable.importance
ggplot(data = data.frame(var = names(importance_scores), value = importance_scores)) +
  geom_point(aes(x = value, y = reorder(var, value))) +
  labs(x = "Variable Importance", y = "Variables")

testing_rf <- testing |> select(-Survey.ResponseID, -Q.amazon.use.hh.size.num)
rf_predictions <- learner_rf$predict_newdata(testing_rf)$response |>
  round() |>
  pmax(training_range_rf[1]) |>
  pmin(training_range_rf[2])

hh_size_rf <- data.frame(
  Survey.ResponseID = testing$Survey.ResponseID,
  Q.amazon.use.hh.size.num = rf_predictions
)

write.csv(hh_size_rf, "hh_size_predictions3.csv", row.names = FALSE)
```

## XGBoost Model

```{r}
task_xgb <- as_task_regr(training |> select(-Survey.ResponseID), target = "Q.amazon.use.hh.size.num")

# Final tuned hyperparameters from the search step.
learner_xgb_final <- lrn(
  "regr.xgboost",
  eta = 0.09926108,
  max_depth = 3,
  min_child_weight = 9,
  nrounds = 1571,
  subsample = 0.9756695,
  colsample_bytree = 0.8403539,
  gamma = 3.487319,
  nthread = 1,
  verbose = 0
)

graph_xgb <- po("imputemedian") %>>%
  po("encode", method = "one-hot") %>>%
  learner_xgb_final

learner_xgb_final <- as_learner(graph_xgb)
learner_xgb_final$train(task_xgb)

testing_xgb <- testing |> select(-Survey.ResponseID, -Q.amazon.use.hh.size.num)
xgb_predictions <- learner_xgb_final$predict_newdata(testing_xgb)$response
```

## Model Interpretation

```{r}
importance <- learner_xgb_final$importance()

imp_df <- data.frame(
  feature = names(importance),
  importance = as.numeric(importance)
) |>
  arrange(importance) |>
  mutate(feature = factor(feature, levels = feature))

ggplot(imp_df, aes(x = feature, y = importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "XGBoost Model: Feature Importance", x = "", y = "Importance") +
  theme_minimal()
```

## Model Agnostic Checks

```{r}
X <- training |> select(-Survey.ResponseID, -Q.amazon.use.hh.size.num)
y <- training$Q.amazon.use.hh.size.num

predictor <- Predictor$new(
  model = learner_xgb_final,
  data = X,
  y = y,
  predict.fun = function(model, newdata) {
    model$predict_newdata(newdata)$response
  }
)

ice_children <- FeatureEffect$new(predictor, feature = "children_items", method = "ice")
ice_children$plot() + ggtitle("ICE Plot: Children Items")

ice_total_spend <- FeatureEffect$new(predictor, feature = "total_spend", method = "ice")
ice_total_spend$plot() + ggtitle("ICE Plot: Total Spending")
```

## Benchmarking

```{r}
task <- TaskRegr$new(
  id = "hh_size",
  backend = training |> select(-Survey.ResponseID),
  target = "Q.amazon.use.hh.size.num"
)

lrn_lm <- lrn("regr.lm")

graph_rf <- po("imputemedian") %>>%
  lrn("regr.ranger", importance = "impurity")
lrn_rf <- as_learner(graph_rf)

graph_xgb <- po("imputemedian") %>>%
  po("encode", method = "one-hot") %>>%
  lrn(
    "regr.xgboost",
    eta = 0.09926108,
    max_depth = 3,
    min_child_weight = 9,
    nrounds = 1571,
    subsample = 0.9756695,
    colsample_bytree = 0.8403539,
    gamma = 3.487319,
    nthread = 1,
    verbose = 0
  )
lrn_xgb <- as_learner(graph_xgb)

learners <- list(lrn_lm, lrn_rf, lrn_xgb)

resampling <- rsmp("cv", folds = 10)
resampling$instantiate(task)

design <- benchmark_grid(
  tasks = list(task),
  learners = learners,
  resamplings = list(resampling)
)

bmr <- benchmark(design)

bmr$aggregate(msr("regr.rmse"))
autoplot(bmr) + ggtitle("Benchmarking RMSE")
```

## Final Predictions

```{r}
hh_size_final <- data.frame(
  Survey.ResponseID = testing$Survey.ResponseID,
  Q.amazon.use.hh.size.num = xgb_predictions
)

write.csv(hh_size_final, "hh_size_predictions_final.csv", row.names = FALSE)
```
