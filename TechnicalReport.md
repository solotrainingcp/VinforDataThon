# Datathon 2026 Round 1 — Revenue and COGS Forecasting

## 1. Project Overview

This repository contains my solution for the revenue forecasting task in **Datathon 2026 Round 1**. The objective is to predict daily `Revenue` and `COGS` for the period from `2023-01-01` to `2024-07-01`.

The final solution is a leakage-aware time-series ensemble pipeline. It combines:

1. A **Recent-Period Seasonal Ensemble**
2. A **Direct XGBoost Time-Series Model**
3. A **Final Calibration and Blending Layer**

The main idea is that the latest years in the training data are more representative of the hidden test period than older historical years. Therefore, the model gives stronger importance to recent sales patterns while still using older data to learn long-term seasonality.

---

## 2. Repository Structure

The main notebook contains the full end-to-end workflow:

* data loading
* data cleaning
* exploratory data analysis
* feature engineering
* time-based validation
* seasonal ensemble modeling
* XGBoost modeling
* final blending and calibration
* submission file generation
* feature importance / explainability analysis

All source code needed to reproduce the final submission is included in the notebook.

---

## 3. Data Sources

The main target file is `sales.csv`, which contains daily historical values of `Date`, `Revenue`, and `COGS`.

The future prediction dates are taken from `sample_submission.csv`.

The dataset also includes auxiliary business tables such as `orders.csv`, `order_items.csv`, `products.csv`, `payments.csv`, `shipments.csv`, `returns.csv`, `web_traffic.csv`, and `inventory.csv`.

These auxiliary tables were inspected during EDA and feature analysis. However, because future values of orders, payments, web traffic, and inventory are not available for the test period, the final best model mainly relies on date-based and time-series forecasting features. This prevents leakage from future business signals that would not be known at prediction time.

---

## 4. Pipeline Summary

The final pipeline follows this structure:

1. Load and validate all input files.
2. Clean the daily sales target.
3. Build calendar, seasonality, lag, rolling, and recent-regime features.
4. Train a seasonal ensemble using multiple recent-period assumptions.
5. Train direct XGBoost models for `Revenue` and `COGS`.
6. Validate all models using chronological validation.
7. Blend the best seasonal model and XGBoost model.
8. Apply a final calibration factor.
9. Export the submission file.

This pipeline is designed to be reproducible, interpretable, and safe from future-data leakage.

---

## 5. Data Cleaning

The raw daily sales data is cleaned using the following steps:

1. Convert `Date` into datetime format.
2. Sort records chronologically.
3. Aggregate duplicated dates if any exist.
4. Reindex the dataset into a complete daily date range.
5. Fill missing target values using interpolation and median fallback.
6. Apply conservative outlier clipping using a rolling median and rolling median absolute deviation.

The outlier clipping is intentionally conservative. It reduces extreme abnormal noise but keeps valid seasonal spikes because sudden sales peaks may represent real business events.

This cleaning step improves the stability of the downstream models while preserving important time-series patterns.

---

## 6. Feature Engineering

## 6.1 Calendar Features

The model extracts standard calendar features from the `Date` column:

* `year`
* `month`
* `day`
* `day_of_week`
* `day_of_year`
* `week_of_year`
* `quarter`
* `is_weekend`
* `is_month_start`
* `is_month_end`
* `is_quarter_start`
* `is_quarter_end`

These features help capture regular business cycles such as weekly, monthly, quarterly, and yearly effects.

## 6.2 Fourier Seasonality Features

To model smooth periodic behavior, the pipeline creates Fourier features from `day_of_year` and `day_of_week`.

These features help the model capture annual and weekly seasonality more smoothly than raw categorical date variables.

They are useful because revenue often follows repeating patterns across the year, such as higher or lower demand during specific periods.

## 6.3 Lag Features

The XGBoost model uses multiple lag features:

* `lag_1`
* `lag_7`
* `lag_14`
* `lag_28`
* `lag_56`
* `lag_91`
* `lag_182`
* `lag_365`

These features allow the model to learn short-term memory, weekly recurrence, quarterly effects, half-year effects, and same-period-last-year behavior.

## 6.4 Rolling Statistical Features

The model also uses rolling statistics over several windows:

* 7 days
* 14 days
* 28 days
* 56 days
* 91 days
* 182 days
* 365 days

For each window, the pipeline computes:

* rolling mean
* rolling standard deviation
* rolling minimum
* rolling maximum

These features help the model understand recent trend, volatility, and deviation from historical baselines.

## 6.5 Recent-Regime Features

The model includes indicators for the latest observed business regime. These features allow the model to treat recent years differently from earlier years.

This is important because the most recent data has a different level and trend compared with older history. Older years remain useful for learning seasonality, but recent years are more useful for predicting the future level of sales.

---

## 7. Modeling Approach

## 7.1 Recent-Period Seasonal Ensemble

The first major component is a recent-period seasonal ensemble.

This model generates forecasts using multiple assumptions:

* same-day-last-year behavior
* day-of-year seasonal profile
* month-day seasonal profile
* month-weekday seasonal profile
* recent-period growth trend
* recent-year weighted seasonality

Several growth assumptions are tested, including:

* zero growth
* last-year growth
* two-year CAGR
* recent-period growth
* latest-year recovery growth

Each configuration is evaluated using chronological validation. The best configurations are then combined using inverse validation error weighting. This means models with lower validation error receive larger weights in the ensemble.

This approach provides strong seasonality modeling while reducing the risk of relying on only one assumption.

## 7.2 Direct XGBoost Time-Series Model

The second major component is a direct XGBoost model trained separately for `Revenue` and `COGS`.

XGBoost uses the engineered features described above:

* calendar features
* Fourier features
* lag features
* rolling features
* recent-regime features

The target is log-transformed during training using `log1p(target)` and converted back using `expm1(prediction)`. This stabilizes the target distribution and reduces the impact of extreme values.

The model also uses recency weighting. More recent years receive larger sample weights, which helps the model adapt to the latest sales regime.

## 7.3 Final Blending and Calibration

The final prediction blends the seasonal model and the XGBoost model.

The final structure is:

Final prediction = `0.825 × Seasonal_Model_Scaled + 0.175 × XGBoost_Model`

The best final settings were:

* `scale = 1.18`
* `xgb_weight = 0.15`

The scale factor corrects the overall prediction level because earlier models consistently underpredicted the target level. The XGBoost component provides additional local corrections based on lag and rolling patterns.

---

## 8. Cross-Validation Strategy

The solution uses **time-based cross-validation**, not random shuffling.

The main validation setup is:

* Training period: all dates before `2022-01-01`
* Validation period: from `2022-01-01` to `2022-12-31`

This validation strategy simulates the real forecasting task: the model must predict future dates using only past data.

The seasonal ensemble also evaluates configurations on recent chronological folds. Configurations that perform better on recent validation periods receive more importance.

This design is more appropriate for time-series forecasting than random train-test splitting, because random splitting would leak future information into the training process.

---

## 9. Leakage Control

Leakage prevention is a central part of the pipeline.

## 9.1 Test File Usage

`sample_submission.csv` is used only to obtain the future dates for prediction. No target values from the test period are used.

## 9.2 Chronological Training

All validation is chronological. The training set always comes before the validation set in time.

## 9.3 Lag and Rolling Features

Lag and rolling features are created only from past observations. Rolling windows are shifted before being used so that the current target value is not included in its own feature.

## 9.4 Recursive Forecasting

For future prediction, XGBoost uses recursive forecasting. This means that when predicting 2023–2024, the model uses historical values and its own previous predictions for future lags. It never uses true future target values.

## 9.5 Auxiliary Tables

Auxiliary business tables are not used as future-known predictors because their future values are unavailable. They are used only for EDA and historical analysis. This avoids leakage from information that would not exist at real prediction time.

---

## 10. Explainability

The solution includes model explainability through:

1. XGBoost feature importance
2. SHAP values

## 10.1 XGBoost Feature Importance

Feature importance is used to identify which features contribute most to model prediction.

The most important feature groups are expected to include:

* `lag_365`
* `lag_182`
* `roll_mean_365`
* `roll_mean_91`
* `day_of_year`
* yearly Fourier features
* recent-regime indicators
* month and weekday features

These features show that the model relies on yearly seasonality, same-period-last-year patterns, recent trend, and recent level changes.

## 10.2 SHAP Explanation

SHAP values can be used to explain how each feature pushes the prediction higher or lower.

A SHAP summary plot provides a global explanation of model behavior. It shows which features have the largest impact and whether high or low feature values tend to increase or decrease the prediction.

This improves transparency and helps translate the model behavior into business insights.

---

## 11. Business Interpretation of Model Drivers

The model identifies the following main revenue drivers.

## 11.1 Annual Seasonality

Features such as `day_of_year`, yearly Fourier terms, and `lag_365` are important. This means revenue follows a strong annual cycle. Some periods of the year consistently generate higher or lower revenue.

## 11.2 Recent Sales Level

The model performs better when recent years are weighted more heavily. This suggests that the latest business level is more relevant to the hidden test period than older historical levels.

## 11.3 Same-Period Historical Behavior

The same-day-last-year and yearly lag features are important. This indicates that revenue on a future date is strongly related to revenue around the same calendar date in previous years.

## 11.4 Short-Term Momentum

Rolling means and recent lags help the model understand whether sales are currently above or below the recent baseline. This allows the model to adjust predictions based on local upward or downward trends.

## 11.5 Relationship Between Revenue and COGS

`COGS` is closely related to `Revenue`. The final calibration scales both targets together, reflecting the business assumption that higher sales generally lead to proportionally higher cost of goods sold.

---

## 12. Reproducibility

The notebook is designed to be reproducible.

A fixed random seed is used:

* `SEED = 2026`
* `random.seed(SEED)`
* `np.random.seed(SEED)`
* `random_state = SEED` for XGBoost

The repository includes the full notebook with all steps:

* data loading
* data cleaning
* feature engineering
* validation
* model training
* model explainability
* final prediction
* submission export

Therefore, the full pipeline can be rerun from start to finish.

---

## 13. Final Result Summary

The solution improved through several stages:

1. Basic seasonal baseline
2. Recent-period seasonal ensemble
3. Uniform scale calibration
4. Direct XGBoost blending
5. Fine-tuning of scale and XGBoost weight

The strongest final setup was:

* `scale = 1.18`
* `xgb_weight = 0.175`

This setup achieved the best leaderboard result among the tested approaches.

---

## 14. Limitations

The main limitation is that future values of auxiliary business signals such as future orders, payments, web traffic, and inventory are unavailable. These variables are useful historically, but they cannot be directly used for future prediction unless they are forecasted first.

Another limitation is that final calibration may slightly depend on leaderboard feedback. Therefore, calibration was kept simple and small to reduce overfitting risk.

---

## 15. Future Improvements

Potential improvements include:

1. Forecasting business components such as order volume, average order value, and COGS ratio.
2. Adding LightGBM or CatBoost as extra ensemble members.
3. Using SHAP-based feature selection to remove noisy features.
4. Training separate models for 2023 and 2024.
5. Modeling `COGS` as a function of predicted `Revenue`.
6. Using more robust multi-fold time-series validation.
7. Building a stacked ensemble from seasonal models, XGBoost, and business-component models.

---

## 16. Conclusion

This solution combines domain-driven time-series modeling with machine learning correction. The strongest insight is that recent years are more useful for predicting the future level of revenue, while older years still help the model learn seasonality.

The final model is reproducible, leakage-aware, and explainable. It uses chronological validation, lag-safe feature engineering, feature importance, and SHAP-based interpretation to provide both accurate predictions and meaningful business insights.
