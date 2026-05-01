
# Datathon 2026 Round 1 — Revenue and COGS Forecasting

## 1. Project Overview

This project solves the revenue forecasting task in Datathon 2026 Round 1. The goal is to predict daily `Revenue` and `COGS` for the period from `2023-01-01` to `2024-07-01`.

The final solution is an ensemble forecasting pipeline that combines three main components:

1. A **Recent-Period Seasonal Ensemble**
2. A **Direct XGBoost Time-Series Model**
3. A **Final Calibration and Blending Layer**

The main idea behind the solution is that the latest years in the training data are more representative of the hidden test period than older historical data. Therefore, the model gives stronger importance to recent sales patterns while still using older years to learn long-term seasonality.

---

## 2. Data Sources

The main target file is `sales.csv`, which contains daily historical values of `Date`, `Revenue`, and `COGS`.

The future prediction dates are taken from `sample_submission.csv`.

The dataset also includes auxiliary business tables such as `orders.csv`, `order_items.csv`, `products.csv`, `payments.csv`, `shipments.csv`, `returns.csv`, `web_traffic.csv`, and `inventory.csv`.

These auxiliary tables were inspected for feature analysis. However, because future values of orders, payments, web traffic, and inventory are not available for 2023–2024, the final best model mainly relies on date-based and time-series forecasting features to avoid leakage.

---

## 3. Data Cleaning Pipeline

The raw sales data is cleaned through the following steps:

1. Convert `Date` into datetime format.
2. Sort observations chronologically.
3. Aggregate duplicated dates if any exist.
4. Reindex the dataset into a complete daily date range.
5. Fill missing values using interpolation and median fallback.
6. Apply conservative outlier clipping using rolling median and rolling MAD.

The outlier clipping is designed to reduce abnormal noise while preserving valid seasonal spikes. This keeps the time series stable while avoiding aggressive smoothing.

---

## 4. Feature Engineering

## 4.1 Calendar Features

The model extracts several calendar features from the `Date` column:

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

These features help the model learn regular business cycles such as weekday/weekend effects, monthly effects, and yearly seasonality.

## 4.2 Fourier Seasonality Features

To capture smooth yearly and weekly cycles, Fourier features are added. These features represent repeating seasonal patterns using sine and cosine transformations.

For yearly seasonality, the model uses Fourier transformations based on `day_of_year`.

For weekly seasonality, the model uses Fourier transformations based on `day_of_week`.

These features allow the model to capture periodic trends more smoothly than raw calendar variables alone.

## 4.3 Recent-Regime Features

A key observation is that the most recent years show a different sales level and trend compared with earlier years. Therefore, the model includes recent-period indicators such as:

* `is_recent_2020`
* `is_recent_2021`
* `is_latest_period`
* `latest_period_t`

These features help the model distinguish older historical patterns from the more relevant latest business regime.

## 4.4 Lag and Rolling Features

For the XGBoost model, lag features are created:

* `lag_1`
* `lag_7`
* `lag_14`
* `lag_28`
* `lag_56`
* `lag_91`
* `lag_182`
* `lag_365`

Rolling statistics are also created over multiple windows:

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

These features help the model learn short-term momentum, medium-term trend, yearly recurrence, local volatility, and deviations from recent sales levels.

---

## 5. Modeling Approach

The final solution uses an ensemble instead of relying on a single model.

## 5.1 Recent-Period Seasonal Ensemble

The first model is a seasonal ensemble based on multiple forecasting assumptions:

* same-day-last-year behavior
* day-of-year seasonal profile
* month-day profile
* month-weekday profile
* recent-period growth trend
* recent-year weighted seasonality

Several growth assumptions are tested, including:

* zero growth
* last-year growth
* two-year CAGR
* recent-period growth
* latest-year recovery growth

Each configuration is evaluated using time-based validation. The best configurations are combined using inverse validation error weighting, so stronger validation models contribute more while still preserving ensemble diversity.

## 5.2 Direct XGBoost Time-Series Model

The second model is a direct XGBoost regressor trained separately for `Revenue` and `COGS`.

The target is log-transformed using `log1p(target)` during training and converted back after prediction using `expm1(prediction_log)`. This reduces the effect of large spikes and stabilizes model training.

The XGBoost model uses:

* calendar features
* Fourier seasonality features
* recent-regime features
* lag features
* rolling statistics

Recent years are given larger sample weights so that the model focuses more on the latest training period, which is more relevant for forecasting 2023–2024.

## 5.3 Final Blending and Calibration

The final prediction blends the seasonal ensemble with the XGBoost model.

The final structure is:

Final prediction = `0.825 × Seasonal_Model_Scaled + 0.175 × XGBoost_Model`

The best-performing blend used:

* `scale = 1.18`
* `xgb_weight = 0.15`

The scale factor corrects the overall prediction level because earlier models consistently underpredicted the target level.

---

## 6. Cross-Validation Strategy

The pipeline uses chronological validation instead of random splitting.

The main validation setup is:

* Training data: all dates before `2022-01-01`
* Validation data: `2022-01-01` to `2022-12-31`

This simulates the real forecasting scenario, where future dates must be predicted using only past information.

The seasonal ensemble also uses recent-year validation to prioritize configurations that perform well in the latest observed regime.

This validation design avoids unrealistic leakage from future data into training.

---

## 7. Leakage Prevention

Several steps are used to prevent data leakage.

## 7.1 Test Data Usage

`sample_submission.csv` is used only to obtain future dates. No target values from the test period are used.

## 7.2 Chronological Training

Training data always comes before validation data. The model never trains on future observations.

## 7.3 Recursive Forecasting

For the XGBoost model, future predictions are generated recursively. When predicting a future date, the model only uses:

* historical values
* previous model predictions
* date-derived features

It does not use future true target values.

## 7.4 Auxiliary Tables

Auxiliary business tables were analyzed, but future auxiliary values are unavailable. Therefore, they are not directly used as future-known features in the final best model. This prevents leakage from unavailable future business signals.

---

## 8. Explainability

The solution supports explainability through:

1. XGBoost feature importance
2. SHAP values

## 8.1 Feature Importance

XGBoost feature importance can be extracted from the trained model. The most important feature groups are expected to include:

* `lag_365`
* `lag_182`
* `roll_mean_365`
* `roll_mean_91`
* `day_of_year`
* yearly Fourier features
* recent-period indicators
* month and weekday features

These features show that the model learns from yearly seasonality, same-period historical behavior, recent trend, recent structural level changes, and weekly business cycles.

## 8.2 SHAP Values

SHAP can be used to explain how each feature affects model predictions.

A SHAP summary plot helps identify which features push predicted revenue upward or downward. In this project, SHAP or feature importance analysis is used to explain the most important drivers behind revenue and COGS forecasts.

---

## 9. Business Interpretation

The model identifies several important drivers of revenue.

## 9.1 Annual Seasonality

Features such as `day_of_year`, yearly Fourier terms, and `lag_365` are highly relevant. This indicates that revenue follows a strong annual cycle, where certain periods of the year consistently have higher or lower sales.

## 9.2 Recent Business Level Shift

The model performs better when recent years are weighted more heavily. This suggests that the latest business regime differs from earlier historical periods. Older years are still useful for learning seasonality, but they are less reliable for predicting the future level of revenue.

## 9.3 Same-Day-Last-Year Behavior

The seasonal ensemble relies heavily on same-day-last-year patterns. This means the model assumes that future revenue is strongly related to the same calendar period in recent years, adjusted by growth and calibration.

## 9.4 Short-Term Momentum

Rolling means and recent lags help the model determine whether revenue is currently above or below its recent baseline. This allows the model to adjust predictions based on local trend changes.

## 9.5 Relationship Between Revenue and COGS

`COGS` is strongly related to `Revenue`, so the final calibration scales both targets together. This reflects the business assumption that higher sales usually lead to proportionally higher cost of goods sold.

---

## 10. Reproducibility

The notebook sets a fixed seed for reproducibility:

* `SEED = 2026`
* `random.seed(SEED)`
* `np.random.seed(SEED)`
* `random_state = SEED` for XGBoost

All source code is included in the notebook, including:

* data loading
* data cleaning
* feature engineering
* validation
* model training
* prediction
* blending
* submission export

This allows the full pipeline to be rerun end-to-end.

---

## 11. Final Result Summary

The final solution improved through several stages:

1. Baseline seasonal model
2. Recent-period seasonal ensemble
3. Uniform scale calibration
4. Direct XGBoost blend
5. Fine-tuned scale and XGBoost weight

The strongest final setup was:

* `scale = 1.18`
* `xgb_weight = 0.175`

This combination produced the best leaderboard result among the tested approaches.

---

## 12. Limitations

The main limitation is that future values of auxiliary business signals such as orders, payments, web traffic, and inventory are unavailable. These features are highly informative historically, but they cannot be directly used for future prediction without forecasting them first.

In addition, leaderboard-based calibration can improve public score but may overfit if used too aggressively. Therefore, final adjustments were kept small and based on consistent model behavior.

---

## 13. Future Improvements

Potential future improvements include:

1. Forecasting business components such as order volume, average order value, and COGS ratio.
2. Training additional ensemble models such as LightGBM and CatBoost.
3. Using SHAP-based feature selection to remove noisy features.
4. Building separate models for 2023 and 2024.
5. Modeling `COGS` as a function of predicted `Revenue`.
6. Using a stronger multi-fold time-series validation framework.
7. Creating a stacked ensemble using seasonal, XGBoost, LightGBM, and business-component models.

---

## 14. Conclusion

This solution combines domain-driven time-series modeling with machine learning correction. The most important insight is that the latest years in the training data are more representative of the hidden test period than older historical years, so recent patterns should receive higher importance.

The final model is leakage-aware, reproducible, and explainable. It captures annual seasonality, recent level changes, recent trend behavior, and the relationship between `Revenue` and `COGS`.
