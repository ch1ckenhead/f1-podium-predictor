# Analyze multiple ML submissions for F1 predictions
_Exported on 11/10/2025 at 11:52:06 GMT+11 from Cursor (2.0.69)_

---

**User**

i have multiple folders for different ML submissions to predict the outcomes of an F1 race.



Iterate through all 6 of them to determine:

1. What they are all predicting

2. Data processing

3. Their feature engineering + any EDA findings

4. Which machine learning techniques they used

5. Success metrics

---

**Cursor**

### Article F1-predictor-master
- **Prediction target**: Regresses each driver’s finishing position (`result`) for upcoming races using historical race results enriched with driver, constructor, and circuit context.【43:55:Article F1-predictor-master/training.py】【97:143:Article F1-predictor-master/data_utils.py】
- **Data processing**: Merges Ergast race, driver, constructor, result, and qualifying feeds; fills missing grid slots with defaults; one-hot encodes categorical IDs; and standard-scales `result`, `start_position`, and race date parts.【97:143:Article F1-predictor-master/data_utils.py】
- **Feature engineering / EDA**: Adds temporal features (year/month/day) and start-position imputations to contextualize outcomes but otherwise leans on merged raw stats; exploratory work is limited to preprocessing statistics rather than visual EDA.
- **ML techniques**: Trains a three-hidden-layer PyTorch regressor with ReLU activations and dropout, optimized by Adam with a ReduceLROnPlateau scheduler.【43:88:Article F1-predictor-master/training.py】
- **Success metrics**: Tracks train/validation MSE per epoch and reports final test-set MSE loss before generating scenario-based finish predictions.【73:104:Article F1-predictor-master/training.py】【191:204:Article F1-predictor-master/training.py】

```43:88:Article F1-predictor-master/training.py
def prepare_data():
    race_data = load_and_process_data()
    encoder = create_and_fit_encoder(race_data, ['circuit', 'name', 'constructor'])
    race_data = apply_one_hot_encoding(encoder, race_data, ['circuit', 'name',
                                                            'constructor'])
    scalers = scale_numeric_features(race_data, ['result', 'start_position',
                                                 'year', 'month', 'day'])
    ...
    model = RaceOutcomePredictor(input_dim)
    loss_function = nn.MSELoss()
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(model_optimizer, 'min', patience=10,
                                  factor=0.1, verbose=True)
```

```97:143:Article F1-predictor-master/data_utils.py
drivers_df = load_drivers()
circuits_df = load_circuits()
races_df = load_races()
races_df = races_df.merge(circuits_df, on='circuitId').drop(columns=['circuitId'])
...
results_df = results_df.merge(df, on=on, how='left')
cleanup_data(results_df)
return results_df
```

```191:204:Article F1-predictor-master/training.py
def calculate_test_loss(predictor, test_set, batch_size):
    ...
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")
```

### Claude f1-2025-ai-predict-main
- **Prediction target**: Binary classification of race winners plus season simulations driven by a `winner` flag derived from finishing position.【150:191:Claude f1-2025-ai-predict-main/data_loader.py】
- **Data processing**: Loads Ergast-style tables, coerces numeric fields, and joins race, driver, constructor, circuit, qualifying, and standings data into a unified per-entry dataset.【11:118:Claude f1-2025-ai-predict-main/data_loader.py】
- **Feature engineering / EDA**: Creates rolling averages for qualifying and race points, circuit win counts, championship standings, and constructor aggregates to capture form.【66:127:Claude f1-2025-ai-predict-main/data_loader.py】
- **ML techniques**: Trains a tuned `RandomForestClassifier`, retains feature importances, and reuses encoders inside an interactive Streamlit workflow.【337:385:Claude f1-2025-ai-predict-main/f1_predictor.py】
- **Success metrics**: Reports train/validation/test accuracy, ROC-AUC, and test precision/recall/F1 to judge race-winner performance before powering simulations.【337:384:Claude f1-2025-ai-predict-main/f1_predictor.py】

```11:191:Claude f1-2025-ai-predict-main/data_loader.py
self.data[file] = pd.read_csv(f'{self.data_path}/{file}.csv')
...
df['points_moving_avg'] = df.sort_values('raceId').groupby('driverId')['points'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)
...
df['winner'] = (df['position'] == 1).astype(int)
```

```337:384:Claude f1-2025-ai-predict-main/f1_predictor.py
self.model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
...
metrics = {
    'Train Accuracy': accuracy_score(y_train, train_pred),
    'Validation Accuracy': accuracy_score(y_val, val_pred),
    'Test Accuracy': accuracy_score(y_test, test_pred),
    'Train ROC AUC': roc_auc_score(y_train, train_proba),
    'Validation ROC AUC': roc_auc_score(y_val, val_proba),
    'Test ROC AUC': roc_auc_score(y_test, test_proba)
}
```

### Feature Rich F1-Race-Predictor-main
- **Prediction target**: Same winner classification as the Claude project, extended with predicted finishing order via a win-probability/grid scoring heuristic and customizable reliability/error sliders.【130:189:Feature Rich F1-Race-Predictor-main/f1_predictor.py】
- **Data processing**: Reuses the loader pipeline to ingest and merge race, driver, constructor, qualifying, circuit, and standings data while encoding categorical attributes.【1:180:Feature Rich F1-Race-Predictor-main/data_loader.py】
- **Feature engineering / EDA**: Builds moving averages, circuit wins, championship standings, and constructor aggregates; simulation tabs surface interactive “what-if” analyses rather than static EDA.
- **ML techniques**: Trains the same Random Forest specification and surfaces feature importances within Streamlit, while adding championship simulations shaped by user-tuned incident probabilities.【337:416:Feature Rich F1-Race-Predictor-main/f1_predictor.py】
- **Success metrics**: Displays train/validation/test accuracy, ROC-AUC, and precision/recall/F1 within the UI after each training run.【337:415:Feature Rich F1-Race-Predictor-main/f1_predictor.py】

```1:189:Feature Rich F1-Race-Predictor-main/f1_predictor.py
win_probs = self.model.predict_proba(pred_df[feature_columns])[:, 1]
results['Score'] = alpha * results['Win Probability'] - beta * results['Grid']
results = results.sort_values('Score', ascending=False).reset_index(drop=True)
results['Predicted Position'] = range(1, len(results) + 1)
```

### F1-Race-Predictor-Full-Stack-main
- **Prediction target**: Classifies whether each driver finishes in the top three of a given round (`Top 3 Finish`).【3:126:F1-Race-Predictor-Full-Stack-main/server/src/feature_engineering.py】
- **Data processing**: Loads constructors, drivers, races, and results; filters seasons from 1982 onward; and merges into a race/driver-level frame.【12:26:F1-Race-Predictor-Full-Stack-main/server/src/data_preprocessing.py】
- **Feature engineering / EDA**: Derives annual and in-season driver/constructor podium rates and average positions, plus one-hot encodes categorical IDs after computing correlations and descriptive stats.【3:105:F1-Race-Predictor-Full-Stack-main/server/src/feature_engineering.py】【4:26:F1-Race-Predictor-Full-Stack-main/server/src/eda.py】
- **ML techniques**: Performs grid-search style tuning over Logistic Regression, KNN, SVC, Random Forest, Decision Tree, and Gaussian NB, keeping best configs and ROC curves for comparison.【64:147:F1-Race-Predictor-Full-Stack-main/server/src/model_training.py】
- **Success metrics**: Uses validation/test ROC-AUC during tuning, then optimizes a final Random Forest threshold to report accuracy and F1 on 2017–2023 holdout data; confusion matrices accompany model comparisons.【114:147:F1-Race-Predictor-Full-Stack-main/server/src/model_training.py】【220:241:F1-Race-Predictor-Full-Stack-main/server/src/model_training.py】

```3:105:F1-Race-Predictor-Full-Stack-main/server/src/feature_engineering.py
df['Top 3 Finish'] = df['positionOrder'].le(3).astype(int)
...
df['Driver Top 3 Finish Percentage (This Year till last race)'] = df.apply(
    lambda row: calculate_driver_top_3_percentage_before_round(row, df), axis=1)
```

```4:26:F1-Race-Predictor-Full-Stack-main/server/src/eda.py
correlations = df_final.corr()['Top 3 Finish'].sort_values(ascending=False)
df_final_encoded = pd.get_dummies(df_final, columns=['circuitId', 'driverId', 'constructorId'])
```

```64:147:F1-Race-Predictor-Full-Stack-main/server/src/model_training.py
param_grid = {
    'LogisticRegression': {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'random_state': [42]},
    ...
}
...
auc_test = roc_auc_score(y_test, pred_test[:, 1])
...
print(f"Model: {model_name}, File: {info['model']}, Test AUC-ROC: {info['auc_roc']:.4f}, Test Accuracy: {info['accuracy']:.4f}")
```

```220:241:F1-Race-Predictor-Full-Stack-main/server/src/model_training.py
optimal_threshold = thresholds[optimal_idx]
pred_test = (data_prob >= optimal_threshold).astype(int)
...
accuracy = accuracy_score(y_test, pred_test)
f1 = f1_score(y_test, pred_test)
```

### Dec Tree, Logist, RF F1nalyze-main
- **Prediction target**: Forecasts final driver finishing positions in Formula 1 races (competition standings).【1:9:Dec Tree, Logist, RF F1nalyze-main/README.md】
- **Data processing**: Cleans training/test/validation tables by dropping columns with >100 `\N`, imputing modes, and label-encoding categorical variables such as nationality, constructor, and status.【18:29:Dec Tree, Logist, RF F1nalyze-main/README.md】
- **Feature engineering / EDA**: Focuses on curated subsets of the supplied competition features rather than bespoke engineering; exploratory work is minimal per README.
- **ML techniques**: Benchmarks Decision Tree, Random Forest, and Logistic Regression (with feature scaling) to minimize leaderboard RMSE.【31:45:Dec Tree, Logist, RF F1nalyze-main/README.md】
- **Success metrics**: Uses validation RMSE for model selection (best 3.46918) and reports Kaggle leaderboard score (3.46918, 23rd/50) to gauge success.【35:65:Dec Tree, Logist, RF F1nalyze-main/README.md】

```18:65:Dec Tree, Logist, RF F1nalyze-main/README.md
Dropped columns in the training dataset with more than 100 occurrences of `\N`.
...
Logistic Regression ... RMSE on validation data: **3.46918**.
...
Our best entry placed us **23rd** on the leaderboard with a score of **3.46918**.
```

### Good EDA f1_race_predictor-main
- **Prediction target**: Binary classification of race winners with downstream betting ROI analysis on 2024 races.【0:16:Good EDA f1_race_predictor-main/Notebooks/9. F1_Conclusions_and_Next_Steps.ipynb】
- **Data processing**: Creates a combined driver/race dataset, drops redundant columns, converts binaries to booleans, and one-hot encodes race, engine, constructor nationality, and driver nationality; removes early rounds lacking history before modeling.【0:64:Good EDA f1_race_predictor-main/Notebooks/3. F1_preprocessing.ipynb】
- **Feature engineering / EDA findings**: Engineers cumulative constructor/driver points and podium percentages, plus pre-race averages; interpretability work highlighted grid position, season points, experience, and team strength as dominant predictors.【33:104:Good EDA f1_race_predictor-main/Notebooks/3. F1_preprocessing.ipynb】【0:12:Good EDA f1_race_predictor-main/Notebooks/9. F1_Conclusions_and_Next_Steps.ipynb】
- **ML techniques**: Iteratively evaluates scaled Logistic Regression, tuned Decision Trees, Random Forest, Random Forest with SMOTE, and XGBoost (also with SMOTE) using pipelines, grid search, and PCA where relevant.【0:121:Good EDA f1_race_predictor-main/Notebooks/6. F1_Models_v2.ipynb】
- **Success metrics**: Tracks accuracy, precision, recall, F1 (overall and Class 1), plus macro averages; Random Forest + SMOTE reached Class 1 F1 ≈0.51, recall 0.50, and overall accuracy ~0.95, which underpinned a reported 46% ROI on selective bets.【32:47:Good EDA f1_race_predictor-main/Notebooks/6. F1_Models_v2.ipynb】【9:12:Good EDA f1_race_predictor-main/Notebooks/9. F1_Conclusions_and_Next_Steps.ipynb】

```0:64:Good EDA f1_race_predictor-main/Notebooks/3. F1_preprocessing.ipynb
process_df = process_df.rename(columns={'Index': 'race_index'})
...
driver_yearly_stats['Driver Top 3 Finish Percentage (This Year)'] = (driver_yearly_stats['Top_3_Finishes'] / driver_yearly_stats['Total_Races']) * 100
...
df_with_dummies = pd.get_dummies(df_merged_with_driver_points, columns=['race','engine_manufacturer', 'constructor_nationality', 'Nationality'])
```

```33:104:Good EDA f1_race_predictor-main/Notebooks/3. F1_preprocessing.ipynb
df = pd.merge(df, constructor_last_year_stats[['year', 'constructorId', 'round', 'Constructor Top 3 Finish Percentage (Last Year)']], on=['year', 'constructorId', 'round'], how='left')
...
df['Driver Average Position (This Year till last race)'] = df.apply(lambda row: calculate_driver_avg_position_before_round(row, df), axis=1)
```

```0:121:Good EDA f1_race_predictor-main/Notebooks/6. F1_Models_v2.ipynb
model = LogisticRegression(max_iter=1000)
...
dt_grid_search = GridSearchCV(estimator=pipe_dt,
        param_grid=dt_param_grid,
        scoring='f1',
        cv=5)
...
grid_search_objects.append(gs)
```

```9:12:Good EDA f1_race_predictor-main/Notebooks/9. F1_Conclusions_and_Next_Steps.ipynb
Random Forest model using SMOTE ... attained an F1 score of 0.51, ... accuracy of 0.95 and ... recall of 0.50 for Class 1.
...
total earnings ... amounted to £92 ... representing a 46% return on investment.
```

---

**Next steps**: Consider standardizing evaluation reports (e.g., confusion matrices, calibration) across repos for easier comparison, and ensure reproducible pipelines (dataset sourcing, deterministic splits) are documented—especially for the SMOTE workflows where leakage risk is higher.

```python
def prepare_data():
    race_data = load_and_process_data()
    encoder = create_and_fit_encoder(race_data, ['circuit', 'name', 'constructor'])
    race_data = apply_one_hot_encoding(encoder, race_data, ['circuit', 'name',
                                                            'constructor'])
    scalers = scale_numeric_features(race_data, ['result', 'start_position',
                                                 'year', 'month', 'day'])
    ...
    model = RaceOutcomePredictor(input_dim)
    loss_function = nn.MSELoss()
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(model_optimizer, 'min', patience=10,
                                  factor=0.1, verbose=True)
```

```python
drivers_df = load_drivers()
circuits_df = load_circuits()
races_df = load_races()
races_df = races_df.merge(circuits_df, on='circuitId').drop(columns=['circuitId'])
...
results_df = results_df.merge(df, on=on, how='left')
cleanup_data(results_df)
return results_df
```

```python
def calculate_test_loss(predictor, test_set, batch_size):
    ...
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")
```

```python
self.data[file] = pd.read_csv(f'{self.data_path}/{file}.csv')
...
df['points_moving_avg'] = df.sort_values('raceId').groupby('driverId')['points'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)
...
df['winner'] = (df['position'] == 1).astype(int)
```

```python
self.model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
...
metrics = {
    'Train Accuracy': accuracy_score(y_train, train_pred),
    'Validation Accuracy': accuracy_score(y_val, val_pred),
    'Test Accuracy': accuracy_score(y_test, test_pred),
    'Train ROC AUC': roc_auc_score(y_train, train_proba),
    'Validation ROC AUC': roc_auc_score(y_val, val_proba),
    'Test ROC AUC': roc_auc_score(y_test, test_proba)
}
```

```python
win_probs = self.model.predict_proba(pred_df[feature_columns])[:, 1]
results['Score'] = alpha * results['Win Probability'] - beta * results['Grid']
results = results.sort_values('Score', ascending=False).reset_index(drop=True)
results['Predicted Position'] = range(1, len(results) + 1)
```

```python
df['Top 3 Finish'] = df['positionOrder'].le(3).astype(int)
...
df['Driver Top 3 Finish Percentage (This Year till last race)'] = df.apply(
    lambda row: calculate_driver_top_3_percentage_before_round(row, df), axis=1)
```

```python
correlations = df_final.corr()['Top 3 Finish'].sort_values(ascending=False)
df_final_encoded = pd.get_dummies(df_final, columns=['circuitId', 'driverId', 'constructorId'])
```

```python
param_grid = {
    'LogisticRegression': {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'random_state': [42]},
    ...
}
...
auc_test = roc_auc_score(y_test, pred_test[:, 1])
...
print(f"Model: {model_name}, File: {info['model']}, Test AUC-ROC: {info['auc_roc']:.4f}, Test Accuracy: {info['accuracy']:.4f}")
```

```python
optimal_threshold = thresholds[optimal_idx]
pred_test = (data_prob >= optimal_threshold).astype(int)
...
accuracy = accuracy_score(y_test, pred_test)
f1 = f1_score(y_test, pred_test)
```

```plaintext
Dropped columns in the training dataset with more than 100 occurrences of `\N`.
...
Logistic Regression ... RMSE on validation data: **3.46918**.
...
Our best entry placed us **23rd** on the leaderboard with a score of **3.46918**.
```

```plaintext
process_df = process_df.rename(columns={'Index': 'race_index'})
...
driver_yearly_stats['Driver Top 3 Finish Percentage (This Year)'] = (driver_yearly_stats['Top_3_Finishes'] / driver_yearly_stats['Total_Races']) * 100
...
df_with_dummies = pd.get_dummies(df_merged_with_driver_points, columns=['race','engine_manufacturer', 'constructor_nationality', 'Nationality'])
```

```plaintext
df = pd.merge(df, constructor_last_year_stats[['year', 'constructorId', 'round', 'Constructor Top 3 Finish Percentage (Last Year)']], on=['year', 'constructorId', 'round'], how='left')
...
df['Driver Average Position (This Year till last race)'] = df.apply(lambda row: calculate_driver_avg_position_before_round(row, df), axis=1)
```

```plaintext
model = LogisticRegression(max_iter=1000)
...
dt_grid_search = GridSearchCV(estimator=pipe_dt,
        param_grid=dt_param_grid,
        scoring='f1',
        cv=5)
...
grid_search_objects.append(gs)
```

```plaintext
Random Forest model using SMOTE ... attained an F1 score of 0.51, ... accuracy of 0.95 and ... recall of 0.50 for Class 1.
...
total earnings ... amounted to £92 ... representing a 46% return on investment.
```

