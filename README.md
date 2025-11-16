# 🏎️ F1 Podium Predictor

A comprehensive machine learning project to predict Formula 1 driver podium probabilities (top-3 finish) using historical race data, telemetry, and weather information.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Installation](#installation)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Models](#models)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project combines historical Formula 1 data from multiple sources to build predictive models for podium finishes. The project follows a complete machine learning pipeline from data collection and verification to feature engineering, modeling, and interpretability analysis.

**Key Objectives:**
- Predict the probability of a driver finishing in the top 3 positions (podium)
- Understand which factors most influence podium finishes
- Provide interpretable predictions for race analysis

**Data Coverage:**
- **Historical Data (1994-2024)**: Race results, qualifying, standings, circuits, drivers, constructors
- **Advanced Data (2018-2024)**: Lap times, telemetry, weather conditions

## ✨ Features

- **Comprehensive Data Integration**: Combines Kaggle historical data with FastF1 API data
- **Robust Data Verification**: Completeness, quality, and cross-source consistency checks
- **Intelligent Feature Engineering**: 
  - Static features (grid position, circuit characteristics)
  - Rolling historical features (driver/constructor performance trends)
  - Placeholder features for telemetry and weather data
- **Multiple ML Models**: LightGBM, CatBoost, and TabNet implementations
- **Model Interpretability**: SHAP analysis, permutation importance, partial dependence plots
- **Resume Functionality**: Data extraction can be paused and resumed
- **Rate Limit Handling**: Robust retry logic for API rate limits and connection issues

## 📁 Project Structure

```
F1/
├── data/
│   ├── raw/
│   │   ├── kaggle/              # Historical F1 data (1994-2024)
│   │   │   ├── races.csv
│   │   │   ├── results.csv
│   │   │   ├── drivers.csv
│   │   │   ├── constructors.csv
│   │   │   ├── circuits.csv
│   │   │   ├── qualifying.csv
│   │   │   ├── driver_standings.csv
│   │   │   ├── constructor_standings.csv
│   │   │   ├── constructor_results.csv
│   │   │   ├── sprint_results.csv
│   │   │   └── ...
│   │   └── fastf1_2018plus/     # FastF1 API data (2018-2024)
│   │       ├── ALL_RESULTS_*.csv
│   │       ├── ALL_LAPS_*.csv
│   │       ├── ALL_TELEMETRY_*.csv
│   │       └── ALL_WEATHER_*.csv
│   └── processed/
│       ├── master_races.csv      # Combined master dataset
│       ├── features.csv          # Engineered features for modeling
│       └── master_races_schema.md # Data schema documentation
├── notebooks/
│   ├── 00_data_fastf1.ipynb     # FastF1 data extraction
│   ├── 01_data_verification.ipynb # Data quality checks
│   ├── 02_data_combining.ipynb   # Combine Kaggle CSVs
│   ├── 03_exploratory_data_analysis.ipynb # EDA and visualizations
│   ├── 04_feature_engineering.ipynb # Feature creation
│   ├── 05_modeling.ipynb         # Model training and evaluation
│   └── 06_interpretability_report.ipynb # Model explanations
├── models/                        # Trained model files
├── docs/                          # Additional documentation
├── reports/                       # Analysis reports
└── README.md
```

## 📊 Data Sources

### 1. Kaggle F1 Dataset
- **Source**: [Kaggle Formula 1 World Championship (1950 - 2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
- **Coverage**: Complete F1 history from 1950-2024
- **Data Types**: Race results, qualifying, standings, circuits, drivers, constructors, lap times, pit stops
- **Usage**: Primary source for historical data (1994+ used in this project)

### 2. FastF1 API
- **Source**: [FastF1 Python Library](https://github.com/theOehrly/Fast-F1)
- **Coverage**: Detailed race data from 2018-2024
- **Data Types**: 
  - **Results**: Detailed race results with timing data
  - **Laps**: Per-lap timing and sector information
  - **Telemetry**: Speed, throttle, brake, gear, DRS status (50Hz sampling)
  - **Weather**: Air temperature, track temperature, rainfall, wind speed/direction
- **Usage**: Advanced features for recent seasons (2018+)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/ch1ckenhead/f1-podium-predictor.git
cd f1-podium-predictor
```

2. **Create a virtual environment (recommended):**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n f1-predictor python=3.10
conda activate f1-predictor
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `fastf1` - F1 data API
- `scikit-learn` - Machine learning utilities
- `lightgbm` - LightGBM gradient boosting
- `catboost` - CatBoost gradient boosting
- `pytorch-tabnet` - TabNet deep learning model
- `shap` - Model interpretability
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations
- `jupyter` - Notebook environment

4. **Download Data:**
   - **Kaggle Data**: Download from [Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) and place CSV files in `data/raw/kaggle/`
   - **FastF1 Data**: Run `notebooks/00_data_fastf1.ipynb` to extract data (or use pre-extracted CSVs in `data/raw/fastf1_2018plus/`)

## 📖 Usage

### Workflow

The project follows a sequential notebook workflow:

1. **Data Extraction** (`00_data_fastf1.ipynb`)
   - Extract FastF1 data for 2018-2024
   - Handles rate limiting and resume functionality
   - Outputs: Yearly CSVs for RESULTS, LAPS, TELEMETRY, WEATHER

2. **Data Verification** (`01_data_verification.ipynb`)
   - Completeness checks (year, race, driver, constructor coverage)
   - Quality checks (missing values, outliers, data types)
   - Cross-source consistency (Kaggle vs FastF1 for 2018-2024)
   - Outputs: Verification report JSON

3. **Data Combining** (`02_data_combining.ipynb`)
   - Combines all Kaggle CSVs into master table
   - Intelligent column deduplication
   - Filters to 1994+ data
   - Outputs: `master_races.csv`

4. **Exploratory Data Analysis** (`03_exploratory_data_analysis.ipynb`)
   - Target variable analysis (podium distribution)
   - Feature distributions and relationships
   - Temporal trends and era comparisons
   - Outputs: EDA insights JSON, visualizations

5. **Feature Engineering** (`04_feature_engineering.ipynb`)
   - Static features (grid, circuit, driver age, constructor)
   - Rolling historical features (3, 5, 10 race windows)
   - Placeholder features for telemetry/weather
   - Outputs: `features.csv`

6. **Modeling** (`05_modeling.ipynb`)
   - Train/Val/Test split (1994-2022 / 2023 / 2024)
   - Train LightGBM, CatBoost, TabNet models
   - Hyperparameter tuning
   - Model evaluation and comparison
   - Outputs: Trained models, evaluation metrics

7. **Interpretability** (`06_interpretability_report.ipynb`)
   - SHAP value analysis
   - Permutation importance
   - Partial dependence plots
   - Feature importance rankings
   - Outputs: Interpretability report

### Quick Start

```python
# Load the master dataset
import pandas as pd
master = pd.read_csv('data/processed/master_races.csv')

# Load features
features = pd.read_csv('data/processed/features.csv')

# Load trained model
import pickle
with open('models/best_podium_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict_proba(features[feature_columns])[:, 1]
```

## 📓 Notebooks

### 00_data_fastf1.ipynb
**Purpose**: Extract FastF1 API data for 2018-2024 seasons

**Features**:
- Robust error handling (rate limits, connection errors, missing data)
- Resume functionality (skips already-generated CSVs)
- Retry logic with exponential backoff
- Comprehensive logging

**Outputs**:
- `ALL_RESULTS_YYYY.csv` - Race results by year
- `ALL_LAPS_YYYY.csv` - Lap timing data by year
- `ALL_TELEMETRY_YYYY.csv` - Telemetry data by year
- `ALL_WEATHER_YYYY.csv` - Weather data by year

### 01_data_verification.ipynb
**Purpose**: Comprehensive data quality and consistency verification

**Checks**:
- **Completeness**: Year coverage, race coverage, driver/constructor presence
- **Quality**: Missing values, outliers, data type consistency
- **Cross-Source Consistency**: Kaggle vs FastF1 comparison (2018-2024)

**Outputs**:
- `verification_report.json` - Summary of all checks

### 02_data_combining.ipynb
**Purpose**: Combine all Kaggle CSVs into a single master table

**Key Features**:
- Intelligent column deduplication (drops identical columns, prefixes different ones)
- Merge-first approach ensures all rows are considered
- One row per (raceId, driverId) for 1994+ races
- Placeholder columns for future FastF1 features

**Outputs**:
- `master_races.csv` - Combined master dataset
- `master_races_schema.md` - Schema documentation

### 03_exploratory_data_analysis.ipynb
**Purpose**: Comprehensive exploratory data analysis

**Analyses**:
- Target variable distribution (podium rate ~14%)
- Feature distributions and relationships
- Correlation analysis
- Temporal trends (1994-2017 vs 2018-2024)
- Circuit and driver performance patterns

**Outputs**:
- EDA visualizations
- `eda_insights.json` - Key findings and recommendations

### 04_feature_engineering.ipynb
**Purpose**: Create all features for modeling

**Feature Categories**:
1. **Static Features**:
   - Grid position (grid, grid_top3, grid_top10, grid_pole)
   - Circuit characteristics (name, country)
   - Driver age
   - Constructor information
   - Qualifying position

2. **Rolling Historical Features** (with shift to prevent data leakage):
   - Driver podium rate (3, 5, 10 race windows)
   - Driver average points (3, 5, 10 race windows)
   - Driver average position (3, 5, 10 race windows)
   - Constructor podium rate (3, 5, 10 race windows)
   - Constructor average points (3, 5, 10 race windows)
   - Total podiums and races completed

3. **Placeholder Features** (for future FastF1 integration):
   - `lap_time_variance`
   - `throttle_variance`
   - `overtake_attempts`
   - `avg_pit_stops`
   - Weather features (air_temp, track_temp, rainfall_mm, wind_speed, wind_direction)

**Outputs**:
- `features.csv` - Final feature set for modeling

### 05_modeling.ipynb
**Purpose**: Train and evaluate multiple ML models

**Models**:
1. **LightGBM**: Gradient boosting with categorical feature support
2. **CatBoost**: Gradient boosting with built-in categorical handling
3. **TabNet**: Deep learning with attention-based feature selection

**Train/Val/Test Split**:
- **Train**: 1994-2022 (chronological)
- **Validation**: 2023
- **Test**: 2024

**Evaluation Metrics**:
- ROC-AUC
- Precision-Recall AUC
- Log Loss
- Classification Report (Precision, Recall, F1)

**Outputs**:
- Trained model files (`.pkl`)
- Evaluation metrics and comparison

### 06_interpretability_report.ipynb
**Purpose**: Explain model predictions and provide insights

**Analyses**:
- **SHAP Values**: Feature contribution to individual predictions
- **Permutation Importance**: Overall feature importance
- **Partial Dependence Plots**: Feature effect on predictions
- **Feature Importance Rankings**: Top drivers of podium probability

**Outputs**:
- Interpretability visualizations
- Feature importance report

## 🤖 Models

### LightGBM
- **Type**: Gradient Boosting Decision Tree
- **Strengths**: Fast training, handles categorical features, good default performance
- **Hyperparameters**: Learning rate, num_leaves, max_depth, min_data_in_leaf

### CatBoost
- **Type**: Gradient Boosting Decision Tree
- **Strengths**: Built-in categorical encoding, robust to overfitting
- **Hyperparameters**: Learning rate, depth, iterations, l2_leaf_reg

### TabNet
- **Type**: Deep Learning (Attention-based)
- **Strengths**: Automatic feature selection, interpretable attention maps
- **Hyperparameters**: Learning rate, n_d, n_a, n_steps, gamma

## 📈 Results

### Model Performance
(Results will be updated after model training)

**Expected Performance Metrics:**
- ROC-AUC: > 0.85
- Precision-Recall AUC: > 0.60
- Log Loss: < 0.40

### Key Insights
(To be populated after EDA and modeling)

## 🔮 Future Work

### Short-term
- [ ] Integrate FastF1 telemetry features (lap time variance, throttle variance)
- [ ] Integrate FastF1 weather features (air temp, track temp, rainfall, wind)
- [ ] Extract overtake attempts from telemetry data
- [ ] Calculate average pit stops from pit_stops.csv
- [ ] Hyperparameter optimization for all models
- [ ] Ensemble model combining all three approaches

### Medium-term
- [ ] Real-time prediction API
- [ ] Web dashboard for predictions and insights
- [ ] Driver-specific model fine-tuning
- [ ] Circuit-specific model fine-tuning
- [ ] Weather-based model variants

### Long-term
- [ ] Predict race winner (not just podium)
- [ ] Predict qualifying positions
- [ ] Predict fastest lap
- [ ] Incorporate betting odds data
- [ ] Time-series forecasting for championship standings

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 for Python code
- Add docstrings to functions and classes
- Include markdown notes in notebooks for clarity
- Update README.md if adding new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Kaggle** for providing comprehensive historical F1 data
- **FastF1** library developers for the excellent API
- **Formula 1** for making race data available
- The open-source ML community for tools and inspiration

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

---

**Note**: This project is for educational and research purposes. Predictions should not be used for gambling or betting purposes.

