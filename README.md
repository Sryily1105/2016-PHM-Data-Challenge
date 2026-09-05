# CMP Removal Rate Prediction

**Project Information**

- Course: Manufacturing Data Science
- Instructor: Yu-Hsin Hung
- Topic: CMP removal rate prediction
- Dataset: 2016 PHM Data Challenge CMP dataset

**Project Overview**

This project analyzes the CMP dataset from the 2016 PHM Data Challenge and builds machine learning models to predict wafer removal rate.

The original notebook was reorganized into separate Python modules to improve readability, reproducibility, and GitHub project structure. The main analysis workflow is kept in `CMP_removal_rate_prediction.ipynb`, while reusable functions are stored in the `src/` folder.

**Repository Structure**

```text
2016-PHM-Data-Challenge/
├── .gitignore
├── CMP_removal_rate_prediction.ipynb
├── README.md
├── requirements.txt
│
├── CMP-data/
│   ├── testing/
│   ├── training/
│   ├── validation/
│   ├── CMP-test-removalrate.csv
│   ├── CMP-training-removalrate.csv
│   └── CMP-validation-removalrate.csv
│
├── src/
│   ├── data.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── model_building.py
│   ├── training.py
│   └── visualization.py
│
├── docs/
│   └── PHM16_Data_Challenge_CFP.pdf
│
└── results/
    ├── tables/
    └── figures/
```
The file docs/PHM16 Data Challenge CFP.pdf provides the background information for the PHM 2016 Data Challenge.


**Analysis Workflow**

The notebook follows the original CMP analysis logic and is organized into the following steps:
1. Load raw CMP process files and removal rate labels
2. Clean missing values, duplicated records, abnormal target values, and sensor outliers
3. Compress time-series process data into wafer-level statistical features
4. Perform feature filtering using correlation analysis and XGBoost feature importance
5. Train machine learning models with time-series cross-validation
6. Compare group-wise models and ensemble predictions
7. Export prediction results and summary tables

**Python Modules**

- src/data.py:  data loading, column standardization, wafer-stage keys, and label merging
- src/data_cleaning.py:  missing value handling, duplicate consolidation, target filtering, and sensor outlier cleaning
- src/feature_engineering.py:  time-series compression, statistical features, trend features, and segment features
- src/feature_selection.py:  correlation filtering and XGBoost-based feature selection
- src/model_building.py:  model definitions, prediction functions, and evaluation metrics
- src/training.py:  time-series cross-validation, Optuna tuning, group-wise training, and ensemble weighting
- src/visualization.py:  EDA plots, target distribution plots, correlation heatmaps, prediction plots, and feature importance plots

**How to Run**

Install the required packages:
- pip install -r requirements.txt

Run the main notebook:
- CMP_removal_rate_prediction.ipynb

**Results**

Generated summary tables are saved under:
- results/tables/

Figures can be saved under:
- results/figures/

Typical output files include model comparison tables, selected feature summaries, validation predictions, final test predictions, and group-wise performance reports.

**Notes**

This project keeps the modeling logic consistent with the original notebook while improving the code organization for GitHub. The notebook is intended to show the complete analysis flow, while the Python files provide reusable functions for data processing, modeling, training, and visualization.

**Reference**

- 2016 PHM Data Challenge CMP dataset
- PHM16 Data Challenge CFP document
