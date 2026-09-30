# Kaggle Churn and Salary Analysis

Two Kaggle notebooks: exploratory data analysis and machine learning on public datasets.

| Notebook | Dataset | What it does |
|---|---|---|
| [`churn-eda-and-predictions.ipynb`](churn-eda-and-predictions.ipynb) | [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | EDA of why telecom customers leave, then logistic regression, random forest, SVC, AdaBoost and XGBoost compared against a baseline |
| [`ds-salaries-for-non-resident-and-predict.ipynb`](ds-salaries-for-non-resident-and-predict.ipynb) | [Data Science Salaries 2023](https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023) | EDA of data-science salaries (with a focus on people working for companies abroad), k-means clustering, and salary prediction with Ridge, gradient boosting and a PyTorch network |

## Results

**Churn prediction** (test set, 1,407 customers, 27% churn)

| Model | Accuracy | ROC-AUC | Recall (churn) |
|---|---|---|---|
| Baseline (always "no churn") | 0.734 | 0.500 | 0.00 |
| Logistic regression | 0.804 | 0.836 | 0.57 |
| AdaBoost | 0.795 | 0.841 | 0.50 |
| XGBoost (tuned) | 0.789 | 0.836 | 0.52 |
| Random forest (grid search) | 0.796 | 0.834 | 0.51 |

The strongest signals are short tenure, month-to-month contracts and fiber optic internet.

**Salary prediction** (test set, 517 rows, after removing duplicates)

| Model | MAE (USD) | R² |
|---|---|---|
| Baseline (median salary) | 53,555 | 0.00 |
| Ridge regression | 38,242 | 0.38 |
| Gradient boosting | 38,848 | 0.37 |
| PyTorch MLP | 38,972 | 0.34 |

Company country (US or not) and experience level explain most of the salary differences.

## Run it

**On Kaggle:** use the "Open in Kaggle" badge at the top of each notebook, or upload the notebook and attach the dataset linked above. Note that the Kaggle-hosted versions are older than the notebooks in this repo.

**Locally:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the two CSVs from the Kaggle links above and put them in a `data/` folder:

```
data/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── ds_salaries.csv
```

Then open the notebooks with `jupyter lab`. Each notebook looks for its CSV in `/kaggle/input/...` first and falls back to `data/`.

Tested with Python 3.12. On an Intel Mac, `pip install torch` only goes up to 2.2 and needs `numpy<2`; `requirements.txt` pins versions that work together. XGBoost on macOS also needs OpenMP (`brew install libomp`).
