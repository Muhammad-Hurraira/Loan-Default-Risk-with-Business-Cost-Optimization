# Loan Default Risk with Business Cost Optimization

This project predicts **loan default risk** and integrates **business cost optimization** so that decisions minimize expected financial loss (balancing false positives vs. false negatives).

## Key Features
- **Preprocessing**: One-hot encoding, median/mode imputation, feature scaling for numeric columns.
- **Models**: Logistic Regression (with class weights) and CatBoost.
- **Cost Model**: Explicit FP/FN business costs; optimal probability threshold chosen to minimize total cost.
- **Evaluation**: Confusion matrix, accuracy, and total business cost at the optimized threshold.

## Why This Project?
Traditional models optimize for accuracy/AUC. Lenders care about **money**. This repo shows how to turn model scores into **profit-aware** decisions by tuning the threshold using real costs.

## Tech Stack
- **Python**: 3.8+  
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `catboost`, `jupyter`

---

## Usage

### 1) Setup
```bash
git clone https://github.com/Muhammad-Hurraira/loan-default-risk.git
cd loan-default-risk
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install pandas numpy scikit-learn catboost jupyter
```

> Optional: create `requirements.txt` and run `pip install -r requirements.txt`.

### 2) Data
Place the dataset file in the repo root (or update the path in the notebook):
- Expected filename: **`home_credit_synthetic_positive.csv`**
- Required columns used in the notebook:
  - `TARGET` (binary label: 1 = default, 0 = non-default)
  - `SK_ID_CURR` (dropped)
  - `AMT_CREDIT` (used to derive business costs)
- Any categorical columns will be auto one-hot encoded.

### 3) Run the notebook
```bash
jupyter notebook Loan_Default_Risk_with_Business_Cost_Optimization.ipynb
```
Then **Run All** cells. The notebook will:
1. Load & preprocess data (encode, impute, scale numerics).
2. Train Logistic Regression on scaled features.
3. Train CatBoost on encoded features.
4. Define business costs (see next step).
5. Sweep thresholds ∈ [0,1], compute total cost, and pick the **optimal threshold** for each model.
6. Print metrics and the minimum total cost achieved.

### 4) Customize business costs (important)
By default, the notebook estimates:
- **COST_FP** (approve but default): `0.5 * mean(AMT_CREDIT)`
- **COST_FN** (reject but would repay): `0.1 * mean(AMT_CREDIT)`

To use your **own** costs, edit the lines that define `COST_FP` and `COST_FN` and set fixed numbers, e.g.:
```python
COST_FP = 5000.0   # your estimated loss per defaulted approval
COST_FN = 800.0    # your estimated missed profit per good rejection
```
Re-run the threshold sweep cells to update the optimal threshold and total cost.

### 5) Use the optimal threshold
The notebook prints something like:
```
Optimal Threshold for Logistic Regression: 0.37
Minimum Total Business Cost for Logistic Regression: 123456.78
```
Apply this threshold to the model’s predicted probabilities to convert scores → approve/reject decisions.

### 6) (Optional) Save artifacts
If you want to persist the pipeline:
```python
import joblib
joblib.dump(scaler, "artifacts/scaler.joblib")               # for Logistic Regression
joblib.dump(log_reg_model, "artifacts/log_reg_model.joblib")
cat_model.save_model("artifacts/cat_model.cbm")               # CatBoost native format
```

### 7) Troubleshooting
- **File not found**: Ensure `home_credit_synthetic_positive.csv` is in the working directory.
- **Missing columns**: Verify `TARGET` and `AMT_CREDIT` exist and have valid values.
- **Imbalance**: The notebook uses `class_weight='balanced'` (LR) and class weights (CatBoost). You can also try SMOTE or different weights.

---

## Future Work

- **Data & Features**
  - Add domain features (DTI, utilization, delinquency counts).
  - Time-aware splits if data is temporal.
  - Robust outlier handling and missing-value strategies.

- **Modeling**
  - Add LightGBM/XGBoost; compare via cross-validated AUC/PR and **cost at optimal threshold**.
  - Hyperparameter tuning (Optuna/GridSearch) across models and thresholds jointly.
  - Score calibration (Platt/Isotonic) to improve probability quality.

- **Cost Optimization**
  - Segment-specific thresholds (income bands, product types).
  - Sensitivity analysis over a grid of `(COST_FP, COST_FN)`; plot profit/cost curves.
  - Incorporate expected recovery rate, LGD/EAD, and CLV for profit-true decisions.

- **Imbalance & Validation**
  - Try focal loss / custom CatBoost loss.
  - Stratified CV with consistent cost evaluation; nested CV for tuning.

- **Explainability & Governance**
  - SHAP for CatBoost; coefficient analysis for LR.
  - Bias/fairness checks (group metrics, parity constraints).
  - Model cards, decision logs, and auditability.

- **MLOps & Deployment**
  - Wrap the best model + threshold in a `Pipeline` and serve with FastAPI.
  - Batch scoring script + CSV I/O templates.
  - CI (GitHub Actions), reproducible env (Conda/Poetry), and Docker.
  - Data & model versioning (DVC/MLflow).

- **Monitoring**
  - Drift detection (feature/score drift), alerting, and periodic threshold re-tuning based on latest cost assumptions.

-  **Contact**

For any queries or feedback, feel free to connect!
- LinkedIn: https://www.linkedin.com/in/muhammad-hurraira-0993a4346/

---

## License
MIT — see [`LICENSE`](LICENSE).
