import shap
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import Booster

# Load model and data
model = Booster(model_file='lgb_model.txt')
df = pd.read_parquet("data/processed/credit.parquet")
X = df.drop('default', axis=1)

# SHAP values
e = shap.TreeExplainer(model)
shap_vals = e.shap_values(X)

# Summary plot
shap.summary_plot(shap_vals, X)
plt.savefig("reports/shap_summary.png")