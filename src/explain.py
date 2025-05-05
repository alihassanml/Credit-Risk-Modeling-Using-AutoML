import shap
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import Booster

model = Booster(model_file='../model/lgb_model.txt', params={"predict_disable_shape_check": True})

df = pd.read_parquet("../data/processed/credit.parquet")

df.drop('ID',axis=1,inplace=True)
df.reset_index(drop=True,inplace=True)

X = df.drop('default', axis=1)

# SHAP values
e = shap.TreeExplainer(model)
shap_vals = e.shap_values(X)

# Summary plot
shap.summary_plot(shap_vals, X)
plt.savefig("reports/shap_summary.png")