import pandas as pd
from autogluon.tabular import TabularPredictor

# Load processed data as pandas
df = pd.read_parquet("data/processed/credit.parquet")
label = 'default'
train_data = df.sample(frac=0.8, random_state=42)

predictor = TabularPredictor(label=label, eval_metric="roc_auc", path="ag_models/")
predictor.fit(train_data, time_limit=3600)
# Leaderboard
print(predictor.leaderboard(silent=True))