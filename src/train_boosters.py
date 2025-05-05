import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load data
df = pd.read_parquet("data/processed/credit.parquet")
X = df.drop('default', axis=1)
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# LightGBM
d_train = lgb.Dataset(X_train, label=y_train)
params = {'objective': 'binary', 'metric':'auc'}
model_lgb = lgb.train(params, d_train, num_boost_round=100)
pred_lgb = model_lgb.predict(X_test)
print("LGB AUC:", roc_auc_score(y_test, pred_lgb))

# XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {'objective': 'binary:logistic', 'eval_metric':'auc'}
bst = xgb.train(params, dtrain, num_boost_round=100)
pred_xgb = bst.predict(dtest)
print("XGB AUC:", roc_auc_score(y_test, pred_xgb))