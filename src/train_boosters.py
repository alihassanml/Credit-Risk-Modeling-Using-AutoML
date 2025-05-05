import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import numpy as np


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



# Wrap model in scikit-learn API
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='auc')

# Define parameter grid
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5]
}

# Perform randomized search
random_search = RandomizedSearchCV(
    xgb_clf, param_distributions=param_dist, 
    n_iter=20, scoring='roc_auc', cv=3, verbose=1, n_jobs=-1, random_state=42
)

random_search.fit(X_train, y_train)

# Evaluate
best_model = random_search.best_estimator_
preds = best_model.predict_proba(X_test)[:, 1]
print("Tuned XGB AUC:", roc_auc_score(y_test, preds))
print("Best Parameters:", random_search.best_params_)
