import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
import warnings ; warnings.filterwarnings('ignore')

class Ensemble:

    def __init__(self, n_estimators=5000, alpha=3, lam=6):
        base_params1 = {'n_estimators': n_estimators, 'num_leaves': 120, 'learning_rate': 0.01,
                        'colsample_bytree': 0.8, 'subsample': 0.9, 'max_depth': 7,
                        'reg_alpha': alpha, 'reg_lambda': lam}

        self.lightgbm1 = LGBMRegressor(objective='l1', subsample_freq=1, silent=False, random_state=18,
                                  importance_type='gain', **base_params1)

        base_params2 = {'n_estimators': n_estimators, 'num_leaves': 500, 'learning_rate': 0.01,
                        'colsample_bytree': 0.8, 'subsample': 0.9, 'max_depth': 9,
                        'reg_alpha': alpha+.1, 'reg_lambda': lam-.1}

        self.lightgbm2 = LGBMRegressor(objective='l1', subsample_freq=1, silent=False, random_state=7,
                                  importance_type='gain', **base_params2)

        base_params3 = {'n_estimators': n_estimators, 'num_leaves': 250, 'learning_rate': 0.01,
                        'colsample_bytree': 0.8, 'subsample': 0.9, 'max_depth': 8,
                        'reg_alpha': alpha-.1, 'reg_lambda': lam+.1}

        self.lightgbm3 = LGBMRegressor(objective='l1', subsample_freq=1, silent=False, random_state=1,
                                  importance_type='gain', **base_params3)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.lightgbm1.fit(self.X, self.y)
        self.lightgbm2.fit(self.X, self.y)
        self.lightgbm3.fit(self.X, self.y)

        return self

    def score_cv(self, cv=5):
        start = time.time()
        score = -cross_val_score(self.model, self.X, self.y, cv=cv, scoring='neg_mean_absolute_error').mean()
        stop = time.time()
        print(f"Validation Time : {round((stop - start) / 60, 2)} mins")
        return score

    def predict(self, X):
        pred_lgbm1 = self.lightgbm1.predict(X)
        pred_lgbm2 = self.lightgbm2.predict(X)
        pred_lgbm3 = self.lightgbm3.predict(X)
        pred = (pred_lgbm1 + pred_lgbm2 + pred_lgbm3)/3

        return pred

