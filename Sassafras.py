import xgboost
import pandas as pd
import numpy as np
import os.path
from os import path
import matplotlib.pyplot as plt
import time
import pprint
import joblib
from functools import partial
from xgboost import XGBRegressor, DMatrix
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Categorical, Integer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split


class solver():
    
    """
    this class takes the name of a data set and its path, trains an xgboost on it. It can also optimize the xgboost hyper parameters using Bayesian Optimization
    """
    
    def __init__(self, file_name, file_location = None):

        self.preped_data = False
        self.train_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_stratified = None
        self.best_params = {'colsample_bytree': 0.7622818192496497,
             'learning_rate': 0.10288470066757835,
             'max_depth': 5,
             'n_estimators': 4084,
             'reg_alpha': 93.64478624692029,
             'reg_lambda': 82.48387604293973,
             'subsample': 0.2567295058365108}
        self.cv_strategy = None
        self.XBGregressor = None

        if not file_location:
            if not path.isfile(file_name):
                print("File doesnt exist in any paths, please check your filename, or add the file to a path and try again")
            else:
                self.train_df = pd.read_csv(file_name)
        else:
            if not path.isfile(file_location + "/" + file_name):
                print("No such file exists in the path, please check the name or the path and try again")
            else:
                self.train_df = pd.read_csv(file_location + "/" + file_name)

    def prep_the_data(self):
        if self.train_df is None:
            return

        self.X_train, self.X_test = train_test_split(self.train_df, test_size=0.2, random_state=440)

        self.y_train = self.X_train.Value
        self.X_train = self.X_train.set_index('Id').drop('Value', axis='columns')
        self.y_test = self.X_test.Value
        self.X_test = self.X_test.set_index('Id').drop('Value', axis='columns')

        self.y_stratified = pd.cut(self.y_train.rank(method='first'), bins=10, labels=False)
        self.y_train = np.array(winsorize(self.y_train, [0.002, 0.0]))  # to help reduce effects of outliers

        # in this data set categorical values are columns with G or I in their name
        categorical = [item for item in self.X_train.columns if 'G' or 'I' in item]
        dummies = pd.get_dummies(self.X_train[categorical])
        self.X_train[dummies.columns] = dummies.iloc[:len(self.X_train), :]
        del dummies
        self.preped_data = True

    def optimizer_report(self, optimizer, X, y, title="model", callbacks=None):
        start = time()

        if callbacks is not None:
            optimizer.fit(X, y, callback=callbacks)
        else:
            optimizer.fit(X, y)

        d = pd.DataFrame(optimizer.cv_results_)
        best_score = optimizer.best_score_
        best_score_std = d.iloc[optimizer.best_index_].std_test_score
        best_params = optimizer.best_params_

        print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
               + u"\u00B1" + " %.3f") % (time() - start,
                                         len(optimizer.cv_results_['params']),
                                         best_score,
                                         best_score_std))
        print('Best parameters:')
        pprint.pprint(best_params)
        print()
        return best_params

    def optimize(self):
        if self.train_df is None:
            return
        if not self.preped_data:
            self.prep_the_data()

        scoring = make_scorer(partial(mean_squared_error, squared = False), greater_is_better= False)
        skf = StratifiedKFold(n_splits= 5, shuffle=True, random_state= 440)
        self.cv_strategy = list(skf.split(self.X_train, self.y_stratified))
        regressor = XGBRegressor(objective = 'reg:squarederror', tree_method = 'gpu_hist', random_state = 440, booster = 'gbtree')

        search_spaces = {'learning_rate': Real(0.01, 1.0, 'uniform'),
                         'max_depth': Integer(2, 12),
                         'subsample': Real(0.1, 1.0, 'uniform'),
                         'colsample_bytree': Real(0.1, 1.0, 'uniform'),
                         'reg_lambda': Real(1e-9, 100., 'uniform'),
                         'reg_alpha': Real(1e-9, 100., 'uniform'),
                         'n_estimators': Integer(50, 5000)
                         }
        optim = BayesSearchCV(estimator=regressor,
                              search_spaces = search_spaces,
                              scoring = scoring,
                              cv = self.cv_strategy,
                              n_iter = 120,
                              n_points = 1,
                              n_jobs = 1,
                              iid=False,
                              return_train_score=False,
                              refit= False,
                              optimizer_kwargs={'base_estimator' : 'GP'}, # for Gaussian
                              random_state = 440,
                              )

        control = DeltaYStopper(delta = 0.001) # stop when the optimization doesnt improve more than 0.001
        time_control = DeadlineStopper(total_time= 60 * 10) # stop after maximum of 10 minutes
        self.best_params = self.optimizer_report(optim, self.X_train, self.y_train, 'XGBoost_regression', callbacks=[control, time_control])


    def Train_on_data(self, folds = 7):
        if self.train_df is None:
            return

        if not self.best_params:
            print("Need to optimize parameters first")
            return
        if not self.preped_data:
            self.prep_the_data()

        reg = XGBRegressor(objective = 'reg:squarederror', tree_method = 'gpu_hist', random_state = 440, **self.best_params)
        skf = StratifiedKFold(n_splits= folds, shuffle=True, random_state=440 )
        predictions = np.zeros(len(self.X_test))
        rmse = list()

        for k, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_stratified)):
            reg.fit(self.X_train.iloc[train_idx, :], self.y_train[train_idx])
            val_preds = reg.predict(self.X_train.iloc[val_idx, :])
            val_rmse = mean_squared_error(y_true=self.y_train[val_idx], y_pred=val_preds, squared=False)
            print(f"Fold {k} RMSE: {val_rmse:0.3f}")
            rmse.append(val_rmse)
            predictions += reg.predict(self.X_test).ravel()

        self.XBGregressor = reg
        predictions /= folds
        print(f"repeated CV RMSE: {np.mean(rmse):0.5f} (std={np.std(rmse):0.5f})")


    def tree_plot(self):
        if self.train_df is None:
            return

        if self.XBGregressor is None:
            print("Regressor hasn't been trained yet")
            return

        xgboost.plot_tree(self.XBGregressor, num_trees= 0)
        plt.rcParams['figure.figsize'] = [50, 10]
        plt.show()


    def importance_plot(self, max_num_features = None):
        if self.train_df is None:
            return

        if self.XBGregressor is None:
            print("Regressor hasn't been trained yet")
            return

        xgboost.plot_importance(self.XBGregressor, max_num_features= max_num_features)
        plt.rcParams['figure.figsize'] = [5,5]
        plt.show()






