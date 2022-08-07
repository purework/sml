import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


class FeatureValidation:
    def __init__(self, train_fi, test_fi):
        training_set = pd.read_csv(train_fi, header=0, index_col=0)
        validation_set = pd.read_csv(test_fi, header=0, index_col=0)
        self.X_train = training_set.iloc[:, 0:-1].values
        self.y_train = training_set["label"].values
        self.X_test = validation_set.iloc[:, 0:-1].values
        self.y_test = validation_set["label"].values

    def select_topk_feature(self, feature_importance, topk):
        X_train = self.X_train
        X_test = self.X_test

        sort_idx = np.argsort(-feature_importance)
        topk = np.minimum(topk, len(sort_idx))

        topk_X_train = X_train[:, sort_idx[0:topk]]
        topk_X_test = X_test[:, sort_idx[0:topk]]

        self.topk_X_train = topk_X_train
        self.top_X_test = topk_X_test

        # return topk_X_train, topk_X_train

    def do_grid_search(self, params, kfold, n_jobs):
        X_train = self.topk_X_train
        y_train = self.y_train
        classifier = RandomForestClassifier(random_state=42)
        clf = GridSearchCV(classifier, params, n_jobs=n_jobs, cv=kfold)

        clf.fit(X_train, y_train)

        best_params = clf.best_params_
        best_score = clf.best_score_
        best_estimator = clf.best_estimator_

        return best_estimator, best_params, best_score

    def evaluation(self, estimator):
        X_test = self.top_X_test
        y_test = self.y_test
        acc = estimator.score(X_test, y_test)

        return acc











