import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV


def create_training_validtion(data, label, test_size, dir):

    sample_num = data.shape[0]
    idx = np.arange(sample_num)
    X_train_idx, X_test_idx = train_test_split(idx, test_size=test_size,
                                               random_state=42, shuffle=True)

    X_train_df = data.iloc[X_train_idx, :].copy()
    label_train = list(label.iloc[X_train_idx, 0].values)
    X_train_df.loc[:, "label"] = label_train

    X_test_df = data.iloc[X_test_idx, :].copy()
    label_test = list(label.iloc[X_test_idx, 0].values)
    X_test_df.loc[:, "label"] = label_test

    X_train_df.to_csv(dir+"training_set.csv", header=True, index=True)
    X_test_df.to_csv(dir+"validation_set.csv", header=True, index=True)


def grid_search(X, y, classifier, parameters, kfold, n_jobs):
    clf = GridSearchCV(classifier, parameters, n_jobs=n_jobs, cv=kfold)
    clf.fit(X, y)
    best_params = clf.best_params_
    best_score = clf.best_score_
    return best_params, best_score


def get_feature_importance(X, y, feature_name, fout, params):
    clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                 min_samples_split=params["min_samples_split"], random_state=42)

    clf.fit(X, y)
    feature_importance = clf.feature_importances_
    feature_dict = {"feature": feature_name, "weight": feature_importance}

    feature_df = pd.DataFrame.from_dict(feature_dict)
    feature_df.to_csv(fout, header=True, index=False)
    return feature_importance


