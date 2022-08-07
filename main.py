from feature_validation import *
from data_loader import *
import os
import matplotlib.pyplot as plt

logfi = "./log.txt"
if os.path.exists(logfi):
    os.remove(logfi)

data_dir = "./data/"
feature_fi = data_dir + "mat-feature.csv"
meta_fi = data_dir + "mat-meta.csv"

feature = pd.read_csv(feature_fi, header=0, index_col=0)
meta = pd.read_csv(meta_fi, header=0, index_col=None)

f = open(logfi, "a+")
if feature.shape[0] != meta.shape[0]:
    assert "sample-number check fails!"

# preprocessing
feature_name = list(feature.columns)
sample_name = meta["Sample"]
data = feature.loc[sample_name, :]
label = list(meta["SampleType"].values)
label = pd.DataFrame([1 if "Tumor" == x else 0 for x in label])

# split data into training set and validation set
test_size = 0.3
create_training_validtion(data, label, test_size, data_dir)

# compute feature importance with cross validation using training_set
training_fi = data_dir + "training_set.csv"
training_set = pd.read_csv(training_fi, header=0, index_col=0)
X_train = training_set.iloc[:, 0:-1].values
y_train = training_set["label"].values
chck0 = "#samples in training set: %d(#normal:%d and #tumor %d)" \
        % (training_set.shape[0], len(y_train)-sum(y_train), sum(y_train))
print(chck0)
f.write(chck0+"\n")

kfold = 5
n_jobs = 8

parameters = {"n_estimators": [3, 5, 7, 10, 50, 100, 200, 500, 1000, 2000],
              "max_depth": [3, 5, 7, 10, 20, 50],
              "min_samples_split": [2, 5, 10]}

classifier = RandomForestClassifier(random_state=42)
best_params, best_score = grid_search(X_train, y_train, classifier, parameters, kfold, n_jobs)

chck1 = "The best parameter settings found in grid search"
f.write(chck1+"\n")
print(chck1)
for key in best_params.keys():
    chck2 = key + "=" + str(best_params[key])
    f.write(chck2 + "\n")
    print(chck2)

chck3 = "The best parameter settings achieved accuracy: %.5f" % best_score
f.write(chck3+"\n")
print(chck3)
#
# # save feature importance under the best parameter settings
feature_importance_out = data_dir + "feature_importance.csv"
feature_importance = get_feature_importance(X_train, y_train, feature_name, feature_importance_out, best_params)

chck4 = "feature importance is saved in " + feature_importance_out
f.write(chck4+"\n")

# validation for feature importance using validation set
validation_fi = data_dir + "validation_set.csv"
feature_validation = FeatureValidation(training_fi, validation_fi)

topk_ls = [50, 100, 200, 500, len(feature_name)]
# topk_ls = [len(feature_name)]
acc_ls = []
for topk in topk_ls:
    # update topk features
    feature_validation.select_topk_feature(feature_importance, topk)

    # find the best parameter settings
    val_best_estimator, val_best_params, val_best_score = feature_validation.do_grid_search(parameters, kfold, n_jobs)

    chck5 = "The best parameter settings with top%d features" % topk
    f.write(chck5+"\n")
    print(chck5)
    for key in val_best_params.keys():
        chck6 = key + "=" + str(val_best_params[key])
        f.write(chck6+"\n")
        print(chck6)

    chck7 = "The best parameter settings achieved accuracy: %.5f" % val_best_score
    f.write(chck7+"\n")
    print(chck7)

    val_acc = feature_validation.evaluation(val_best_estimator)
    acc_ls.append(val_acc)
    chck8 = "With top%d features, the validation set achieved accuracy:%.5f" % (topk, val_acc)
    f.write(chck8+"\n")
    print(chck8)

f.close()

