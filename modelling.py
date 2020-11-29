import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import KFold,StratifiedKFold
import lightgbm as lgb
import splitting as sp


def modelling():

    # RUNNING SPLITTING
    df_cleaned, X, y, X_train, y_train, X_test, y_test = sp.splitting()

    # CREATING 5-FOLD AND CHECKIN THE VALIDATION SETS
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # CREATING THE MODEL
    model_lgb = lgb.LGBMClassifier(
        n_jobs=4,
        n_estimators=100000,
        boost_from_average='false',
        learning_rate=0.01,
        num_leaves=64,
        num_threads=4,
        max_depth=-1,
        tree_learner="serial",
        feature_fraction=0.7,
        bagging_freq=5,
        bagging_fraction=0.7,
        min_data_in_leaf=100,
        silent=-1,
        verbose=-1,
        max_bin=255,
        bagging_seed=11,
    )

    # USING THE K-FOLD CREATED AND TRAINING(FITTING) THE MODEL IN THE FOR LOOP
    accuracies = []
    f1s = []
    aucs = []
    second_aucs = []

    models = []

    for train_idx, valid_idx in kf.split(X, y):
        X_train = X.loc[train_idx]
        y_train = y[train_idx]

        X_valid = X.loc[valid_idx]
        y_valid = y[valid_idx]

        model = model_lgb.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='auc',
                              verbose=200, early_stopping_rounds=300)
        print(model)

        predictions = model_lgb.predict_proba(X_valid, num_iteration=model_lgb.best_iteration_)[:, 1]
        print(predictions[:10])

        # print("Accuracy: ", accuracy_score(y_true = y_valid, y_pred = predictions))
        # print("F1: ", f1_score(y_true = y_valid, y_pred = predictions))
        print("AUC: ", roc_auc_score(y_true=y_valid, y_score=predictions))
        fp, tp, threshold = roc_curve(y_valid, predictions)
        print("2ND AUC: ", auc(fp, tp))
        print("\n")

        # accuracies.append(accuracy_score(y_true = y_valid, y_pred = predictions))
        # f1s.append(f1_score(y_true = y_valid, y_pred = predictions))
        aucs.append(roc_auc_score(y_true=y_valid, y_score=predictions))
        second_aucs.append(auc(fp, tp))

        models.append(model)


    # SEEING THE LIST OF METRICS
    print("AUCs: ", aucs)
    print("Second way calculating AUCs: ", second_aucs)
    print("\n")

    # SEEING THE MEAN OF THE METRICS
    print("Avg AUCs: ", np.mean(aucs))
    print("Avg Second second way calculating AUCs: ", np.mean(second_aucs))