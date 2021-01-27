from os import name
import numpy as np
from numpy.lib.twodim_base import mask_indices
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import seaborn as sns
import re
from sklearn import ensemble
from sklearn import model_selection
#模型融合及测试
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import learning_curve
#利用不同的模型来对特征进行筛选


def get_top_n_features(titanic_train_data_X, titanic_train_data_Y, top_n_features):

    # # random forest
    # rf_est = RandomForestClassifier(random_state=0)
    # rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
    # rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
    # rf_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    # joblib.dump(rf_grid,'/mnt/mydisk/model/tatanic/rf_grid.json')

    # # AdaBoost
    # ada_est =AdaBoostClassifier(random_state=0)
    # ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
    # ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
    # ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    # joblib.dump(ada_grid,'/mnt/mydisk/model/tatanic/ada_grid.json')

    # # ExtraTree
    # et_est = ExtraTreesClassifier(random_state=0)
    # et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}
    # et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
    # et_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    # joblib.dump(et_grid,'/mnt/mydisk/model/tatanic/et_grid.json')

    # # GradientBoosting
    # gb_est =GradientBoostingClassifier(random_state=0)
    # gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
    # gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=25, cv=10, verbose=1)
    # gb_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    # joblib.dump(gb_grid,'/mnt/mydisk/model/tatanic/gb_grid.json')

    # # DecisionTree
    # dt_est = DecisionTreeClassifier(random_state=0)
    # dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
    # dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=25, cv=10, verbose=1)
    # dt_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    # joblib.dump(dt_grid,'/mnt/mydisk/model/tatanic/dt_grid.json')

    rf_grid = joblib.load('/mnt/mydisk/model/tatanic/rf_grid.json')
    ada_grid = joblib.load('/mnt/mydisk/model/tatanic/ada_grid.json')
    et_grid = joblib.load('/mnt/mydisk/model/tatanic/et_grid.json')
    gb_grid = joblib.load('/mnt/mydisk/model/tatanic/gb_grid.json')
    dt_grid = joblib.load('/mnt/mydisk/model/tatanic/dt_grid.json')

    feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': rf_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']

    feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),
                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']

    feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': et_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']

    feature_imp_sorted_gb = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': gb_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']

    feature_imp_sorted_gb = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': gb_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']

    feature_imp_sorted_dt = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': dt_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']

    # merge the three models
    features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt],
                               ignore_index=True).drop_duplicates()

    features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et,
                                     feature_imp_sorted_gb, feature_imp_sorted_dt], ignore_index=True)

    return features_top_n, features_importance


def get_out_fold(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    s
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), verbose=0):  # 绘制学习曲线
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


if __name__ == '__main__':
    train_data = pd.read_csv('Tatanic/data/train2.csv')
    test_data = pd.read_csv('Tatanic/data/test2.csv')
    test_df_org = pd.read_csv('Tatanic/data/test.csv')
    PassengerId = test_df_org['PassengerId']

    titanic_train_data_X = train_data.drop(['Survived'], axis=1)
    titanic_train_data_Y = train_data['Survived']
    titanic_test_data_X = test_data.drop(['Survived'], axis=1)

    #用模型筛选特征
    feature_to_pick = 30
    feature_top_n, feature_importance = get_top_n_features(
        titanic_train_data_X, titanic_train_data_Y, feature_to_pick)

    #用筛选出的特征构建训练集和测试集
    titanic_train_data_X = pd.DataFrame(titanic_train_data_X[feature_top_n])
    titanic_test_data_X = pd.DataFrame(titanic_test_data_X[feature_top_n])

    #stack第一层

    # Some useful parameters which will come in handy later on
    ntrain = titanic_train_data_X.shape[0]
    ntest = titanic_test_data_X.shape[0]
    SEED = 0  # for reproducibility
    NFOLDS = 7  # set folds for out-of-fold prediction
    kf = KFold(n_splits=NFOLDS, random_state=SEED, shuffle=False)

    rf = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt', max_depth=6,
                                min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)
    ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)
    et = ExtraTreesClassifier(
        n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)
    gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.008,
                                    min_samples_split=3, min_samples_leaf=2, max_depth=5, verbose=0)
    dt = DecisionTreeClassifier(max_depth=8)
    knn = KNeighborsClassifier(n_neighbors=2)
    svm = SVC(kernel='linear', C=0.025)

    #pandas转arrays:
    x_train = titanic_train_data_X.values
    x_test = titanic_test_data_X.values
    y_train = titanic_train_data_Y.values

   # Create our OOF train and test predictions. These base results will be used as new features
    rf_oof_train, rf_oof_test = get_out_fold(
        rf, x_train, y_train, x_test)  # Random Forest
    ada_oof_train, ada_oof_test = get_out_fold(
        ada, x_train, y_train, x_test)  # AdaBoost
    et_oof_train, et_oof_test = get_out_fold(
        et, x_train, y_train, x_test)  # Extra Trees
    gb_oof_train, gb_oof_test = get_out_fold(
        gb, x_train, y_train, x_test)  # Gradient Boost
    dt_oof_train, dt_oof_test = get_out_fold(
        dt, x_train, y_train, x_test)  # Decision Tree
    knn_oof_train, knn_oof_test = get_out_fold(
        knn, x_train, y_train, x_test)  # KNeighbors
    svm_oof_train, svm_oof_test = get_out_fold(
        svm, x_train, y_train, x_test)  # Support Vector

    #我们利用XGBoost，使用第一层预测的结果作为特征对最终的结果进行预测。
    x_train = np.concatenate((rf_oof_train, ada_oof_train, et_oof_train,
                              gb_oof_train, dt_oof_train, knn_oof_train, svm_oof_train), axis=1)
    x_test = np.concatenate((rf_oof_test, ada_oof_test, et_oof_test,
                             gb_oof_test, dt_oof_test, knn_oof_test, svm_oof_test), axis=1)

    gbm = XGBClassifier(n_estimators=2000, max_depth=4, min_child_weight=2, gamma=0.9, subsample=0.8,
                        colsample_bytree=0.8, objective='binary:logistic', nthread=-1, scale_pos_weight=1).fit(x_train, y_train)
    predictions = gbm.predict(x_test)

    StackingSubmission = pd.DataFrame(
        {'PassengerId': PassengerId, 'Survived': predictions})
    StackingSubmission.to_csv('StackingSubmission.csv', index=False, sep=',')

    #观察不同的学习曲线
    # RandomForest
    rf_parameters = {'n_jobs': -1, 'n_estimators': 500, 'warm_start': True, 'max_depth': 6, 'min_samples_leaf': 2,
                     'max_features': 'sqrt', 'verbose': 0}
    # AdaBoost

    ada_parameters = {'n_estimators': 500, 'learning_rate': 0.1}
    # ExtraTrees
    et_parameters = {'n_jobs': -1, 'n_estimators': 500,
                     'max_depth': 8, 'min_samples_leaf': 2, 'verbose': 0}
    # GradientBoosting
    gb_parameters = {'n_estimators': 500, 'max_depth': 5,
                     'min_samples_leaf': 2, 'verbose': 0}
    # DecisionTree
    dt_parameters = {'max_depth': 8}
    # KNeighbors
    knn_parameters = {'n_neighbors': 2}
    # SVM
    svm_parameters = {'kernel': 'linear', 'C': 0.025}
    # XGB
    gbm_parameters = {'n_estimators': 2000, 'max_depth': 4, 'min_child_weight': 2, 'gamma': 0.9, 'subsample': 0.8,
                      'colsample_bytree': 0.8, 'objective': 'binary:logistic', 'nthread': -1, 'scale_pos_weight': 1}
    title = "Learning Curves"
    plot_learning_curve(RandomForestClassifier(**rf_parameters), title, x_train, y_train,
                        cv=None,  n_jobs=4, train_sizes=[50, 100, 150, 200, 250, 350, 400, 450, 500])
    plt.show()
