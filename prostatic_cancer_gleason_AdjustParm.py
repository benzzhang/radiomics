import csv
import os

import numpy as np
import pandas as pd
import sklearn.svm
import yaml
import time
from matplotlib import pyplot as plt
from scipy.stats import levene, ttest_ind
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, scale, normalize, label_binarize
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score


def featurs_deal(csv_file, type):
    print('reading data from ', csv_file)

    data_ori = pd.read_csv(csv_file)
    header_row = next(csv.reader(open(csv_file)))
    str_idx = []
    # find the columns index where the type of cell is 'string'
    for idx, i in enumerate(data_ori.iloc[1, :]):
        if isinstance(i, str):
            str_idx.append(idx)

    data = data_ori
    for i in str_idx:
        data = data.drop(columns=[header_row[i]])

    # low level cases and high level cases in data are 59 and 116
    # make 'low:high' in testset is 1:1
    # (59-42):(116-98) = 17:18 in testset,
    # 42:98 in trainset.
    data_L = data[:][data['Label'] == 1]
    data_H = data[:][data['Label'] == 2]

    # shuffle
    data_L = data_L.sample(frac=1.0)
    data_H = data_H.sample(frac=1.0)

    data_train_L = data_L.iloc[:42, :]
    data_train_H = data_H.iloc[:98, :]

    data_test_L = data_L.iloc[42:, :]
    data_test_H = data_H.iloc[98:, :]

    # T test
    index = []
    for colName in data.columns[:]:
        if levene(data_train_L[colName], data_train_H[colName])[1] > 0.05:
            if ttest_ind(data_train_L[colName], data_train_H[colName])[1] < 0.05:
                # 独立样本T检验结果具有显著性差异(p > 0.05), 特征索引colName加入index
                index.append(colName)
        else:
            if ttest_ind(data_train_L[colName], data_train_H[colName], equal_var=False)[1] < 0.05:
                index.append(colName)

    print(len(index), 'features obtained after T test: ', index)

    # concat LABEL1 and LABEL2
    data_train_L = data_train_L[index]
    data_train_H = data_train_H[index]
    data_train = pd.concat([data_train_L, data_train_H])
    # shuffle
    data_train = shuffle(data_train)
    x_train = data_train[data_train.columns[1:]]

    data_test_L = data_test_L[index]
    data_test_H = data_test_H[index]
    data_test = pd.concat([data_test_L, data_test_H])
    data_test = shuffle(data_test)
    x_test = data_test[data_test.columns[1:]]

    # keep columns-name and normalize
    columns = x_train.columns
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # x_train = scaler.transform(x_train)
    x_train = pd.DataFrame(x_train, columns=columns)
    # x_test = scaler.transform(x_test)
    x_test = pd.DataFrame(x_test, columns=columns)
    y_train = data_train['Label']
    y_test = data_test['Label']
    if type == 2:
        # Label 0 -> 1 , 1 -> 2
        y_train = label_binarize(y_train, classes=[1, 2])
        y_train = y_train.ravel()
        y_test = label_binarize(y_test, classes=[1, 2])
        y_test = y_test.ravel()

    elif type == 3:
        # Label 0 ->  , 1 -> , 2 ->
        y_train = label_binarize(y_train, classes=[0, 1, 2])
        y_train = y_train.ravel()
        y_test = label_binarize(y_test, classes=[0, 1, 2])
        y_test = y_test.ravel()
    return x_train, y_train, x_test, y_test


def features_reduction(clf, data):
    import warnings
    warnings.filterwarnings('ignore')

    filter = clf['filter']
    x_train, y_train, x_test, y_test = data

    if filter == 'LASSO':
        alphas = np.logspace(-4, 1, 50)
        model_lassoCV = LassoCV(alphas=alphas, max_iter=100000, cv=5).fit(x_train, y_train)
        coef = pd.Series(model_lassoCV.coef_, index=x_train.columns)
        print('α - %.4f , %s %d ' % (model_lassoCV.alpha_, 'Lasso picked', sum(coef != 0)))
        index = coef[coef != 0].index
        x_train = x_train[index]
        x_test = x_test[index]

    elif filter == 'RFE':
        if clf['RFE_based'] == 'LR':
            model = LogisticRegression()
            rfe = RFE(estimator=model, n_features_to_select=10)
            selector = rfe.fit(x_train, y_train)

            support = pd.Series(selector.support_, index=x_train.columns)
            index = support[support == True].index
            x_train = x_train[index]
            x_test = x_test[index]


        elif clf['RFE_based'] == 'SVM':
            model = SVC(kernel='linear', probability=True, random_state=42)
            rfe = RFE(estimator=model, n_features_to_select=10)
            selector = rfe.fit(x_train, y_train)

            # method 1:
            support = pd.Series(selector.support_, index=x_train.columns)
            index = support[support == True].index
            x_train = x_train[index]
            x_test = x_test[index]

            # method 2:
            # type of x_train: DataFrame -> np.array
            # x_train = selector.transform(x_train)
            # x_test = selector.transform(x_test)

    # PCA - necessary ?
    if x_train.shape[1] > 10:
        pca = PCA(n_components=10)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)

    return x_train, y_train, x_test, y_test


def classify(clf, data):
    # StratifiedKFold - 分层抽样：训练集，测试集中各类别样本的比例与原始数据集中相同
    kflod = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    x_train, y_train, x_test, y_test = data

    classifier = None
    parameters_grid = None
    model = clf['model']
    if model == 'RF':
        classifier = RandomForestClassifier(n_jobs=-1, random_state=42)

        parameters_grid = [
            {
                'n_estimators': [int(x) for x in np.linspace(start=10, stop=200, num=20)],
                # 'max_features': ['auto', 'sqrt'],
                'max_depth': [int(x) for x in np.linspace(5, 105, num=25)],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        ]

    elif model == 'KNN':
        classifier = KNeighborsClassifier()

        parameters_grid = [
            {
                "n_neighbors": np.arange(1, 20, 1),
                "weights": ["uniform"]
            },
            {
                "n_neighbors": np.arange(2, 20, 1),
                "weights": ["distance"],
                "p": [i for i in range(1, 6)]
            }
        ]

    elif model == 'SVM':
        classifier = SVC(kernel='linear',
                         probability=True,
                         class_weight='balanced',
                         random_state=0)

        parameters_grid = [
            {
                "gamma": [0.0001, 0.0005, 0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 10, 100],
                "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
            }
        ]

    elif model == 'Bayes':
        classifier = GaussianNB()
        classifier.fit(x_train, y_train)
        pre = classifier.predict(x_test)
        accuracy, precision, recall, f1 = accuracy_score(y_test, pre), \
                                          precision_score(y_test, pre), \
                                          recall_score(y_test, pre), \
                                          f1_score(y_test, pre)

        return accuracy, precision, recall, f1

    if parameters_grid is not None:
        grid_search = GridSearchCV(estimator=classifier,
                                   param_grid=parameters_grid,
                                   n_jobs=-1,
                                   cv=kflod)
        grid_result = grid_search.fit(x_train, y_train)
        # print('Best ' + model + ' Train Score: %.4f using %s' % (grid_result.best_score_, grid_search.best_params_))
        print('Best ' + model + ' Model: ', grid_search.best_estimator_)
        pre_proba = grid_search.best_estimator_.predict_proba(x_test)
        pre = grid_result.best_estimator_.predict(x_test)
        print(x_test)
        print(y_test)
        print(pre_proba)
        print(pre)
        accuracy, precision, recall, f1 = accuracy_score(y_test, pre), \
                                          precision_score(y_test, pre), \
                                          recall_score(y_test, pre), \
                                          f1_score(y_test, pre)

        print(' accuracy:%.2f \n precision:%.2f \n recall:%.2f \n f1:%.2f' % (accuracy, precision, recall, f1))
        # test_score = accuracy
        # test_score = grid_search.best_estimator_.fit(x_train, y_train).score(x_test, y_test)
        # print(model + ' Test Score: ', test_score)

        return accuracy, precision, recall, f1


def main(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    modality = config['object']['series']
    roi = config['object']['roi']
    type = config['object']['type']
    classifier = config['classifier']

    csv_file = './R' + modality + '-' + roi + '.csv'
    data = featurs_deal(csv_file, type)
    data = features_reduction(classifier, data)
    accuracy, precision, recall, f1 = classify(classifier, data)

    file = os.path.join(config['save_path']['log'], 'log.txt')
    if not os.path.isfile(file):
        f = open(file, mode="w", encoding="utf-8")
        f.write('%s%s%s%s%s%s%s\n' % ('features'.center(15, ' '),
                                      'model'.center(15, ' '),
                                      'select method'.center(15, ' '),
                                      'accuracy'.center(15, ' '),
                                      'precision'.center(15, ' '),
                                      'recall'.center(15, ' '),
                                      'f1'.center(15, ' ')))
        f.close()

    with open(file, 'a') as f:
        method = classifier['filter']
        if classifier['filter'] == 'RFE':
            method = classifier['filter'] + '(' + classifier['RFE_based'] + ')'
        f.write('%s%s%s%s%s%s%s%s\n' % (csv_file.center(15, ' '),
                                        classifier['model'].center(15, ' '),
                                        method.center(15, ' '),
                                        str(np.around(accuracy, 4)).center(15, ' '),
                                        str(np.around(precision, 4)).center(15, ' '),
                                        str(np.around(recall, 4)).center(15, ' '),
                                        str(np.around(f1, 4)).center(15, ' '),
                                        time.strftime('%Y-%m-%d %H:%M', time.localtime()).center(20, ' ')))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prostatic Cancer Classifier on Gleason')
    parser.add_argument('--config-file', type=str, default='./config_AdjustParm.yaml')
    args = parser.parse_args()
    main(args.config_file)
