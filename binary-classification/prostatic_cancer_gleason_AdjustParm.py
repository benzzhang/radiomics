import csv
import os

import numpy as np
import pandas as pd
import sklearn.svm
import yaml
import time
import seaborn as sns
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
from scipy.stats import levene, ttest_ind
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, scale, normalize, label_binarize
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

seed = 42


def featurs_deal(csv_file):
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

    # the number of low and high level cases are A and B
    # make 'low:high' in testset to be 1:1
    # make train:test to be 7:3
    # (59-26):(116-26) = 33:90 in trainset,
    # 26:26 in trainset.

    data_L = data[:][data['Label'] == 1]
    data_H = data[:][data['Label'] == 2]

    # shuffle
    data_L = data_L.sample(frac=1.0, random_state=seed)
    data_H = data_H.sample(frac=1.0, random_state=seed)

    data_train_L = data_L.iloc[:33, :]
    data_train_H = data_H.iloc[:90, :]

    data_test_L = data_L.iloc[33:, :]
    data_test_H = data_H.iloc[90:, :]

    # T test
    # index = []
    # for colName in data.columns[:]:
    #     if levene(data_train_L[colName], data_train_H[colName])[1] > 0.05:
    #         if ttest_ind(data_train_L[colName], data_train_H[colName])[1] < 0.05:
    #             # 独立样本T检验结果具有显著性差异(p > 0.05), 特征索引colName加入index
    #             index.append(colName)
    #     else:
    #         if ttest_ind(data_train_L[colName], data_train_H[colName], equal_var=False)[1] < 0.05:
    #             index.append(colName)
    #
    # print(len(index), 'features obtained after T test: ', index)

    # concat LABEL1 and LABEL2
    # data_train_L = data_train_L[index]
    # data_train_H = data_train_H[index]
    data_train = pd.concat([data_train_L, data_train_H])
    # shuffle
    data_train = shuffle(data_train, random_state=seed)
    x_train = data_train[data_train.columns[1:]]

    # data_test_L = data_test_L[index]
    # data_test_H = data_test_H[index]
    data_test = pd.concat([data_test_L, data_test_H])
    data_test = shuffle(data_test, random_state=seed)
    x_test = data_test[data_test.columns[1:]]

    # keep columns-name and normalize
    columns = x_train.columns
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = pd.DataFrame(x_train, columns=columns)
    x_test = pd.DataFrame(x_test, columns=columns)
    y_train = data_train['Label']
    y_test = data_test['Label']

    # 原本标签1&2 变成[0]&[1]
    y_train = label_binarize(y_train, classes=[1, 2])
    y_train = y_train.ravel()
    y_test = label_binarize(y_test, classes=[1, 2])
    y_test = y_test.ravel()

    print('original features nums:', x_train.shape[1])
    # drop columns include 'NaN'
    is_NaN = x_train.isnull().any()  # False - 无缺失值
    if sum(is_NaN[is_NaN == True]) > 0:
        # print index name
        # NaN_index = is_NaN[is_NaN == True].index
        # print(NaN_index)

        # drop these columns include 'NaN'
        # not_NaN_index = is_NaN[is_NaN == False].index
        # x_train = x_train[not_NaN_index]
        # x_test = x_test[not_NaN_index]

        # fill 'NaN' with mean()
        x_train.fillna(x_train.mean(), inplace=True)
        x_test.fillna(x_test.mean(), inplace=True)

    # handle outlier. dataset must be normally distributed
    # for i in x_train.columns:
    #     mean_data = x_train[i].mean()
    #     std_data = x_train[i].std()
    #     cols = (x_train[i] > mean_data+3*std_data) | (x_train[i] < mean_data-3*std_data)
    #     if cols.any():
    #         print(i+' has outlier '+sum(cols == True))

    # oversampling
    print("Before OverSampling, counts of label '1' in trainset: {}".format(sum(y_train == 1)))
    print("Before OverSampling, counts of label '0' in trainset: {}".format(sum(y_train == 0)))
    print("oversampling ...")

    sm = SMOTE(random_state=2)
    x_train_res, y_train_res = sm.fit_sample(x_train, y_train)

    print("After OverSampling, counts of label '1' in trainset: {}".format(sum(y_train_res == 1)))
    print("After OverSampling, counts of label '0' in trainset: {}".format(sum(y_train_res == 0)))

    # shuffle the resampled trainset
    x_train_res.insert(loc=0, column='Label', value=y_train_res)
    x_train_res = x_train_res.sample(frac=1.0, random_state=seed)
    x_train = x_train_res[x_train_res.columns[1:]]
    y_train = x_train_res['Label'].ravel()

    return x_train, y_train, x_test, y_test


def features_reduction(clf, data):
    filter = clf['filter']
    x_train, y_train, x_test, y_test = data

    if filter == 'LASSO':
        lassoCV = LassoCV(max_iter=10000, cv=5, random_state=seed).fit(x_train, y_train)
        coef = pd.Series(lassoCV.coef_, index=x_train.columns)
        print('α - %.4f , %s %d ' % (lassoCV.alpha_, 'Lasso picked', sum(coef != 0)))

        # Draw MSE with different alphas
        plt.figure()
        # 随着alpha值的变化，均方误差的变化曲线
        plt.plot(lassoCV.alphas_, lassoCV.mse_path_, ':')
        # 验证过程中，随着alpha值的变化，均方误差的平均曲线
        plt.plot(lassoCV.alphas_, lassoCV.mse_path_.mean(axis=-1),
                 label='Average MSE Across Folds', linewidth=2, color='cyan')
        # 每次验证系统认为的最合适的alpha值
        plt.axvline(lassoCV.alpha_, linestyle='--', label='CV Estimate of Best alpha', color='blue')
        plt.semilogx()
        plt.legend()
        plt.xlabel('alpha')
        plt.ylabel('Mean Square Error')
        plt.axis('tight')
        plt.savefig('../output/lassoCV_picked_{}.png'.format(sum(coef != 0)))
        plt.show()

        index = coef[coef != 0].index
        x_train = x_train[index]
        x_test = x_test[index]

    elif filter == 'RFE':
        if clf['RFE_based'] == 'LR':
            model = LogisticRegression(random_state=seed)
            rfe = RFE(estimator=model, n_features_to_select=10)
            selector = rfe.fit(x_train, y_train)

            support = pd.Series(selector.support_, index=x_train.columns)
            index = support[support == True].index
            x_train = x_train[index]
            x_test = x_test[index]

        elif clf['RFE_based'] == 'RFC':
            model = RandomForestClassifier()
            rfe = RFE(estimator=model, n_features_to_select=10)
            selector = rfe.fit(x_train, y_train)

            support = pd.Series(selector.support_, index=x_train.columns)
            index = support[support == True].index
            x_train = x_train[index]
            x_test = x_test[index]

        elif clf['RFE_based'] == 'SVM':
            model = SVC(kernel='linear', probability=True, random_state=seed)
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

    elif filter == 'PCA':
        print('applying PCA ...')
        pca = PCA(n_components=0.95)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)

    # Spearman rank correlation test
    x_train_df = pd.DataFrame(x_train)
    corr_matrix = x_train_df.corr(method='spearman').abs()
    # draw heatmap
    sns.set(font_scale=1.0)
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr_matrix, cmap="YlOrRd", square=True, ax=ax)
    f.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=800)

    print('features nums after dimensionality reduction:', x_train.shape[1])

    return x_train, y_train, x_test, y_test


def classify(clf, data):
    # StratifiedKFold - 分层抽样：训练集，测试集中各类别样本的比例与原始数据集中相同
    kflod = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    x_train, y_train, x_test, y_test = data

    # 调参，确定n_estimators大概范围
    # scorel = []
    # for i in range(45, 75, 1):
    #     rfc = RandomForestClassifier(n_estimators=i + 1, n_jobs=-1, random_state=42)
    #     score = cross_val_score(rfc, x_train, y_train, cv=5).mean()
    #     scorel.append(score)
    #     print(max(scorel), (scorel.index(max(scorel)) * 1) + 45)
    #     plt.figure(figsize=[20, 5])
    # plt.plot(range(46, 76, 1), scorel)
    # plt.show()  # list.index([object])，返回这个object在列表list中的索引

    classifier = None
    parameters_grid = None
    model = clf['model']
    if model == 'RF':
        classifier = RandomForestClassifier(n_jobs=-1)

        parameters_grid = {
            'n_estimators': [int(x) for x in np.arange(30, 50)],
            # 'max_depth': [3, 4, 5, 6, 7, 8],
            # 'max_features': ["auto", "sqat", "log2"],
            # 'class_weight': [None, "balanced"]
        }


    elif model == 'KNN':
        classifier = KNeighborsClassifier(n_jobs=-1)

        parameters_grid = {
            # "n_neighbors":  [int(x) for x in np.arange(5, 20)],
            "leaf_size": [int(x) for x in np.arange(1, 10)]
        }


    elif model == 'LR':
        classifier = LogisticRegression(n_jobs=-1)

        parameters_grid = {
            "C": [int(x) for x in np.arange(1, 20)],
        }


    elif model == 'SVM':
        classifier = SVC(probability=True, gamma='auto')

        parameters_grid = {
            # "kernel": ['linear', 'rbf', 'poly'],
            # "C": [0.0001, 0.001, 0.01, 0.5, 0.1, 1, 5, 10, 100]
            "C": [0.5, 0.6 , 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        }


    elif model == 'Bayes':
        classifier = GaussianNB()

        parameters_grid = {
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        }

    # GridSearch
    if parameters_grid is not None:
        grid_search = GridSearchCV(estimator=classifier,
                                   param_grid=parameters_grid,
                                   cv=kflod,
                                   scoring='accuracy'  # tp+tn / tp+tn+fp+fn
                                   # Positive and Negative are both important
                                   )
        # grid_search = RandomizedSearchCV(estimator=classifier,
        #                                  param_distributions=parameters_grid,
        #                                  n_iter=30)
        grid_result = grid_search.fit(x_train, y_train)
        print('Best ' + model + ' Train Score: %.4f using %s' % (grid_result.best_score_, grid_search.best_params_))
        print('Best ' + model + ' Model: ', grid_search.best_estimator_)
        pre_proba = grid_search.best_estimator_.predict_proba(x_test)
        pre = grid_result.best_estimator_.predict(x_test)
        # print(x_test)
        print(pre_proba)
        print(y_test)
        print(pre)
        print(grid_result.best_estimator_.score(x_test, y_test))
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
    import warnings
    warnings.filterwarnings('ignore')

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    modality = config['object']['series']
    roi = config['object']['roi']
    classifier = config['classifier']

    csv_file = 'R' + modality + '-' + roi + '.csv'
    data = featurs_deal(os.path.join(config['file_path']['csv'], csv_file))
    data = features_reduction(classifier, data)
    result = classify(classifier, data)

    file = os.path.join(config['file_path']['log'], 'log.txt')
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
                                        str(np.around(result[0], 4)).center(15, ' '),
                                        str(np.around(result[1], 4)).center(15, ' '),
                                        str(np.around(result[2], 4)).center(15, ' '),
                                        str(np.around(result[3], 4)).center(15, ' '),
                                        time.strftime('%Y-%m-%d %H:%M', time.localtime()).center(20, ' '),))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prostatic Cancer Classifier on Gleason')
    parser.add_argument('--config-file', type=str, default='./config_AdjustParm.yaml')
    args = parser.parse_args()
    main(args.config_file)
