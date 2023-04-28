'''
@Author     : Jian Zhang
@Init Date  : 2022-10-28
@File       : prostatic_cancer_xgboost.py
@IDE        : PyCharm
@Description: 根据mode_dict的组合进行train&test, 对最优模型的前nums个特征制图并输出到.csv作为后续R的相关性分析
'''

import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, auc, roc_curve, confusion_matrix
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
from matplotlib import rcParams


def remove_str_data(data, header_row):
    str_idx = []
    # .iloc[1,:] 索引为1的行
    for idx, i in enumerate(data.iloc[1, :]):
        if isinstance(i, str):
            str_idx.append(idx)
    for i in str_idx:
        data = data.drop(columns=[header_row[i]])
    return data


def data_read(path, is_order=False):
    data_ori = pd.read_csv(path)
    header_row = next(csv.reader(open(path)))
    if is_order:
        data_ori['ID'] = data_ori['ID'].apply(lambda x: int(x[1:]))
        data_ori = data_ori.sort_values(by=['ID'], ascending=True)
    data_1 = remove_str_data(data_ori, header_row)
    for item in data_1.columns:
        modality = path.split('/')[-1][1]
        if not(item=='Label' or item=='ID'):
            data_1.rename(columns={item: item+'_'+modality}, inplace=True)
    # data_1.drop(columns=['ID'], inplace=True)
    return data_1


def two_class_data_split(data):
    data_L = data[:][data['Label'] == 1]
    data_H = data[:][data['Label'] == 2]
    data_all = pd.concat([data_L, data_H])
    data_x = data_all[data.columns[1:]]
    data_y = data_all['Label']
    # %%
    std = StandardScaler()
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=7)
    y_train = label_binarize(y_train, classes=[1, 2])
    y_train = y_train.ravel()
    y_test = label_binarize(y_test, classes=[1, 2])
    y_test = y_test.ravel()
    x_train_std = std.fit_transform(x_train)
    y_train_std = std.fit_transform(y_train)
    return x_train_std, x_test, y_train_std, y_test


def data_concat(path1, path2):
    data1 = data_read(path1, True)
    data2 = data_read(path2, True)

    data2 = data2[data2.columns[2:]]
    data_res = pd.concat([data1, data2], axis=1)
    return data_res


def data_concat_three(path1, path2, path3):
    data1 = data_read(path1, True)
    data2 = data_read(path2, True)
    data3 = data_read(path3, True)

    data2 = data2[data2.columns[2:]]
    data_res_1 = pd.concat([data1, data2], axis=1)
    data3 = data3[data3.columns[2:]]
    data_res_2 = pd.concat([data_res_1, data3], axis=1)

    return data_res_2


def data_add(path_1, path_2):
    data1 = data_read(path_1, True)
    data2 = data_read(path_2, True)
    # for item in data2.columns:
    #     if item in data1.columns:
    #         if item != "Label":
    #             data1[item] = data1[item]
    # data1.drop(columns=['ID'], inplace=True)
    return data1


def data_train_test(data, random_seed=7):
    data_L = data[:][data['Label'] == 1]
    data_H = data[:][data['Label'] == 2]
    data_all = pd.concat([data_L, data_H])
    # 去掉data_all前两列'Label'和'ID'
    data_x = data_all[data_all.columns[2:]]
    data_y = data_all['Label']

    std = StandardScaler()

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=random_seed)
    y_train = label_binarize(y_train, classes=[1, 2])
    y_train = y_train.ravel()
    y_test = label_binarize(y_test, classes=[1, 2])
    y_test = y_test.ravel()

    # x_train = mean_norm(x_train)
    # x_test = mean_norm(x_test)
    # x_train = std.fit_transform(x_train)
    # x_test = std.fit_transform(x_test)

    return x_train, x_test, y_train, y_test


def mean_norm(df_input):
    return df_input.apply(lambda x: (x - x.mean()) / x.std(), axis=0)


def minmax_norm(df_input):
    return (df_input - df_input.min()) / (df_input.max() - df_input.min())


if __name__ == '__main__':

    # 找到最优模型后填入↓ , 例如 "T2WI+ADC+DWI-P"
    # best_model = "T2WI+ADC+DWI-P"
    best_model = None

    path1 = "../data2/RT-C.csv"
    path2 = "../data2/RT-M.csv"
    path3 = "../data2/RT-P.csv"
    path4 = "../data2/RA-C.csv"
    path5 = "../data2/RA-M.csv"
    path6 = "../data2/RA-P.csv"
    path7 = "../data2/RD-C.csv"
    path8 = "../data2/RD-M.csv"
    path9 = "../data2/RD-P.csv"

    path10 = "../data2/merge_ADC_CP.csv"
    path11 = "../data2/merge_DWI_CP.csv"
    path12 = "../data2/merge_T2_CP.csv"

    mode_dict = {
                 "T2WI-C": [path1],
                 "T2WI-M": [path2],
                 "T2WI-P": [path3],
                 # "T2WI-C+P": [path12],
                 "ADC-C": [path4],
                 "ADC-M": [path5],
                 "ADC-P": [path6],
                 # "ADC-C+P": [path10],
                 "DWI-C": [path7],
                 "DWI-M": [path8],
                 "DWI-P": [path9],
                 # "DWI-C+P": [path11],
                 "T2WI+DWI-C": [path1, path7],
                 "T2WI+DWI-M": [path2, path8],
                 "T2WI+DWI-P": [path3, path9],
                 # "T2WI+DWI-C+P": [path12, path11],
                 "ADC+DWI-C": [path4, path7],
                 "ADC+DWI-M": [path5, path8],
                 "ADC+DWI-P": [path6, path9],
                 # "ADC+DWI-C+P": [path10, path11],
                 "T2WI+ADC-C": [path1, path4],
                 "T2WI+ADC-M": [path2, path5],
                 "T2WI+ADC-P": [path3, path6],
                 # "T2WI+ADC-C+P": [path12, path10],
                 "T2WI+ADC+DWI-C": [path1, path4, path7],
                 "T2WI+ADC+DWI-M": [path2, path5, path8],
                 "T2WI+ADC+DWI-P": [path3, path6, path9],
                 # "T2WI+ADC+DWI-C+P": [path12, path11, path10]
                 }

    global data
    for key in mode_dict.keys():
        mode = key
        if len(mode_dict[key]) == 1:
            path = mode_dict[key][0]
            data = data_read(path, True)
        elif len(mode_dict[key]) == 2:
            path1 = mode_dict[key][0]
            path2 = mode_dict[key][1]
            data = data_concat(path1, path2)
        elif len(mode_dict[key]) == 3:
            path1 = mode_dict[key][0]
            path2 = mode_dict[key][1]
            path3 = mode_dict[key][2]
            data = data_concat_three(path1, path2, path3)

        nums = 972
        x_train, x_test, y_train, y_test = data_train_test(data, random_seed=nums)

        model = XGBClassifier(learning_rate=0.001,
                              eta=0.1,
                              n_estimators=300,  # 树的个数
                              max_depth=10,  # 树的深度
                              min_child_weight=1,  # 叶子节点最小权重
                              gamma=0.1,
                              reg_alpha=0.01,
                              reg_lambda=0.9,
                              subsample=0.8,  # 随机选择80%样本建立决策树
                              objective='binary:logistic',  # 指定损失函数
                              scale_pos_weight=1,  # 解决样本个数不平衡的问题
                              random_state=3407  # 随机数
                              )

        # 训练
        model.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=20, verbose=True)

        # 重要特征作图, 只对 best_model 进行作图
        # 知道最重要特征后，注释掉 mode_dict 中除 best_model 外的其他组合，减少for循环用时
        # 取得重要特征值保存到 .csv, 使用R绘制相关性热图
        if mode == best_model:
            # 字体参数
            config = {"font.family": 'Arial'}
            rcParams.update(config)
            # 设置画布尺寸，默认为(6.4, 4.8)
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            # 进入'plot_importance'源码加一个返回值labels; 修改line94参数
            # return ax, labels
            # 取得前nums个重要特征，根据数值调整画布尺寸到合适
            nums = 50
            _, important_labels = plot_importance(model,
                                                  height=0.5,
                                                  ax=ax,
                                                  max_num_features=nums,
                                                  importance_type='gain',
                                                  show_values=True)
            plt.tight_layout()
            plt.rcParams['savefig.dpi'] = 300
            plt.rcParams['figure.dpi'] = 300
            plt.savefig('../dataAnalysis/importance_{}.tiff'.format(nums))
            plt.show()

            important_labels_list = list(important_labels)
            print(important_labels_list)

            important_train = x_train
            important_train_columns = x_train.columns
            for i in important_train_columns:
                if i not in important_labels_list:
                    important_train = important_train.drop(i, axis=1)
            important_train.to_csv('../important_features_{}.csv'.format(nums))

        # 训练集结果
        y_train_pred_proba = model.predict_proba(x_train)
        y_train_pred = [(y[1] >= 0.5) * 1 for y in y_train_pred_proba]
        # 训练集指标
        accuracy_train = accuracy_score(y_train, y_train_pred)
        pre_train = precision_score(y_train, y_train_pred)
        recall_train = recall_score(y_train, y_train_pred)
        f1_train = f1_score(y_train, y_train_pred)
        auc_train = roc_auc_score(y_train, y_train_pred_proba[:, 1])
        # a = confusion_matrix(y_test, y_pred)
        # spec = a[0][0] / (a[0][1] + a[0][0])

        print("训练集Acc: %.2f%%" % (accuracy_train * 100.0))
        print("训练集Pre: %.2f%%" % (pre_train * 100.0))
        print("训练集Recall: %.2f%%" % (recall_train * 100.0))
        print("训练集F1: %.2f%%" % (f1_train * 100.0))
        print("训练集AUC: %.2f%%" % (auc_train * 100.0))

        # 测试集结果
        y_test_pred_proba = model.predict_proba(x_test)
        y_test_pred = [(y[1] >= 0.5) * 1 for y in y_test_pred_proba]
        # 测试集指标
        accuracy_test = accuracy_score(y_test, y_test_pred)
        pre_test = precision_score(y_test, y_test_pred)
        recall_test = recall_score(y_test, y_test_pred)
        f1_test = f1_score(y_test, y_test_pred)
        auc_test = roc_auc_score(y_test, y_test_pred_proba[:, 1])
        # a = confusion_matrix(y_test, y_pred)
        # spec = a[0][0] / (a[0][1] + a[0][0])

        print("测试集Acc: %.2f%%" % (accuracy_test * 100.0))
        print("测试集Pre: %.2f%%" % (pre_test * 100.0))
        print("测试集Recall: %.2f%%" % (recall_test * 100.0))
        print("测试集F1: %.2f%%" % (f1_test * 100.0))
        print("测试集AUC: %.2f%%" % (auc_test * 100.0))

        # perm_importance = permutation_importance(model, x_test, y_test)
        # sorted_idx = perm_importance.importances_mean.argsort()
        # plt.barh(x_test.feature_names[sorted_idx], perm_importance.importances_mean[sorted_idx])

        # NPV: TN / (TN+FN)
        TN, FN = 0, 0
        for y1,y2 in zip(list(y_test), y_test_pred):
            if y2 == 0:
                if y1 == y2:
                    TN += 1
                else:
                    FN += 1
        NPV = TN / (TN+FN+1e-10)

        with open('../experiment-CMP-2.txt', 'a+') as f:
            f.write(mode)
            f.write('\n')
            f.write('<train_data>\n')
            f.write(" ".join(map(str, y_train)))
            f.write('\n')
            f.write(" ".join(map(str, y_train_pred)) + "\t")
            f.write('\n')
            f.write(" ".join(map(str, y_train_pred_proba[:, 1])) + "\t")
            f.write('\n')
            f.write(" accuracy: %.2f%%" % (accuracy_train * 100.0) + "\n")
            f.write("precision: %.2f%%" % (pre_train * 100.0) + "\n")
            f.write("   recall: %.2f%%" % (recall_train * 100.0) + "\n")
            f.write("       F1: %.2f%%" % (f1_train * 100.0) + "\n")
            f.write("      AUC: %.2f%%" % (auc_train * 100.0) + "\n")
            f.write('\n')
            f.write('<test_data>\n')
            f.write(" ".join(map(str, y_test)) + "\t")
            f.write('\n')
            f.write(" ".join(map(str, y_test_pred)) + "\t")
            f.write('\n')
            f.write(" ".join(map(str, y_test_pred_proba[:, 1])) + "\t")
            f.write('\n')
            f.write(" accuracy: %.2f%%" % (accuracy_test * 100.0) + "\n")
            f.write("precision: %.2f%%" % (pre_test * 100.0) + "\n")
            f.write("   recall: %.2f%%" % (recall_test * 100.0) + "\n")
            f.write("       F1: %.2f%%" % (f1_test * 100.0) + "\n")
            f.write("      AUC: %.2f%%" % (auc_test * 100.0) + "\n")
            # f.write("      NPV: %.2f%%" % (round(NPV,2) * 100.0) + "\n")
            f.write('*----------------------------------------------------------------------------------------------------------\n')

    print("done!")
