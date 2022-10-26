'''
@Author     : Jian Zhang
@Init Date  : 2022-10-26 17:03
@File       : dataAnalysis.py
@IDE        : PyCharm
@Description: For 'ROC Curve' & 'Delong Test' & 'HeatMap'
'''

import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import scipy.stats as st
import seaborn as sns
import xlrd

# color_map: Referenced from https://blog.csdn.net/Bit_Coders/article/details/121383126
def RGB_to_Hex(rgb):
    """
    RGB格式颜色转换为16进制颜色格式
    Args:
        rgb: tuple

    Returns:
        color: str
    """
    RGB = list(rgb)
    color = '#'
    for i in RGB:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color

def generate_colors(N=7, colormap='hsv'):
    """
    生成颜色列表
    Args:
        N: 生成颜色列表中的颜色个数
        colormap: plt中的色表，如'cool'、'autumn'等

    Returns:
        rgb_list: list, 每个值(r,g,b)在0~255范围
        hex_list: list, 每个值为十六进制颜色码类似：#FAEBD7
    """
    step = max(int(255/N),1)
    cmap = plt.get_cmap(colormap)
    rgb_list = []
    hex_list = []
    for i in range(N):
        id = step*i # cmap(int)->(r,g,b,a) in 0~1
        id = 255 if id>255 else id
        rgba_color = cmap(id)
        rgb = [int(d*255) for d in rgba_color[:3]]
        rgb_list.append(tuple(rgb))
        hex_list.append(RGB_to_Hex(rgb))
    return rgb_list,hex_list

def config_show(type, exp):
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像
    plt.ylim([-0.05, 1.05])
    font1 = {'size':10}
    if type=='ROC':
        plt.xlabel('False Positive Rate',font1)
        plt.ylabel('True Positive Rate',font1)
        plt.title('ROC Curve')
    elif type=='PR':
        plt.xlabel('Recall',font1)
        plt.ylabel('Precision',font1)
        plt.title('PR Curve')
    plt.legend(loc="lower right")
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_color('gray')
    ax.spines['top'].set_color('gray')
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.savefig('./dataAnalysis/{}_{}.tiff'.format(type, exp))
    plt.show()

# Delong Test: Referenced from https://blog.csdn.net/weixin_42404713/article/details/120146245
class DelongTest():
    def __init__(self, preds1, preds2, label, threshold=0.05):
        '''
        preds1:the output of model1
        preds2:the output of model2
        label :the actual label
        '''
        self._preds1 = preds1
        self._preds2 = preds2
        self._label = label
        self.threshold = threshold
        self._show_result()

    def _auc(self, X, Y) -> float:
        return 1 / (len(X) * len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])

    def _kernel(self, X, Y) -> float:
        '''
        Mann-Whitney statistic
        '''
        return .5 if Y == X else int(Y < X)

    def _structural_components(self, X, Y) -> list:
        V10 = [1 / len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]
        V01 = [1 / len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]
        return V10, V01

    def _get_S_entry(self, V_A, V_B, auc_A, auc_B) -> float:
        return 1 / (len(V_A) - 1) * sum([(a - auc_A) * (b - auc_B) for a, b in zip(V_A, V_B)])

    def _z_score(self, var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B) / ((var_A + var_B - 2 * covar_AB) ** (.5) + 1e-8)

    def _group_preds_by_label(self, preds, actual) -> list:
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y

    def _compute_z_p(self):
        X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)
        X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)

        V_A10, V_A01 = self._structural_components(X_A, Y_A)
        V_B10, V_B01 = self._structural_components(X_B, Y_B)

        auc_A = self._auc(X_A, Y_A)
        auc_B = self._auc(X_B, Y_B)

        # Compute entries of covariance matrix S (covar_AB = covar_BA)
        var_A = (self._get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_A01, auc_A,auc_A) * 1 / len(V_A01))
        var_B = (self._get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1 / len(V_B10) + self._get_S_entry(V_B01, V_B01, auc_B,auc_B) * 1 / len(V_B01))
        covar_AB = (self._get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_B01,auc_A,auc_B) * 1 / len(V_A01))

        # Two tailed test
        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)
        p = st.norm.sf(abs(z)) * 2

        return z, p

    def _show_result(self):
        z, p = self._compute_z_p()
        print(f"z score = {z:.5f};\np value = {p:.5f};")
        if p < self.threshold:
            print("There is a significant difference")
        else:
            print("There is NO significant difference")


if __name__ == '__main__':

    # 字体参数
    config = {
        "font.family":'Arial',
    }
    rcParams.update(config)

    # 保存实验结果的.txt
    results = 'best_CMP.txt'
    #--------------Prepare Data for drawing--------------#
    seriesWithROI, AUCs = [], []
    label_train, preds_train, label_test, preds_test = [], [], [], []

    with open(results, 'r') as f:
        for i in  np.arange(len(open(results).readlines())):
            line = f.readline()
            if i%21 == 0:
                seriesWithROI.append(line.strip())
            if i%21 == 19:
                AUCs.append(line.strip())
            if i%21 == 2:
                label_train.append(line.strip())
            if i%21 == 4:
                preds_train.append(line.strip())

            if i%21 == 12:
                label_test.append(line.strip())
            if i%21 == 14:
                preds_test.append(line.strip())

    color_map_RGB, color_map_HEX = generate_colors(len(seriesWithROI),'gist_ncar')
    print('color_map:', color_map_HEX)
    #--------------Prepare Data for drawing--------------#

    #--------------Draw ROC in Train&Test--------------#

    # for roiName, color, label, pred in zip(seriesWithROI, color_map_HEX, label_train, preds_train):
    #     labelList = [int(i) for i in label.split(' ')]
    #     predList = [float(i) for i in pred.split(' ')]
    #     fpr, tpr, threshold = metrics.roc_curve(labelList, predList)
    #     plt.plot(fpr, tpr, '--', color=color, label='{}(AUC = {:.2f})'.format(roiName, auc(fpr, tpr)), lw=2)
    # plt.plot([0, 1], [0, 1], '--',color='#000000', lw=1)
    # config_show('ROC', 'Train_'+ results.split('.')[0])

    # plt.cla()
    for roiName, color, label, pred in zip(seriesWithROI, color_map_HEX, label_test, preds_test):
        labelList = [int(i) for i in label.split(' ')]
        predList = [float(i) for i in pred.split(' ')]
        fpr, tpr, threshold = metrics.roc_curve(labelList, predList)
        plt.plot(fpr, tpr, '--', color=color, label='{}(AUC = {:.2f})'.format(roiName, auc(fpr, tpr)), lw=2)
    plt.plot([0, 1], [0, 1], '--',color='#000000', lw=1)
    config_show('ROC', 'Test_'+ results.split('.')[0])
    #--------------Draw ROC in Train&Test--------------#

    #--------------Draw PR in Train&Test--------------#

    # plt.cla()
    # for roiName, color, label, pred in zip(seriesWithROI, color_map_HEX, label_train, preds_train):
    #     labelList = [int(i) for i in label.split(' ')]
    #     predList = [float(i) for i in pred.split(' ')]
    #     fpr, tpr, threshold = metrics.roc_curve(labelList, predList)
    #     plt.plot(fpr, tpr, '--', color=color, label='{}(AUC = {:.2f})'.format(roiName, auc(fpr, tpr)), lw=2)
    # config_show('PR', 'Train_'+ results.split('.')[0])

    # plt.cla()
    for roiName, color, label, pred in zip(seriesWithROI, color_map_HEX, label_test, preds_test):
        labelList = [int(i) for i in label.split(' ')]
        predList = [float(i) for i in pred.split(' ')]
        precision, recall, threshold = metrics.precision_recall_curve(labelList, predList)
        plt.plot(precision, recall, '--', color=color, label='{}'.format(roiName), lw=2)
    config_show('PR', 'Test_'+ results.split('.')[0])
    #--------------Draw PR in Train&Test--------------#

    #--------------form .txt--------------#
    label_each = [int(i) for i in label_test[0].split(' ')]
    preds_each = []
    for pred in preds_test:
        predList = [float(i) for i in pred.split(' ')]
        preds_each.append(predList)

    with open('./dataAnalysis/RStudio_DelongTest.txt', 'w') as f:

        for idx, label in enumerate(label_each):
            f.write(str(label) + '\t'
                    + str(preds_each[0][idx]) + '\t'
                    + str(preds_each[1][idx]) + '\t'
                    + str(preds_each[2][idx]) + '\n')
    #--------------form .txt--------------#

    #--------------Delong Test--------------#
    preds_each = []
    roiNames_each =[]
    for roiName, pred in zip(seriesWithROI, preds_test):
        predList = [float(i) for i in pred.split(' ')]
        roiNames_each.append(roiName)
        preds_each.append(predList)
    label_each = [int(i) for i in label_test[0].split(' ')]

    for i in np.arange(len(preds_each)):
        for j in np.arange(i+1, len(preds_each)):
            print('*----------------------------------------------------------')
            # print('Model A:', preds_each[i])
            # print('Model B:', preds_each[j])
            # print('   True:',label_each)
            print(roiNames_each[i]+' vs. '+roiNames_each[j])
            DelongTest(preds_each[i], preds_each[j], label_each, threshold=0.05)
    #--------------Delong Test--------------#

    #--------------HeatMap--------------#

    book = xlrd.open_workbook('二分类test.xls')
    sheet1 = book.sheet_by_index(0)
    rois = []
    series =[]
    aucs = []
    for row in range(sheet1.nrows):
        for col in range(sheet1.ncols):
            if row==0 and col>0:
                x = sheet1.cell(row, col).value.strip()
                if x != '':
                    rois.append(x)
            if col==0 and row>1:
                x = sheet1.cell(row, col).value.strip()
                if x != '':
                    series.append(x)
            if row>1 and (col==5 or col==10 or col==15):
                x = sheet1.cell(row, col).value
                if x != '':
                    aucs.append(x)

    auc = np.array(aucs).reshape(len(series), len(rois)) / 100  # list转array

    # plt.cla()
    sns.set()
    df = pd.DataFrame(auc, index=series , columns=rois)
    df.head()
    sns.heatmap(df, annot=True, cmap='YlGnBu')
    plt.tight_layout()
    plt.savefig('./dataAnalysis/HeatMap.tiff')
    plt.show()
    #--------------HeatMap--------------#
