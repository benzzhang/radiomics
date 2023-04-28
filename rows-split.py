'''
@Author     : Jian Zhang
@Init Date  : 2023-04-27 16:39
@File       : rows-split.py
@IDE        : PyCharm
@Description: 根据TZ/PZ标签挑出PZ/TZ类型数据的特征，组合成新表放入不同data文件夹
'''
import os

import pandas as pd

def split(xlsxFile, df_dic):
    PZfile = './data-PZ/' + os.path.split(xlsxFile)[-1]
    TZfile = './data-TZ/' + os.path.split(xlsxFile)[-1]

    df_rows = pd.read_csv(xlsxFile)
    df_PZ_new, df_TZ_new = pd.DataFrame(columns=df_rows.columns), pd.DataFrame(columns=df_rows.columns)

    for l1, n1, c in zip(df_dic['Label'], df_dic['编号'], df_dic['位置']):
        for idx, (l2, n2) in enumerate(zip(df_rows['Label'], df_rows['ID'])):
            if '外' in c:
                if l1 == l2 and n1[1:] == n2[1:].zfill(3):
                    df_PZ_new = df_PZ_new.append(df_rows.iloc[idx])
                    break
            elif '移行' == c:
                if l1 == l2 and n1[1:] == n2[1:].zfill(3):
                    df_TZ_new = df_TZ_new.append(df_rows.iloc[idx])
                    break

    df_PZ_new.to_csv(PZfile, index=False)
    df_TZ_new.to_csv(TZfile, index=False)
    print(PZfile+' saved'+' & '+TZfile+' saved')

if __name__ == '__main__':
    files = os.listdir('./data2')
    dict = '瘤周分区.xlsx'
    df_dic = pd.read_excel(dict)

    file_list = []
    for i in files:
        if i.endswith('.csv') and i.split('.')[0][-1] == 'P':
            file_list.append(i)
    print(file_list, 'needed to be dealt')

    for i in file_list:
        try:
            split(os.path.join('./data2', i), df_dic)
        except Exception as e:
            print('Error in ', i, ':', e)
            continue




