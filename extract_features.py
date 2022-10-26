import numpy as np
from radiomics import featureextractor, getFeatureClasses
import os
import SimpleITK as sitk
import pandas as pd
import csv

ori_path = r'D:\prostatic\Primary-data\YT'
mask_path = r'D:\prostate\nii-Primary-data\MASK-DATA\RT-P'
save_path = r'D:\nii-Primary-data'


# path_read:读取dicom的文件路径  path_save:保存nii的文件路径
def dcm2nii(path_read, path_save):
    # GetGDCMSeriesIDs读取序列号相同的dcm文件
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path_read)
    # GetGDCMSeriesFileNames读取序列号相同dcm文件的路径，series[0]代表第一个序列号对应的文件
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path_read, series_id[0])
    print(len(series_file_names))
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3d = series_reader.Execute()
    sitk.WriteImage(image3d, path_save)
# dcm2nii()

# dcm 转 nii
def dcm2nii_folder():
    for ori_name in os.listdir(ori_path):
        new_name = ori_name[1:]
        new_name = ori_name[0].upper() + str(new_name).zfill(3)

        dcm2nii(os.path.join(ori_path, ori_name), os.path.join(os.path.join(save_path, ori_name[0]), new_name + '.nii'))
# dcm2nii_folder()

# mask 重命名
def rename(path):
    for ori_name in os.listdir(path):
        cut1 = ori_name.split('_')[0]
        cut2 = ori_name.split('_')[1]
        new_cut1 = cut1[0].upper() + str(cut1[1:]).zfill(3)
        new_name = new_cut1 + '_' + cut2
        print('trans ', os.path.join(path, ori_name), ' to ', os.path.join(path, new_name))
        os.rename(os.path.join(path, ori_name), os.path.join(path, new_name))
# rename(r'D:\prostatic\Primary-data\RT-P')

def rename_folder(path):
    for ori_name in os.listdir(path):
        new_name = ori_name[0].upper() + str(ori_name[1:]).zfill(3)
        print('trans ', os.path.join(path, ori_name), ' to ', os.path.join(path, new_name))
        os.rename(os.path.join(path, ori_name), os.path.join(path, new_name))
# rename_folder(r'D:\prostatic\Primary-data\YT')

# line 53-76 为提取特征代码; 修改 raw mask 参数地址调整提取序列; 生成csv后需要手动添加一列标签
raw = './RAW-DATA/T'
mask = './merge_T2_CP'
save = './merge_features'

def extract_features(raw_path, mask_path, save_path):
    all_data = pd.DataFrame()

    for raw, mask in zip(os.listdir(raw_path), os.listdir(mask_path)):
        print(raw, mask)
        raw_ = sitk.ReadImage(os.path.join(raw_path, raw))
        mask_ = sitk.ReadImage(os.path.join(mask_path, mask))

        config = './Params.yaml'
        extractor = featureextractor.RadiomicsFeatureExtractor(config)
        featureVector = extractor.execute(raw_, mask_)
        # print(featureVector.values())
        data = pd.DataFrame.from_dict(featureVector.values()).T
        data.columns = featureVector.keys()
        data.index = [raw.split('.')[0]]

        all_data = pd.concat([all_data, data])
    all_data.to_csv(os.path.join(save_path, mask_path[2:]+'.csv'))
# extract_features(raw, mask, save)

# 校正标签顺序; 从001-999
def label_out(mask_path):
    features_data = pd.read_csv(os.path.join('./', mask_path))
    label = features_data.iloc[:, 0:2]
    for i in range(len(label['ID'])):
        label['ID'][i] = label['ID'][i][0].upper() + str(label['ID'][i][1:]).zfill(3)
    print(label)
    print(features_data)
    features_data.iloc[:, 0:2] = label
    print(features_data)
    label.to_csv('correct.csv')
label_out('RT-C.csv')


def size_set():
    path = r'D:\nii-Primary-data\RAW-DATA\D'
    set1 = set()
    set2 = set()
    for i in os.listdir(path):
        itk_img = sitk.ReadImage(os.path.join(path, i))
        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
        num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
        origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())
        print(height, width)
        set1.add(height)
        set2.add(width)
        # print(spacing)
    print(set1)
    print(set2)
# size_set()