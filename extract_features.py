import numpy as np
from radiomics import featureextractor, getFeatureClasses
import os
import SimpleITK as sitk
import pandas as pd
import csv

ori_path = r'D:\prostate\Primary-data\YA'
mask_path = r'D:\prostate\nii-Primary-data\MASK-DATA\RT-P'

save_path = r'D:\prostate\nii-Primary-data'


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


# dcm 转 nii
def dcm2nii_folder():
    for ori_name in os.listdir(ori_path):
        new_name = ori_name[1:]
        new_name = ori_name[0].upper() + str(new_name).zfill(3)

        dcm2nii(os.path.join(ori_path, ori_name), os.path.join(os.path.join(save_path, ori_name[0]), new_name + '.nii'))


# mask 重命名
def rename(path):
    for ori_name in os.listdir(path):
        cut1 = ori_name.split('_')[0]
        cut2 = ori_name.split('_')[1]
        new_cut1 = cut1[0].upper() + str(cut1[1:]).zfill(3)
        new_name = new_cut1 + '_' + cut2
        print('trans ', os.path.join(path, ori_name), ' to ', os.path.join(path, new_name))
        os.rename(os.path.join(path, ori_name), os.path.join(path, new_name))


raw = r'D:\nii-Primary-data\RAW-DATA\A'
mask = r'D:\nii-Primary-data\MASK-DATA\RA-C'
save = r'D:\nii-Primary-data\features'

def extract_features(raw_path, mask_path, save_path):
    all_data = pd.DataFrame()

    for raw, mask in zip((os.listdir(raw_path)), (os.listdir(mask_path))):
        print(raw, mask)
        raw_ = os.path.join(raw_path, raw)
        mask_ = os.path.join(mask_path, mask)

        config = './Params1.yaml'
        extractor = featureextractor.RadiomicsFeatureExtractor(config)
        featureVector = extractor.execute(raw_, mask_)
        # print(featureVector.values())
        data = pd.DataFrame.from_dict(featureVector.values()).T
        data.columns = featureVector.keys()
        data.index = [raw_.split('\\')[-1].split('.')[0]]

        all_data = pd.concat([all_data, data])
    all_data.to_csv(os.path.join(save_path, mask_path.split('\\')[-1]+'2.csv'))


def label_out(mask_path):
    features_data = pd.read_csv(os.path.join(r'./data2', mask_path.split('\\')[-1] + '.csv'))
    label = features_data.iloc[:, 0:2]
    for i in range(len(label['ID'])):
        label['ID'][i] = label['ID'][i][0].upper() + str(label['ID'][i][1:]).zfill(3)
    print(label)
    print(features_data)
    features_data.iloc[:, 0:2] = label
    print(features_data)
    label.to_csv('correct.csv')

# label_out(mask_path)
# extract_features(raw, mask, save)

# itk_img = sitk.ReadImage(r'D:\nii-Primary-data\RAW-DATA\A\A157.nii')
# img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
# num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
# origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
# spacing = np.array(itk_img.GetSpacing())
# print(num_z, height, width)
# print(spacing)
# itk_img = sitk.ReadImage(r'D:\nii-Primary-data\MASK-DATA\RA-C2\A157_MergeExternalPlus1Shrink2.nii')
# img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
# num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
# origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
# spacing = np.array(itk_img.GetSpacing())
# print(num_z, height, width)
# print(spacing)
