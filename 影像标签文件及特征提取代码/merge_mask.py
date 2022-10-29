import numpy as np
import os
import nibabel as nib
import nrrd

def merge_mask(path_C, path_P, path_merge):
    if not os.path.exists(path_merge):
        os.mkdir(path_merge)
    maskList_C = os.listdir(path_C)
    maskList_P = os.listdir(path_P)

    for c,p in zip(maskList_C, maskList_P):
        global mask_data_C
        global mask_data_P
        global options
        if c.endswith('.nii'):
            mask_data_C = nib.load(os.path.join(path_C, c)).get_data()
            mask_data_C = np.array(mask_data_C)
        if c.endswith('.nrrd'):
            # data:图片的多维矩阵; options:图片的相关信息
            mask_data_C, options = nrrd.read(os.path.join(path_C, c))
        if p.endswith('.nii'):
            mask_data_P = nib.load(os.path.join(path_P, p)).get_data()
            mask_data_P = np.array(mask_data_P)
        if p.endswith('.nrrd'):
            mask_data_P, options = nrrd.read(os.path.join(path_P, p))
        print(options)
        print(c,'\n',p)
        print(mask_data_C.shape,'\n',mask_data_P.shape)
        print('--------------------')
        mask_merge = np.logical_or(mask_data_C, mask_data_P) * 1
        nrrd.write(os.path.join(path_merge, c.split('_')[0]+'_MergeCP.nrrd'),
                   mask_merge, options)

series_name_C = ['RT-C', 'RD-C', 'RA-C']
series_name_P = ['RT-P', 'RD-P', 'RA-P']
series_name_merge = ['merge_T2_CP', 'merge_DWI_CP', 'merge_ADC_CP']

# 标准化命名后的mask文件目录
path_mask = './MASK-DATA'

for C, P, Merge in zip(series_name_C, series_name_P, series_name_merge):
    path_C = os.path.join(path_mask, C)
    path_P = os.path.join(path_mask, P)
    path_merge = os.path.join(path_mask, Merge)

    merge_mask(path_C, path_P, path_merge)