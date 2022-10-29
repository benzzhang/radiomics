# radiomics
binary classify using ML model (TODO: or DL model)


```bash
│  .gitignore
│  AUC总表.xls
│  best_CMP.txt
│  C.txt ' "C"ROI的所有实验结果 '
│  dataAnalysis.py
│  experiment-CMP-2-with_NPV.txt
│  experiment-CMP-2. '二分类实验结果汇总'
│  important_features_20.csv '最重要前20特征表, 用以R绘corr图'
│  important_features_50.csv
│  M.txt ' "M"ROI的所有实验结果 '
│  P.txt ' "P"ROI的所有实验结果 '
│  README.md
│  各序列各ROI指标总表.xls
│  最优模型的指标表.xls
│
├─binary-classification '模型训练及对应配置文件'
│      config.yaml
│      config_AdjustParm.yaml
│      prostatic_cancer_gleason.py
│      prostatic_cancer_gleason_AdjustParm.py
│      prostatic_cancer_xgboost.py 'XGBoost'
│
├─data2 '带二分类标签的特征表文件目录'
├─dataAnalysis '保存 AUC热图、ROC曲线、PR曲线、最重要前20/50特征系数图及相关性分析图、DelongTest结果'
│      corr_20.tiff
│      corr_50.tiff
│      DelongTest_Results.txt
│      HeatMap_AUC.tiff
│      importance_20_cover.tiff
│      importance_20_gain.tiff
│      importance_20_weight.tiff
│      importance_50_cover.tiff
│      importance_50_gain.tiff
│      importance_50_weight.tiff
│      PR_Test_best_CMP.tiff
│      PR_Test_C.tiff
│      PR_Test_M.tiff
│      PR_Test_P.tiff
│      ROC_Test_best_CMP.tiff
│      ROC_Test_C.tiff
│      ROC_Test_M.tiff
│      ROC_Test_P.tiff
│      RStudio_DelongTest.txt
│
├─output '非XGBoost实验结果保存目录'
│      log.txt
│
└─影像标签文件及特征提取代码
    │  extract_features.py '特征提取'
    │  merge_mask.py '将2类mask融合'
    │  Params.yaml '特征提取配置文件'
    │  统一命名的nii文件.txt
    │
    ├─features '提取的特征表 及 "融合ROI"的提取特征表'
    │  └─merge_features
    ├─MASK-DATA 'ROI文件'
    │  ├─merge_ADC_CP
    │  ├─merge_DWI_CP
    │  ├─merge_T2_CP
    │  ├─RA-C
    │  ├─RA-M
    │  ├─RA-P
    │  ├─RD-C
    │  ├─RD-M
    │  ├─RD-P
    │  ├─RT-C
    │  ├─RT-M
    │  └─RT-P
    └─RAW-DATA '各序列影像文件'
        ├─A
        ├─D
        └─T
```
