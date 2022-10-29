# radiomics
classify using ML model (TODO: or DL model)

#

##folder
  - 'binary-classification': code and .yaml in binary classification task
  - 'three-classification': code and .yaml in three classification task
  - 'output': the log to save experimental result
  - 'merge_xxx_CP': mask files which merging ROI-C and ROI-P
  - 'merge_features': features files  which extracting from folder 'merge_xxx_CP' and RAW-DATA
  
##file
  - 'prostatic_cancer_gleason_AdjustParm.py': you can select optimal parms in specific model and feature filtering methods through this file
  - 'prostatic_cancer_gleason.py': use this .py to perform a combination of different models and feature filtering methods
  - 'extract_features.py': funcs
  - 'merge_mask.py': funcs used for merging different ROI
