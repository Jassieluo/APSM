The APSM standalone module is stored in apsm\_works/apsm\_module/apsm\_module.py.



We have already integrated this module into Ultralytics' block structure, allowing direct modification of the training scripts under apsm\_works/model\_tools to select which model to use and whether to integrate APSM for training.



However, before conducting validation, it is necessary to first use apsm\_works/apsm\_tools/add\_noise.py to build a noisy dataset for evaluation.



Then modify the validation scripts under apsm\_works/model\_tools to perform comparative testing with and without noise.



The configured model configuration files with APSM integration are located in the apsm\_works/model\_config folder.

