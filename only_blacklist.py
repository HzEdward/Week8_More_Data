# some images only appear in the blacklist, and not in the whitelist
#**原始数据集：**
# 训练数据集：320（160黑名单，160白名单）
# 验证数据集：80（40黑名单，40白名单）
# solution: Based on ./data.csv find images that are whitelist and have the same class as the blacklist, add them to the whitelist
# **仅在黑名单中**
#* ["train/blacklist", "train/whitelist", "test/blacklist", "test/whitelist"]
# Class: **Marker**, Imbalance Report: {'distribution': [10.625, 0.0, 27.5, 0.0]}
# 10.625/100*160 = 17, 27.5/100*40 = 11, image_required = 28, ratio = 17/(17+11) = 0.607
# Class: **Iris Hooks**, Imbalance Report: {'distribution': [6.875, 0.0, 2.5, 0.0]}
# 6.875/100*160 = 11, 2.5/100*40 = 1, image_required = 12, ratio = 11/(11+1) = 0.917
# Class: **Rycroft Cannula Handle**, Imbalance Report: {'distribution': [2.5, 0.0, 2.5, 0.0]}
# 2.5/100*160 = 4, 2.5/100*40 = 1, image_required = 5, ratio = 4/(4+1) = 0.8
"""
length of train_list: 16
length of test_list: 12

length of train_list: 11
length of test_list: 1

length of train_list: 4
length of test_list: 1
"""
import pandas as pd
import os
import shutil
from only_whitelist import get_image_list, copy_image, split_train_test

def create_whitelist(class_name:str, num:int , ratio:float):
    copy_image(class_name, get_image_list(class_name, "./data.csv", num))
    split_train_test(class_name, ratio)

if "__main__" == __name__:
    create_whitelist("Marker", 28, 0.607)
    print("\n")
    create_whitelist("Iris Hooks", 12, 0.917)
    print("\n")
    create_whitelist("Rycroft Cannula Handle", 5, 0.8)
    print("\n")
    print("Done!")




    




