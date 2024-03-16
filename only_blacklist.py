# some images only appear in the blacklist, and not in the whitelist
#**原始数据集：**
# 训练数据集：320（160黑名单，160白名单）
# 验证数据集：80（40黑名单，40白名单）
# solution: Based on ./data.csv find images that are whitelist and have the same class as the blacklist, add them to the whitelist
# **仅在黑名单中**
# Class: **Marker**, Imbalance Report: {'distribution': [10.625, 0.0, 27.5, 0.0]}
# 10.625/100 * 40 = 4.25, 27.5/100 * 40 = 11, image_required = 15.25
# Class: **Iris Hooks**, Imbalance Report: {'distribution': [6.875, 0.0, 2.5, 0.0]}
# 6.875/100 * 40 = 2.75, 2.5/100 * 40 = 1, image_required = 3.75
# Class: **Rycroft Cannula Handle**, Imbalance Report: {'distribution': [2.5, 0.0, 2.5, 0.0]}
# 2.5/100 * 40 = 1, 2.5/100 * 40 = 1, image_required = 2

import pandas as pd
import os
import shutil
from only_whitelist import get_image_list, copy_image




if "__main__" == __name__:
    copy_image("Marker", get_image_list("Marker", "./data.csv", 15))



