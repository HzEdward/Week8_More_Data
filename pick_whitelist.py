#**原始数据集：**

# 训练数据集：320（160黑名单，160白名单）

# 验证数据集：80（40黑名单，40白名单）

# 仅在白名单中
# ["train/blacklist", "train/whitelist", "test/blacklist", "test/whitelist"]
# Class: **Secondary Knife**, Imbalance Report: {'distribution': [0.0, 4.375, 0.0, 10.0]}
# 4.375 / 100 * 320 = 14， 10 / 100 * 320 = 32
# Class: **Secondary Knife Handle**, Imbalance Report: {'distribution': [0.0, 3.125, 0.0, 2.5]}
# 3.125 / 100 * 320 = 10， 2.5 / 100 * 320 = 8
# Class: **Lens Injector Handle**, Imbalance Report: {'distribution': [0.0, 1.875, 0.0, 2.5]}
# 1.875 / 100 * 320 = 6， 2.5 / 100 * 320 = 8
# Class: **Primary Knife Handle**, Imbalance Report: {'distribution': [0.0, 0.625, 0.0, 2.5]}
# 0.625 / 100 * 320 = 2， 2.5 / 100 * 320 = 8
# 解决方案：加入黑名单数据, 人为制造包含这些类的黑名单
import pandas as pd
import os
import shutil
import sys

# 给定类别的图像，从./data.csv中提取含有该类别的图像路径，
# 输入：类别名称，data.csv地址，需要提取的数量
# 输出：含有该类别的图像路径列表
def get_image_list(class_name, data_path, num):
    data = pd.read_csv(data_path)
    class_data = data[data[class_name] == 1]
    class_data = class_data.sample(n=num, random_state=1)

    return class_data['img_path'].tolist()


if 








