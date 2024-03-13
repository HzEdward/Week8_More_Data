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

from data_argumentation.data_argumentation import DataArgumentation

# 给定类别的图像，从./data.csv中提取含有该类别的图像路径：如果某一行的该类别的值不为0且blacklist为0，则将该行的img_path加入到列表中
# 输入：类别名称，data.csv地址，需要提取的地址数量。
# 输出：含有该类别的图像路径列表

def get_image_list(class_name, data_csv, num):
    data = pd.read_csv(data_csv)
    class_data = data[data[class_name] != 0]
    class_data = class_data[class_data['blacklisted'] == 0]
    image_list = class_data['img_path'].tolist()
    image_list = image_list[:num]
    return image_list

# 输入：获取完成的图像路径列表之后，将列表中的图片以及对应的mask复制到不同的文件夹中。
# 输出：创建文件夹，将图像和mask复制到不同的文件夹中。
def copy_image(image_list, class_name):
    for img in image_list:
    
        img_name = img.split('/')[-1]
        mask_name = img_name.split('.')[0] + '_mask.png'
        shutil.copy(img, '.whitelist_pick/train/whitelist/' + class_name + '/' + img_name)
        shutil.copy(img, '.whitelist_pick/test/whitelist/' + class_name + '/' + img_name)
        shutil.copy(img, '.whitelist_pick/train/whitelist/' + class_name + '/' + mask_name)
        shutil.copy(img, '.whitelist_pick/test/whitelist/' + class_name + '/' + mask_name)
   


def copy_image(class_name: str, list_name: list):
    """
    将给定列表中的图像和遮罩复制到不同的文件夹中。如果没有文件夹则创建文件夹。

    Args:
        class_name (list): 需要复制的图像和遮罩的类别。e.g. "surgical_tape_list", "eye_retractors_list"

    Returns:
        None

    Raises:
        None
    """
    for index, i in enumerate(list_name, start=1):
        image_name = i.split('/')[-1]
        folder_name = f"image_pair_{index}"
        mask_path = i.replace('Images', 'Labels')
        mask_name = mask_path.split('/')[-1]
        # 将surgical_tape_list编程class_name
        folder_path = './' + class_name + '/' + folder_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        shutil.copyfile("../segmentation/"+i, folder_path + '/' + "image_"+ image_name)
        shutil.copyfile("../segmentation/"+mask_path, folder_path + '/' + "label_"+ mask_name)




if "__main__" == __name__:
    Secondary_Knife_List = get_image_list("Secondary Knife", "./data.csv", 14) 
    copy_image("Secondary_Knife", Secondary_Knife_List )
    Secondary_Knife_Folder = DataArgumentation("./Secondary_Knife")




    










