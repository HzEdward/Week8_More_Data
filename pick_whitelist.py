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
import cv2
import numpy as np
import random


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

# a new class based on DataArgumentation
class DataArgumentation_new(DataArgumentation):
    def __init__(self, folder_path):
        super().__init__(folder_path)

    def shift(self, image):
        """
        对给定的图像进行微移动。这是一种数据增强的方式，可以模拟mislabelled mask的情况。

        Args:
            image (numpy.ndarray): 要进行微移动的图像。

        Returns:
            shifted_image (numpy.ndarray): 微移动后的图像。

        Raises:
            None
        """
        # 随机生成微移动的距离
        random_x = random.randint(-10, 10)
        random_y = random.randint(-10, 10)
        # 对图像进行微移动
        shifted_image = np.roll(image, random_x, axis=1)
        shifted_image = np.roll(shifted_image, random_y, axis=0)
        return shifted_image

    def new_data_argumentation(self):
        """
        对给定的文件夹中的图像进行数据增强。但是数据增强的方式更加贴近于模拟mislabelled mask的情况

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # Ensure directories are created before performing data augmentation        
        file_list = os.listdir(self.folder_path) 
        # remove hidden files
        file_list = [file for file in file_list if not file.startswith(".")]
        
        for file_name in file_list:
            file_path = os.path.join(self.folder_path, file_name)
            condition = {"condition 1": file_name.endswith('.jpg') or file_name.endswith('.png'),
                         "condition 2": file_name.startswith(".") == False,
                         "condition 3": file_name.startswith("image_")}
            
            if all(condition.values()):
                image = cv2.imread(file_path)

                # 新方法一：图片微移动
                shifted_image = self.shift(image)
                file_name_split = file_name.split(".")[0]
                shifted_file_path = os.path.join(self.shift_path, file_name_split + '_shifted.png')
                self.create_directories(self.shift_path)
                cv2.imwrite(shifted_file_path, shifted_image)

                # 新方法二：图片微旋转，与旧方法直接旋转代码一样
                angle = abs(random.uniform(-10, 10))
                rotated_image = self.rotate(image, angle)
                rotated_file_path = os.path.join(self.rotate_path, file_name_split + "_rotated.png")
                self.create_directories(self.rotate_path)
                cv2.imwrite(rotated_file_path, rotated_image)

                # 新方法三：图片镜像翻转
                flipped_image = self.flip(image)
                flipped_file_path = os.path.join(self.flip_path, file_name_split + '_flipped.png')
                self.create_directories(self.flip_path)
                cv2.imwrite(flipped_file_path, flipped_image)

                # 新方法四：直接和另外一张图片对换
                random_image = random.choice(file_list)
                random_image_path = os.path.join(self.folder_path, random_image)
                random_image = cv2.imread(random_image_path)
                swapped_image = self.swap(image, random_image)
                swapped_file_path = os.path.join(self.swap_path, file_name_split + "_swapped.png")
                self.create_directories(self.swap_path)
                cv2.imwrite(swapped_file_path, swapped_image)

                # 不变copy
                original_file_path = os.path.join(self.original_path, file_name_split+"_original.png")
                self.create_directories(self.original_path)
                cv2.imwrite(original_file_path, image)

                # print(f"Data augmentation completed for {file_name}.")
                # print("---------------------------------------------------")
                # print(f"Flipped image saved as {os.path.basename(flipped_file_path)}")
                # print(f"Rotated image saved as {os.path.basename(rotated_file_path)}")
                # print(f"Brightness adjusted image saved as {os.path.basename(brightness_file_path)}")
                # print(f"Contrast adjusted image saved as {os.path.basename(contrast_file_path)}")
                # print(f"Noise added image saved as {os.path.basename(noisy_file_path)}")
                # print(f"Blurred image saved as {os.path.basename(blur_file_path)}")
                # print(f"Original image saved as {os.path.basename(original_file_path)}")
                # print("---------------------------------------------------")
                # print("---------------------------------------------")
                            
            elif condition["condition 3"] == False:
                print(f"{file_name} is a mask image.")

            elif condition ["condition 2"] == True:
                print(f"{file_name} is a hidden file.")

        
    



if "__main__" == __name__:
    Secondary_Knife_List = get_image_list("Secondary Knife", "./data.csv", 14) 
    copy_image("Secondary_Knife", Secondary_Knife_List )
    # for every subfolder in Secondary_Knife, do data argumentation
    for subfolder in os.listdir("./Secondary_Knife"):
        subfolder_path = os.path.join("./Secondary_Knife", subfolder)
        if os.path.isdir(subfolder_path):
            Secondary_Knife_Folder = DataArgumentation_new(subfolder_path)
            Secondary_Knife_Folder.new_data_argumentation()




    










