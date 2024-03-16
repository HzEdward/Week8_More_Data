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
import torch


from data_argumentation.data_argumentation import DataArgumentation
from data_argumentation.data_argumentation import check_same_image

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
def copy_image_unused(image_list, class_name):
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
class DataArgumentation_art_mislabelled(DataArgumentation):
    def __init__(self, folder_path):
        super().__init__(folder_path)
        
        # 取self.folder_path的上一级目录文件夹名字
        # e.g. desktop/test/image_pair_1 to desktop/test
        self.middle_folder_path = os.path.dirname(self.folder_path)
        self.mis_shifted_path = os.path.join(self.middle_folder_path, "mis_shifted")
        self.mis_rotated_path = os.path.join(self.middle_folder_path, "mis_rotated")
        self.mis_flipped_path = os.path.join(self.middle_folder_path, "mis_flipped")
        self.mis_replaced_path = os.path.join(self.middle_folder_path, "mis_replaced")

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
        random_x = random.randint(-20, 20)
        random_y = random.randint(-20, 20)
        # 对图像进行微移动
        shifted_image = np.roll(image, random_x, axis=1)
        shifted_image = np.roll(shifted_image, random_y, axis=0)
        return shifted_image
    
    def replace_image(self, original_image_path: str):
        """
            
        给定一个图像的路径，把这个图像路径替换成另图像

        参数:
            be_replaced_path (str): 要替换的图像image的路径。
            (请注意，是图像的路径)

        返回:
            numpy.ndarray: 替换后的图像作为NumPy数组。
        """
        whiteblack_folder_layer = "./"+original_image_path.split("/")[-3]+"/"
        pair_folder_layer = os.listdir(whiteblack_folder_layer)


        # 定义过滤条件
        # filter_conditions = [
        #     lambda file: not file.startswith("."),
        #     lambda file: file.startswith("image_"),
        #     lambda file: "mis" not in file,
        #     lambda file: file != original_image_path.split("/")[-2]
        # ]
        # # 应用过滤条件
        # for condition in filter_conditions:
        #     pair_folder_layer = [file for file in pair_folder_layer if condition(file)]

        # print("pair_folder_layer: ", pair_folder_layer)


        # pair_folder_layer is made by file

        pair_folder_layer = [i for i in pair_folder_layer if not i.startswith(".")]
        pair_folder_layer = [i for i in pair_folder_layer if i.startswith("image_")]
        pair_folder_layer = [i for i in pair_folder_layer if "mis" not in i]
        pair_folder_layer = [i for i in pair_folder_layer if i != original_image_path.split("/")[-2]]

        substitute_folder_name = random.choice(pair_folder_layer)
        substitute_folder_full_path = os.path.join("./"+original_image_path.split("/")[-3]+"/"+substitute_folder_name)
        for image in os.listdir(substitute_folder_full_path):
            if not image.startswith(".") and image.startswith("image_") and "mis" not in image:
                substitute_image_full_path = os.path.join(substitute_folder_full_path, image)
                subsitute_image = cv2.imread(substitute_image_full_path)
            else:
                continue
            
        return subsitute_image
      
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
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure directories are created before performing data augmentation        
        file_list = os.listdir(self.folder_path) 
        file_list = [file for file in file_list if not file.startswith(".")] # remove hidden files
        
        for file_name in file_list:
            file_path = os.path.join(self.folder_path, file_name)
           # file path= ./test/image_pair_1/image_Video1_frame000090.png

            condition = {"condition 1": file_name.endswith('.jpg') or file_name.endswith('.png'),
                         "condition 2": not file_name.startswith("."),
                         "condition 3": file_name.startswith("image_")}
            
            if all(condition.values()):
                image_path = os.path.join(self.folder_path, file_name)
                print("\n")
                print(f"Data augmentation started for {image_path}.")

                image = cv2.imread(file_path)
                file_name_split = os.path.splitext(file_name)[0]

                #test/image_pair_1/image_Video1_frame000090.png to test/image_pair_1/label_Video1_frame000090.png
                mask_path = os.path.join(self.folder_path, file_name_split.replace("image_", "label_")+ ".png")

                # 新方法一：图片微移动
                shifted_image = self.shift(image)
                image_pair_path = self.folder_path.split("/")[-1] + "_mis_shifted"
                shifted_file_path = os.path.join(self.middle_folder_path, image_pair_path, file_name_split + '_shifted.png')
                # Create directories if they don't exist
                self.create_directories(os.path.join(self.middle_folder_path, image_pair_path))
                # Save shifted image
                cv2.imwrite(shifted_file_path, shifted_image)
                # Save mask image
                mask_image_path = os.path.join(self.middle_folder_path, image_pair_path, file_name_split.replace("image_", "label_") + "_shifted.png")
                shutil.copy(mask_path, mask_image_path)
                # Check if shifted image is the same as the original image
                if check_same_image(file_path, shifted_file_path) == True:
                    print("shifted_image is the same as the original image")

                # 新方法二：图片微旋转，与旧方法直接旋转代码一样
                # make sure the angle is not too small
                angle = abs(random.uniform(-10, 10))
                angle =0.1907822277782607
                print("angle: ", angle)
                rotated_image = self.rotate(image, angle)
                image_pair_path = self.folder_path.split("/")[-1] + "_mis_rotated"
                rotated_image_path = os.path.join(self.middle_folder_path, image_pair_path, file_name_split + '_rotated.png')
                self.create_directories(os.path.join(self.middle_folder_path, image_pair_path))
                cv2.imwrite(rotated_image_path, rotated_image)
                # Save mask image
                mask_image_path = os.path.join(self.middle_folder_path, image_pair_path, file_name_split.replace("image_", "label_") + "_rotated.png")
                shutil.copy(mask_path, mask_image_path)
                # Check if rotated image is the same as the original image
                if check_same_image(file_path, rotated_image_path) == True:
                    print("rotated_image is the same as the original image")

                # 新方法三：图片镜像翻转
                flipped_image = self.flip(image)
                image_pair_path = self.folder_path.split("/")[-1] + "_mis_flipped"
                flipped_file_path = os.path.join(self.middle_folder_path, image_pair_path, file_name_split + '_flipped.png')
                self.create_directories(os.path.join(self.middle_folder_path, image_pair_path))
                cv2.imwrite(flipped_file_path, flipped_image)
                # Save mask image
                mask_image_path = os.path.join(self.middle_folder_path, image_pair_path, file_name_split.replace("image_", "label_") + "_flipped.png")
                shutil.copy(mask_path, mask_image_path)
                # Check if flipped image is the same as the original image  
                if check_same_image(file_path, flipped_file_path) == True:
                    print("flipped_image is the same as the original image")


                # 新方法四：将另外一个类别的image(e.g. 在../层另一个文件夹的"image_"开头的png中 替换当前文件夹下的image图像，这很符合mislabelled mask缺少帧数而导致的情况
                replace_image = self.replace_image(file_path)
                image_pair_path = self.folder_path.split("/")[-1] + "_mis_replaced"
                replaced_file_path = os.path.join(self.middle_folder_path, image_pair_path, file_name_split + '_replaced.png')
                self.create_directories(os.path.join(self.middle_folder_path, image_pair_path))
                cv2.imwrite(replaced_file_path, replace_image)
                # Save mask image
                mask_image_path = os.path.join(self.middle_folder_path, image_pair_path, file_name_split.replace("image_", "label_") + "_replaced.png")
                shutil.copy(mask_path, mask_image_path)
                # Check if replaced image is the same as the original image
                if check_same_image(file_path, replaced_file_path) == True:
                    print("replaced_image is the same as the original image")


                # 不变copy,用做比较
                # original_file_path = os.path.join(self.original_path, 
                #                                   file_name_split+"_original.png")
                # self.create_directories(self.original_path)
                # cv2.imwrite(original_file_path, image)

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
            else:
                continue

def remove_files(folder_path: str, keyword: str):
    """
    删除文件夹中的mis文件夹

    Args:
        folder_path (str): 要删除含有mis文件夹的文件夹路径。

    Returns:
        None

    Raises:
        None
    """
    if keyword == "mis":
        for file in os.listdir(folder_path):
            if file.__contains__("mis"):
                shutil.rmtree(os.path.join(folder_path, file))
            else:
                continue
            
    elif keyword == "original":
        for file in os.listdir(folder_path):
            if not file.__contains__("mis"):
                shutil.rmtree(os.path.join(folder_path, file))
            else:
                continue
        

    
        
if "__main__" == __name__:
    
    for file in os.listdir("./test"):
        condition = {
                    "condition 1": lambda file: not file.startswith("."),
                    # contain no "mis" e.g. "hello_mis_shifted", "mis_rotated"
                    "condition 2": lambda file: "mis" not in file
                    }            
        if all(cond(file) for cond in condition.values()):
            full_file_path= os.path.join("./test", file)
            print("full_file_path: ", full_file_path)
            Secondary_Knife_Folder = DataArgumentation_art_mislabelled(full_file_path)
            Secondary_Knife_Folder.new_data_argumentation()
        else:
            continue
        
    remove_files("./test", "mis")
        



    

    


    










    










