import os
import cv2
import numpy as np
import random
import sys
import torch

import torchvision.transforms as transforms

class DataArgumentation:
    """
    A class for performing data augmentation on images in a given folder.

    Args:
        folder_path (str): The path to the folder containing the images.
        本类只能处理单个文件夹下的图片，不能处理多个文件夹下的图片。
        如果要处理多个文件夹下的图片，需要在外部调用本类的方法。

    Attributes:
        folder_path (str): The path to the folder containing the images.
        flip_path (str): The path to the folder where flipped images will be saved.
        rotate_path (str): The path to the folder where rotated images will be saved.
        brightness_path (str): The path to the folder where brightness adjusted images will be saved.
        contrast_path (str): The path to the folder where contrast adjusted images will be saved.
        noise_path (str): The path to the folder where noisy images will be saved.
        blur_path (str): The path to the folder where blurred images will be saved.

    Methods:
        create_directories(): Creates the necessary directories for saving the augmented images.
        rotate(image, angle): Rotates the given image by the specified angle.
        flip(image): Flips the given image horizontally.
        add_noise(image): Adds random noise to the given image.
        apply_blur(image): Applies Gaussian blur to the given image.
        data_augmentation(): Performs data augmentation on the images in the folder.
    
    """

    def __init__(self, folder_path):
        self.folder_path = folder_path
        # 改动：将base_path的值改为folder_path上一级目录
        self.base_path = os.path.dirname(os.path.abspath(folder_path)) # Get parent directory of folder_path
        self.folder_path_pure = self.folder_path.split("./")[1].rstrip("/")
        self.flip_path = os.path.join(self.base_path, self.folder_path_pure+ "_(flipped)")
        self.rotate_path = os.path.join(self.base_path, self.folder_path_pure+"_(rotated)")
        self.brightness_path = os.path.join(self.base_path, self.folder_path_pure+"_(brightness)")
        self.contrast_path = os.path.join(self.base_path, self.folder_path_pure+"_(contrast)")
        self.noise_path = os.path.join(self.base_path, self.folder_path_pure+"_(noise)")
        self.blur_path = os.path.join(self.base_path, self.folder_path_pure+"_(blur)")
        self.original_path = os.path.join(self.base_path, self.folder_path_pure+"_(original)")

    # def create_directories(self):
    #     for directory in [self.flip_path, self.rotate_path, self.brightness_path, self.contrast_path, self.noise_path, self.blur_path]:
    #         if not os.path.exists(directory):
    #             os.makedirs(directory)
    
    def create_directories(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def rotate(self, image, angle):
        tensor_image = transforms.ToTensor()(image)
        rotation_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(angle),
            transforms.ToTensor()
        ])
        rotated_image = rotation_transform(tensor_image)
        rotated_image = transforms.ToPILImage()(rotated_image).convert("RGB")
        rotated_image = np.array(rotated_image)
        return rotated_image

    def flip(self, image):
        flipped_image = cv2.flip(image, 1)
        return flipped_image

    def add_noise(self, image):
        noise = np.random.normal(0, 25, image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image

    def apply_blur(self, image):
        blur_kernel_size = random.choice([3, 5, 7])  # Choose a random blur kernel size
        blurred_image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)
        return blurred_image

#TODO：一边制造数据增强之后的图片，一边创建新的文件夹，并且把文件夹创建在和原文件夹同一个目录下（同一级别）
    def data_augmentation(self):
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure directories are created before performing data augmentation        
        file_list = os.listdir(self.folder_path) 
        # remove hidden files
        file_list = [file for file in file_list if not file.startswith(".")]
        # print("file_list:", file_list)
        
        for file_name in file_list:
            # print("file_name:", file_name)
            file_path = os.path.join(self.folder_path, file_name)
            # print("Processing", file_name)
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                image = cv2.imread(file_path)

                # 方法一：镜像翻转
                flipped_image = self.flip(image)
                file_name_split = file_name.split(".")[0]
                flipped_file_path = os.path.join(self.flip_path, file_name_split + '_flipped.png')
                self.create_directories(self.flip_path)
                cv2.imwrite(flipped_file_path, flipped_image)

                # 方法二：直接旋转
                angle = abs(random.uniform(-10, 10))  # 将角度转换为正数
                rotated_image = self.rotate(image, angle)
                rotated_file_path = os.path.join(self.rotate_path, file_name_split + "_rotated.png")
                self.create_directories(self.rotate_path)
                cv2.imwrite(rotated_file_path, rotated_image)

                # 方法三：亮度调整
                brightness_image = random.uniform(0.5, 1.5)
                brightness_file_path = os.path.join(self.brightness_path,  file_name_split+ "_brightness.png")
                self.create_directories(self.brightness_path)
                cv2.imwrite(brightness_file_path, image * brightness_image)

                # 方法四：对比度调整
                contrast_image = random.uniform(0.5, 1.5)
                contrast_file_path = os.path.join(self.contrast_path, file_name_split + "_contrast.png")
                self.create_directories(self.contrast_path)
                cv2.imwrite(contrast_file_path, np.clip(image * contrast_image, 0, 255).astype(np.uint8))

                # 方法五：添加噪声
                noisy_image = self.add_noise(image)
                noisy_file_path = os.path.join(self.noise_path, file_name_split + "_noisy.png")
                self.create_directories(self.noise_path)
                cv2.imwrite(noisy_file_path, noisy_image)

                # 方法六：模糊处理
                blurred_image = self.apply_blur(image)
                blur_file_path = os.path.join(self.blur_path, file_name_split + "_blur.png")
                self.create_directories(self.blur_path)
                cv2.imwrite(blur_file_path, blurred_image)

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
            
            else:
                print(f"Skipping {file_name} as it is not an image file")

# 测试两张RGB图片是否相同
def check_same_image(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("One or both images could not be read.")
        return False

    if img1.shape == img2.shape:
        difference = cv2.subtract(img1, img2)
        b, g, r = cv2.split(difference)
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            print("The images are completely Equal")
            return True
        else:
            print("The images are NOT equal")
            return False
    else:
        print("The images are NOT equal")
        return False

if __name__ == "__main__":
    # 遍历./dataset/下的所有文件夹，对每个文件夹下的图片进行数据增强
    dataset_path = "./dataset/"
    folder_list = os.listdir(dataset_path)
    # remove hidden files
    folder_list = [folder for folder in folder_list if not folder.startswith(".")]
    # print("folder_list:", folder_list) 

    for folder in folder_list: # folder: test or train
        # remove hidden files
        if folder.startswith("."):
            continue
        else: 
            print(f"Processing ({folder}) folder")
            for sub_folder in os.listdir(os.path.join(dataset_path, folder)): # sub_folder: white or black
                if sub_folder.startswith("."):
                    continue
                else:
                    for sub_sub_folder in os.listdir(os.path.join(dataset_path, folder, sub_folder)):
                        if sub_sub_folder.startswith("."):
                            continue
                        else:
                            # print("sub_sub_folder:", sub_sub_folder)
                            # print(f"Processing ({folder}/{sub_folder}/{sub_sub_folder}) folder")
                            folder_path = os.path.join(dataset_path, folder, sub_folder, sub_sub_folder)
                            # print("folder_path:", folder_path)
                            data_argumentation = DataArgumentation(folder_path)
                            # print(f"Data augmentation completed for ({folder}/{sub_folder}/{sub_sub_folder}) folder.")
                            # print("---------------------------------------------------")

    # folder_path = './simu_data_pair/'
    # data_argumentation = DataArgumentation(folder_path)
    # print("self.base_path:", data_argumentation.base_path)
    # data_argumentation.data_augmentation()




