from past_pick_whitelist.surgical_tape import surgical_tape_list, eye_retractors_list
import pandas as pd
import os
import shutil
import sys


'''
    trial.py 通过给定的类来提供含有该类的图像，并且制作成训练集和验证集。

    trial.py contain the following 3 functions:

    1. check_list(surgical_tape_list, eye_retractors_list) 只是起到了检查的功能
    从data.csv文件找到和surgical_tape_list和eye_retractors_list中的元素具有相同img_path的行
    并且查看这些行在surgical_tape或eye_retractors列是否为0。

    2. copy_image(surgical_tape_list, eye_retractors_list) 非常有用
    将surgical_tape_list和eye_retractors_list中的元素对应的图片和mask复制到不同的文件夹中。

    3. train_valid_split(folder_path, train_percentage, valid_percentage)
    将folder_path中的子文件夹按照train_percentage和valid_percentage的比例分配到train和valid文件夹中。
'''
def check_list(surgical_tape_list, eye_retractors_list):
    """
    检查 "data.csv "文件中 "img_path "与 "surgical_tape_list "和 "eye_retractors_list "中元素相同的行中 "surgical_tape "或 "eye_retractors "是否为 0。

    Args:
        surgical_tape_list（列表）： 手术磁带的图像路径列表。
        eye_retractors_list （列表）： 眼球回缩器的图像路径列表。

    Returns:
        None

    Raises:
        None
    """
    data = pd.read_csv('./data.csv')
    surgical_tape = data[data['img_path'].isin(surgical_tape_list)]
    eye_retractors = data[data['img_path'].isin(eye_retractors_list)]
    surgical_tape = surgical_tape[['img_path', 'Surgical Tape']]
    eye_retractors = eye_retractors[['img_path', 'Eye Retractors']]
    surgical_tape = surgical_tape[surgical_tape['Surgical Tape'] == 0]
    eye_retractors = eye_retractors[eye_retractors['Eye Retractors'] == 0]

    print(surgical_tape)
    print(eye_retractors)

def copy_image(surgical_tape_list, eye_retractors_list):
    """
    将给定列表中的图像和遮罩复制到不同的文件夹中。

    Args:
        surgical_tape_list （列表）： 手术胶带的图像路径列表。
        eye_retractors_list （列表）： 眼球牵开器的图像路径列表。

    Returns:
        None

    Raises:
        None
    """
    for index, i in enumerate(surgical_tape_list, start=1):
        image_name = i.split('/')[-1]
        folder_name = f"image_pair_{index}"
        mask_path = i.replace('Images', 'Labels')
        mask_name = mask_path.split('/')[-1]
        folder_path = './surgical_tape_list/' + folder_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        shutil.copyfile("../segmentation/"+i, folder_path + '/' + "image_"+ image_name)
        shutil.copyfile("../segmentation/"+mask_path, folder_path + '/' + "label_"+ mask_name)

    for index, i in enumerate(eye_retractors_list, start=1):
        image_name = i.split('/')[-1]
        folder_name = f"image_pair_{index}"
        mask_path = i.replace('Images', 'Labels')
        mask_name = mask_path.split('/')[-1]
        folder_path = './eye_retractors_list/' + folder_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        shutil.copyfile("../segmentation/"+i, folder_path + '/' + "image_"+ image_name)
        shutil.copyfile("../segmentation/"+mask_path, folder_path + '/' + "label_"+ mask_name)

def train_valid_split(folder_path, train_percentage, valid_percentage):
    """
    根据指定的百分比将给定文件夹内的子文件夹分成训练集和验证集。

    Args:
        folder_path (str):包含要分割的子文件夹的文件夹路径。
        train_percentage(float): 要包含在训练集中的子文件夹的百分比。
        valid_percentage (float): 要包含在验证集中的子文件夹的百分比。

    Returns:
        None

    Raises:
        None
    """
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return
    
    if not os.path.exists('./argumented/train'):
        os.makedirs('./argumented/train')
    if not os.path.exists('./argumented/valid'):
        os.makedirs('./argumented/valid')
    
    subfolder = os.listdir(folder_path)
    num_subfolders = len(subfolder)
    
    if num_subfolders == 0:
        print(f"There are no subfolders inside the {folder_path} folder.")
        return
    
    train_count = int(num_subfolders * train_percentage / 100)
    valid_count = int(num_subfolders * valid_percentage / 100)

    if train_count + valid_count > num_subfolders:
        print("The sum of training and validation percentages exceeds 100.")
        return
    
    train_subfolders = subfolder[:train_count]
    valid_subfolders = subfolder[train_count:train_count+valid_count] 

    for subfolder_name in train_subfolders:
        if subfolder_name == 'train' or subfolder_name == 'valid' or subfolder_name == '.DS_Store':
            continue
        else:
            shutil.move(os.path.join(folder_path, subfolder_name), './argumented/train/' + subfolder_name)
    
    for subfolder_name in valid_subfolders:
        if subfolder_name == 'train' or subfolder_name == 'valid' or subfolder_name == '.DS_Store':
            continue
        else:
            shutil.move(os.path.join(folder_path, subfolder_name), './argumented/valid/' + subfolder_name)

    print(f"Moved {train_count} subfolders to ./argumented/train.") 
    print(f"Moved {valid_count} subfolders to ./argumented/valid.")

    subfolder = os.listdir(folder_path)
    for subfolder_name in subfolder:
        if subfolder_name == 'train' or subfolder_name == 'valid' or subfolder_name == '.DS_Store':
            continue
        else:
            shutil.rmtree(os.path.join(folder_path, subfolder_name))
            print(f"Removed {subfolder_name} from {folder_path}.")

    print("Splitting is done.")

if __name__ == '__main__':
    train_valid_split('./eye_retractors_list', 80, 20)



    
    



