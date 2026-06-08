from email.mime import image

import numpy as np
import os
import tarfile #解压tar.gz文件
import urllib.request #下载数据集
import pickle #读取CIFAR-10数据集官方batch文件

from PIL import Image #图像处理库

URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz' #CIFAR-10数据集下载链接
DATA_DIR = './cifar-10-data' #数据集存储目录

TAR_PATH = os.path.join(DATA_DIR, 'cifar-10-python.tar.gz') #下载的tar.gz文件路径
EXTRACT_DIR = os.path.join(DATA_DIR, 'cifar-10-batches-py') #解压后的数据目录

NUMPY_DIR = os.path.join(DATA_DIR, 'cifar-10-numpy') #转换为NumPy数组后的数据目录
IMAGES_DIR = os.path.join(DATA_DIR, 'cifar-10-images') #保存图像文件的目录

#检查目录是否存在，如果不存在则创建目录
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        print(f"Directory '{path}' already exists.")

def check_path_without_print(path):
    if not os.path.exists(path):
        os.makedirs(path)

check_path(DATA_DIR)
check_path(NUMPY_DIR)
check_path(IMAGES_DIR)

#下载CIFAR-10数据集,如果已经下载则跳过下载步骤,避免重复下载浪费时间和带宽.
def download_cifar10():

    if not os.path.exists(TAR_PATH):
        print("Downloading CIFAR-10 dataset...")
        urllib.request.urlretrieve(URL, TAR_PATH)
        print("Download completed.")
    else:
        print("CIFAR-10 dataset already downloaded.")

#解压CIFAR-10数据集,如果已经解压则跳过解压步骤,避免重复解压浪费时间和存储空间.
def extract_cifar10():
    if not os.path.exists(EXTRACT_DIR):
        print("Extracting CIFAR-10 dataset...")
        with tarfile.open(TAR_PATH, 'r:gz') as tar:
            tar.extractall(path=DATA_DIR)
        print("Extraction completed.")
    else:
        print("CIFAR-10 dataset already extracted.")


#读取CIFAR-10数据集官方batch文件,返回一个包含图像数据和标签的字典.
#每个batch文件包含10000张32x32像素的彩色图像,以及对应的标签.
def load_pickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


#加载一个batch文件,返回图像数据和标签数据的NumPy数组.
def load_batch(filepath):
    batch = load_pickle(filepath)

    image = batch['data'] #图像数据,形状为(10000, 3072),每行表示一张图像的像素值.
    labels = batch['labels'] #标签数据,形状为(10000,),每个元素表示对应图像的类别标签.

    #将图像数据从(10000, 3072)的形状转换为(10000, 3, 32, 32),
    #每张图像有3个颜色通道(红、绿、蓝),每个通道是32x32像素.
    image = image.reshape(-1, 3, 32, 32)
    #将图像数据从(10000, 3, 32, 32)转换为(10000, 32, 32, 3),符合常见的图像数据格式.
    image = image.transpose(0, 2, 3, 1) 
    labels = np.array(labels, dtype=np.int64) #将标签列表转换为NumPy数组,方便后续处理.
    
    return image, labels


#加载CIFAR-10数据集的所有训练和测试数据,返回图像数据和标签数据的NumPy数组.
def load_all_cifar10():
    images = []
    labels = []

    #CIFAR-10数据集包含5个训练batch文件和1个测试batch文件.
    for i in range(1, 6):
        batch_file = os.path.join(EXTRACT_DIR, f'data_batch_{i}')
        image, label = load_batch(batch_file)
        images.append(image)
        labels.append(label)

    #加载测试batch文件
    test_batch_file = os.path.join(EXTRACT_DIR, 'test_batch')
    test_image, test_label = load_batch(test_batch_file)
    images.append(test_image)
    labels.append(test_label)

    #将所有训练和测试图像数据合并成一个大的NumPy数组.
    images = np.concatenate(images, axis=0) #合并图像数据,形状为(60000, 32, 32, 3).
    labels = np.concatenate(labels, axis=0) #合并标签数据,形状为(60000,).

    return images, labels


#加载类别名称,从batches.meta文件中读取类别名称列表,每个元素是一个字符串表示类别名称.
def load_classes_name():
    meta_file = os.path.join(EXTRACT_DIR, 'batches.meta')
    meta = load_pickle(meta_file)
    return meta['label_names'] #返回类别名称列表,长度为10,每个元素是一个字符串表示类别名称.


#将图像数据、标签数据和类别名称保存为NumPy数组文件,方便后续使用和加载.
def save_numpy_data(images, labels, class_names):
    np.save(os.path.join(NUMPY_DIR, 'cifar10_images.npy'), images) #将图像数据保存为NumPy数组文件.
    np.save(os.path.join(NUMPY_DIR, 'cifar10_labels.npy'), labels) #将标签数据保存为NumPy数组文件.
    np.save(os.path.join(NUMPY_DIR, 'cifar10_classes.npy'), np.array(class_names)) #将类别名称保存为NumPy数组文件.
    print("NumPy data saved successfully.")


#将图像数据保存为PNG文件,按照类别名称组织目录.每个类别一个子目录,图像文件命名为"image_i.png".
def save_images(images, labels, class_names):
    for i in range(len(images)):
        image = images[i] #获取第i张图像数据,形状为(32, 32, 3).
        label = labels[i] #获取第i张图像的标签,是一个整数表示类别索引.
        class_name = class_names[label] #根据标签索引获取对应的类别名称.

        #创建类别名称对应的目录,如果目录不存在则创建.
        class_dir = os.path.join(IMAGES_DIR, class_name)
        check_path_without_print(class_dir)

        #保存图像文件,文件名格式为"image_i.png",其中i是图像的索引.
        image_path = os.path.join(class_dir, f'image_{i}.png')
        img = Image.fromarray(image) #将NumPy数组转换为PIL图像对象.
        img.save(image_path) #保存图像文件.

    print("Images saved successfully.")


def main():
    download_cifar10() #下载CIFAR-10数据集
    extract_cifar10() #解压CIFAR-10数据集

    images, labels = load_all_cifar10() #加载CIFAR-10数据集的所有图像和标签数据
    class_names = load_classes_name() #加载类别名称列表

    print(f"Loaded {len(images)} images and {len(labels)} labels.") #打印加载的图像和标签数量
    print(f"Class names: {class_names}") #打印类别名称列表
    print(f"Image shape: {images[0].shape}") #打印第一张图像的形状,应该是(32, 32, 3)
    print(f"Label of first image: {labels[0]} ({class_names[labels[0]]})") #打印第一张图像的标签索引和对应的类别名称
    print(f"Unique labels: {np.unique(labels)}") #打印唯一的标签索引,应该是0-9

    save_images(images, labels, class_names) #将图像保存为PNG文件,按照类别名称组织目录.
    save_numpy_data(images, labels, class_names) #将图像数据、标签数据和类别名称保存为NumPy数组文件.


if __name__ == "__main__":
    main()

