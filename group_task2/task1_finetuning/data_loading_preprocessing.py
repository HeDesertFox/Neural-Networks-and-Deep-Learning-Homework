import os
import requests
import tarfile
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import pandas as pd

def download_dataset(url, data_dir):
    """
    下载并解压数据集。如果数据集的.tgz文件尚未存在，则进行下载和解压。
    参数:
        url (str): 数据集下载链接。
        data_dir (str): 数据存放目录。
    """
    tgz_path = os.path.join(data_dir, 'CUB_200_2011.tgz')
    if not os.path.exists(tgz_path):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        print("正在下载数据集...")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        try:
            with open(tgz_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()

            if os.path.getsize(tgz_path) != total_size_in_bytes:
                raise Exception("下载文件大小与预期不符。")

            print("正在解压数据集...")
            with tarfile.open(tgz_path, 'r:gz') as tar_ref:
                tar_ref.extractall(data_dir)
        except Exception as e:
            print(f"发生错误：{e}")
            os.remove(tgz_path)  # 删除损坏的文件
    else:
        print("数据集已存在，无需再次下载。")

def create_filename_to_split_mapping(data_dir):
    """
    根据数据目录创建文件名到训练/验证集划分的映射。
    参数:
        data_dir (str): 数据集的目录。
    返回:
        dict: 包含文件路径与数据集划分（训练/验证）的映射。
    """
    images_path = os.path.join(data_dir, 'CUB_200_2011', 'images.txt')
    split_path = os.path.join(data_dir, 'CUB_200_2011', 'train_test_split.txt')

    image_id_to_filename = pd.read_csv(images_path, delim_whitespace=True, header=None, index_col=0)[1].to_dict()
    image_id_to_split = pd.read_csv(split_path, delim_whitespace=True, header=None, index_col=0)[1].to_dict()

    filename_to_split = {}
    base_image_path = os.path.join(data_dir, 'CUB_200_2011', 'images')
    for image_id, relative_path in image_id_to_filename.items():
        full_path = os.path.normpath(os.path.join(base_image_path, relative_path))
        filename_to_split[full_path] = image_id_to_split[image_id]

    return filename_to_split

def load_data(data_dir, batch_size, download_url=None):
    """
    加载并预处理数据集。
    参数:
        data_dir (str): 数据集的目录。
        batch_size (int): 批处理大小。
        download_url (str, optional): 数据集的下载URL。
    返回:
        tuple: 包含训练集和验证集的数据加载器。
    """
    if download_url:
        download_dataset(download_url, data_dir)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'CUB_200_2011', 'images'), transform=transform)

    filename_to_split = create_filename_to_split_mapping(data_dir)

    train_indices = [i for i, (img_path, _) in enumerate(full_dataset.imgs)
                     if filename_to_split[os.path.normpath(img_path)] == 1]
    val_indices = [i for i, (img_path, _) in enumerate(full_dataset.imgs)
                   if filename_to_split[os.path.normpath(img_path)] == 0]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
