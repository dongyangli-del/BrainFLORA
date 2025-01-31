import os
import pickle
import numpy as np
import mne
import pandas as pd
from collections import defaultdict
import logging
import random
import shutil

# 设置日志格式和级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_crop_epochs(fif_file):
    """加载并裁剪 MEG 事件数据。"""
    logging.info(f"加载 MEG 数据：{fif_file}")
    epochs = mne.read_epochs(fif_file, preload=True)
    epochs.crop(tmin=0, tmax=1.0)
    logging.info("裁剪 epochs 时间窗口到 0 到 1.0 秒")

    # 按事件 ID 对 epochs 进行排序
    sorted_indices = np.argsort(epochs.events[:, 2])
    epochs = epochs[sorted_indices]

    # 过滤掉事件 ID 为 999999 的事件
    filtered_epochs = epochs[epochs.events[:, 2] != 999999]
    logging.info(f"过滤掉事件 ID 为 999999 的事件，剩余 {len(filtered_epochs)} 个事件")

    # 返回过滤后的 epochs
    logging.info(f"总共读取了 {len(filtered_epochs)} 个 epochs")
    return filtered_epochs

def save_data(data, filename):
    """保存数据到指定文件。"""
    logging.info(f"保存数据到 {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def build_category_prefix_mapping(meg_image_dir):
    """构建类别名称到带前缀的类别名称的映射。"""
    logging.info(f"构建类别前缀映射：{meg_image_dir}")
    category_prefix_mapping = {}
    for split in ["training_images", "test_images"]:
        split_dir = os.path.join(meg_image_dir, split)
        if os.path.isdir(split_dir):
            for category_with_prefix in os.listdir(split_dir):
                if "_" in category_with_prefix:
                    prefix, category = category_with_prefix.split("_", 1)
                    category_prefix_mapping[category] = category_with_prefix
    logging.info(f"共找到 {len(category_prefix_mapping)} 个类别映射")
    return category_prefix_mapping

def get_full_image_event_mapping(csv_img_file_path, category_prefix_mapping):
    """根据 CSV 文件创建完整的图像事件映射。"""
    logging.info(f"读取图像路径文件：{csv_img_file_path}")
    full_image_event_mapping = {}
    df = pd.read_csv(csv_img_file_path, header=None)
    for event_id, image_path in enumerate(df[0], start=1):
        parts = image_path.strip().split("/")
        if len(parts) > 1:
            category = parts[1]  # 假设类别是路径的第二部分
            if category in category_prefix_mapping:
                category_with_prefix = category_prefix_mapping[category]
                img = os.path.basename(image_path)
                full_image_event_mapping[event_id] = (category_with_prefix, img)
    logging.info(f"创建了 {len(full_image_event_mapping)} 个事件映射")
    return full_image_event_mapping

def exclude_event_ids_with_exact_count(epochs, count_to_exclude=12):
    """排除出现次数恰好为指定次数的事件 ID。"""
    logging.info(f"排除出现次数恰好为 {count_to_exclude} 次的事件 ID")
    event_id_counts = defaultdict(int)
    for event_id in epochs.events[:, 2]:
        event_id_counts[event_id] += 1

    # 找出需要排除的事件 ID
    event_ids_to_exclude = {event_id for event_id, count in event_id_counts.items() if count == count_to_exclude}
    logging.info(f"共有 {len(event_ids_to_exclude)} 个事件 ID 将被排除")

    # 过滤 epochs
    indices_to_keep = [idx for idx, event in enumerate(epochs.events) if event[2] not in event_ids_to_exclude]
    filtered_epochs = epochs[indices_to_keep]
    logging.info(f"过滤后剩余 {len(filtered_epochs)} 个 epochs")
    return filtered_epochs

def split_categories_randomly(category_list, test_category_count, seed=42):
    """随机划分类别为测试集和训练集。"""
    logging.info("开始随机划分类别为测试集和训练集")
    random.seed(seed)
    test_categories = random.sample(category_list, test_category_count)
    train_categories = [category for category in category_list if category not in test_categories]
    logging.info(f"测试集类别数量：{len(test_categories)}，训练集类别数量：{len(train_categories)}")
    return test_categories, train_categories

def assign_epochs_to_sets(epochs, full_image_event_mapping, test_categories, train_categories, max_epochs_per_class=12):
    """根据类别将 epochs 分配到测试集和训练集，并限制每个类别的最大 epochs 数量。"""
    logging.info("根据类别将 epochs 分配到测试集和训练集，限制每个类别的最大 epochs 数量")
    test_indices = []
    train_indices = []
    event_category_mapping = {}
    category_epoch_counts = defaultdict(int)

    for idx in range(len(epochs)):
        event_id = epochs.events[idx, 2]
        if event_id in full_image_event_mapping:
            category_with_prefix, _ = full_image_event_mapping[event_id]
            # 去掉原有的数字前缀，只保留类别名称
            if "_" in category_with_prefix:
                _, category_name = category_with_prefix.split("_", 1)
            else:
                category_name = category_with_prefix
            event_category_mapping[event_id] = category_name
            if category_epoch_counts[category_name] >= max_epochs_per_class:
                continue  # 超过最大次数，跳过
            category_epoch_counts[category_name] += 1
            if category_name in test_categories:
                test_indices.append(idx)
            elif category_name in train_categories:
                train_indices.append(idx)
    logging.info(f"分配后训练集 epochs 数量：{len(train_indices)}，测试集 epochs 数量：{len(test_indices)}")
    return train_indices, test_indices, event_category_mapping

def arrange_epochs_alphabetically(epochs, indices, event_ids, event_category_mapping):
    """根据类别的字母顺序排列 epochs。"""
    logging.info("根据类别的字母顺序排列 epochs")
    categories = [event_category_mapping[event_id] for event_id in event_ids]
    sorted_pairs = sorted(zip(categories, indices), key=lambda x: x[0])
    sorted_indices = [idx for _, idx in sorted_pairs]
    arranged_epochs = epochs[sorted_indices]
    arranged_categories = [event_category_mapping[epochs.events[idx, 2]] for idx in sorted_indices]
    return arranged_epochs, arranged_categories

def copy_images(arranged_categories, full_image_event_mapping, event_ids, meg_image_dir, output_image_dir, set_type):
    """复制对应的刺激图片到指定目录。"""
    logging.info(f"开始复制 {set_type} 刺激图片")
    # 创建类别到新数字前缀的映射
    unique_categories = sorted(set(arranged_categories))
    category_to_index = {category: f"{idx+1:05d}_{category}" for idx, category in enumerate(unique_categories)}

    for idx, event_id in enumerate(event_ids):
        if event_id in full_image_event_mapping:
            original_category_with_prefix, img = full_image_event_mapping[event_id]
            # 去掉原有的数字前缀，只保留类别名称
            if "_" in original_category_with_prefix:
                _, category_name = original_category_with_prefix.split("_", 1)
            else:
                category_name = original_category_with_prefix
            if category_name in category_to_index:
                folder_name = category_to_index[category_name]
                # 尝试从原始训练集或测试集目录中查找图片
                src_image_path_train = os.path.join(meg_image_dir, "training_images", original_category_with_prefix, img)
                src_image_path_test = os.path.join(meg_image_dir, "test_images", original_category_with_prefix, img)
                if os.path.exists(src_image_path_train):
                    src_image_path = src_image_path_train
                elif os.path.exists(src_image_path_test):
                    src_image_path = src_image_path_test
                else:
                    logging.warning(f"无法找到图片 {img}，跳过")
                    continue  # 如果图片不存在，跳过

                dest_folder = os.path.join(output_image_dir, set_type, folder_name)
                os.makedirs(dest_folder, exist_ok=True)
                dest_image_path = os.path.join(dest_folder, img)
                if not os.path.exists(dest_image_path):
                    shutil.copyfile(src_image_path, dest_image_path)
    logging.info(f"{set_type} 刺激图片复制完成")

def main():
    # 设置路径
    csv_img_file_path = "/mnt/dataset0/ldy/4090_Workspace/4090_THINGS/osfstorage/THINGS/Metadata/Image-specific/image_paths.csv"
    meg_image_dir = "/mnt/dataset0/ldy/datasets/THINGS_MEG/images_set"
    output_dir = "/mnt/dataset0/ldy/datasets/THINGS_MEG/preprocessed_random/sub-02"
    output_image_dir = "/mnt/dataset0/ldy/datasets/THINGS_MEG/random_image_set"
    base_fif_dir = "/mnt/dataset0/ldy/datasets/meg_dataset/original_preprocessed/preprocessed"

    # 创建必要的目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_image_dir, "test_images"), exist_ok=True)
    os.makedirs(os.path.join(output_image_dir, "training_images"), exist_ok=True)

    # 构建映射
    category_prefix_mapping = build_category_prefix_mapping(meg_image_dir)
    full_image_event_mapping = get_full_image_event_mapping(csv_img_file_path, category_prefix_mapping)

    # 定义受试者列表
    subjects = [
        ('sub-02', 'preprocessed_P2-epo.fif'),
        # 可以在此添加更多受试者
    ]

    for subject_id, fif_filename in subjects:
        logging.info(f"开始处理受试者 {subject_id}")
        fif_file = os.path.join(base_fif_dir, fif_filename)
        epochs = load_and_crop_epochs(fif_file)

        # 排除出现次数恰好为12次的事件 ID
        epochs = exclude_event_ids_with_exact_count(epochs, count_to_exclude=12)

        # 获取所有类别列表
        all_categories = set()
        for event_id in epochs.events[:, 2]:
            if event_id in full_image_event_mapping:
                category_with_prefix, _ = full_image_event_mapping[event_id]
                # 去掉原有的数字前缀，只保留类别名称
                if "_" in category_with_prefix:
                    _, category_name = category_with_prefix.split("_", 1)
                else:
                    category_name = category_with_prefix
                all_categories.add(category_name)
        all_categories = sorted(all_categories)

        # 随机划分类别为测试集和训练集
        test_categories, train_categories = split_categories_randomly(all_categories, test_category_count=200, seed=42)

        # 分配 epochs 到训练集和测试集，限制每个类别最多12个epochs
        max_epochs_per_class = 12
        train_indices, test_indices, event_category_mapping = assign_epochs_to_sets(
            epochs, full_image_event_mapping, test_categories, train_categories, max_epochs_per_class)

        # 提取事件 ID 列表
        train_event_ids = [epochs.events[idx, 2] for idx in train_indices]
        test_event_ids = [epochs.events[idx, 2] for idx in test_indices]

        # 根据类别字母顺序排列 epochs
        train_epochs, train_categories_ordered = arrange_epochs_alphabetically(
            epochs, train_indices, train_event_ids, event_category_mapping)
        test_epochs, test_categories_ordered = arrange_epochs_alphabetically(
            epochs, test_indices, test_event_ids, event_category_mapping)

        # 提取数据
        train_data = train_epochs.get_data()
        test_data = test_epochs.get_data()
        ch_names = train_epochs.ch_names
        times = train_epochs.times

        # 统计训练集和测试集的类别数量和每个类别的epochs数量
        unique_train_categories = sorted(set(train_categories_ordered))
        unique_test_categories = sorted(set(test_categories_ordered))

        train_category_counts = defaultdict(int)
        for category in train_categories_ordered:
            train_category_counts[category] += 1

        test_category_counts = defaultdict(int)
        for category in test_categories_ordered:
            test_category_counts[category] += 1

        logging.info(f"测试集包含 {len(unique_test_categories)} 个类别，{len(test_epochs)} 个 epochs")
        logging.info(f"训练集包含 {len(unique_train_categories)} 个类别，{len(train_epochs)} 个 epochs")
        
        # 保存数据
        save_data({'meg_data': train_data, 'ch_names': ch_names, 'times': times},
                  os.path.join(output_dir, "preprocessed_meg_training.pkl"))
        save_data({'meg_data': test_data, 'ch_names': ch_names, 'times': times},
                  os.path.join(output_dir, "preprocessed_meg_test.pkl"))

        # 复制训练集和测试集的图片
        copy_images(train_categories_ordered, full_image_event_mapping, train_event_ids,
                    meg_image_dir, output_image_dir, set_type="training_images")
        copy_images(test_categories_ordered, full_image_event_mapping, test_event_ids,
                    meg_image_dir, output_image_dir, set_type="test_images")

    logging.info("所有受试者处理完成！")

if __name__ == "__main__":
    main()
