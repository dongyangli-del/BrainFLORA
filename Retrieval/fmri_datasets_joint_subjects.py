import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
import pickle
import os
import clip

# proxy = 'http://127.0.0.1:7890'
# os.environ['http_proxy'] = proxy
# os.environ['https_proxy'] = proxy
cuda_device_count = torch.cuda.device_count()
print(cuda_device_count)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# vlmodel, preprocess = clip.load("ViT-B/32", device=device)
model_type = 'ViT-H-14'

# import open_clip
# vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
#     model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device=device)

import json
from omegaconf import OmegaConf
import os

cfg = OmegaConf.load(os.path.join("/mnt/dataset1/ldy/Workspace/FLORA/configs/config.yaml"))
cfg = OmegaConf.structured(cfg)
# Access the paths from the config
img_directory_training = cfg.fmridataset.img_directory_training
img_directory_test = cfg.fmridataset.img_directory_test
import itertools
import torch.nn.functional as F

class fMRIDataset():
    """
    subjects = ['sub-01', 'sub-02', 'sub-03']
    """
    def __init__(self, data_path, adap_subject=None, subjects=None, train=True, time_window=[0, 1.0], classes = None, pictures = None):
        self.data_path = data_path
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 720 if train else 100
        self.classes = classes
        self.pictures = pictures
        self.adap_subject = adap_subject  # 保存这个参数
        self.modal = 'fmri'
        # assert any subjects in subject_list
        assert any(sub in self.subject_list for sub in self.subjects)

        self.data, self.labels, self.text, self.img = self.load_data()
        
        # self.data = self.extract_eeg(self.data, time_window)

        # Calculate the length of data for each subject
        self.subject_data_lens = [data.shape[0] for data in self.data]
        self.cumulative_data_lens = [0] + list(itertools.accumulate(self.subject_data_lens))  # Cumulative lengths for indexing
        
        
        if self.classes is None and self.pictures is None:
            # Try to load the saved features if they exist
            features_filename = "/mnt/dataset1/ldy/Workspace/EEG_Image_decode/Retrieval/ori_fMRI_ViT-H-14_features_train.pt" if self.train else "/mnt/dataset1/ldy/Workspace/EEG_Image_decode/Retrieval/ori_fMRI_ViT-H-14_features_test.pt"
            
            if os.path.exists(features_filename) :
                saved_features = torch.load(features_filename)
                self.text_features = saved_features['text_features']
                self.img_features = saved_features['img_features']
            else:
                # self.text_features = self.Textencoder(self.text)
                # self.img_features = self.ImageEncoder(self.img)
                # torch.save({
                #     'text_features': self.text_features.cpu(),
                #     'img_features': self.img_features.cpu(),
                # }, features_filename)
                saved_features = torch.load(features_filename)
                self.text_features = saved_features['text_features']
                self.img_features = saved_features['img_features']
        else:
            self.text_features = self.Textencoder(self.text)
            self.img_features = self.ImageEncoder(self.img)

    def load_data(self):
        data_list = []
        label_list = []
        texts = []
        images = []

        # 判断是训练集还是测试集
        if self.train:
            directory = img_directory_training
        else:
            directory = img_directory_test

        # 获取该路径下的所有目录
        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()
        if self.classes is not None:
            dirnames = [dirnames[i] for i in self.classes]

        # 构建文本描述
        for dir in dirnames:
            new_description = f"This picture is {dir}"
            texts.append(new_description)

        # 获取所有图片路径
        if self.train:
            img_directory = img_directory_training
        else:
            img_directory = img_directory_test

        all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()
        images = [os.path.join(os.path.join(img_directory, folder), img) for folder in all_folders for img in os.listdir(os.path.join(img_directory, folder)) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for subject in self.subjects:
            if self.train:
                # if subject == self.adap_subject:
                #     continue
                file_name = 'train_responses.pkl'
            else:
                if subject == self.adap_subject or self.adap_subject is None:
                    file_name = 'test_responses.pkl'
                else:
                    continue
            file_path = os.path.join(self.data_path, subject, file_name)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                preprocessed_eeg_data = torch.from_numpy(data).float().detach()
                preprocessed_eeg_data = preprocessed_eeg_data.view(-1, *preprocessed_eeg_data.shape[1:])

                if self.train:
                    # 训练集处理
                    n_classes, samples_per_class = 720, 12
                    subject_data = []
                    subject_labels = []
                    for i in range(n_classes):
                        start_index = i * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index + samples_per_class]
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
                        subject_data.append(preprocessed_eeg_data_class)
                        subject_labels.append(labels)
                    
                    data_tensor = torch.cat(subject_data, dim=0).view(-1, *subject_data[0].shape[2:])
                    label_tensor = torch.cat(subject_labels, dim=0)

                else:
                    # 测试集处理，进行平均操作
                    n_classes, samples_per_class = 100, 1
                    subject_data = []
                    subject_labels = []
                    for i in range(n_classes):
                        if self.classes is not None and i not in self.classes:
                            continue
                        start_index = i * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index + samples_per_class]
                        # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                        preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), 0)  # 进行平均操作
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
                        subject_data.append(preprocessed_eeg_data_class.unsqueeze(0))  # 添加维度以便后续拼接
                        # subject_data.append(preprocessed_eeg_data_class)  # 添加维度以便后续拼接
                        subject_labels.append(labels)
                    # data_tensor = torch.cat(subject_data, dim=0).view(-1, *subject_data[0].shape[2:])
                    data_tensor = torch.cat(subject_data, dim=0)
                    label_tensor = torch.cat(subject_labels, dim=0)

                data_list.append(data_tensor)
                label_list.append(label_tensor)


        
        # if self.train:                           
        #     data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])
        #     print("data_tensor", data_tensor.shape)
        # else:                       
        #     data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)
        #     # data_tensor = data_tensor[:, 0, :]        
        # label_tensor = torch.cat(label_list, dim=0)
        

        # print(f"Data tensor shape: {data_tensor.shape}, label tensor shape: {label_tensor.shape}, text length: {len(texts)}, image length: {len(images)}")
        # 不再拼接不同被试的数据，直接返回各自的列表
        if self.train:
            for i in range(len(self.subjects)):            
                print("data_list", data_list[i].shape[1])
                # print("label_list", len(label_list))
                print(f"Data list length: {len(data_list[i])}, label list length: {len(label_list[i])}, text length: {len(texts)}, image length: {len(images)}")
        else:            
            print(f"Data list length: {len(data_list[0])}, label list length: {len(label_list[0])}, text length: {len(texts)}, image length: {len(images)}")    

        return data_list, label_list, texts, images




    def __getitem__(self, index):
        # Step 1: Find which subject the index belongs to
        subject_idx = None
        for i, cum_len in enumerate(self.cumulative_data_lens[1:]):
            if index < cum_len:
                subject_idx = i
                break
        subject_offset = index - self.cumulative_data_lens[subject_idx]
        
        # Step 2: Get the data and label for the specific subject and offset
        x = self.data[subject_idx][subject_offset]
        label = self.labels[subject_idx][subject_offset]
        
        # Get the subject ID from fmri_subjects list
        subject_id = self.subjects[subject_idx]  # 获取被试标识符
        
        # Pad the fmri data (x) to 7000 if necessary
        # target_length = 7000
        target_length = 11000
        if x.shape[0] < target_length:
            # If x is smaller than the target size, pad it
            padding_size = target_length - x.shape[0]
            x = F.pad(x, (0, padding_size), value=0)  # Pad at the end with zeros
        elif x.shape[0] > target_length:
            # If x is larger than the target size, truncate it
            x = x[:target_length]

        # Step 3: Calculate text and img indices based on subject-specific parameters
        index_n_sub_train = self.n_cls * 12 * 1
        index_n_sub_test = self.n_cls * 12 * 1

        if self.train:
            text_index = (subject_offset % index_n_sub_train) // (12 * 1)
            img_index = (subject_offset % index_n_sub_train) // (1)
        else:
            text_index = (subject_offset % index_n_sub_test) // (1)
            img_index = (subject_offset % index_n_sub_test) // (1)
        # print("self.img", len(self.img))
        
        # print("self.img_features", self.img_features.shape)
        # print("img_index", img_index)
        # 获取文本、图像和特征
        text = self.text[text_index]
        img = self.img[img_index]
        text_features = self.text_features[text_index]
        img_features = self.img_features[img_index]

        return self.modal, x, label, text, text_features, img, img_features, subject_id  # 返回subject_id

    # def __getitem__(self, index):
    #     # Step 1: Find which subject the index belongs to
    #     subject_idx = None
    #     for i, cum_len in enumerate(self.cumulative_data_lens[1:]):
    #         if index < cum_len:
    #             subject_idx = i
    #             break
    #     subject_offset = index - self.cumulative_data_lens[subject_idx]
        
    #     # Step 2: Get the data and label for the specific subject and offset
    #     x = self.data[subject_idx][subject_offset]
    #     label = self.labels[subject_idx][subject_offset]
        
    #     # Get the subject ID from fmri_subjects list
    #     subject_id = self.subjects[subject_idx]  # 获取被试标识符
        
    #     # Step 3: Calculate text and img indices based on subject-specific parameters
    #     index_n_sub_train = self.n_cls * 12 * 1
    #     index_n_sub_test = self.n_cls * 12 * 1

    #     if self.train:
    #         text_index = (subject_offset % index_n_sub_train) // (12 * 1)
    #         img_index = (subject_offset % index_n_sub_train) // (1)
    #     else:
    #         text_index = (subject_offset % index_n_sub_test) // (1)
    #         img_index = (subject_offset % index_n_sub_test) // (1)

    #     # 获取文本、图像和特征
    #     text = self.text[text_index]
    #     img = self.img[img_index]
    #     text_features = self.text_features[text_index]
    #     img_features = self.img_features[img_index]

    #     return self.modal, x, label, text, text_features, img, img_features, subject_id  # 返回subject_id

    def __len__(self):
        return sum(self.subject_data_lens)  # 所有被试的数据总长度

    # def __getitem__(self, index):
    #     # Step 1: Find which subject the index belongs to
    #     subject_idx = None
    #     for i, cum_len in enumerate(self.cumulative_data_lens[1:]):
    #         if index < cum_len:
    #             subject_idx = i
    #             break
    #     subject_offset = index - self.cumulative_data_lens[subject_idx]
        
    #     # Step 2: Get the data and label for the specific subject and offset
    #     x = self.data[subject_idx][subject_offset]
    #     label = self.labels[subject_idx][subject_offset]

    #     # Step 3: Calculate text and img indices based on subject-specific parameters
    #     if self.pictures is None:
    #         if self.classes is None:
    #             index_n_sub_train = self.n_cls * 12 * 1
    #             index_n_sub_test = self.n_cls * 12 * 1
    #         else:
    #             index_n_sub_test = len(self.classes) * 12 * 1
    #             index_n_sub_train = len(self.classes) * 12 * 1

    #         if self.train:
    #             text_index = (subject_offset % index_n_sub_train) // (12 * 1)
    #             img_index = (subject_offset % index_n_sub_train) // (1)
    #         else:
    #             text_index = (subject_offset % index_n_sub_test) // (1)
    #             img_index = (subject_offset % index_n_sub_test) // (1)
    #     else:
    #         if self.classes is None:
    #             index_n_sub_train = self.n_cls * 1 * 1
    #             index_n_sub_test = self.n_cls * 1 * 12
    #         else:
    #             index_n_sub_test = len(self.classes) * 1 * 12
    #             index_n_sub_train = len(self.classes) * 1 * 1

    #         if self.train:
    #             text_index = (subject_offset % index_n_sub_train) // (1 * 1)
    #             img_index = (subject_offset % index_n_sub_train) // (1)
    #         else:
    #             text_index = (subject_offset % index_n_sub_test) // (1)
    #             img_index = (subject_offset % index_n_sub_test) // (1)

    #     # 获取文本、图像和特征
    #     text = self.text[text_index]
    #     img = self.img[img_index]
    #     text_features = self.text_features[text_index]
    #     img_features = self.img_features[img_index]

    #     return self.modal, x, label, text, text_features, img, img_features, -1
    
        
    # def __getitem__(self, index):
    #     # Get the data and label corresponding to "index"
    #     # index: (subjects * classes * 12 * 1)
    #     x = self.data[index]
    #     label = self.labels[index]
        
    #     if self.pictures is None:
    #         if self.classes is None:
    #             index_n_sub_train = self.n_cls * 12 * 1
    #             index_n_sub_test = self.n_cls * 12 * 1
    #         else:
    #             index_n_sub_test = len(self.classes)* 12 * 1
    #             index_n_sub_train = len(self.classes)* 12 * 1
    #         # text_index: classes
    #         if self.train:
    #             text_index = (index % index_n_sub_train) // (12 * 1)
    #         else:
    #             text_index = (index % index_n_sub_test) // (1)
    #         # img_index: classes * 10
    #         if self.train:
    #             img_index = (index % index_n_sub_train) // (1)
    #         else:
    #             img_index = (index % index_n_sub_test) // (1)
    #     else:
    #         if self.classes is None:
    #             index_n_sub_train = self.n_cls * 1 * 1
    #             index_n_sub_test = self.n_cls * 1 * 12
    #         else:
    #             index_n_sub_test = len(self.classes)* 1 * 12
    #             index_n_sub_train = len(self.classes)* 1 * 1
    #         # text_index: classes
    #         if self.train:
    #             text_index = (index % index_n_sub_train) // (1 * 1)
    #         else:
    #             text_index = (index % index_n_sub_test) // (1 * 12)
    #         # img_index: classes * 10
    #         if self.train:
    #             img_index = (index % index_n_sub_train) // (1)
    #         else:
    #             img_index = (index % index_n_sub_test) // (12)
                
    #     text = self.text[text_index]
    #     # print("self.img", len(self.img))
    #     # print("img_index", img_index)
    #     img = self.img[img_index]
        
    #     text_features = self.text_features[text_index]
    #     img_features = self.img_features[img_index]
        
    #     return x, label, text, text_features, img, img_features

    # def __len__(self):
    #     return self.data[0].shape[0]  # or self.labels.shape[0] which should be the same

if __name__ == "__main__":
    # Instantiate the dataset and dataloader
    data_path = "/mnt/dataset0/ldy/datasets/fmri_dataset/Preprosessed"  # Replace with the path to your data
    data_path = data_path
    train_dataset = fMRIDataset(data_path, subjects = ['sub-01', 'sub-02', 'sub-03'], train=True)    
    test_dataset = fMRIDataset(data_path, subjects = ['sub-01', 'sub-02', 'sub-03'], train=False)
    # train_dataset = MEGDataset(data_path, adap_subject = 'sub-01', train=True)    
    # test_dataset = MEGDataset(data_path, adap_subject = 'sub-01', train=False)    
    # train_dataset = MEGDataset(data_path, train=True) 
    # test_dataset = MEGDataset(data_path, train=False) 
    # 训练的eeg数据：torch.Size([16540, 4, 17, 100]) [训练图像数量，训练图像重复数量，通道数，脑电信号时间点]
    # 测试的eeg数据：torch.Size([200, 80, 17, 100])
    # 1秒 'times': array([-0.2 , -0.19, -0.18, ... , 0.76,  0.77,  0.78, 0.79])}
    # 17个通道'ch_names': ['Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2']
    # 100 Hz
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    i = 80*1-1
    x, label, text, text_features, img, img_features  = test_dataset[i]
    print(f"Index {i}, Label: {label}, text: {text}")
    Image.open(img)
            
    
        
    