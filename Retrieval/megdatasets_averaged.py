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

# proxy = 'http://127.0.0.1:7890'
# os.environ['http_proxy'] = proxy
# os.environ['https_proxy'] = proxy
cuda_device_count = torch.cuda.device_count()
print(cuda_device_count)
device = "cuda:1" if torch.cuda.is_available() else "cpu"
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
img_directory_training = cfg.megdataset.img_directory_training
img_directory_test = cfg.megdataset.img_directory_test


class MEGDataset():
    """
    subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04']
    """
    def __init__(self, data_path, adap_subject=None, subjects=None, train=True, time_window=[0, 1.0], classes = None, pictures = None):
        self.data_path = data_path
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 1654 if train else 200
        self.classes = classes
        self.pictures = pictures
        self.adap_subject = adap_subject  # 保存这个参数
        self.modal = 'meg'
        # assert any subjects in subject_list
        # assert any(sub in self.subject_list for sub in self.subjects)

        self.data, self.labels, self.text, self.img = self.load_data()
        
        self.data = self.extract_eeg(self.data, time_window)
        
        
        if self.classes is None and self.pictures is None:
            # Try to load the saved features if they exist
            features_filename = os.path.join(f'/mnt/dataset1/ldy/Workspace/FLORA/data_preparing/newsplit_MEG_ViT-H-14_features_train.pt') if self.train else os.path.join(f'/mnt/dataset1/ldy/Workspace/FLORA/data_preparing/newsplit_MEG_ViT-H-14_features_test.pt')
            
            if os.path.exists(features_filename) :
                saved_features = torch.load(features_filename)
                self.text_features = saved_features['text_features']
                self.img_features = saved_features['img_features']
            else:
                self.text_features = self.Textencoder(self.text)
                self.img_features = self.ImageEncoder(self.img)
                torch.save({
                    'text_features': self.text_features.cpu(),
                    'img_features': self.img_features.cpu(),
                }, features_filename)
        else:
            self.text_features = self.Textencoder(self.text)
            self.img_features = self.ImageEncoder(self.img)
            
    def load_data(self):
        data_list = []
        label_list = []
        texts = []
        images = []
        
        if self.train:
            directory = img_directory_training
        else:
            directory = img_directory_test
        # 获取该路径下的所有目录
        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()
        
        if self.classes is not None:
            dirnames = [dirnames[i] for i in self.classes]

        for dir in dirnames:
            # 尝试找到第一个'_'的位置
            # try:
            #     idx = dir.index('_')
            #     description = dir[idx+1:]  # 从第一个'_'之后取得所有内容
            # except ValueError:
            #     print(f"Skipped: {dir} due to no '_' found.")
            #     continue
            # description = dir    
            new_description = f"This picture is {dir}"
            texts.append(new_description)

        if self.train:
            img_directory = img_directory_training  # 请将其替换为你的新地址
        else:
            img_directory = img_directory_test
        
        all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()  # 保证文件夹的顺序


        
        images = []  # 初始化images列表
        for folder in all_folders:
            folder_path = os.path.join(img_directory, folder)
            all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()  
            images.extend(os.path.join(folder_path, img) for img in all_images)

            
        print("self.subjects", self.subjects)
        print("adap_subject", self.adap_subject)
        for subject in self.subjects:
            if self.train:
                if subject == self.adap_subject:  # 跳过被排除的被试
                    continue            
                # print("subject:", subject)    
                file_name = 'preprocessed_meg_training.pkl'

                file_path = os.path.join(self.data_path, subject, file_name)
                print(f"{file_path}")
                # 读取pkl文件
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                    
                    preprocessed_eeg_data = torch.from_numpy(data['meg_data']).float().detach()                
                    # preprocessed_eeg_data = preprocessed_eeg_data.view(-1, *preprocessed_eeg_data.shape[2:])
                    # print("preprocessed_eeg_data", preprocessed_eeg_data.shape)
                    times = torch.from_numpy(data['times']).detach()
                    ch_names = data['ch_names']  # 保留为 Python 列表，或者进行适当的编码

                    n_classes = 1654  # 每个类包含10张图片
                    samples_per_class = 12  # 一个类有十个数据
                    

                    for i in range(n_classes):
                        start_index = i * samples_per_class
                        # if self.adap_subject==None:
                        #     preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        # else:
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index + samples_per_class]
                        # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)


                        # print("train preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                        # preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class, 1)
                        # preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class, 0)
                        # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  # 添加类标签
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)

                 
            else:
                if subject == self.adap_subject or self.adap_subject==None:                                          
                    file_name = 'preprocessed_meg_test.pkl'
                    file_path = os.path.join(self.data_path, subject, file_name)
                    # 读取pkl文件
                    with open(file_path, 'rb') as file:
                        data = pickle.load(file)
                        preprocessed_eeg_data = torch.from_numpy(data['meg_data']).float().detach()
                        # preprocessed_eeg_data = preprocessed_eeg_data.view(-1, *preprocessed_eeg_data.shape[2:])
                        
                        times = torch.from_numpy(data['times']).detach()
                        ch_names = data['ch_names']  # 保留为 Python 列表，或者进行适当的编码
                        n_classes = 200  # Each class contains 1 images
                        print("preprocessed_eeg_data", preprocessed_eeg_data.shape)
                        samples_per_class = 12  # 一个类有1个数据

                        for i in range(n_classes):
                            if self.classes is not None and i not in self.classes:  # If we've defined specific classes and the current class is not in the list, skip
                                continue
                            start_index = i * samples_per_class  # Update start_index for each class
                            preprocessed_eeg_data_class = preprocessed_eeg_data[start_index:start_index+samples_per_class]
                            
                            # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                            labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  # Add class labels
                            preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class, 0)
                            # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                            # preprocessed_eeg_data_class = preprocessed_eeg_data_class.squeeze(0)
                            # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                            # preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), 0)
                            # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                            data_list.append(preprocessed_eeg_data_class)
                            label_list.append(labels)  # Add labels to the label list
                else:
                    continue
        # datalist: (subjects * classes) * (10 * 4 * 17 * 100)
        # data_tensor: (subjects * classes * 10 * 4) * 17 * 100
        # data_list = np.mean(data_list, )
        # print("data_list", len(data_list))
        if self.train:
            # print("data_list", *data_list[0].shape[1:])            
            # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)                 
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[1:])
            # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)   
            # print("label_tensor", label_tensor.shape)
            print("data_tensor", data_tensor.shape)
            label_tensor = torch.cat(label_list, dim=0)
        else:           
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)  
            # data_tensor = data_tensor[:, 0, :, :]
            # label_tensor = torch.cat(label_list, dim=0)
            # print("label_tensor", label_tensor.shape)
            # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])
            label_tensor = torch.cat(label_list, dim=0)[::12]
            print("data_tensor", data_tensor.shape)
        # label_list: (subjects * classes) * 10
        # label_tensor: (subjects * classes * 10)
        # print("label_tensor = torch.cat(label_list, dim=0)")
        # print(label_list)            
        # label_tensor = torch.cat(label_list, dim=0)
        # print(label_tensor[:300])
        if self.train:
            # label_tensor: (subjects * classes * 10 * 4)
            label_tensor = label_tensor.repeat_interleave(1)
            if self.classes is not None:
                unique_values = list(label_tensor.numpy())
                lis = []
                for i in unique_values:
                    if i not in lis:
                        lis.append(i)
                unique_values = torch.tensor(lis)        
                mapping = {val.item(): index for index, val in enumerate(unique_values)}   
                label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)
                
        else:
            # label_tensor = label_tensor.repeat_interleave(80)
            # if self.classes is not None:
            #     unique_values = torch.unique(label_tensor, sorted=False)
           
            #     mapping = {val.item(): index for index, val in enumerate(torch.flip(unique_values, [0]))}
            #     label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)
            pass      

                    
        self.times = times
        self.ch_names = ch_names

        print(f"Data tensor shape: {data_tensor.shape}, label tensor shape: {label_tensor.shape}, text length: {len(texts)}, image length: {len(images)}")
        
        return data_tensor, label_tensor, texts, images

    def extract_eeg(self, eeg_data, time_window):

        start, end = time_window

        # Get the indices of the times within the specified window
        indices = (self.times >= start) & (self.times <= end)
        # print("self.times", self.times.shape)
        # print("indices", indices)
        # print("indices", indices.shape)
        # print("eeg_data", eeg_data.shape)
        # Use these indices to select the corresponding data
        extracted_data = eeg_data[..., indices]
        # print(f"extracted_data shape: {extracted_data.shape}")

        return extracted_data
    
    def Textencoder(self, text):   
            # 使用预处理器将文本转换为模型的输入格式
            text_inputs = torch.cat([clip.tokenize(t) for t in text]).to(device)

            # 使用CLIP模型来编码文本
            with torch.no_grad():
                text_features = vlmodel.encode_text(text_inputs)
            
            text_features = F.normalize(text_features, dim=-1).detach()
       
            return text_features
        
    def ImageEncoder(self,images):
        batch_size = 20  # 设置为合适的值
        image_features_list = []
      
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            image_inputs = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images]).to(device)

            with torch.no_grad():
                batch_image_features = vlmodel.encode_image(image_inputs)
                batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)

            image_features_list.append(batch_image_features)

        image_features = torch.cat(image_features_list, dim=0)
        
        return image_features
    
    def __getitem__(self, index):
        # Get the data and label corresponding to "index"
        # index: (subjects * classes * 12 * 1)
        x = self.data[index]
        label = self.labels[index]
        
        if self.pictures is None:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 12 * 1
                index_n_sub_test = self.n_cls * 1 * 12
            else:
                index_n_sub_test = len(self.classes)* 1 * 12
                index_n_sub_train = len(self.classes)* 12 * 1
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (12 * 1)
            else:
                text_index = (index % index_n_sub_test) // (1)
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // (1)
            else:
                img_index = (index % index_n_sub_test) // (1)
        else:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 1 * 1
                index_n_sub_test = self.n_cls * 1 * 12
            else:
                index_n_sub_test = len(self.classes)* 1 * 12
                index_n_sub_train = len(self.classes)* 1 * 1
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (1)
            else:
                text_index = (index % index_n_sub_test) // (1)
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // (1)
            else:
                img_index = (index % index_n_sub_test) // (1)
                
        text = self.text[text_index]
        # print("self.img", len(self.img))
        # print("img_index", img_index)
        # img = self.img[img_index]
        
        text_features = self.text_features[text_index]
        # img_features = self.img_features[img_index]
        if self.train:
            img_features = self.img_features[img_index]
            img = self.img[img_index]
        else:
            img_features = self.img_features[::12][img_index]
            img = self.img[::12][img_index]        
        # return self.modal, x, label, text, text_features, img, img_features, 'sub-00'
        return self.modal, x, label, text, text_features, img, img_features, index, img_index, 'sub-00'

    def __len__(self):
        return self.data.shape[0]  # or self.labels.shape[0] which should be the same

if __name__ == "__main__":
    # Instantiate the dataset and dataloader
    data_path = "/home/ldy/THINGS-MEG/preprocessed_newsplit"  # Replace with the path to your data
    data_path = data_path
    train_dataset = MEGDataset(data_path, subjects = ['sub-01', 'sub-02'], train=True)    
    test_dataset = MEGDataset(data_path, subjects = ['sub-01'], train=False)
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
    modal, x, label, text, text_features, img, img_features, _  = test_dataset[i]
    print(f"Index {i}, Label: {label}, text: {text}")
    Image.open(img)
            
    
        
    
    