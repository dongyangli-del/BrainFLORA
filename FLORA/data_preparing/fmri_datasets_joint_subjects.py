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
from transformers import CLIPVisionModel
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers.utils import load_image
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2, 3, 4, 5, 6, 7"
proxy = 'http://10.20.37.38:7890'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy
cuda_device_count = torch.cuda.device_count()
print(cuda_device_count)
import open_clip
device = "cuda:5" if torch.cuda.is_available() else "cpu"

# vlmodel, preprocess = clip.load("ViT-B/32", device=device)


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

class CLIPEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
        self.clip_size = (224, 224)
        self.device = device
        preproc = transforms.Compose([
        transforms.Resize(size=self.clip_size[0], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(size=self.clip_size),
        # transforms.ToTensor(), # only for debug
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
        self.preprocess = preproc

        # for param in self.clip.parameters():
        #     param.requires_grad = False
        
    def clip_encode_image(self, x):
        
        x = x.reshape(x.shape[0], x.shape[1], -1) #([batchsize, 1024, 256])
        x = x.permute(0, 2, 1) 

        # print("embed", self.clip.vision_model.embeddings.class_embedding.to(x.dtype).shape)
        class_embedding = self.clip.vision_model.embeddings.class_embedding.to(x.dtype)

        class_embedding = class_embedding.repeat(x.shape[0], 1, 1)  # ([batchsize, 1, 1024])
        # print("class_embedding", class_embedding.shape)   
        
        x = torch.cat([class_embedding, x], dim=1)
        
        # x = torch.cat([self.clip.vision_model.embeddings.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
        #               x.shape[-1], dtype=x.dtype, device=self.device), x], dim=1)  
        pos_embedding = self.clip.vision_model.embeddings.position_embedding # Embedding(257, 1024)
        
        position_ids =  torch.arange(0, 257).unsqueeze(0).to(self.device)
        x = x + pos_embedding(position_ids)
        x = self.clip.vision_model.pre_layrnorm(x)
        x = self.clip.vision_model.encoder(x, output_hidden_states=True)
        
        select_hidden_state_layer = -2
        select_hidden_state = x.hidden_states[select_hidden_state_layer]#torch.Size([1, 256, 1024])
        
        image_features = select_hidden_state[:, 1:] # torch.Size([1, 256, 1024]
        return image_features

    def encode_image(self, x):
        x = x.to(self.device)
        x = self.clip.vision_model.embeddings.patch_embedding(x) #x torch.Size([1024, 16, 16])
        # print("x", x.shape)
        image_feats = self.clip_encode_image(x)

        return image_feats
class fMRIDataset():
    """
    subjects = ['sub-01', 'sub-02', 'sub-03']
    """
    def __init__(self, data_path, adap_subject=None, subjects=None, train=True, use_caption=False, time_window=[0, 1.0], classes = None, pictures = None):
        self.data_path = data_path
        self.train = train
        self.use_caption=use_caption        
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
        
        

            
        # Define the features_filename based on the condition
        if self.use_caption:
            model_type = 'ViT-L-14'
            features_filename = os.path.join(f'/mnt/dataset1/ldy/Workspace/FLORA/data_preparing/fMRI_{model_type}_features_multimodal_train.pt') if self.train else os.path.join(f'/mnt/dataset1/ldy/Workspace/FLORA/data_preparing/fMRI_{model_type}_features_multimodal_test.pt')
        else:
            model_type = 'ViT-H-14'     
            features_filename = os.path.join(f'/mnt/dataset1/ldy/Workspace/FLORA/data_preparing/fMRI_{model_type}_features_train.pt') if self.train else os.path.join(f'/mnt/dataset1/ldy/Workspace/FLORA/data_preparing/fMRI_{model_type}_features_test.pt')

        # Try to load the saved features if they exist
        if os.path.exists(features_filename):
            saved_features = torch.load(features_filename, weights_only=True)
            if self.use_caption:
                self.img_features = saved_features['img_features']
                self.text_features = torch.zeros((self.img_features.shape[0], 1, 1024)).cpu()
            else:
                self.text_features = saved_features['text_features']
                self.img_features = saved_features['img_features']
        else:
            if self.use_caption:                
                self.clip_encoder = CLIPEncoder(device)
                self.img_features = self.ImageEncoder(self.img, self.use_caption)
                torch.save({
                    'img_features': self.img_features.cpu(),
                    'text_features': torch.zeros((self.img_features.shape[0], 1, 1024)).cpu()
                }, features_filename)
            else:                
                self.vlmodel, self.preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
                    model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device=device)
                
                self.text_features = self.Textencoder(self.text)
                self.img_features = self.ImageEncoder(self.img)
                torch.save({
                    'text_features': self.text_features.cpu(),
                    'img_features': self.img_features.cpu(),
                }, features_filename)            

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
                # print("data_list", len(data_list))
                # print("label_list", len(label_list))
                print(f"Data list length: {len(data_list[i])}, label list length: {len(label_list[i])}, text length: {len(texts)}, image length: {len(images)}")
        else:            
            print(f"Data list length: {len(data_list[0])}, label list length: {len(label_list[0])}, text length: {len(texts)}, image length: {len(images)}")    

        return data_list, label_list, texts, images




            
    
    def Textencoder(self, text):           
        # 使用预处理器将文本转换为模型的输入格式
        text_inputs = torch.cat([open_clip.tokenize(t) for t in text]).to(device)

        # 使用CLIP模型来编码文本
        with torch.no_grad():
            text_features = self.vlmodel.encode_text(text_inputs)
            
        return text_features
        
    def ImageEncoder(self, images, use_caption=False):
        batch_size = 20
        image_features_list = []
        transform = transforms.ToTensor()
        
        # 选择预处理和编码器
        if use_caption:
            encoder = self.clip_encoder
            preprocess_fn = lambda img: self.clip_encoder.preprocess(transform(Image.open(img)))
        else:
            encoder = self.vlmodel
            preprocess_fn = lambda img: self.preprocess_train(Image.open(img).convert("RGB"))
        
        # 批量处理
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            image_inputs = torch.stack([preprocess_fn(img) for img in batch_images])

            with torch.no_grad():
                image_features = encoder.encode_image(image_inputs)
            image_features_list.append(image_features)
            del batch_images

        # 合并所有图像特征
        image_features = torch.cat(image_features_list, dim=0)
        return image_features
    

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
        target_length = 7000
        # target_length = 11000
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
        if self.use_caption:
            text_features = torch.zeros((1, 1, 1024))
        else:
            text_features = self.text_features[text_index]  

        img_features = self.img_features[img_index]

        return self.modal, x, label, text, text_features, img, img_features, index, img_index, subject_id  # 返回subject_id


    def __len__(self):
        return sum(self.subject_data_lens)  # 所有被试的数据总长度



if __name__ == "__main__":
    # Instantiate the dataset and dataloader
    data_path = "/mnt/dataset0/ldy/datasets/fmri_dataset/Preprosessed"  # Replace with the path to your data
    data_path = data_path
    train_dataset = fMRIDataset(data_path, subjects = ['sub-01'], train=True, use_caption=True)    
    test_dataset = fMRIDataset(data_path, subjects = ['sub-01'], train=False, use_caption=True)
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
    _, x, label, text, text_features, img, img_features, index, img_index, subject_id   = test_dataset[i]
    print(f"Index {i}, Label: {label}, text: {text}")
    Image.open(img)
            
    
        
    