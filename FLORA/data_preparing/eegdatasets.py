import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
from transformers import CLIPVisionModel
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers.utils import load_image
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2, 3, 4, 5, 6, 7"
# proxy = 'http://10.20.37.38:7890'
proxy = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy

device = "cuda:1" if torch.cuda.is_available() else "cpu"
import open_clip
import json
from omegaconf import OmegaConf
import os

cfg = OmegaConf.load(os.path.join("/mnt/dataset1/ldy/Workspace/FLORA/configs/config.yaml"))
cfg = OmegaConf.structured(cfg)

# Access the paths from the config
# img_directory_training = cfg.eegdataset.img_directory_training
# img_directory_test = cfg.eegdataset.img_directory_test
img_directory_training = "/mnt/dataset0/ldy/datasets/THINGS_EEG/images_set/training_images"
img_directory_test = "/mnt/dataset0/ldy/datasets/THINGS_EEG/images_set/test_images"



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
        x = self.preprocess(x) #x torch.Size([3, 224, 224])
        x = self.clip.vision_model.embeddings.patch_embedding(x) #x torch.Size([1024, 16, 16])
        # print("x", x.shape)
        image_feats = self.clip_encode_image(x)

        return image_feats
    

class EEGDataset():
    """
    subjects = ['sub-01', 'sub-02', 'sub-05', 'sub-04', 'sub-03', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    """
    def __init__(self, data_path, adap_subject=None, subjects=None, train=True, use_caption=False, time_window=[0, 1.0], classes = None, pictures = None):
        self.data_path = data_path
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 1654 if train else 200
        self.classes = classes
        self.pictures = pictures
        self.adap_subject = adap_subject
        self.modal='eeg'
        # assert any subjects in subject_list
        assert any(sub in self.subject_list for sub in self.subjects)
        self.use_caption = use_caption
        self.data, self.labels, self.text, self.img = self.load_data()
        
        self.data = self.extract_eeg(self.data, time_window)
                    
        # Define the features_filename based on the condition
        if self.use_caption:
            model_type = 'ViT-L-14'
            features_filename = os.path.join(f'/mnt/dataset1/ldy/Workspace/FLORA/data_preparing/{model_type}_features_multimodal_train.pt') if self.train else os.path.join(f'/mnt/dataset1/ldy/Workspace/FLORA/data_preparing/{model_type}_features_multimodal_test.pt')
        else:
            model_type = 'ViT-H-14'     
            features_filename = os.path.join(f'/mnt/dataset1/ldy/Workspace/FLORA/data_preparing/{model_type}_features_train.pt') if self.train else os.path.join(f'/mnt/dataset1/ldy/Workspace/FLORA/data_preparing/{model_type}_features_test.pt')

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
        
        if self.train:
            directory = img_directory_training
        else:
            directory = img_directory_test

        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()
        
        if self.classes is not None:
            dirnames = [dirnames[i] for i in self.classes]

        for dir in dirnames:

            try:
                idx = dir.index('_')
                description = dir[idx+1:]
            except ValueError:
                print(f"Skipped: {dir} due to no '_' found.")
                continue
                
            new_description = f"This picture is {description}"
            texts.append(new_description)

        if self.train:
            img_directory = img_directory_training
        else:
            img_directory = img_directory_test
        
        all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()

        if self.classes is not None and self.pictures is not None:
            images = []
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                pic_idx = self.pictures[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    if pic_idx < len(all_images):
                        images.append(os.path.join(folder_path, all_images[pic_idx]))
        elif self.classes is not None and self.pictures is None:
            images = []
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    images.extend(os.path.join(folder_path, img) for img in all_images)
        elif self.classes is None:
            images = []
            for folder in all_folders:
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()  
                images.extend(os.path.join(folder_path, img) for img in all_images)
        else:

            print("Error")
            
        print("self.subjects", self.subjects)
        print("adap_subject", self.adap_subject)
        for subject in self.subjects:
            if self.train:
                if subject == self.adap_subject:
                    continue            
                # print("subject:", subject)    
                file_name = 'preprocessed_eeg_training.npy'

                file_path = os.path.join(self.data_path, subject, file_name)
                data = np.load(file_path, allow_pickle=True)
                
                preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()                
                times = torch.from_numpy(data['times']).detach()[50:]
                ch_names = data['ch_names']

                n_classes = 1654
                samples_per_class = 10
                
                if self.classes is not None and self.pictures is not None:
                    for c, p in zip(self.classes, self.pictures):
                        start_index = c * 1 + p
                        if start_index < len(preprocessed_eeg_data):
                            preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+1]
                            labels = torch.full((1,), c, dtype=torch.long).detach()
                            data_list.append(preprocessed_eeg_data_class)
                            label_list.append(labels)

                elif self.classes is not None and self.pictures is None:
                    for c in self.classes:
                        start_index = c * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        labels = torch.full((samples_per_class,), c, dtype=torch.long).detach()
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)

                else:
                    for i in range(n_classes):
                        start_index = i * samples_per_class
                        # if self.adap_subject==None:
                        #     preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        # else:
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                        # preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class, 1)
                        # preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class, 0)
                        # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)

                 
            else:
                if subject == self.adap_subject or self.adap_subject==None:
                    file_name = 'preprocessed_eeg_test.npy'
                    file_path = os.path.join(self.data_path, subject, file_name)
                    data = np.load(file_path, allow_pickle=True)
                    preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()
                    times = torch.from_numpy(data['times']).detach()[50:]
                    ch_names = data['ch_names']
                    n_classes = 200  # Each class contains 1 images
                    
                    samples_per_class = 1

                    for i in range(n_classes):
                        if self.classes is not None and i not in self.classes:  # If we've defined specific classes and the current class is not in the list, skip
                            continue
                        start_index = i * samples_per_class  # Update start_index for each class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index:start_index+samples_per_class]
                        # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  # Add class labels
                        preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), 0)
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
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])                 
            # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[1:])
            # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)   
            # print("label_tensor", label_tensor.shape)
            print("data_tensor", data_tensor.shape)
        else:           
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)   
            # label_tensor = torch.cat(label_list, dim=0)
            # print("label_tensor", label_tensor.shape)
            # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])
        # print("data_tensor", data_tensor.shape)
        # label_list: (subjects * classes) * 10
        # label_tensor: (subjects * classes * 10)
        # print("label_tensor = torch.cat(label_list, dim=0)")
        # print(label_list)
        label_tensor = torch.cat(label_list, dim=0)
        # label_tensor = torch.cat(label_list, dim=0)
        # print(label_tensor[:300])
        if self.train:
            # label_tensor: (subjects * classes * 10 * 4)
            label_tensor = label_tensor.repeat_interleave(4)
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
        text_inputs = torch.cat([open_clip.tokenize(t) for t in text]).to(device)

        with torch.no_grad():
            text_features = vlmodel.encode_text(text_inputs)
        
        text_features = F.normalize(text_features, dim=-1).detach()
    
        return text_features

    def ImageEncoder(self, images, use_caption=False):
        batch_size = 512   
        image_features_list = []
        transform = transforms.ToTensor()
        
        if use_caption:         
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                image_inputs = torch.stack([transform(Image.open(img)) for img in batch_images])                
                
                with torch.no_grad():
                    image_feature = self.clip_encoder.encode_image(image_inputs)
                image_features_list.append(image_feature)
                del batch_images

        else:
            vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
                model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device=device)
            
            batch_size = 512        
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                image_inputs = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images])

                with torch.no_grad():
                    batch_image_features = vlmodel.encode_image(image_inputs)
                image_features_list.append(batch_image_features)
                del batch_images

        image_features = torch.cat(image_features_list, dim=0)                        
        return image_features


    
    def __getitem__(self, index):
        # Get the data and label corresponding to "index"
        # index: (subjects * classes * 10 * 4)
        x = self.data[index]
        label = self.labels[index]
        
        if self.pictures is None:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 10 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes)* 1 * 80
                index_n_sub_train = len(self.classes)* 10 * 4
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (10 * 4)
            else:
                text_index = (index % index_n_sub_test) 
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // (4)
            else:
                img_index = (index % index_n_sub_test) 
        else:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 1 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes)* 1 * 80
                index_n_sub_train = len(self.classes)* 1 * 4
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (1 * 4)
            else:
                text_index = (index % index_n_sub_test) 
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // (4)
            else:
                img_index = (index % index_n_sub_test) 
                
        text = self.text[text_index]
        img = self.img[img_index]
        if self.use_caption:
            text_features = torch.zeros((1, 1, 1024))
        else:
            text_features = self.text_features[text_index]
        img_features = self.img_features[img_index]
        
        return self.modal, x, label, text, text_features, img, img_features, index, img_index, 'sub-00'

    def __len__(self):
        return self.data.shape[0]  # or self.labels.shape[0] which should be the same

if __name__ == "__main__":
    # Instantiate the dataset and dataloader
    data_path = "/mnt/dataset0/ldy/datasets/THINGS_EEG/Preprocessed_data_250Hz"  # Replace with the path to your data
    train_dataset = EEGDataset(data_path, subjects = ['sub-01'], train=True, use_caption=True)    
    test_dataset = EEGDataset(data_path, subjects = ['sub-01'], train=False, use_caption=True)
    # train_dataset = EEGDataset(data_path, adap_subject = 'sub-01', train=True)    
    # test_dataset = EEGDataset(data_path, adap_subject = 'sub-01', train=False)    
    # train_dataset = EEGDataset(data_path, train=True) 
    # test_dataset = EEGDataset(data_path, train=False) 

    # 100 Hz
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    i = 80*1-1
    _, x, label, text, text_features, img, img_features, index, img_index, _ = test_dataset[i]
    print(f"Index {i}, Label: {label}, text: {text}")
    Image.open(img)
            
    
        
    
