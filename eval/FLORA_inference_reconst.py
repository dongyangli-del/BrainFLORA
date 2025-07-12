import sys
import os
import random
import re
import argparse
import warnings

# 设置环境变量
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
os.environ["WANDB_SILENT"] = "true"

# 获取当前工作目录（假设 Notebook 位于 parent_dir）
current_dir = os.getcwd()

# 构建项目根目录的路径（假设 parent_dir 和 model 同级）
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.path.insert(0, "/mnt/dataset1/ldy/Workspace/FLORA/")    

# 现在可以使用绝对导入
from model.unified_encoder_multi_tower import UnifiedEncoder

# 导入必要的库
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

import wandb
wandb.init(mode="disabled")

from data_preparing.eegdatasets import EEGDataset
from data_preparing.megdatasets_averaged_copy import MEGDataset
from data_preparing.fmri_datasets_joint_subjects import fMRIDataset
from data_preparing.datasets_mixer import MetaEEGDataset, MetaMEGDataset, MetafMRIDataset, MetaDataLoader

from sklearn.metrics import confusion_matrix

from loss import ClipLoss
from model.diffusion_prior import Pipe, EmbeddingDataset, DiffusionPriorUNet
from model.custom_pipeline import Generator4Embeds
# 忽略警告
warnings.filterwarnings("ignore")

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
proxy = 'http://10.20.37.38:7890'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy


def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

def get_eegfeatures(unified_model, dataloader, device, text_features_all, img_features_all, k, eval_modality, test_classes):
    unified_model.eval()
    text_features_all = text_features_all[eval_modality].to(device).float()
    if eval_modality=='eeg' or eval_modality=='fmri':
        img_features_all = (img_features_all[eval_modality]).to(device).float()
    elif eval_modality=='meg':
        img_features_all = (img_features_all[eval_modality][::12]).to(device).float()  
    total_loss = 0
    correct = 0
    top5_correct_count=0
    total = 0
    loss_func = ClipLoss() 
    all_labels = set(range(text_features_all.size(0)))
    save_features = False
    features_list = []  # List to store features    
    features_tensor = torch.zeros(0, 0)
    with torch.no_grad():
        for batch_idx, (modal, data, labels, text, text_features, img, img_features, _, _, sub_ids) in enumerate(dataloader):
            data = data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            batch_size = data.size(0) 
            subject_ids = [extract_id_from_string(sub_id) for sub_id in sub_ids]
            subject_ids = torch.tensor(subject_ids, dtype=torch.long).to(device)
            neural_features = unified_model(data, subject_ids, modal=eval_modality)
            
            logit_scale = unified_model.logit_scale.float()            
            features_list.append(neural_features)
               
            img_loss = loss_func(neural_features, img_features, logit_scale)
            loss = img_loss        
            total_loss += loss.item()
            
            for idx, label in enumerate(labels):

                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                

                logits_img = logit_scale * neural_features[idx] @ selected_img_features.T
                # logits_text = logit_scale * neural_features[idx] @ selected_text_features.T
                # logits_single = (logits_text + logits_img) / 2.0
                logits_single = logits_img
                # print("logits_single", logits_single.shape)

                # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) \in {0, 1, ..., n_cls-1}
                if predicted_label == label.item():
                    correct += 1        
                if k==test_classes:
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                            
                    # Check if the ground truth label is among the top-5 predictions
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                 
                total += 1              
        
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total    
    top5_acc = top5_correct_count / total    
    return average_loss, accuracy, top5_acc, labels, features_tensor.cpu()



# Define Parameters
encoder_paths_list = [
    'eeg=/mnt/dataset1/ldy/Workspace/EEG_Image_decode/Retrieval/models/contrast/across/ATMS/01-06_01-46/150.pth',
    'meg=/mnt/dataset1/ldy/Workspace/EEG_Image_decode/Retrieval/models/contrast/across/ATMS/01-11_14-50/150.pth',
    'fmri=/mnt/dataset0/ldy/Workspace/EEG_Image_decode/Retrieval/models/contrast/across/ATMS/01-18_01-35/50.pth'
]
eval_modality = 'fmri'  # Modality to evaluate on

# Subjects Configuration
test_subjects = ['sub-02', 'sub-03']
# test_subjects = ['sub-01']
eeg_subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
meg_subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04']
fmri_subjects = ['sub-01', 'sub-02', 'sub-03']

modalities = ['eeg', 'meg', 'fmri']  # Modalities to include in inference
test_classes = 100
# Update test_subjects and test_classes based on eval_modality
# if eval_modality == 'eeg':
#     test_subjects = eeg_subjects
#     test_classes = 200
# elif eval_modality == 'meg':
#     test_subjects = meg_subjects
#     test_classes = 200
# elif eval_modality == 'fmri':
#     test_subjects = fmri_subjects
#     test_classes = 100
# else:
#     raise ValueError(f"Unsupported modality: {eval_modality}")

# Example usage
print(f"Evaluation Modality: {eval_modality}")
print(f"Test Subjects: {test_subjects}")
print(f"Number of Test Classes: {test_classes}")

# Dataset Paths
# eeg_data_path = "/home/ldy/4090_Workspace/4090_THINGS/Preprocessed_data_250Hz"
# meg_data_path = "/home/ldy/THINGS-MEG/preprocessed_newsplit"
# fmri_data_path = "/home/ldy/fmri_dataset/Preprocessed"

# parser.add_argument('--eeg_data_path', type=str, default="/mnt/dataset0/ldy/datasets/THINGS_EEG/Preprocessed_data_250Hz", help='Path to the EEG dataset')
# parser.add_argument('--meg_data_path', type=str, default="/mnt/dataset0/ldy/datasets/THINGS_MEG/preprocessed_newsplit", help='Path to the MEG dataset')    
# parser.add_argument('--fmri_data_path', type=str, default="/mnt/dataset0/ldy/datasets/fmri_dataset/Preprocessed", help='Path to the fMRI dataset')     

eeg_data_path = "/mnt/dataset0/ldy/datasets/THINGS_EEG/Preprocessed_data_250Hz"
meg_data_path = "/mnt/dataset0/ldy/datasets/THINGS_MEG/preprocessed_newsplit"
fmri_data_path = "/mnt/dataset0/ldy/datasets/fmri_dataset/Preprocessed"

# Output and Logging Configuration (Not needed for inference, but kept for completeness)
output_dir = './outputs/contrast'
project = "train_pos_img_text_rep"
entity = "sustech_rethinkingbci"
name = "lr=3e-4_img_pos_pro_eeg"

# Inference Parameters
device_preference = 'cuda:5'  # e.g., 'cuda:0' or 'cpu'
device_type = 'gpu'  # 'cpu' or 'gpu'

# Process encoder_paths into a dictionary
encoder_paths = {}
for path in encoder_paths_list:
    key, value = path.split('=')
    encoder_paths[key] = value

# Set device based on the argument
device = torch.device(device_preference if device_type == 'gpu' and torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize empty datasets for each modality
text_features_test_all = {}
img_features_test_all = {}




#####################################################################################
# Initialize the Unified Encoder Model
unified_model = UnifiedEncoder(encoder_paths, device, user_caption=False)
# unified_model.load_state_dict(torch.load("/mnt/dataset1/ldy/Workspace/FLORA/models/contrast/across/Unified_EEG+MEG+fMRI_EEG/01-26_14-01/150.pth"))
# unified_model.load_state_dict(torch.load("/mnt/dataset1/ldy/Workspace/FLORA/models/contrast/across/Unified_EEG+MEG+fMRI_EEG/01-27_14-29/300.pth"))
unified_model.load_state_dict(torch.load("/mnt/dataset1/ldy/Workspace/FLORA/models/contrast/across/Unified_EEG+MEG+fMRI_EEG/01-29_01-18/300.pth"))
# unified_model.load_state_dict(torch.load("/mnt/dataset1/ldy/Workspace/FLORA/models/contrast/across/Unified_EEG+MEG+fMRI_EEG/01-29_10-25/300.pth"))
unified_model.to(device)
unified_model.eval()  # Set model to evaluation mode

# diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
# high_pipe = Pipe(diffusion_prior, device=device)
# high_pipe.diffusion_prior.load_state_dict(torch.load("/mnt/dataset0/ldy/models/contrast/across/Unified_EEG+MEG+fMRI_EEG/01-22_18-16/prior_diffusion/60.pth"))
# high_pipe.diffusion_prior.to(device)
# high_pipe.diffusion_prior.eval()  # Set model to evaluation mode


# Print model parameters info
def format_num(num):
    for unit in ['','K','M','B','T']:
        if num < 1000:
            return f"{num:.2f}{unit}"
        num /= 1000
    return f"{num:.2f}P"

total_params = sum(p.numel() for p in unified_model.parameters())
trainable_params = sum(p.numel() for p in unified_model.parameters() if p.requires_grad)
print(f"Total parameters: {format_num(total_params)}")
print(f"Trainable parameters: {format_num(trainable_params)}")

if total_params > 0:
    trainable_percentage = (trainable_params / total_params) * 100
    print(f"Trainable parameters percentage: {trainable_percentage:.2f}%")
else:
    print("Total parameters count is zero, cannot compute percentage.")
#####################################################################################


from IPython.display import Image, display


diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
high_pipe = Pipe(diffusion_prior, device=device)
# high_pipe.diffusion_prior.load_state_dict(torch.load("/mnt/dataset1/ldy/Workspace/FLORA/models/contrast/across/Unified_EEG+MEG+fMRI_EEG/01-26_14-01/prior_diffusion/150.pth"))
# high_pipe.diffusion_prior.load_state_dict(torch.load("/mnt/dataset1/ldy/Workspace/FLORA/models/contrast/across/Unified_EEG+MEG+fMRI_EEG/01-27_14-29/prior_diffusion/300.pth"))
high_pipe.diffusion_prior.load_state_dict(torch.load("/mnt/dataset1/ldy/Workspace/FLORA/models/contrast/across/Unified_EEG+MEG+fMRI_EEG/01-29_01-18/prior_diffusion/300.pth"))
# high_pipe.diffusion_prior.load_state_dict(torch.load("/mnt/dataset1/ldy/Workspace/FLORA/models/contrast/across/Unified_EEG+MEG+fMRI_EEG/01-29_10-25/prior_diffusion/300.pth"))
high_pipe.diffusion_prior.to(device)
high_pipe.diffusion_prior.eval()  # Set model to evaluation mode

# set a seed value
seed_value = 42
generator = Generator4Embeds(num_inference_steps=4, device=device)
gen = torch.Generator(device=device)
gen.manual_seed(seed_value)
folder = f'/mnt/dataset1/ldy/Workspace/FLORA/eval/{eval_modality}_generated_imgs'
os.makedirs(folder, exist_ok=True)

def get_priorfeatures(sub, unified_model, dataloader, device, text_features_all, img_features_all, k, eval_modality, test_classes):
    unified_model.eval()
    text_features_all = text_features_all[eval_modality].to(device).float()
    if eval_modality=='eeg' or eval_modality=='fmri':
        img_features_all = (img_features_all[eval_modality]).to(device).float()
    elif eval_modality=='meg':
        img_features_all = (img_features_all[eval_modality][::12]).to(device).float()  
    total_loss = 0
    correct = 0
    top5_correct_count=0
    total = 0
    loss_func = ClipLoss() 
    all_labels = set(range(text_features_all.size(0)))
    save_features = False
    features_list = []  # List to store features    
    features_tensor = torch.zeros(0, 0)
    count = 0
    with torch.no_grad():
        for batch_idx, (modal, data, labels, text, text_features, img, img_features, _, _, sub_ids) in enumerate(dataloader):
            data = data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            batch_size = data.size(0) 
            subject_ids = [extract_id_from_string(sub_id) for sub_id in sub_ids]
            subject_ids = torch.tensor(subject_ids, dtype=torch.long).to(device)
            neural_features = unified_model(data, subject_ids, modal=eval_modality)
            # print("neural_features", neural_features.shape)
            for i in range(neural_features.shape[0]):
                h = high_pipe.generate(c_embeds=neural_features[i].unsqueeze(0), num_inference_steps=10, guidance_scale=2.0)

                # image_1 = generator.generate(eeg_embeds_1[index], generator=gen)  
                # display(image_1)

                image_2 = generator.generate(h, generator=gen)  
                # display(image_2)          
                # 设置保存图像的路径和文件名
                dir_path = os.path.join(folder, sub)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                file_path = os.path.join(dir_path, f'image_{count+1}.png')  # 图像名称为 image_1.png, image_2.png, 等等                
                # 保存图像
                image_2.save(file_path)  # 使用 PIL 或图像对象的 .save 方法保存图像  
                count+=1
            logit_scale = unified_model.logit_scale.float()            
            features_list.append(neural_features)         

        if save_features:
            features_tensor = torch.cat(features_list, dim=0)
            print("features_tensor", features_tensor.shape)
            torch.save(features_tensor.cpu(), f"neural_features_eval_{eval_modality}_{sub}_test.pt")  # Save features as .pt file 
    return features_tensor.cpu()


for sub in test_subjects:
    # Prepare test dataset based on eval_modality and test_subjects
    if eval_modality == 'eeg':
        test_dataset = EEGDataset(eeg_data_path, subjects=[sub], train=False)
    elif eval_modality == 'meg':
        test_dataset = MEGDataset(meg_data_path, subjects=[sub], train=False)
    elif eval_modality == 'fmri':
        test_dataset = fMRIDataset(fmri_data_path, adap_subject=sub, subjects=[sub], train=False)
    
    # Collect test features
    text_features_test_all[eval_modality] = test_dataset.text_features
    img_features_test_all[eval_modality] = test_dataset.img_features

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    eeg_features_test = get_priorfeatures(
        sub, unified_model, test_loader, device, text_features_test_all, img_features_test_all, k=test_classes, eval_modality=eval_modality, test_classes=test_classes
    )