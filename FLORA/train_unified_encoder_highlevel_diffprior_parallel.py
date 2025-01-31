import os

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from data_preparing.eegdatasets import EEGDataset
from data_preparing.megdatasets_averaged import MEGDataset
from data_preparing.fmri_datasets_joint_subjects import fMRIDataset
from data_preparing.datasets_mixer import MetaEEGDataset, MetaMEGDataset, MetafMRIDataset, MetaDataLoader

from einops.layers.torch import Rearrange, Reduce

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from util import wandb_logger

import csv
from torch import Tensor
import itertools
import math
import re
from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding
import numpy as np
from loss import ClipLoss, mixco_nce, soft_clip_loss, mixco_1d
import argparse
from torch import nn
from torch.optim import AdamW
from model.unified_encoder_multi_tower import UnifiedEncoder
from model.diffusion_prior import Pipe, EmbeddingDataset, DiffusionPriorUNet
from model.custom_pipeline import Generator4Embeds
import argparse
import datetime
import numpy as np
import os
import copy
import time
from pathlib import Path
import functools
import multiprocessing
import utils.misc as misc

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
os.environ["WANDB_SILENT"] = "true"

from torch.cuda.amp import GradScaler, autocast
import wandb
wandb.init(mode="disabled")
from torch.nn.utils import clip_grad_norm_
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import sys
from accelerate import Accelerator
import torch
import os
import re
import random
import torch.nn as nn
import torch.optim as optim

# Initialize Accelerator
accelerator = Accelerator(device_placement=True, split_batches=True, mixed_precision='bf16')
print("PID of this process =", os.getpid())
print = accelerator.print

device = accelerator.device
print("device:", device)

num_devices = torch.cuda.device_count()
if num_devices == 0:
    num_devices = 1
num_workers = num_devices
print(accelerator.state)

local_rank = accelerator.state.local_process_index
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
print("distributed =", distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size)

import warnings
warnings.filterwarnings("ignore")

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

def train_model(epoch, unified_model, high_pipe, dataloader, optimizer, device, text_features_all, img_features_all, config, accelerator, eval_modality='eeg'):
    unified_model.train()    
    if eval_modality == 'eeg':
        img_features_all = img_features_all[eval_modality][::10].to(device).float()
    elif eval_modality in ['meg', 'fmri']:
        img_features_all = img_features_all[eval_modality][::12].to(device).float()
        
    text_features_all = text_features_all[eval_modality].to(device).float()  # (n_cls, d)
    
    total_loss = 0
    correct = 0
    total = 0
    features_list = []  # List to store features
    loss_func = ClipLoss()

    num_voxels = {1: 6036, 2: 5944, 3: 5238}
    
    # Ensure all features are in float32
    img_features_all = img_features_all.float()
    text_features_all = text_features_all.float()
    mixup_pct = .33
    clip_scale = 1    
    mse_loss_fn = nn.MSELoss(reduction='mean')
    prior_loss_sum = 0
    prior_criterion = nn.MSELoss(reduction='mean')
    
    # Parallelize with Accelerator
    for batch_idx, (modal, data, labels, text, text_features, img, img_features, index, img_index, sub_ids) in enumerate(dataloader):
        data = data.to(device).float()
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        batch_size = data.size(0)
        subject_ids = [extract_id_from_string(sub_id) for sub_id in sub_ids]
        subject_ids = torch.tensor(subject_ids, dtype=torch.long).to(device)
        
        # Forward pass
        neural_features = unified_model(data, subject_ids, modal=modal[0])        
        regress_loss = mse_loss_fn(neural_features, img_features)
        
        if config.use_prior:     
            num_train_timesteps = high_pipe.scheduler.config.num_train_timesteps
            c_embeds = neural_features
            h_embeds = img_features
            N = h_embeds.shape[0]

            # 1. randomly replacing c_embeds to None
            if torch.rand(1) < 0.1:
                c_embeds = None

            # 2. Generate noisy embeddings as input
            noise = torch.randn_like(h_embeds)

            # 3. sample timestep
            timesteps = torch.randint(0, num_train_timesteps, (N,), device=device)

            # 4. add noise to h_embedding
            perturbed_h_embeds = high_pipe.scheduler.add_noise(
                h_embeds,
                noise,
                timesteps
            )

            # 5. predict noise
            noise_pre = high_pipe.diffusion_prior(perturbed_h_embeds, timesteps, c_embeds)
            
            # 6. loss function weighted by sigma
            prior_loss = prior_criterion(noise_pre, noise)
            prior_loss_sum += prior_loss.item()

        logit_scale = unified_model.logit_scale.float()
        
        neural_features_clone, perm, betas, select = mixco_1d(neural_features.clone())  
        
        neural_features_norm = nn.functional.normalize(neural_features_clone.flatten(1), dim=-1)
        img_features_norm = nn.functional.normalize(img_features.flatten(1), dim=-1)
        
        if epoch < int(mixup_pct * config.epochs):                
            loss_clip = mixco_nce(
                neural_features_norm,
                img_features_norm,
                temp=.006,
                perm=perm, betas=betas, select=select)
        else:
            loss_clip = soft_clip_loss(
                neural_features_norm,
                img_features_norm,
                temp=logit_scale)
        
        loss_clip *= clip_scale
        loss = loss_clip + regress_loss + prior_loss
        
        # Backward pass and optimization
        accelerator.backward(loss)
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(unified_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(high_pipe.diffusion_prior.parameters(), 1.0)
        high_pipe.diffusion_prior.lr_scheduler.step()
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Compute the corresponding logits
        logits_img = logit_scale * neural_features @ img_features_all.T
        predicted = torch.argmax(logits_img, dim=1)  # (n_batch,)
        
        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()
        
        del modal, data, labels, text, text_features, img, img_features, index, img_index, sub_ids
    
    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    return average_loss, accuracy




def evaluate_model(epoch, unified_model, high_pipe, dataloader, device, text_features_all, img_features_all, k, config, eval_modality='eeg'):
    unified_model.eval()
    text_features_all = text_features_all[eval_modality].to(device).float()
    if eval_modality=='eeg' or eval_modality=='fmri':
        img_features_all = (img_features_all[eval_modality]).to(device).float()
    elif eval_modality=='meg':
        img_features_all = (img_features_all[eval_modality][::12]).to(device).float()    
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.99
    top5_correct = 0
    top5_correct_count = 0
    # Get all unique classes
    all_labels = set(range(text_features_all.size(0)))
    top5_acc = 0
    loss_func = ClipLoss() 

    batch_idx = 0
    img_features_all = img_features_all.float()  # 转换为 float16
    text_features_all = text_features_all.float()  # 转换为 float16    
    num_voxels = {1: 6036, 2: 5944, 3: 5238} 
    mixup_pct=.33
    clip_scale=1

    with torch.no_grad():
        for batch_idx, (modal, data, labels, text, text_features, img, img_features, index, img_index, sub_ids) in enumerate(dataloader):
        # for batch_idx, (modal, data, labels, text, text_features, img, img_features, subject_id) in enumerate(dataloader):    
            # # 使用高斯分布生成与 eeg_data 形状完全相同的张量
            # noise = torch.randn_like(data).to(device)  # 标准正态分布，均值=0，标准差=1

            # # 将 eeg_data 替换为噪声
            # data = noise
            data = data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            batch_size = data.size(0) 
            subject_ids = [extract_id_from_string(sub_id) for sub_id in sub_ids]
            subject_ids = torch.tensor(subject_ids, dtype=torch.long).to(device)
            
            # with autocast():  # 在推理中启用半精度计算
            neural_features = unified_model(data, subject_ids, modal=eval_modality)
            
            logit_scale = unified_model.logit_scale.float()

            neural_features, perm, betas, select = mixco_1d(neural_features)  
            neural_features_norm = nn.functional.normalize(neural_features.flatten(1), dim=-1)
            img_features_norm = nn.functional.normalize(img_features.flatten(1), dim=-1)            
            # img_loss = eeg_model.loss_func(neural_features, img_features, logit_scale)
            # text_loss = eeg_model.loss_func(neural_features, text_features, logit_scale)
            
            if epoch < int(mixup_pct * config.epochs):                
                loss_clip = mixco_nce(
                    neural_features_norm,
                    img_features_norm,
                    temp=.006,
                    perm=perm, betas=betas, select=select)
                    # loss_clip = img_loss
            else:
                # epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                loss_clip = soft_clip_loss(
                    neural_features_norm,
                    img_features_norm,
                    temp=logit_scale)            
            
            
            loss_clip *= clip_scale
            loss = loss_clip                    
        
            total_loss += loss.item()
            
            for idx, label in enumerate(labels):
                # First, select k-1 classes excluding the correct class
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                selected_text_features = text_features_all[selected_classes]
                
                if k==200:
                    # Compute the corresponding logits
                    logits_img = logit_scale * neural_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    # print("logits_single", logits_single.shape)
                    # Get the predicted class
                    # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) in {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        # print("predicted_label", predicted_label)
                        correct += 1
                    
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                   
                    # Check if the true label is in the top-5 predictions
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                elif k == 50 or k == 100:
                    # For k=50 or 100, select k classes for evaluation
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]

                    logits_img = logit_scale * neural_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    
                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    if predicted_label == label.item():
                        correct += 1
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                   
                    # Check if the true label is in the top-5 predictions
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                elif k==2 or k==4 or k==10:
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                    # Compute the corresponding logits
                    logits_img = logit_scale * neural_features[idx] @ selected_img_features.T
                    # logits_text = logit_scale * neural_features[idx] @ selected_text_features.T
                    # logits_single = (logits_text + logits_img) / 2.0
                    logits_single = logits_img
                    # print("logits_single", logits_single.shape)
                    # Get the predicted class
                    # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) in {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        correct += 1
                    total += 1
                else:
                    print("Error.")
            del modal, data, labels, text, text_features, img, img_features, index, img_index, sub_ids
    # print("total_loss", total_loss)
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total
    return average_loss, accuracy, top5_acc

def main_train_loop(test_subjects, current_time, unified_model, high_pipe, train_dataloader, test_dataloader, optimizer, device, text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, logger=None, accelerator=accelerator, eval_modality='eeg'):
    logger = wandb_logger(config) if logger else None
    logger.watch(unified_model,logger) 
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs = []
    v4_accs = []
    v10_accs = []

    best_accuracy = 0.0
    best_model_weights = None
    best_epoch_info = {}
    results = []  # List to store results for each epoch
    scaler = GradScaler()  # 初始化GradScaler    
    if config.use_prior:
        high_pipe.diffusion_prior.train()
        from diffusers.optimization import get_cosine_schedule_with_warmup
        high_pipe.diffusion_prior.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=(len(train_dataloader) * config.epochs),
        )
    for epoch in range(config.epochs):
        # Train the model
        train_loss, train_accuracy = train_model(epoch, unified_model, high_pipe, train_dataloader, optimizer, device, text_features_train_all, img_features_train_all, config=config, accelerator=accelerator, eval_modality=eval_modality)
        if (epoch +1) % 10 == 0:                    
            # Get the current time and format it as a string (e.g., '2024-01-17_15-30-00')                  
            os.makedirs(f"./models/contrast/across/{config.encoder_type}/{current_time}", exist_ok=True)             
            file_path = f"./models/contrast/across/{config.encoder_type}/{current_time}/{epoch+1}.pth"
            torch.save(unified_model.state_dict(), file_path)
            
            os.makedirs(f"./models/contrast/across/{config.encoder_type}/{current_time}/prior_diffusion", exist_ok=True)             
            prior_file_path = f"./models/contrast/across/{config.encoder_type}/{current_time}/prior_diffusion/{epoch+1}.pth"
            torch.save(high_pipe.diffusion_prior.state_dict(), prior_file_path)            
            print(f"Unified model saved in {file_path}!")
            print(f"prior diffusion model saved in {prior_file_path}!")
            
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate the model            
        if eval_modality == 'fmri':                
            test_loss, test_accuracy, top5_acc = evaluate_model(epoch, unified_model, high_pipe, test_dataloader, device, text_features_test_all, img_features_test_all,k=100, config=config,  eval_modality=eval_modality)
        else:
            test_loss, test_accuracy, top5_acc = evaluate_model(epoch, unified_model, high_pipe, test_dataloader, device, text_features_test_all, img_features_test_all,k=200, config=config,  eval_modality=eval_modality)    
        _, v2_acc, _ = evaluate_model(epoch, unified_model, high_pipe, test_dataloader, device, text_features_test_all, img_features_test_all, k = 2, config=config,  eval_modality=eval_modality)
        _, v4_acc, _ = evaluate_model(epoch, unified_model, high_pipe, test_dataloader, device, text_features_test_all, img_features_test_all, k = 4, config=config,  eval_modality=eval_modality)
        _, v10_acc, _ = evaluate_model(epoch, unified_model, high_pipe, test_dataloader, device, text_features_test_all, img_features_test_all, k = 10, config=config,  eval_modality=eval_modality)
        _, v50_acc, v50_top5_acc = evaluate_model(epoch, unified_model, high_pipe, test_dataloader, device, text_features_test_all, img_features_test_all,  k=50, config=config,  eval_modality=eval_modality)
        _, v100_acc, v100_top5_acc = evaluate_model(epoch, unified_model, high_pipe, test_dataloader, device, text_features_test_all, img_features_test_all,  k=100, config=config,  eval_modality=eval_modality)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)
        
        # Append results for this epoch
        epoch_results = {
        "epoch": epoch + 1,
        # "train_loss": train_loss,
        # "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "v2_acc": v2_acc,
        "v4_acc": v4_acc,
        "v10_acc": v10_acc,
        "top5_acc":top5_acc,
        "v50_acc": v50_acc,
        "v100_acc": v100_acc,
        "v50_top5_acc":v50_top5_acc,
        "v100_top5_acc": v100_top5_acc
        }

        results.append(epoch_results)
        # If the test accuracy of the current epoch is the best, save the model and related information
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            # best_model_weights = model.state_dict().copy()
            
            best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "v2_acc":v2_acc,
                "v4_acc":v4_acc,
                "v10_acc":v10_acc
            }
        logger.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy,
            "v2 Accuracy": v2_acc,
            "v4 Accuracy": v4_acc,
            "v10 Accuracy": v10_acc,
            "Epoch": epoch
        })

        print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
        print(f"Epoch {epoch + 1}/{config.epochs} - v2 Accuracy:{v2_acc} - v4 Accuracy:{v4_acc} - v10 Accuracy:{v10_acc} - v50 Accuracy:{v50_acc} - v100 Accuracy:{v100_acc}")

    # # Load the best model weights
    # model.load_state_dict(best_model_weights)

    # # # Save the best model
    # torch.save(model.state_dict(), '{train_pos_img_text}.pth')

    # Create 5 subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # Loss curve
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(test_losses, label='Test Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    # Overall accuracy curve
    axs[0, 1].plot(train_accuracies, label='Train Accuracy')
    axs[0, 1].plot(test_accuracies, label='Test Accuracy')
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")

    # The following are the three new plots you added, assuming you've already calculated the corresponding accuracies
    # 2-class accuracy plot
    axs[1, 0].plot(v2_accs, label='2-class Accuracy')
    axs[1, 0].legend()
    axs[1, 0].set_title("2-Class Accuracy Curve")

    # 4-class accuracy plot
    axs[1, 1].plot(v4_accs, label='4-class Accuracy')
    axs[1, 1].legend()
    axs[1, 1].set_title("4-Class Accuracy Curve")

    # 10-class accuracy plot
    axs[2, 0].plot(v10_accs, label='10-class Accuracy')
    axs[2, 0].legend()
    axs[2, 0].set_title("10-Class Accuracy Curve")

    # Construct the string information for annotation
    info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
                f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
                f"Test Accuracy: {best_epoch_info['test_accuracy']:.4f}\n"
                f"v2_acc:{best_epoch_info['v2_acc']:.4f}\n"
                f"v4_acc:{best_epoch_info['v4_acc']:.4f}\n"
                f"v10_acc:{best_epoch_info['v10_acc']:.4f}")

    axs[2, 1].axis('off')  
    axs[2, 1].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[2, 1].transAxes)

    plt.tight_layout()

    # Add main title
    plt.suptitle('pos_img_text', fontsize=16, y=1.05)
    plt.savefig('pos_img_text')
    logger.finish()
    return results

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import torch
from torch.optim import AdamW
import itertools

def create_optimizer_for_multiple_models(models, use_prior=True, max_lr=1e-3):
    opt_grouped_parameters = []

    # 遍历所有模型并将其优化器参数加入
    for model in models:
        if hasattr(model, 'diffusion_prior') and use_prior:
            opt_grouped_parameters.append({
                'params': model.parameters()
            })
        else:
            # 将模型的所有参数加入优化器
            opt_grouped_parameters.append({
                'params': model.parameters()
            })

    # 创建最终的优化器
    optimizer = AdamW(opt_grouped_parameters, lr=max_lr)
    return optimizer


import argparse
import os
import datetime
import torch
from torch.utils.data import DataLoader
import itertools
from transformers import AdamW
import csv
from adabelief_pytorch import AdaBelief
import os
# CUDA_VISIBLE_DEVICES="0, 2, 3, 4, 5, 6, 7"
def main():
    # Use argparse to parse the command-line arguments
    parser = argparse.ArgumentParser(description='EEG Transformer Training Script')
    parser.add_argument(
        '--encoder_paths',
        nargs='+',
        required=False,
        default=['eeg=/mnt/dataset1/ldy/Workspace/EEG_Image_decode/Retrieval/models/contrast/across/ATMS/01-06_01-46/150.pth',
                 'meg=/mnt/dataset1/ldy/Workspace/EEG_Image_decode/Retrieval/models/contrast/across/ATMS/01-11_14-50/150.pth',
                'fmri=/mnt/dataset0/ldy/Workspace/EEG_Image_decode/Retrieval/models/contrast/across/ATMS/01-18_01-35/50.pth'
                 ])           
    # New parameter to specify which modalities to train on
    parser.add_argument('--modalities', nargs='+', choices=['eeg', 'meg', 'fmri'], default=['eeg', 'meg', 'fmri'], help='List of modalities to train on (e.g., eeg, meg, fmri)')
    # 新增参数，用于指定评估的模态
    parser.add_argument('--eval_modality', type=str, choices=['eeg', 'meg', 'fmri'], default='fmri', help='Modality to evaluate on')
    parser.add_argument('--depth', type=int, default=8, help='Depth of the model (default: 4)')
    # parser.add_argument('--eeg_data_path', type=str, default="/home/ldy/THINGS-EEG/Preprocessed_data_250Hz", help='Path to the EEG dataset')
    # parser.add_argument('--meg_data_path', type=str, default="/home/ldy/THINGS-MEG/preprocessed_newsplit", help='Path to the MEG dataset')    
    # parser.add_argument('--fmri_data_path', type=str, default="/home/ldy/fmri_dataset/Preprocessed", help='Path to the MEG dataset')        
    
    # # # # Paths for datasets  
    # parser.add_argument('--eeg_data_path', type=str, default="/home/ldy/4090_Workspace/4090_THINGS/Preprocessed_data_250Hz", help='Path to the EEG dataset')
    # parser.add_argument('--meg_data_path', type=str, default="/home/ldy/THINGS-MEG/preprocessed_newsplit", help='Path to the MEG dataset')    
    # parser.add_argument('--fmri_data_path', type=str, default="/mnt/dataset0/ldy/datasets/fmri_dataset/Preprocessed", help='Path to the fMRI dataset')        
    
    
    parser.add_argument('--eeg_data_path', type=str, default="/mnt/dataset0/ldy/datasets/THINGS_EEG/Preprocessed_data_250Hz", help='Path to the EEG dataset')
    parser.add_argument('--meg_data_path', type=str, default="/mnt/dataset0/ldy/datasets/THINGS_MEG/preprocessed_newsplit", help='Path to the MEG dataset')    
    parser.add_argument('--fmri_data_path', type=str, default="/mnt/dataset0/ldy/datasets/fmri_dataset/Preprocessed", help='Path to the fMRI dataset')      
    
    
    # Output directory and logging configuration
    parser.add_argument('--output_dir', type=str, default='./outputs/contrast', help='Directory to save output results')    
    parser.add_argument('--project', type=str, default="train_pos_img_text_rep", help='WandB project name')
    parser.add_argument('--entity', type=str, default="sustech_rethinkingbci", help='WandB entity name')
    parser.add_argument('--name', type=str, default="lr=3e-4_img_pos_pro_eeg", help='Experiment name')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=250, help='Batch size')
    parser.add_argument('--logger', type=bool, default=True, help='Enable WandB logging')
    parser.add_argument('--gpu', type=str, default='cuda:3', help='GPU device to use')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Device to run on (cpu or gpu)')    
    parser.add_argument('--insubject', type=bool, default=True, help='In-subject mode or cross-subject mode')
    parser.add_argument('--encoder_type', type=str, default='Unified_EEG+MEG+fMRI_EEG', help='Encoder type')
    parser.add_argument('--test_subjects', nargs='+', default=['sub-02'], help='Subject ID to test on')        
    parser.add_argument('--eeg_subjects', nargs='+', default=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'], help='List of subject IDs (default: sub-01 to sub-10)')                
    # parser.add_argument('--eeg_subjects', nargs='+', default=['sub-01'], help='List of EEG subject IDs')
    parser.add_argument('--meg_subjects', nargs='+', default=['sub-01', 'sub-02', 'sub-03', 'sub-04'], help='List of MEG subject IDs')    
    # parser.add_argument('--meg_subjects', nargs='+', default=['sub-01'], help='List of MEG subject IDs')    
    # parser.add_argument('--fmri_subjects', nargs='+', default=['sub-02'], help='List of fMRI subject IDs')    
    parser.add_argument('--fmri_subjects', nargs='+', default=['sub-01', 'sub-02', 'sub-03'], help='List of fMRI subject IDs')    
    parser.add_argument("--use_prior", action=argparse.BooleanOptionalAction, default=True, help="whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
    )    
    args = parser.parse_args()
        # Initialize Accelerator
    accelerator = Accelerator(device_placement=True)

    encoder_paths = {}
    for path in args.encoder_paths:
        key, value = path.split('=')
        encoder_paths[key] = value
    # Set device based on the argument
    # device = torch.device(args.gpu if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')

    # Initialize empty datasets for each modality
    eeg_train_dataset = None
    meg_train_dataset = None
    fmri_train_dataset = None
    text_features_train_all = {}
    text_features_test_all = {}
    img_features_train_all = {}
    img_features_test_all = {}

    # Load datasets based on selected modalities
    if 'eeg' in args.modalities:
        eeg_train_dataset = MetaEEGDataset(args.eeg_data_path, args.eeg_subjects, train=True)
        text_features_train_all['eeg'] = eeg_train_dataset.text_features
        img_features_train_all['eeg'] = eeg_train_dataset.img_features

    if 'meg' in args.modalities:
        meg_train_dataset = MetaMEGDataset(args.meg_data_path, args.meg_subjects, train=True)
        text_features_train_all['meg'] = meg_train_dataset.text_features
        img_features_train_all['meg'] = meg_train_dataset.img_features

    if 'fmri' in args.modalities:
        fmri_train_dataset = MetafMRIDataset(args.fmri_data_path, args.fmri_subjects, train=True)
        text_features_train_all['fmri'] = fmri_train_dataset.text_features
        img_features_train_all['fmri'] = fmri_train_dataset.img_features

    # Initialize training loop
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")    
    
    
    unified_model = UnifiedEncoder(encoder_paths, device, num_experts=5, num_heads=args.depth, ff_dim=64*args.depth, num_layers=args.depth)
    
    unified_model.to(device)

    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
    high_pipe = Pipe(diffusion_prior, device=device)
    high_pipe.diffusion_prior.to(device)
    # 将模型列表传递给函数
    models = [unified_model, diffusion_prior]
        
    optimizer = create_optimizer_for_multiple_models(models, use_prior=True, max_lr=1e-3)

    # 替换为 AdaBelief 优化器
    # optimizer = AdaBelief(
    #     itertools.chain(unified_model.parameters()),
    #     lr=args.lr,
    #     betas=(0.9, 0.999),  # 与 AdamW 相同的 beta 参数
    #     eps=1e-16,            # 与 AdamW 相同的 epsilon 参数
    #     weight_decay=1e-2,    # 与 AdamW 相同的 weight decay 参数
    #     rectify=False         # 如果您想使用 Rectified 的 AdaBelief，可以设置为 True
    # )
    for name, param in unified_model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")    
    def format_num(num):
        for unit in ['','K','M','B','T']:
            if num < 1000:
                return f"{num:.2f}{unit}"
            num /= 1000
        return f"{num:.2f}P"

    # 计算并打印模型的总参数量和可训练参数量
    total_params = sum(p.numel() for p in unified_model.parameters())
    trainable_params = sum(p.numel() for p in unified_model.parameters() if p.requires_grad)
    print(f"Total parameters: {format_num(total_params)}")
    print(f"Trainable parameters: {format_num(trainable_params)}")

    # 计算并打印可训练参数的百分比
    if total_params > 0:
        trainable_percentage = (trainable_params / total_params) * 100
        print(f"Trainable parameters percentage: {trainable_percentage:.2f}%")
    else:
        print("Total parameters count is zero, cannot compute percentage.")
            
    # Define the meta data loader dynamically based on selected modalities
    metadataloader = MetaDataLoader(
        eeg_dataset=eeg_train_dataset if 'eeg' in args.modalities else None,
        meg_dataset=meg_train_dataset if 'meg' in args.modalities else None,
        fmri_dataset=fmri_train_dataset if 'fmri' in args.modalities else None,
        batch_size=args.batch_size,
        drop_last=True,
        modalities=args.modalities
    )
    train_loader = metadataloader

    # Prepare test dataset based on eval_modality and test_subjects
    if args.eval_modality == 'eeg':
        test_dataset = EEGDataset(args.eeg_data_path, subjects=args.test_subjects, train=False)
    elif args.eval_modality == 'meg':
        test_dataset = MEGDataset(args.meg_data_path, subjects=args.test_subjects, train=False)
    elif args.eval_modality == 'fmri':
        test_dataset = fMRIDataset(args.fmri_data_path, subjects=args.test_subjects, train=False)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    
    # Collect test features
    text_features_test_all[args.eval_modality] = test_dataset.text_features
    img_features_test_all[args.eval_modality] = test_dataset.img_features    

    # Prepare everything with accelerator
    unified_model, diffusion_prior, optimizer, train_loader, test_loader = accelerator.prepare(
        unified_model, diffusion_prior, optimizer, train_loader, test_loader
    )
    # Perform the main training loop
    results = main_train_loop(args.test_subjects, current_time, unified_model, high_pipe, train_loader, test_loader, optimizer, device, 
                                text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config=args, 
                                logger=args.logger, accelerator=accelerator, eval_modality=args.eval_modality)



if __name__ == '__main__':
    main()

