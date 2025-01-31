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

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
import sys
from einops.layers.torch import Rearrange, Reduce
from loss import ClipLoss
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
import csv
from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet
import argparse
import math
from fmri_datasets_joint_subjects import fMRIDataset
sys.path.insert(0,'/mnt/dataset0/ldy/Workspace/EEG_Image_decode/Retrieval')
sys.path.insert(0,'/mnt/dataset0/ldy/Workspace/EEG_Image_decode/Retrieval/model')
from umbrae import BrainXS_thingsfmri
# sys.path.insert(0,'/mnt/dataset0/ldy/Workspace/MindEyeV2/src')
# from models import BrainNetwork
sys.path.insert(0,'/mnt/dataset0/ldy/Workspace/neuro_v2l/Code/llava/model/fmri_encoder')
from transformers import CLIPVisionConfig
from vit3d import CLIPVision3dModelWithProjection

class NeV2L(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()
        self.config = CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14-336")
        self.config.image_size = (32, 16, 16)
        self.config.num_channels = 1
        self.config.patch_size = 8
        self.proj_fmri = nn.Linear(7000, 8192)
        self.model = CLIPVision3dModelWithProjection(self.config)


    def forward(self, x):
        x = self.proj_fmri(x)
        x = x.reshape(x.size(0), -1, 16, 16)
        x = self.model(x, output_hidden_states=True)
        eeg_features = x.last_hidden_state
        eeg_features = eeg_features.mean(dim=1)
        return eeg_features

#--------------------------------MindEyeV2-----------------------------------#
class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()
        self.proj_fmri = nn.Linear(7000, 1024) # Equivalent to the original
        self.BrainNetwork = BrainNetwork(h=1024, in_dim=1024, seq_len=1, n_blocks=4,
                          clip_size=1024, out_dim=1024, 
                          blurry_recon=False, clip_scale=1)

    def forward(self, x):
        x = self.proj_fmri(x)
        x = x.reshape(x.size(0), -1, 1024)
        x = self.BrainNetwork(x)
        eeg_features,_,_ = x
        # print("eeg_features:", eeg_features.shape)
        eeg_features = eeg_features.mean(dim=1)
        # print("eeg_features:", eeg_features.shape)
        # eeg_features = eeg_features.squeeze(1)
        # print("eeg_features:", eeg_features.shape)
        return eeg_features


#--------------------------------NICE-----------------------------------#
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (8, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)     
        # print("x", x.shape)   
        x = self.tsconv(x)
        # print("tsconv", x.shape)   
        x = self.projection(x)
        # print("projection", x.shape)  
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x

class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )
   
class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=7600, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class NICE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()        
        self.proj_fmri = nn.Linear(7000, 8192)
    def forward(self, data):
        data = self.proj_fmri(data)
        data = data.reshape(data.size(0), -1, 1024)
        eeg_embedding = self.enc_eeg(data)
        out = self.proj_eeg(eeg_embedding)

        return out  
#########################################################################


#-------------------------------EEGNetv4--------------------------------#
class EEGNetv4_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = (8, 1024)
        self.eegnet = EEGNetv4(
            in_chans=self.shape[0],
            n_classes=1024,   
            input_window_samples=self.shape[1],
            final_conv_length='auto',
            pool_mode='mean',
            F1=8,
            D=20,
            F2=160,
            kernel_length=4,
            third_kernel_size=(4, 2),
            drop_prob=0.25
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()
        self.proj_fmri = nn.Linear(7000, 8192)
    def forward(self, data):
        data = self.proj_fmri(data)
        data = data.reshape(data.size(0), -1, 1024)
        data = data.unsqueeze(0)
        data = data.reshape(data.shape[1], data.shape[2], data.shape[3], data.shape[0])
        # print(data.shape)
        prediction = self.eegnet(data)
        return prediction
#########################################################################


#--------------------------EEGConformer_Encoder-------------------------#
class EEGConformer_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = (63, 250)
        self.eegConformer = EEGConformer(n_outputs=None, 
                                   n_chans=self.shape[0], 
                                   n_filters_time=40, 
                                   filter_time_length=10, 
                                   pool_time_length=25, 
                                   pool_time_stride=5, 
                                   drop_prob=0.25, 
                                   att_depth=2, 
                                   att_heads=1, 
                                   att_drop_prob=0.5, 
                                   final_fc_length=1760, 
                                   return_features=False, 
                                   n_times=None, 
                                   chs_info=None, 
                                   input_window_seconds=None,
                                   n_classes=1024, 
                                   input_window_samples=self.shape[1], 
                                   add_log_softmax=True)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()
    def forward(self, data):
        # data = data.unsqueeze(0)
        # data = data.reshape(data.shape[1], data.shape[2], data.shape[3], data.shape[0])
        # print(data.shape)
        prediction = self.eegConformer(data)
        return prediction
#########################################################################


#-----------------------------EEGITNet_Encoder--------------------------#
class EEGITNet_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = (63, 250)
        self.eegEEGITNet = EEGITNet(n_outputs=1024, 
                                  n_chans=self.shape[0], 
                                  n_times=None, 
                                  drop_prob=0.4, 
                                  chs_info=None, 
                                  input_window_seconds=1.0, 
                                  sfreq=250, 
                                  input_window_samples=self.shape[1],
                                  add_log_softmax=True)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()
    def forward(self, data):
        prediction = self.eegEEGITNet(data)
        return prediction
#########################################################################


#--------------------------------MLP------------------------------------#
def make_block(h_c, h_l,dropout_rate=0.25):
    block = nn.Sequential(
        nn.LayerNorm(h_l),
        nn.Linear(h_l, h_l), 
        nn.GELU(),
        nn.Dropout(dropout_rate),  
        Rearrange('B C L->B L C'),
        nn.LayerNorm(h_c),
        nn.Linear(h_c, h_c), 
        nn.GELU(),
        nn.Dropout(dropout_rate),  
        Rearrange('B L C->B C L'),
    )
    return block

class Projector(nn.Module):

    def __init__(self, in_features, h_dim=(64, 1024), n_hidden_layer=2,dropout_rate=0.25):
        # in_features: (c, l)
        super().__init__()
        c, l = in_features
        h_c, h_l = h_dim
        c_o, l_o = 1, 1024

        self.input_layer = nn.Sequential(
            nn.LayerNorm(l),
            nn.Linear(l, h_l), 
            nn.GELU(),
            nn.Dropout(dropout_rate),  
            Rearrange('B C L->B L C'),
            nn.LayerNorm(c),
            nn.Linear(c, h_c), 
            nn.GELU(),
            nn.Dropout(dropout_rate),  
            Rearrange('B L C->B C L'),
        )
        
        self.output_layer = nn.Sequential(
            nn.LayerNorm(h_l),
            nn.Linear(h_l, l_o), 
            nn.GELU(),
            nn.Dropout(dropout_rate),  
            Rearrange('B C L->B L C'),
            nn.LayerNorm(h_c),
            nn.Linear(h_c, c_o), 
            nn.GELU(),
            nn.Dropout(dropout_rate),  
            Rearrange('B L C->B (C L)'),
        )
        
        self.blocks = nn.Sequential(*[
            make_block(h_c, h_l) for _ in range(n_hidden_layer)
        ])
        
        self.projector = nn.Sequential(*[
            self.input_layer,
            self.blocks,
            self.output_layer,
        ])

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.01))
                
        self.loss_func = ClipLoss()
    
    def forward(self, eeg_embeds):
        
        eeg_embeds = self.projector(eeg_embeds)
        # print("eeg_embeds")
        # print(eeg_embeds.shape)
        eeg_features = F.normalize(eeg_embeds, dim=-1)
        return eeg_features
#########################################################################


#-------------------------ShallowFBCSPNet_Encoder-----------------------#
class ShallowFBCSPNet_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = (63, 250)
        self.ShallowFBCSPNet = ShallowFBCSPNet(n_chans=self.shape[0],
                                         n_outputs=1024,
                                         n_times=self.shape[1], 
                                         n_filters_time=20, 
                                         filter_time_length=20,
                                         n_filters_spat=20,
                                         pool_time_length=25, 
                                         pool_time_stride=5, 
                                         final_conv_length='auto', 
                                         pool_mode='mean', 
                                         split_first_layer=True,
                                         batch_norm=True, 
                                         batch_norm_alpha=0.1, 
                                         drop_prob=0.5,
                                         chs_info=None, 
                                         input_window_seconds=1.0, 
                                         sfreq=250, 
                                         add_log_softmax=True)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()
    def forward(self, data):
        prediction = self.ShallowFBCSPNet(data)
        return prediction
#########################################################################


#---------------------------ATCNet_Encoder------------------------------#
class ATCNet_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = (63, 250)
        self.eegATCNet = ATCNet(n_chans=self.shape[0], 
                                n_outputs=1024,
                                input_window_seconds=1.0,
                                sfreq=250.,
                                conv_block_n_filters=8,
                                conv_block_kernel_length_1=32,
                                conv_block_kernel_length_2=8,
                                conv_block_pool_size_1=4,
                                conv_block_pool_size_2=3,
                                conv_block_depth_mult=2,
                                conv_block_dropout=0.3,
                                n_windows=5,
                                att_head_dim=4,
                                att_num_heads=2,
                                att_dropout=0.5,
                                tcn_depth=2,
                                tcn_kernel_size=4,
                                tcn_n_filters=16,
                                tcn_dropout=0.3,
                                tcn_activation=nn.ELU(),
                                concat=False,
                                max_norm_const=0.25,
                                chs_info=None,
                                n_times=None,
                                n_channels=None,
                                n_classes=None,
                                input_size_s=None,
                                add_log_softmax=True)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()
    def forward(self, data):
        # print("data", data.shape)
        prediction = self.eegATCNet(data)
        return prediction
#########################################################################


#-------------------------------Meta------------------------------------#
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model + 1, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1)
        x = x + pe
        return x

class EEGAttention(nn.Module):
    def __init__(self, channel, d_model, nhead):
        super(EEGAttention, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.channel = channel
        self.d_model = d_model

    def forward(self, src):
        src = src.permute(2, 0, 1)  # Change shape to [time_length, batch_size, channel]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output.permute(1, 2, 0)  # Change shape back to [batch_size, channel, time_length]

class MetaEEG(nn.Module):
    def __init__(self, num_channels=8, sequence_length=1024, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
        super(MetaEEG, self).__init__()
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)               
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        self.conv_blocks = nn.Sequential(*[ConvBlock(num_channels, sequence_length) for _ in range(num_blocks)],
                                         Rearrange('B C L->B L C'))
        self.linear_projection = nn.Sequential(
                                            Rearrange('B L C->B C L'),
                                            nn.Linear(sequence_length, num_latents),
                                            Rearrange('B C L->B L C'))
        self.temporal_aggregation = nn.Linear(sequence_length, 1)
        self.clip_head = MLPHead(num_latents, num_latents)
        self.mse_head = MLPHead(num_latents, num_latents)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.01))
        self.loss_func = ClipLoss()
        self.proj_fmri = nn.Linear(7000, 8192)

    def forward(self, x):    
    # def forward(self, x, subject_id):
        # print(f'Input shape: {x.shape}')
        # attn_output, _ = self.attention(x, x, x)
        x = self.proj_fmri(x)
        x = x.reshape(x.size(0), -1, 1024)
       
        x = self.attention_model(x)
        # print(f'After attention shape: {x.shape}')
         
        # x = self.subject_wise_linear[subject_id](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        
        x = self.conv_blocks(x)
        # print(f'After convolutional blocks shape: {x.shape}')
        
        # x = self.conv_blocks(x)
        # print(f'After convolutional blocks shape: {x.shape}')
        
        x = self.linear_projection(x)
        # print(f'After linear projection shape: {x.shape}')
        
        x = self.temporal_aggregation(x)
        # print(f'After temporal aggregation shape: {x.shape}')

        clip_out = self.clip_head(x)
        # print(f'Clip head output shape: {clip_out.shape}')
    
        mse_out = self.mse_head(x)
        # print(f'MSE head output shape: {mse_out.shape}')

        # return clip_out, mse_out
        return clip_out

class ConvBlock(nn.Module):
    def __init__(self, num_channels, num_features):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, num_features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)
        self.norm3 = nn.LayerNorm(num_features)
        self.residual_conv = nn.Conv1d(num_channels, num_features, kernel_size=1)

    def forward(self, x):
        # print(f'ConvBlock input shape: {x.shape}')
        residual = self.residual_conv(x)
        # residual = x
        # print(f'residual shape: {residual.shape}')
        
        x = F.gelu(self.conv1(x))
        x = self.norm1(x)
        # print(f'After first convolution shape: {x.shape}')
                
        x = F.gelu(self.conv2(x))
        x = self.norm2(x)
        # print(f'After second convolution shape: {x.shape}')
        
        x = F.gelu(self.conv3(x))
        x = self.norm3(x)
        # print(f'After third convolution shape: {x.shape}')
        
        x += residual
        # print(f'ConvBlock output shape: {x.shape}')
        return x

class MLPHead(nn.Module):
    def __init__(self, in_features, num_latents, dropout_rate=0.25):
        super(MLPHead, self).__init__()

        self.layer1 = nn.Sequential(
            Rearrange('B C L->B L C'),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, num_latents),
            nn.GELU(),
            nn.Dropout(dropout_rate), 
            Rearrange('B L C->B (C L)'),
        )
    def forward(self, x):
        # print(f'MLPHead input shape: {x.shape}')
        x = self.layer1(x)
        # print(f'After first layer of MLPHead shape: {x.shape}')
        return x
#########################################################################


def train_model(model, dataloader, optimizer, device, text_features_all, img_features_all):
    model.train()
    text_features_all = text_features_all.to(device).float() # (n_cls, d)
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha=0.99
    features_list = []  # List to store features
    save_features= True
    for batch_idx, (_, eeg_data, labels, text, text_features, img, img_features, subject_id) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        
        optimizer.zero_grad()
        eeg_features = model(eeg_data).float()
        # eeg_features,_,_ = model(eeg_data)
        eeg_features = eeg_features.float()
        features_list.append(eeg_features)
        logit_scale = model.logit_scale
        
        img_loss = model.loss_func(eeg_features, img_features, logit_scale)
        text_loss = model.loss_func(eeg_features, text_features, logit_scale)
        # loss = img_loss + text_loss
        # print("text_loss", text_loss)
        # print("img_loss", img_loss)
        loss = alpha * img_loss + (1 - alpha) * text_loss
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
        
        # logits = logit_scale * eeg_features @ text_features_all.T # (n_batch, n_cls)
        
        logits_img = logit_scale * eeg_features @ img_features_all.T
        # logits_text = logit_scale * eeg_features @ text_features_all.T
        # logits_single = (logits_text + logits_img) / 2.0
        # logits_text = logit_scale * eeg_features @ text_features_all.T
        logits_single = logits_img
        predicted = torch.argmax(logits_single, dim=1) # (n_batch, ) \in {0, 1, ..., n_cls-1}

        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()


    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    return average_loss, accuracy

def evaluate_model(model, dataloader, device, text_features_all, img_features_all, k):
    model.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.99
    top5_correct = 0
    top5_correct_count = 0
    
    all_labels = set(range(text_features_all.size(0)))
    top5_acc = 0
    with torch.no_grad():
        for batch_idx, (_, eeg_data, labels, text, text_features, img, img_features, subject_id) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            eeg_features = model(eeg_data).float()
            logit_scale = model.logit_scale 
            # print(eeg_features.type, text_features.type, img_features.type)

            img_loss = model.loss_func(eeg_features, img_features, logit_scale)
            text_loss = model.loss_func(eeg_features, text_features, logit_scale)
            loss = img_loss*alpha + text_loss*(1-alpha)
            
            total_loss += loss.item()
            
            for idx, label in enumerate(labels):
                
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                # selected_text_features = text_features_all[selected_classes]
                selected_img_features = img_features_all[selected_classes]
                if k==100:
                    
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    # logits_text = logit_scale * eeg_features[idx] @ selected_text_features.T
                    # logits_single = (logits_text + logits_img) / 2.0
                    logits_single = logits_img
                    # print("logits_single", logits_single.shape)
                    
                    # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) \in {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        # print("predicted_label", predicted_label)
                        correct += 1
                    
                    
                    
                    
                    # print("logits_single", logits_single)
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                           
                    
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:     
                        # print("top5_indices", top5_indices)
                        # print("Yes")               
                        top5_correct_count+=1     
                    # print("*"*50)                               
                    total += 1
                    
                elif k==2 or k==4 or k==10:
                    
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    # logits_text = logit_scale * eeg_features[idx] @ selected_text_features.T
                    # logits_single = (logits_text + logits_img) / 2.0
                    logits_single = logits_img
                    # print("logits_single", logits_single.shape)
                    
                    # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) \in {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        correct += 1
                    total += 1
                else:
                    print("Error.")

                    
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total
    return average_loss, accuracy, top5_acc

def main_train_loop(sub, encoder_type, current_time, model, train_dataloader, test_dataloader, optimizer, device, 
                    text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, logger=None):
    
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs = []
    v4_accs = []
    v10_accs = []

    best_accuracy = 0.0
    best_model_weights = None
    best_epoch_info = {}
    results = []
    for epoch in range(config['epochs']):
        
        train_loss, train_accuracy = train_model(model, train_dataloader, optimizer, device, text_features_train_all, img_features_train_all)
       
        # directory = f"/mnt/dataset0/ldy/Workspace/FLORA/models/{encoder_type}/across/fMRI/{current_time}"
        directory = f"/mnt/dataset0/ldy/Workspace/FLORA/models/{encoder_type}/in_subject/fMRI/{sub}/{current_time}"
        if not os.path.exists(directory):
            os.makedirs(directory)
    
        if (epoch+1)%5 == 0:                        
            # torch.save(model.state_dict(), f"/mnt/dataset0/ldy/Workspace/FLORA/models/{encoder_type}/across/fMRI/{current_time}/{epoch+1}.pth")
            torch.save(model.state_dict(), f"/mnt/dataset0/ldy/Workspace/FLORA/models/{encoder_type}/in_subject/fMRI/{sub}/{current_time}/{epoch+1}.pth")
            print("model save in: ", directory)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        
        test_loss, test_accuracy, top5_acc = evaluate_model(model, test_dataloader, device, text_features_test_all, img_features_test_all,k=100)
        _, v2_acc, _ = evaluate_model(model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 2)
        _, v4_acc, _ = evaluate_model(model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 4)
        _, v10_acc, _ = evaluate_model(model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 10)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)
        # Append results for this epoch
        epoch_results = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "v2_acc": v2_acc,
        "v4_acc": v4_acc,
        "v10_acc": v10_acc,
        "top5_acc":top5_acc
        }
        results.append(epoch_results)
        
        print(f"Epoch {epoch + 1}/{config['epochs']} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
        print(f"Epoch {epoch + 1}/{config['epochs']} - v2 Accuracy:{v2_acc} - v4 Accuracy:{v4_acc} - v10 Accuracy:{v10_acc}")

    return results

import datetime
def main():
    parser = argparse.ArgumentParser(description='Train EEG-Image/Text Model')
    parser.add_argument('--data_path', type=str, default="/mnt/dataset0/ldy/datasets/fmri_dataset/Preprocessed", help='Path to the preprocessed data')
    parser.add_argument('--project', type=str, default="train_pos_img_text_rep", help='Project name')
    parser.add_argument('--entity', type=str, default="sustech_rethinkingbci", help='Entity name')
    parser.add_argument('--name', type=str, default="lr=3e-4_img_pos_pro_eeg", help='Experiment name')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--logger', default=False, help='Enable logging')
    parser.add_argument('--insubject', default=True, help='Train within subject')
    parser.add_argument('--test_subjects', nargs='+', default=['sub-01'], help='Subject ID to test on')  
    parser.add_argument('--encoder_type', type=str, default='NeV2L', help='EEG encoder model type, you can choose from these options: Projector, EEGConformer_Encoder, MetaEEG, EEGNetv4_Encoder, ShallowFBCSPNet_Encoder, NICE, ATCNet_Encoder, EEGITNet_Encoder')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training (e.g., "cuda:0" or "cpu")')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    data_path = args.data_path
    subjects = ['sub-01', 'sub-02', 'sub-03']
    # subjects = ['sub-01']
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    

    # Re-initialize the model for each subject
    model = globals()[args.encoder_type]()

    print(args.encoder_type)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # train_dataset = fMRIDataset(
    #     data_path,
    #     subjects=subjects,
    #     train=True
    # )
    # test_dataset = fMRIDataset(
    #     data_path,
    #     subjects=subjects,
    #     train=False
    # )
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=0,
    #     drop_last=True
    # )
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=1,
    #     shuffle=True,
    #     num_workers=0,
    #     drop_last=True
    # )

    # text_features_train_all = train_dataset.text_features
    # text_features_test_all = test_dataset.text_features
    # img_features_train_all = train_dataset.img_features
    # img_features_test_all = test_dataset.img_features
    # current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    # config = vars(args)
    # results = main_train_loop(
    #     subjects,
    #     args.encoder_type,
    #     current_time,
    #     model,
    #     train_loader,
    #     test_loader,
    #     optimizer,
    #     device,
    #     text_features_train_all,
    #     text_features_test_all,
    #     img_features_train_all,
    #     img_features_test_all,
    #     config,
    #     logger=args.logger
    # )


    for sub in subjects:
        train_dataset = fMRIDataset(
            data_path,
            subjects=[sub],
            train=True
        )
        test_dataset = fMRIDataset(
            data_path,
            subjects=[sub],
            train=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        text_features_train_all = train_dataset.text_features
        text_features_test_all = test_dataset.text_features
        img_features_train_all = train_dataset.img_features
        img_features_test_all = test_dataset.img_features
        current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

        config = vars(args)
        results = main_train_loop(
            sub,
            args.encoder_type,
            current_time,
            model,
            train_loader,
            test_loader,
            optimizer,
            device,
            text_features_train_all,
            text_features_test_all,
            img_features_train_all,
            img_features_test_all,
            config,
            logger=args.logger
        )
    

if __name__ == '__main__':
    main()
    
    
