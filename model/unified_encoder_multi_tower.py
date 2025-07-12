import sys
import torch
import numpy as np
import torch.nn as nn
from einops.layers.torch import Rearrange
import numpy as np
from torch import nn
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
import torch
from torch import nn
from omegaconf import OmegaConf
import os
from .EEG_MedformerTS import eeg_encoder
from .MEG_MedformerTS import  meg_encoder
from .fMRI_MedformerTS import fmri_encoder


cfg = OmegaConf.load(os.path.join("/mnt/dataset1/ldy/Workspace/FLORA/configs/config.yaml"))
cfg = OmegaConf.structured(cfg)

# Initialize linear layers with Xavier uniform initialization
default_linear_init = nn.init.xavier_uniform_

import torch
import torch.nn as nn
import torch.nn.functional as F



# class MoEProjection(nn.Module):
#     def __init__(self, input_dim=250, output_dim=1024, num_experts=3, num_heads=4, ff_dim=2048, num_layers=4):
#         super(MoEProjection, self).__init__()
#         self.num_experts = num_experts
#         self.output_dim = output_dim
        
#         # 使用单个线性层并行计算所有专家的输出
#         self.experts = nn.Linear(input_dim, output_dim * num_experts)
        
#         # 路由器网络：可以使用Transformer来增强路由的能力
#         self.router = nn.Sequential(
#             nn.Linear(input_dim, input_dim * 2),
#             nn.ReLU(),
#             nn.Linear(input_dim * 2, num_experts),
#             nn.Softmax(dim=-1)  # 对每个专家的权重进行归一化
#         )
        
#         # 增强的专家网络，使用Transformer Encoder处理每个专家的输出
#         self.transformer_layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(d_model=output_dim, nhead=num_heads, dim_feedforward=ff_dim)
#             for _ in range(num_layers)
#         ])
        
#     def forward(self, x):
#         """
#         前向传播
#         Args:
#             x: 输入张量，形状 [batch_size, input_dim]
#         Returns:
#             输出张量，形状 [batch_size, output_dim]
#         """
#         # 计算路由权重
#         routing_weights = self.router(x)  # [batch_size, num_experts]
        
#         # 并行计算所有专家的输出
#         # 输出形状 [batch_size, num_experts * output_dim]
#         experts_output = self.experts(x)
        
#         # 重塑为 [batch_size, num_experts, output_dim]
#         experts_output = experts_output.view(x.size(0), self.num_experts, self.output_dim)
        
#         # 将路由权重扩展为 [batch_size, num_experts, 1] 以便广播
#         routing_weights = routing_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
        
#         # 对每个专家的输出应用 Transformer 层
#         for transformer in self.transformer_layers:
#             experts_output = transformer(experts_output)  # [batch_size, num_experts, output_dim]
        
#         # 计算加权输出
#         weighted_output = experts_output * routing_weights  # [batch_size, num_experts, output_dim]
        
#         # 将所有专家的加权输出求和，得到最终输出 [batch_size, output_dim]
#         output = weighted_output.sum(dim=1)
        
#         return output




class HardMoEProjection(nn.Module):
    def __init__(self, input_dim=250, output_dim=1024, num_experts=3):
        super(HardMoEProjection, self).__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim
        
        self.experts = nn.Linear(input_dim, output_dim * num_experts)
        
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, num_experts)
        )
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状 [batch_size, input_dim]
        Returns:
            输出张量，形状 [batch_size, output_dim]
        """
        # 计算路由分数
        routing_scores = self.router(x)  # [batch_size, num_experts]
        
        # 使用 argmax 找到得分最高的专家
        expert_indices = torch.argmax(routing_scores, dim=-1)  # [batch_size]
        
        # 创建 one-hot 向量
        routing_weights = torch.zeros_like(routing_scores)  # [batch_size, num_experts]
        routing_weights.scatter_(1, expert_indices.unsqueeze(-1), 1.0)  # 将选中的专家权重设为1
        
        # 并行计算所有专家的输出
        experts_output = self.experts(x)  # [batch_size, num_experts * output_dim]
        
        # 重塑为 [batch_size, num_experts, output_dim]
        experts_output = experts_output.view(x.size(0), self.num_experts, self.output_dim)
        
        # 将路由权重扩展为 [batch_size, num_experts, 1] 以便广播
        routing_weights = routing_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
        
        # 计算选中专家的输出
        weighted_output = experts_output * routing_weights  # [batch_size, num_experts, output_dim]
        
        # 将所有专家的输出求和（实际上只有选中的专家有输出）
        output = weighted_output.sum(dim=1)  # [batch_size, output_dim]
        
        return output

    def get_selected_expert_indices(self, x):
        """
        获取选中的专家索引，用于分析或调试
        Args:
            x: 输入张量，形状 [batch_size, input_dim]
        Returns:
            专家索引张量，形状 [batch_size]
        """
        with torch.no_grad():
            routing_scores = self.router(x)
            expert_indices = torch.argmax(routing_scores, dim=-1)
        return expert_indices

class MoEProjection(nn.Module):
    def __init__(self, input_dim=250, output_dim=1024, num_experts=3):
        super(MoEProjection, self).__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim
        
        # 使用单个线性层并行计算所有专家的输出
        self.experts = nn.Linear(input_dim, output_dim * num_experts)
        
        # 路由器用于计算每个专家的权重
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, num_experts),
            # nn.Softmax(dim=-1)  # 对每个专家的权重进行归一化
        )
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状 [batch_size, input_dim]
        Returns:
            输出张量，形状 [batch_size, output_dim]
        """
        # 计算路由权重
        routing_weights = self.router(x).sigmoid()  # [batch_size, num_experts]
        
        # 并行计算所有专家的输出
        # 输出形状 [batch_size, num_experts * output_dim]
        experts_output = self.experts(x)
        
        # 重塑为 [batch_size, num_experts, output_dim]
        experts_output = experts_output.view(x.size(0), self.num_experts, self.output_dim)
        
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # 将路由权重扩展为 [batch_size, num_experts, 1] 以便广播
        routing_weights = routing_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
        
        # 计算加权输出
        weighted_output = experts_output * routing_weights  # [batch_size, num_experts, output_dim]
        
        # 将所有专家的加权输出求和，得到最终输出 [batch_size, output_dim]
        output = weighted_output.sum(dim=1)
        
        return output

import torch
import torch.nn as nn

class MoEProjection_upsamp(nn.Module):
    def __init__(self, input_dim=250, output_dim=1024, num_experts=3):
        super(MoEProjection_upsamp, self).__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.clip_seq_dim = 256
        # 使用单个线性层并行计算所有专家的输出
        self.experts = nn.Linear(input_dim, output_dim * num_experts)
        
        # 路由器用于计算每个专家的权重
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, num_experts),
            nn.Softmax(dim=-1)  # 对每个专家的权重进行归一化
        )
        
        # 上采样卷积，用于从 [batch_size, 1, output_dim] 到 [batch_size, output_dim, 257]
        self.upsample = nn.Linear(1, self.clip_seq_dim)
        
        # 投影层，用于从 upsampled_output 变换到 output
        self.projector = self._projector(output_dim * self.clip_seq_dim, output_dim)

    def _projector(self, in_dim, out_dim, h=512):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, out_dim)
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状 [batch_size, input_dim]
        Returns:
            output: 投影后的输出张量，形状 [batch_size, output_dim]
            upsampled_output: 经过上采样卷积得到的输出，形状 [batch_size, 257, output_dim]
        """
        # 计算路由权重
        routing_weights = self.router(x)  # [batch_size, num_experts]
        
        # 并行计算所有专家的输出
        # 输出形状 [batch_size, num_experts * output_dim]
        experts_output = self.experts(x)
        
        # 重塑为 [batch_size, num_experts, output_dim]
        experts_output = experts_output.view(x.size(0), self.num_experts, self.output_dim)
        
        # 将路由权重扩展为 [batch_size, num_experts, 1] 以便广播
        routing_weights = routing_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
        
        # 计算加权输出
        weighted_output = experts_output * routing_weights  # [batch_size, num_experts, output_dim]
        
        # 将所有专家的加权输出求和，得到最终输出 [batch_size, output_dim]
        upsampled_output = weighted_output.sum(dim=1)  # [batch_size, output_dim]

        # 上采样操作：对专家输出进行上采样 [batch_size, output_dim]
        x_upsampled = upsampled_output.unsqueeze(2)  # [batch_size, output_dim, 1]
        upsampled_output = self.upsample(x_upsampled)  # [batch_size, output_dim, 257]
        upsampled_output = upsampled_output.permute(0, 2, 1)  # [batch_size, 257, output_dim]

        # 对 upsampled_output 进行投影操作
        upsampled_output_flatten = upsampled_output.reshape(upsampled_output.size(0), -1)  # [batch_size, output_dim * 257]
        output = self.projector(upsampled_output_flatten)  # [batch_size, output_dim]

        return output, upsampled_output


import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn

class UnifiedEncoder(nn.Module):
    def __init__(self, encoder_paths: dict[str, str] = None, device=None, in_dim=1024, h=1024, out_dim=1024, num_experts=5, num_heads=4, ff_dim=2048, num_layers=4, user_caption=False):
        super().__init__()
        
        # 设备设置
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.user_caption = user_caption
        
        if self.user_caption:
            self.moe_projection = MoEProjection_upsamp(input_dim=in_dim, output_dim=out_dim, num_experts=num_experts)
        else: 
            self.moe_projection = MoEProjection(input_dim=in_dim, output_dim=out_dim, num_experts=num_experts)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))          
        self.modals = ['eeg', 'meg', 'fmri']
                    
        self.encoder = nn.ModuleDict()
        
        # 从头训练或加载预训练模型
        for modal in self.modals:
            if modal == 'eeg':
                encoder = eeg_encoder()
                if encoder_paths is not None and 'eeg' in encoder_paths:
                    encoder.load_state_dict(torch.load(encoder_paths['eeg'], map_location=self.device))
                    encoder.eval()  # 如果加载预训练权重则设为评估模式
                    for param in encoder.parameters():  # 冻结参数
                        param.requires_grad = False
                encoder.to(self.device)
                self.encoder[modal] = encoder
                
            elif modal == 'meg':
                encoder = meg_encoder()
                if encoder_paths is not None and 'meg' in encoder_paths:
                    encoder.load_state_dict(torch.load(encoder_paths['meg'], map_location=self.device))
                    encoder.eval()
                    for param in encoder.parameters():
                        param.requires_grad = False
                encoder.to(self.device)
                self.encoder[modal] = encoder
    
            elif modal == 'fmri':                
                encoder = fmri_encoder()
                if encoder_paths is not None and 'fmri' in encoder_paths:
                    encoder.load_state_dict(torch.load(encoder_paths['fmri'], map_location=self.device))
                    encoder.eval()
                    for param in encoder.parameters():
                        param.requires_grad = False
                encoder.to(self.device)
                self.encoder[modal] = encoder

    def forward(self, x, subject_ids, modal):        
        x = self.encoder[modal](x, subject_ids)                                

        if self.user_caption:
            output, upsampled_output = self.moe_projection(x)
            return output, upsampled_output
        else:
            output = self.moe_projection(x)
            return output