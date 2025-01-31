import sys
import torch
import numpy as np
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor
from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding
import numpy as np
from model.loss import ClipLoss
from torch import nn
from typing import Optional, Tuple
from dataclasses import dataclass
import math
import functools
import copy

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn as nn
from .components import RMSNorm
from flash_attn import flash_attn_func

import open_clip

from functools import partial

from transformers import CLIPVisionModel 
from .perceiver import PerceiverResampler

import torch
from torch import nn
from torchvision import transforms
from .ATMS import ATMS
from .Medformer import Medformer
from .projector import FusionHead

from omegaconf import OmegaConf
import os

cfg = OmegaConf.load(os.path.join("/mnt/dataset0/ldy/Workspace/FLORA/configs/config.yaml"))
cfg = OmegaConf.structured(cfg)

# Initialize linear layers with Xavier uniform initialization
default_linear_init = nn.init.xavier_uniform_

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 2
    n_heads: int = 4
    token_size: int = 1024  # defined later by tokenizer
    multiple_of: int = 256  # Uncommented this line
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim -
             1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads  # Removed model parallel division
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        default_linear_init(self.wq.weight)

        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        default_linear_init(self.wk.weight)

        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        default_linear_init(self.wv.weight)

        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        default_linear_init(self.wo.weight)

        self.flash = True
        self.k_cache, self.v_cache = None, None

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], prompt=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.k_cache is None or self.v_cache is None:
            keys, values = xk, xv
        else:
            self.k_cache = self.k_cache.to(xk)
            self.v_cache = self.v_cache.to(xv)
            self.k_cache[:bsz, start_pos: start_pos + seqlen, :, :] = xk
            self.v_cache[:bsz, start_pos: start_pos + seqlen, :, :] = xv
            keys = self.k_cache[:bsz, :start_pos + seqlen]
            values = self.v_cache[:bsz, :start_pos + seqlen]

        output = flash_attn_func(
            xq, keys, values, dropout_p=0.0, causal=mask is not None)
        output = output.contiguous().view(bsz, seqlen, -1)

        return self.wo(output)

    def allocate_kv_cache(self, max_batch_size: int, max_seq_len: int) -> None:
        kv_cache_shape = (max_batch_size, max_seq_len,
                          self.n_local_heads, self.head_dim)
        if self.k_cache is None or self.k_cache.size() != kv_cache_shape:
            self.k_cache = torch.empty(kv_cache_shape)
        if self.v_cache is None or self.v_cache.size() != kv_cache_shape:
            self.v_cache = torch.empty(kv_cache_shape)

    def destroy_kv_cache(self) -> None:
        self.k_cache, self.v_cache = None, None

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * \
            ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        default_linear_init(self.w1.weight)

        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        default_linear_init(self.w2.weight)

        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        default_linear_init(self.w3.weight)

    def _silu_gating(self, x, y):
        return F.silu(x) * y

    def forward(self, x):
        return self.w2(self._silu_gating(self.w1(x), self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def _forward_ffn(self, h):
        return h + self.feed_forward(self.ffn_norm(h))

    def _forward_attention(self, x, start_pos, freqs_cis, mask, prompt):
        return x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, prompt)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], prompt=None):
        h = self._forward_attention(x, start_pos, freqs_cis, mask, prompt)
        out = self._forward_ffn(h)
        return out

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        default_linear_init(self.fc1.weight)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        default_linear_init(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x



class BrainEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14')        
        self.clip_size = (224, 224)       
        preproc = transforms.Compose([
            transforms.Resize(size=self.clip_size[0], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(size=self.clip_size),
            # transforms.ToTensor(), # only for debug
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.preprocess = preproc

        for param in self.clip.parameters():
            param.requires_grad = False
            # param.data = param.data.half()

        self.clip_width = self.clip.vision_model.embeddings.patch_embedding.out_channels

        self.conv1 = nn.ModuleDict()
        self.position_embedding = nn.ParameterDict()
        self.modals = ['image', 'fmri']
        for modal in self.modals:
            if modal =='image':
                modal_tokens = 256 + 1
                pass
            elif modal == 'fmri':
                modal_tokens = 8 + 1
                self.conv1[modal] = nn.Linear(15724, 8192)
                self.position_embedding[modal] = nn.Embedding(modal_tokens, self.clip_width)

    def clip_encode_image(self, x, modal='image'):
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1) 

        x = torch.cat([self.clip.vision_model.embeddings.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                      x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  

        pos_embedding = self.clip.vision_model.embeddings.position_embedding # Embedding(257, 1024)
        if modal == 'fmri':
            pos_embedding = self.position_embedding[modal]
            
        modal_tokens = 257 if modal == 'image' else 9
        position_ids =  torch.arange(0, modal_tokens).unsqueeze(0).to(x.device)

        x = x + pos_embedding(position_ids)
        x = self.clip.vision_model.pre_layrnorm(x)
        x = self.clip.vision_model.encoder(x, output_hidden_states=True)

        select_hidden_state_layer = -2
        select_hidden_state = x.hidden_states[select_hidden_state_layer] # torch.Size([1, 257, 1024])
        image_features = select_hidden_state[:, 1:] # torch.Size([1, 256, 1024]

        return image_features

    def encode_image(self, x, modal='image'):
        if modal in ['image']:
            x = self.preprocess(x)
            x = self.clip.vision_model.embeddings.patch_embedding(x)  # conv1, shape = [*, width, grid, grid]
        elif modal == 'fmri':
            x = self.conv1[modal](x)
            x = x.reshape(x.size(0), self.clip_width, -1)

        image_feats = self.clip_encode_image(x, modal=modal)

        return image_feats


class Perceiver(nn.Module):
    def __init__(self, patch_embed_dim=1024, hidden_size=512, num_latents=1024):
        super().__init__()
    
        self.ln_vision = nn.LayerNorm(patch_embed_dim)
        self.llm_proj = nn.Linear(
            patch_embed_dim, hidden_size
        )

        self.perceiver = PerceiverResampler(
            dim = patch_embed_dim,
            dim_head = 96,
            depth = 6,
            heads = 16,
            num_latents = num_latents,
            num_media_embeds = 1
        )

    def forward(self, image_features):
        image_features = self.ln_vision(image_features)
        inputs_llm = self.perceiver(image_features)
        return self.llm_proj(inputs_llm)


class MoEProjection(nn.Module):
    def __init__(self, input_dim=250, output_dim=1024, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, num_experts),
            nn.Softmax(dim=-1)  # 对每个专家的权重进行归一化
        )

    def forward(self, x):
        # x 的初始形状假设为 [batch_size, num_latents, input_dim]
        batch_size, num_latents, _ = x.shape

        # 计算路由权重
        routing_weights = self.router(x)  # [batch_size, num_latents, num_experts]
        
        # 逐个专家计算输出并应用权重
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)  # [batch_size, num_latents, output_dim]
            # 使用 `routing_weights` 对应专家的权重，添加 `.unsqueeze(-1)` 以适应输出维度
            weighted_output = expert_output * routing_weights[:, :, i].unsqueeze(-1)  # 广播至 [batch_size, num_latents, output_dim]
            expert_outputs.append(weighted_output)

        # 将所有专家的加权输出求和
        output = torch.stack(expert_outputs, dim=0).sum(dim=0)  # 最终形状 [batch_size, num_latents, output_dim]
        return output


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

# class PatchEmbedding(nn.Module):
#     def __init__(self, emb_size=16):
#         super().__init__()
#         self.tsconv = nn.Sequential(
#             # 第1层：卷积 + 下采样（4倍）
#             nn.Conv2d(1, 16, (1, 256), stride=(1, 4)),  # 输出宽度减少4倍
#             nn.AvgPool2d((1, 25), stride=(1, 5)),      # 输出宽度减少5倍
#             nn.BatchNorm2d(16),
#             nn.ELU(),
            
#             # 第2层：深度卷积
#             nn.Conv2d(16, 16, (8, 1), stride=(1, 1), groups=16),  # 不改变尺寸
#             nn.BatchNorm2d(16),
#             nn.ELU(),
#             nn.Conv2d(16, 16, (1, 1), stride=(1, 1)),            # 不改变尺寸
#             nn.BatchNorm2d(16),
#             nn.ELU(),
            
#             # 第3层：额外的下采样（4倍）
#             nn.AvgPool2d((1, 4), stride=(1, 4)),      # 输出宽度减少4倍
#             nn.Dropout(0.5),
            
#             # 第4层：进一步下采样（步幅10倍）
#             nn.Conv2d(16, 16, (1, 8), stride=(1, 10)),  # 输出宽度减少10倍
#             nn.BatchNorm2d(16),
#             nn.ELU(),
            
#         )

#         self.projection = nn.Sequential(
#             nn.Conv2d(16, emb_size, (1, 1), stride=(1, 1)),  
#             Rearrange('b e (h) (w) -> b (h w) e'),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.unsqueeze(1)  # 假设输入形状为 [batch, length]
#         x = self.tsconv(x)
#         x = self.projection(x)
#         return x


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # Revised from ShallowNet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (899, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)     
        # print("x", x.shape)   
        x = self.tsconv(x)
        # print("tsconv", x.shape)   
        x = self.projection(x)
        # print("projection", x.shape)  
        return x
        
class Proj_neuro(nn.Sequential):
    def __init__(self, emb_size=40, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead(),
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
        
class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class Config:
    def __init__(self):
        self.task_name = 'classification'  # Example task name
        self.seq_len = 1024                      # Sequence length        
        self.pred_len = 96                     # Prediction length        
        self.output_attention = False          # Whether to output attention weights
        self.d_model = 250                     # Model dimension
        self.embed = 'timeF'                   # Time encoding method
        self.freq = 'h'                        # Time frequency
        self.dropout = 0.25                    # Dropout ratio
        self.factor = 1                        # Attention scaling factor
        self.n_heads = 4                       # Number of attention heads
        self.e_layers = 1                     # Number of encoder layers
        self.d_ff = 256                       # Feedforward network dimension
        self.activation = 'gelu'               # Activation function
        self.enc_in = 54                        # Encoder input dimension (example value)
        
        self.single_channel = False
        self.patch_len_list = "2,4,8"
        self.augmentations = "flip,shuffle,frequency,jitter,mask,drop"
        self.no_inter_attn = False
        self.num_class = 250


import hydra
class UnifiedEncoder(nn.Module):
    def __init__(self, in_dim=1024, h=1024, out_dim=1024, num_latents=128, qformer_spec=cfg.hyperparameter):
        super().__init__()
        self.fmri_subs = [i for i in range(1, 4)]
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))                      
        # self.perceiver = Perceiver(patch_embed_dim=h, hidden_size=h, num_latents=num_latents)
        # self.atm = ATMS(joint_train=True, num_subjects=10)
        default_config = Config()
        # self.encoder = iTransformer(default_config, joint_train)   
        self.encoder = Medformer(default_config)   
        # 初始化 proj_neuro 为 ModuleDict
        self.proj_neuro = nn.ModuleDict()   
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()         
        # 加入 MoE 投影模块
        self.moe_projection = MoEProjection(input_dim=in_dim, output_dim=out_dim, num_experts=1)

        # 引入可学习的权重参数，初始化为1.0
        self.log_alpha = nn.Parameter(torch.zeros(1))  # log_alpha = log(1.0) = 0
        self.log_beta = nn.Parameter(torch.zeros(1))   # log_beta = log(1.0) = 0
        
        # 根据不同模态初始化 fusion_head
        self.fusion_heads = nn.ModuleDict()
        self.modals = ['eeg', 'meg', 'fmri']
        # 假设 qformer_spec 中包含每个模态的 proj_neuro 配置
        for modal in self.modals:
            # 直接获取 proj_neuro 配置，而不是实例化 MLP
            proj_config = getattr(qformer_spec, modal).proj_neuro
            self.proj_neuro[modal] = Proj_neuro(
                emb_size=proj_config.emb_size,
                embedding_dim=proj_config.embedding_dim,
                proj_dim=proj_config.proj_dim,
                drop_proj=proj_config.drop_proj
            )
        for modal in self.modals:
            # 实例化不同模态的 fusion_head
            self.fusion_heads[modal] = hydra.utils.instantiate(getattr(qformer_spec, modal))
            self.fusion_heads[modal].init_cross_attn(qformer_spec, modal)
        clip_width = 1024
        self.conv1 = nn.ModuleDict()
        self.positional_embedding = nn.ParameterDict()
        self.num_voxels = {1: 6036, 2: 5944, 3: 5238}
        for modal in self.modals:
            if modal =='image':
                modal_tokens = 256 + 1
                pass
            
            elif modal == 'eeg':
                modal_tokens = 54
                self.conv1[modal] = nn.Conv1d(
                    in_channels=250, out_channels=clip_width, kernel_size=10, bias=False)
                self.positional_embedding[modal] = nn.Parameter(torch.empty([modal_tokens, clip_width]))
                nn.init.normal_(self.positional_embedding[modal], std=0.02)
                
            elif modal == 'meg':
                modal_tokens = 262
                self.conv1[modal] = nn.Conv1d(
                    in_channels=201, out_channels=clip_width, kernel_size=10, bias=False)
                self.positional_embedding[modal] = nn.Parameter(torch.empty([modal_tokens, clip_width]))
                nn.init.normal_(self.positional_embedding[modal], std=0.02)     
                           
            elif modal == 'fmri':                
                # self.conv1[modal] = nn.ModuleDict({
                #     str(sub): nn.Linear(self.num_voxels.get(sub), 8192) for sub in self.fmri_subs
                # })
                self.conv1[modal] = nn.ModuleDict({
                    str(sub): nn.Linear(7000, 8192) for sub in self.fmri_subs
                })                
                self.positional_embedding[modal] = nn.Parameter(torch.empty([8, clip_width]))
                nn.init.normal_(self.positional_embedding[modal], std=0.02)
                
    def forward(self, x, subject_ids, modal):
        
        if modal in ['eeg', 'meg']:
            x = x.transpose(1, 2)
            
        if modal == 'fmri':
            # print(x.shape)
            # print(subject_ids[0].item())
            x = self.conv1[modal][f'{subject_ids[0].item()}'](x)
        else:
            x = self.conv1[modal](x)
        
        if modal in ['eeg', 'meg', 'fmri']:
            pos_embedding = self.positional_embedding[modal]
            
        x = x.reshape(x.size(0), -1, 1024)
        # x = x.transpose(1, 2)            
        # print("pos_embedding", pos_embedding.shape)
        # print("x", x.shape)
        x = x + pos_embedding.to(x.dtype)
        # [B, 250, 63] -> [B, 1024, 63]
        # x = x.reshape(x.size(0), 1024, -1)
        # x = x.transpose(1, 2)
        # x 的初步处理
        # x = self.perceiver(x)
        # print("x", x.shape)
        # x = self.atm(x, subject_ids, modal)
        x = self.encoder(x)
        # print("after medformer: x", x.shape)
        # x = x.transpose(1, 2)
        
        # 通过 MoE 模块投影
        # x = self.moe_projection(x)  # [batch_size, channels, h]

        # x = x.transpose(1, 2)
            
        x = self.enc_eeg(x)
        
        x = self.proj_eeg(x)        
        
        # 使用对应模态的 proj_neuro
        # x = self.proj_neuro[modal](x)
        # print("x", x.shape)
        # 通过 fusion_head 的 compute_latent
        # 通过指定模态的 fusion_head 进行 compute_latent
        # x = self.fusion_heads[modal].compute_latent(x)
        # x = x.squeeze(1)
        # print("x", x.shape)
        
        return x

