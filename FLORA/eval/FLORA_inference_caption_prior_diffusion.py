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

sys.path.insert(0, "/mnt/dataset1/ldy/Workspace/FLORA")    

# 现在可以使用绝对导入
from model.unified_encoder_multi_tower import UnifiedEncoder

# 导入必要的库
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import confusion_matrix

from loss import ClipLoss
from model.diffusion_prior import Pipe, EmbeddingDataset, DiffusionPriorUNet
from model.custom_pipeline import Generator4Embeds
# 忽略警告
warnings.filterwarnings("ignore")

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# proxy = 'http://10.20.37.38:7890'
# os.environ['http_proxy'] = proxy
# os.environ['https_proxy'] = proxy
device = 'cuda:4'

eeg_features_test = torch.load('/mnt/dataset1/ldy/Workspace/FLORA/eval/fMRI_features_sub_01_test.pt')



diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
high_pipe = Pipe(diffusion_prior, device=device)
# high_pipe.diffusion_prior.load_state_dict(torch.load("/mnt/dataset1/ldy/Workspace/FLORA/models/contrast/across/Unified_EEG+MEG+fMRI_EEG/01-29_14-48/prior_diffusion/100.pth"))
high_pipe.diffusion_prior.load_state_dict(torch.load("/mnt/dataset0/ldy/models/contrast/across/Unified_EEG+MEG+fMRI_EEG/01-22_18-16/prior_diffusion/60.pth"))

high_pipe.diffusion_prior.to(device)
high_pipe.diffusion_prior.eval()  # Set model to evaluation mode

from model.diffusion_prior_caption import Pipe, EmbeddingDataset, DiffusionPriorUNet, PriorNetwork, BrainDiffusionPrior
# setup diffusion prior network
clip_emb_dim = 1024
clip_seq_dim = 256
depth = 1
dim_head = 4
heads = clip_emb_dim//4 # heads * dim_head = clip_emb_dim
timesteps = 100
out_dim = clip_emb_dim

prior_network = PriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens = clip_seq_dim,
        learned_query_mode="pos_emb"
    )

high_pipe = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
)
high_pipe.to(device)
high_pipe.eval()

eeg_features_test = eeg_features_test.to(device)
prior_out = high_pipe.p_sample_loop(eeg_features_test.shape, 
                text_cond = dict(text_embed = eeg_features_test), 
                cond_scale = 1., timesteps = 20)

prior_out.shape
torch.save(prior_out, 'fMRI_prior_features_test.pt')

