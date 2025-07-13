import sys
import os
import warnings
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_

# Import custom modules
from model.unified_encoder_multi_tower import UnifiedEncoder
from model.diffusion_prior import Pipe, EmbeddingDataset, DiffusionPriorUNet
from model.diffusion_prior_caption import Pipe as CaptionPipe, PriorNetwork, BrainDiffusionPrior
from model.custom_pipeline import Generator4Embeds
from loss import ClipLoss

# Set environment variables
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
os.environ["WANDB_SILENT"] = "true"

# Set up paths
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, "/mnt/dataset1/ldy/Workspace/FLORA")

# Add project root to Python path if not already present
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Device configuration
device = 'cuda:4'
warnings.filterwarnings("ignore")

# Load test EEG features
eeg_features_test = torch.load('/mnt/dataset1/ldy/Workspace/FLORA/eval/fMRI_features_sub_01_test.pt')

# Initialize and load diffusion prior model (version 1)
diffusion_prior_v1 = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
high_pipe_v1 = Pipe(diffusion_prior_v1, device=device)
high_pipe_v1.diffusion_prior.load_state_dict(
    torch.load("/mnt/dataset0/ldy/models/contrast/across/Unified_EEG+MEG+fMRI_EEG/01-22_18-16/prior_diffusion/60.pth")
)
high_pipe_v1.diffusion_prior.to(device)
high_pipe_v1.diffusion_prior.eval()

# Initialize diffusion prior model (version 2 - caption version)
clip_emb_dim = 1024
clip_seq_dim = 256
depth = 1
dim_head = 4
heads = clip_emb_dim // 4  # heads * dim_head = clip_emb_dim
timesteps = 100
out_dim = clip_emb_dim

prior_network = PriorNetwork(
    dim=out_dim,
    depth=depth,
    dim_head=dim_head,
    heads=heads,
    causal=False,
    num_tokens=clip_seq_dim,
    learned_query_mode="pos_emb"
)

high_pipe_v2 = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
)
high_pipe_v2.to(device)
high_pipe_v2.eval()

# Generate prior features from EEG test features
eeg_features_test = eeg_features_test.to(device)
prior_out = high_pipe_v2.p_sample_loop(
    eeg_features_test.shape,
    text_cond=dict(text_embed=eeg_features_test),
    cond_scale=1.,
    timesteps=20
)

# Save the generated prior features
print(f"Generated prior features shape: {prior_out.shape}")
torch.save(prior_out, 'fMRI_prior_features_test.pt')