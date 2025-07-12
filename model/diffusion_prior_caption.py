import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from torch.utils.data import Dataset
# for prior
from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.dalle2_pytorch import l2norm, default, exists
from dalle2_pytorch.train_configs import DiffusionPriorNetworkConfig
# vd prior
from dalle2_pytorch.dalle2_pytorch import RotaryEmbedding, CausalTransformer, SinusoidalPosEmb, MLP, Rearrange, repeat, rearrange, prob_mask_like, LayerNorm, RelPosBias, Attention, FeedForward


class PriorNetwork(nn.Module):
    def __init__(
        self,
        dim,
        num_timesteps = None,
        num_time_embeds = 1,
        # num_image_embeds = 1,
        # num_brain_embeds = 1,
        num_tokens = 257,
        causal = True,
        learned_query_mode = 'none',
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_time_embeds = num_time_embeds
        self.continuous_embedded_time = not exists(num_timesteps)
        self.learned_query_mode = learned_query_mode

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        if self.learned_query_mode == 'token':
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim))
        if self.learned_query_mode == 'pos_emb':
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim) * scale)
        if self.learned_query_mode == 'all_pos_emb':
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens*2+1, dim) * scale)
        self.causal_transformer = FlaggedCausalTransformer(dim = dim, causal=causal, **kwargs)

        self.null_brain_embeds = nn.Parameter(torch.randn(num_tokens, dim))
        self.null_image_embed = nn.Parameter(torch.randn(num_tokens, dim))

        self.num_tokens = num_tokens
        self.self_cond = False

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, brain_cond_drop_prob = 1., image_cond_drop_prob = 1, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        image_embed,
        diffusion_timesteps,
        *,
        self_cond=None,
        brain_embed=None,
        text_embed=None,
        brain_cond_drop_prob = 0.,
        text_cond_drop_prob = None,
        image_cond_drop_prob = 0.
    ):
        if text_embed is not None:
            brain_embed = text_embed
        if text_cond_drop_prob is not None:
            brain_cond_drop_prob = text_cond_drop_prob
        
        # image_embed = image_embed.view(len(image_embed),-1,16*16)
        # text_embed = text_embed.view(len(text_embed),-1,768)
        # brain_embed = brain_embed.view(len(brain_embed),-1,16*16)
        # print(*image_embed.shape)
        # print(*image_embed.shape, image_embed.device, image_embed.dtype)
        
        batch, _, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype
        # num_time_embeds, num_image_embeds, num_brain_embeds = self.num_time_embeds, self.num_image_embeds, self.num_brain_embeds
        
        # classifier free guidance masks
        brain_keep_mask = prob_mask_like((batch,), 1 - brain_cond_drop_prob, device = device)
        brain_keep_mask = rearrange(brain_keep_mask, 'b -> b 1 1')

        image_keep_mask = prob_mask_like((batch,), 1 - image_cond_drop_prob, device = device)
        image_keep_mask = rearrange(image_keep_mask, 'b -> b 1 1')

        # mask out brain embeddings with null brain embeddings

        # import pdb; pdb.set_trace()
        null_brain_embeds = self.null_brain_embeds.to(brain_embed.dtype)
        brain_embed = torch.where(
            brain_keep_mask,
            brain_embed,
            null_brain_embeds[None]
        )

        # mask out image embeddings with null image embeddings
        null_image_embed = self.null_image_embed.to(image_embed.dtype)
        image_embed = torch.where(
            image_keep_mask,
            image_embed,
            null_image_embed[None]
        )

        # whether brain embedding is used for conditioning depends on whether brain encodings are available for attention (for classifier free guidance, even though it seems from the paper it was not used in the prior ddpm, as the objective is different)
        # but let's just do it right
        if self.continuous_embedded_time:
            # if continuous cast to flat, else keep int for indexing embeddings
            diffusion_timesteps = diffusion_timesteps.type(dtype)
        time_embed = self.to_time_embeds(diffusion_timesteps)

        if self.learned_query_mode == 'token':
            learned_queries = repeat(self.learned_query, 'n d -> b n d', b = batch)
        elif self.learned_query_mode == 'pos_emb':
            pos_embs = repeat(self.learned_query, 'n d -> b n d', b = batch)
            image_embed = image_embed + pos_embs
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        elif self.learned_query_mode == 'all_pos_emb':
            pos_embs = repeat(self.learned_query, 'n d -> b n d', b = batch)
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        else:
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        
        tokens = torch.cat((
            brain_embed,  # 257
            time_embed,  # 1
            image_embed,  # 257
            learned_queries  # 257
        ), dim = -2)
        if self.learned_query_mode == 'all_pos_emb':
            tokens = tokens + pos_embs

        # attend
        tokens = self.causal_transformer(tokens)

        # get learned query, which should predict the image embedding (per DDPM timestep)
        pred_image_embed = tokens[..., -self.num_tokens:, :]

        return pred_image_embed

class FlaggedCausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_in = False,
        norm_out = True,
        attn_dropout = 0.,
        ff_dropout = 0.,
        final_proj = True,
        normformer = False,
        rotary_emb = True,
        causal=True
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = causal, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))

        self.norm = LayerNorm(dim, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim, dim, bias = False) if final_proj else nn.Identity()

    def forward(self, x):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        for attn, ff in self.layers:
            x = attn(x, attn_bias = attn_bias) + x
            x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)


class BrainDiffusionPrior(DiffusionPrior):
    """ 
    Differences from original:
    - Allow for passing of generators to torch random functions
    - Option to include the voxel2clip model and pass voxels into forward method
    - Return predictions when computing loss
    - Load pretrained model from @nousr trained on LAION aesthetics
    """
    def __init__(self, *args, **kwargs):
        voxel2clip = kwargs.pop('voxel2clip', None)
        super().__init__(*args, **kwargs)
        self.voxel2clip = voxel2clip

    @torch.no_grad()
    def p_sample(self, x, t, text_cond = None, self_cond = None, clip_denoised = True, cond_scale = 1.,
                generator=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = t, text_cond = text_cond, self_cond = self_cond, clip_denoised = clip_denoised, cond_scale = cond_scale)
        if generator is None:
            noise = torch.randn_like(x)
        else:
            noise = torch.randn_like(x)
            # noise = torch.randn(x.size(), device=x.device, dtype=x.dtype, generator=generator)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(self, *args, timesteps = None, **kwargs):
        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps
        is_ddim = timesteps < self.noise_scheduler.num_timesteps

        if not is_ddim:
            normalized_image_embed = self.p_sample_loop_ddpm(*args, **kwargs)
        else:
            normalized_image_embed = self.p_sample_loop_ddim(*args, **kwargs, timesteps = timesteps)

        # print("PS removed all image_embed_scale instances!")
        image_embed = normalized_image_embed #/ self.image_embed_scale
        return image_embed

    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape, text_cond, cond_scale = 1., generator=None):
        batch, device = shape[0], self.device

        if generator is None:
            image_embed = torch.randn(shape, device = device)
        else:
            image_embed = torch.randn(shape, device = device, generator=generator)
        x_start = None # for self-conditioning

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step', total=self.noise_scheduler.num_timesteps, disable=True):
            times = torch.full((batch,), i, device = device, dtype = torch.long)

            self_cond = x_start if self.net.self_cond else None
            image_embed, x_start = self.p_sample(image_embed, times, text_cond = text_cond, self_cond = self_cond, cond_scale = cond_scale, 
                                                 generator=generator)

        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    def p_losses(self, image_embed, times, text_cond, noise = None):
        noise = default(noise, lambda: torch.randn_like(image_embed))

        image_embed_noisy = self.noise_scheduler.q_sample(x_start = image_embed, t = times, noise = noise)

        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(image_embed_noisy, times, **text_cond).detach()

        pred = self.net(
            image_embed_noisy,
            times,
            self_cond = self_cond,
            text_cond_drop_prob = self.text_cond_drop_prob,
            image_cond_drop_prob = self.image_cond_drop_prob,
            **text_cond
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise

        loss = nn.functional.mse_loss(pred, target) # mse
        # print("1", loss)
        # loss += (1 - nn.functional.cosine_similarity(pred, target).mean())
        # print("2", (1 - nn.functional.cosine_similarity(pred, target).mean()))
        return loss, pred

    def forward(
        self,
        text = None,
        image = None,
        voxel = None,
        text_embed = None,      # allow for training on preprocessed CLIP text and image embeddings
        image_embed = None,
        text_encodings = None,  # as well as CLIP text encodings
        *args,
        **kwargs
    ):
        assert exists(text) ^ exists(text_embed) ^ exists(voxel), 'either text, text embedding, or voxel must be supplied'
        assert exists(image) ^ exists(image_embed), 'either image or image embedding must be supplied'
        assert not (self.condition_on_text_encodings and (not exists(text_encodings) and not exists(text))), 'text encodings must be present if you specified you wish to condition on it on initialization'

        if exists(voxel):
            assert exists(self.voxel2clip), 'voxel2clip must be trained if you wish to pass in voxels'
            assert not exists(text_embed), 'cannot pass in both text and voxels'
            if self.voxel2clip.use_projector:
                clip_voxels_mse, clip_voxels = self.voxel2clip(voxel)
                text_embed = clip_voxels_mse
            else:
                clip_voxels = self.voxel2clip(voxel)
                text_embed = clip_voxels_mse = clip_voxels
            # text_embed = self.voxel2clip(voxel)

        if exists(image):
            image_embed, _ = self.clip.embed_image(image)

        # calculate text conditionings, based on what is passed in

        if exists(text):
            text_embed, text_encodings = self.clip.embed_text(text)

        text_cond = dict(text_embed = text_embed)

        if self.condition_on_text_encodings:
            assert exists(text_encodings), 'text encodings must be present for diffusion prior if specified'
            text_cond = {**text_cond, 'text_encodings': text_encodings}

        # timestep conditioning from ddpm

        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch)

        # PS: I dont think we need this? also if uncommented this does in-place global variable change
        # scale image embed (Katherine)
        # image_embed *= self.image_embed_scale

        # calculate forward loss

        loss, pred = self.p_losses(image_embed, times, text_cond = text_cond, *args, **kwargs)
        
        # undo the scaling so we can directly use it for real mse loss and reconstruction
        return loss, pred



class DiffusionPrior(nn.Module):

    def __init__(
            self, 
            embed_dim=1024, 
            cond_dim=42,
            hidden_dim=1024,
            layers_per_block=4, 
            time_embed_dim=512,
            act_fn=nn.SiLU,
            dropout=0.0,
        ):
        super().__init__()
        
        self.embed_dim = embed_dim

        # 1. time embedding
        self.time_proj = Timesteps(time_embed_dim, True, 0)
        self.time_embedding = TimestepEmbedding(
            time_embed_dim,
            hidden_dim,
        )

        # 2. conditional embedding 
        self.cond_embedding = nn.Linear(cond_dim, hidden_dim)

        # 3. prior mlp

        # 3.1 input
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn(),
        )

        # 3.2 hidden
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    act_fn(),
                    nn.Dropout(dropout),
                )
                for _ in range(layers_per_block)
            ]
        )

        # 3.3 output
        self.output_layer = nn.Linear(hidden_dim, embed_dim)
        

    def forward(self, x, t, c=None):
        # x (batch_size, embed_dim)
        # t (batch_size, )
        # c (batch_size, cond_dim)

        # 1. time embedding
        t = self.time_proj(t) # (batch_size, time_embed_dim)
        t = self.time_embedding(t) # (batch_size, hidden_dim)

        # 2. conditional embedding 
        c = self.cond_embedding(c) if c is not None else 0 # (batch_size, hidden_dim)

        # 3. prior mlp

        # 3.1 input
        x = self.input_layer(x) 

        # 3.2 hidden
        for layer in self.hidden_layers:
            x = x + t + c
            x = layer(x) + x
            
        # 3.3 output
        x = self.output_layer(x)

        return x


class DiffusionPriorUNet(nn.Module):

    def __init__(
            self, 
            embed_dim=1024, 
            cond_dim=42,
            hidden_dim=[1024, 512, 256, 128, 64],
            time_embed_dim=512,
            act_fn=nn.SiLU,
            dropout=0.0,
        ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        # 1. time embedding
        self.time_proj = Timesteps(time_embed_dim, True, 0)

        # 2. conditional embedding 
        # to 3.2, 3,3

        # 3. prior mlp

        # 3.1 input
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim[0]),
            nn.LayerNorm(hidden_dim[0]),
            act_fn(),
        )

        # 3.2 hidden encoder
        self.num_layers = len(hidden_dim)
        self.encode_time_embedding = nn.ModuleList(
            [TimestepEmbedding(
                time_embed_dim,
                hidden_dim[i],
            ) for i in range(self.num_layers-1)]
        ) # d_0, ..., d_{n-1}
        self.encode_cond_embedding = nn.ModuleList(
            [nn.Linear(cond_dim, hidden_dim[i]) for i in range(self.num_layers-1)]
        )
        self.encode_layers = nn.ModuleList(
            [nn.Sequential(
                    nn.Linear(hidden_dim[i], hidden_dim[i+1]),
                    nn.LayerNorm(hidden_dim[i+1]),
                    act_fn(),
                    nn.Dropout(dropout),
                ) for i in range(self.num_layers-1)]
        )

        # 3.3 hidden decoder
        self.decode_time_embedding = nn.ModuleList(
            [TimestepEmbedding(
                time_embed_dim,
                hidden_dim[i],
            ) for i in range(self.num_layers-1,0,-1)]
        ) # d_{n}, ..., d_1
        self.decode_cond_embedding = nn.ModuleList(
            [nn.Linear(cond_dim, hidden_dim[i]) for i in range(self.num_layers-1,0,-1)]
        )
        self.decode_layers = nn.ModuleList(
            [nn.Sequential(
                    nn.Linear(hidden_dim[i], hidden_dim[i-1]),
                    nn.LayerNorm(hidden_dim[i-1]),
                    act_fn(),
                    nn.Dropout(dropout),
                ) for i in range(self.num_layers-1,0,-1)]
        )

        # 3.4 output
        self.output_layer = nn.Linear(hidden_dim[0], embed_dim)
        

    def forward(self, x, t, c=None):
        # x (batch_size, embed_dim)
        # t (batch_size, )
        # c (batch_size, cond_dim)

        # 1. time embedding
        t = self.time_proj(t) # (batch_size, time_embed_dim)

        # 2. conditional embedding 
        # to 3.2, 3.3

        # 3. prior mlp

        # 3.1 input
        x = self.input_layer(x) 

        # 3.2 hidden encoder
        hidden_activations = []
        for i in range(self.num_layers-1):
            hidden_activations.append(x)
            t_emb = self.encode_time_embedding[i](t) 
            c_emb = self.encode_cond_embedding[i](c) if c is not None else 0
            x = x + t_emb + c_emb
            x = self.encode_layers[i](x)
        
        # 3.3 hidden decoder
        for i in range(self.num_layers-1):
            t_emb = self.decode_time_embedding[i](t)
            c_emb = self.decode_cond_embedding[i](c) if c is not None else 0
            x = x + t_emb + c_emb
            x = self.decode_layers[i](x)
            x += hidden_activations[-1-i]
            
        # 3.4 output
        x = self.output_layer(x)

        return x


class EmbeddingDataset(Dataset):

    def __init__(self, c_embeddings, h_embeddings):
        self.c_embeddings = c_embeddings
        self.h_embeddings = h_embeddings

    def __len__(self):
        return len(self.c_embeddings)

    def __getitem__(self, idx):
        return {
            "c_embedding": self.c_embeddings[idx],
            "h_embedding": self.h_embeddings[idx]
        }

class EmbeddingDatasetVICE(Dataset):
    def __init__(self, path_data):
        image_features_dict = torch.load(os.path.join(path_data, 'openclip_emb/image_features.pt'))
        self.embedding_vise = torch.load(os.path.join(path_data, 'variables/embedding_vise.pt'))
        self.image_features = image_features_dict['image_features']
        self.labels = image_features_dict['labels']
        self.label2index = image_features_dict['l2i']

    def __len__(self):
        return len(self.image_features)

    def __getitem__(self, idx):
        idx_c = self.label2index[self.labels[idx]]
        return {
            "c_embedding": self.embedding_vise[idx_c],
            "h_embedding": self.image_features[idx]
        }
    

# Copied from diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler.add_noise
def add_noise_with_sigma(
    self,
    original_samples: torch.FloatTensor,
    noise: torch.FloatTensor,
    timesteps: torch.FloatTensor,
) -> torch.FloatTensor:
    # Make sure sigmas and timesteps have the same device and dtype as original_samples
    sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
    if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
        # mps does not support float64
        schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
        timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
    else:
        schedule_timesteps = self.timesteps.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)

    step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < len(original_samples.shape):
        sigma = sigma.unsqueeze(-1)

    noisy_samples = original_samples + noise * sigma
    return noisy_samples, sigma


# diffusion pipe
class Pipe:
    
    def __init__(self, diffusion_prior=None, scheduler=None, device='cuda'):
        self.diffusion_prior = diffusion_prior.to(device)
        
        if scheduler is None:
            from diffusers.schedulers import DDPMScheduler
            self.scheduler = DDPMScheduler() 
            # self.scheduler.add_noise_with_sigma = add_noise_with_sigma.__get__(self.scheduler)
        else:
            self.scheduler = scheduler
            
        self.device = device
        
    def train(self, dataloader, num_epochs=10, learning_rate=1e-4):
        self.diffusion_prior.train()
        device = self.device
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(self.diffusion_prior.parameters(), lr=learning_rate)
        from diffusers.optimization import get_cosine_schedule_with_warmup
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=(len(dataloader) * num_epochs),
        )

        num_train_timesteps = self.scheduler.config.num_train_timesteps

        for epoch in range(num_epochs):
            loss_sum = 0
            for batch in dataloader:
                c_embeds = batch['c_embedding'].to(device) if 'c_embedding' in batch.keys() else None
                h_embeds = batch['h_embedding'].to(device)
                N = h_embeds.shape[0]

                # 1. randomly replecing c_embeds to None
                if torch.rand(1) < 0.1:
                    c_embeds = None

                # 2. Generate noisy embeddings as input
                noise = torch.randn_like(h_embeds)

                # 3. sample timestep
                timesteps = torch.randint(0, num_train_timesteps, (N,), device=device)

                # 4. add noise to h_embedding
                perturbed_h_embeds = self.scheduler.add_noise(
                    h_embeds,
                    noise,
                    timesteps
                ) # (batch_size, embed_dim), (batch_size, )

                # 5. predict noise
                noise_pre = self.diffusion_prior(perturbed_h_embeds, timesteps, c_embeds)
                
                # 6. loss function weighted by sigma
                loss = criterion(noise_pre, noise) # (batch_size,)
                loss = (loss).mean()
                            
                # 7. update parameters
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion_prior.parameters(), 1.0)
                lr_scheduler.step()
                optimizer.step()

                loss_sum += loss.item()

            loss_epoch = loss_sum / len(dataloader)
            print(f'epoch: {epoch}, loss: {loss_epoch}')
            # lr_scheduler.step(loss)

    def generate(
            self, 
            c_embeds=None, 
            num_inference_steps=50, 
            timesteps=None,
            guidance_scale=5.0,
            generator=None
        ):
        # c_embeds (batch_size, cond_dim)
        self.diffusion_prior.eval()
        N = c_embeds.shape[0] if c_embeds is not None else 1

        # 1. Prepare timesteps
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, self.device, timesteps)

        # 2. Prepare c_embeds
        if c_embeds is not None:
            c_embeds = c_embeds.to(self.device)

        # 3. Prepare noise
        h_t = torch.randn(N, self.diffusion_prior.embed_dim, generator=generator, device=self.device)

        # 4. denoising loop
        for _, t in tqdm(enumerate(timesteps)):
            t = torch.ones(h_t.shape[0], dtype=torch.float, device=self.device) * t
            # 4.1 noise prediction
            if guidance_scale == 0 or c_embeds is None:
                noise_pred = self.diffusion_prior(h_t, t)
            else:
                noise_pred_cond = self.diffusion_prior(h_t, t, c_embeds)
                noise_pred_uncond = self.diffusion_prior(h_t, t)
                # perform classifier-free guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # 4.2 compute the previous noisy sample h_t -> h_{t-1}
            h_t = self.scheduler.step(noise_pred, t.long().item(), h_t, generator=generator).prev_sample
        
        return h_t

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 1. test prior
    prior = DiffusionPriorUNet(cond_dim=1024)
    x = torch.randn(2, 1024)
    t = torch.randint(0, 1000, (2,))
    c = torch.randn(2, 1024)
    y = prior(x, t, c)
    print(y.shape)



