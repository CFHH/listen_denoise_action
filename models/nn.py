# Copyright 2023 Motorica AB, Inc. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.tisa_transformer import TisaTransformer
from math import sqrt

class Conv1dLayer(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1):
    super().__init__()
    self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
    nn.init.kaiming_normal_(self.conv1d.weight)

  def forward(self, x):
    # (batch_size,150,C)转成(batch_size,C,150)卷积成(batch_size,256,150)再转回(batch_size,150,256)，in_channels=C，out_channels=256
    return self.conv1d(x.permute(0,2,1)).permute(0,2,1)

def silu(x):
  return x * torch.sigmoid(x)

class DiffusionEmbedding(nn.Module):
  def __init__(self, max_steps, in_channels, hidden_channels):
    super().__init__()
    
    self.in_channels = in_channels  # 代码写死128
    self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False) # self.embedding是(150,128)
    self.projection1 = nn.Linear(in_channels, hidden_channels)  # hidden_channels配置是512
    self.projection2 = nn.Linear(hidden_channels, hidden_channels)

  def forward(self, diffusion_step):
    if diffusion_step.dtype in [torch.int32, torch.int64]:
      x = self.embedding[diffusion_step]  # (batch_size, 128)
    else:
      x = self._lerp_embedding(diffusion_step)
    x = self.projection1(x) # (batch_size, 512)
    x = silu(x)
    x = self.projection2(x) # (batch_size, 512)
    x = silu(x)
    return x

  def _lerp_embedding(self, t):
    low_idx = torch.floor(t).long()
    high_idx = torch.ceil(t).long()
    low = self.embedding[low_idx]
    high = self.embedding[high_idx]
    return low + (high - low) * (t - low_idx)

  def _build_embedding(self, max_steps):
    steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
    dims = torch.arange(64).unsqueeze(0)          # [1,64]
    table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    return table
    
class ResidualBlock(nn.Module):
  def __init__(self, residual_channels, 
            embedding_dim, 
            l_cond_dim,
            nn_name,
            nn_args,
            index):

    super().__init__()
    if nn_name=="tisa":
        dilation_cycle = nn_args["dilation_cycle"]  # [0,1,2]
        dilation=dilation_cycle[(index % len(dilation_cycle))]
        self.nn = TisaTransformer(residual_channels, 2 * residual_channels, d_model=residual_channels, num_blocks=nn_args["num_blocks"], num_heads=nn_args["num_heads"], activation=nn_args["activation"], norm=nn_args["norm"], drop_prob=nn_args["dropout"], d_ff=nn_args["d_ff"], seqlen=nn_args["seq_len"], use_preln=nn_args["use_preln"], bias=nn_args["bias"], dilation=dilation)
    elif nn_name=="conv":
        dilation=2**(index % nn_args["dilation_cycle_length"])
        self.nn = Conv1dLayer(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
    else:
        raise ValueError(f"Unknown nn_name: {nn_name}")
        
    self.l_cond_dim = l_cond_dim
    
    self.diffusion_projection = nn.Linear(embedding_dim, residual_channels)
    self.local_cond_projection = nn.Linear(l_cond_dim, residual_channels)
    self.output_projection = Conv1dLayer(residual_channels, 2 * residual_channels, 1)
    self.residual_channels = residual_channels

  def forward(self, x, diffusion_step, local_cond):
    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(1)  # (20,512)->(20,256)->(20,1,256)
    y = x + diffusion_step

    if self.l_cond_dim > 0:
        y += self.local_cond_projection(local_cond)  # (20,150,5)->(20,150,256)
    y = self.nn(y).squeeze(-1)  # (20,150,512)->(20,150,512)

    gate, filter = torch.chunk(y, 2, dim=2)  # (20,150,256), (20,150,256)
    y = torch.sigmoid(gate) * torch.tanh(filter)  # (20,150,256)

    y = self.output_projection(y) # (20,150,512)
    residual, skip = torch.chunk(y, 2, dim=2)
    return (x + residual) / sqrt(2.0), skip


class LDA(nn.Module):
  def __init__(self,
                pose_dim,
                residual_layers,    # 20
                residual_channels,  # 256
                embedding_dim,      # 512
                l_cond_dim,
                g_cond_dim,
                n_noise_schedule,   # 150
                nn_name,            # 'tisa'
                nn_args):
    super().__init__()
    self.input_projection = Conv1dLayer(pose_dim, residual_channels, 1)
    self.diffusion_embedding = DiffusionEmbedding(n_noise_schedule, 128, embedding_dim)

    self.residual_layers = nn.ModuleList([
        ResidualBlock(residual_channels,
            embedding_dim,
            l_cond_dim + g_cond_dim,
            nn_name,
            nn_args,
            i)
        for i in range(residual_layers)
    ])
    self.skip_projection = Conv1dLayer(residual_channels, residual_channels, 1)
    self.output_projection = Conv1dLayer(residual_channels, pose_dim, 1)
    nn.init.zeros_(self.output_projection.conv1d.weight)
    self.l_cond_dim = l_cond_dim
    self.g_cond_dim = g_cond_dim

  def forward(self, x, local_cond, global_cond, diffusion_step):
    # x就是noisy_poses(batch_size, 150, C)，diffusion_step是t(batch_size,), local_cond是音乐数据(batch_size, 150, 3)，global_cond是风格数据(batch_size, 150, 风格数量)
    x = self.input_projection(x)  # (batch_size,150,256)
    x = F.relu(x)  # 激活函数，大于0不变，小于0变0
    
    diffusion_step = self.diffusion_embedding(diffusion_step) # (batch_size=20, 512)
    if self.g_cond_dim > 0:
        local_cond=torch.cat((local_cond, global_cond), dim=2)

    skip = None
    i=1
    for layer in self.residual_layers:
      x, skip_connection = layer(x, diffusion_step, local_cond)
      skip = skip_connection if skip is None else skip_connection + skip
        
    if skip is not None:  # (20,150,256)
        x = skip / sqrt(len(self.residual_layers))  # (20,150,256)
    x = self.skip_projection(x)  # (20,150,256)
    x = F.relu(x)
    x = self.output_projection(x)  # (20,150,动作数据个数)
    return x
