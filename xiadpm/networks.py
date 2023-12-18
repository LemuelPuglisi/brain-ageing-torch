import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


def he_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)


class DownBlock(nn.Module):
    """
    Conv3D -> BN -> ReLU -> Conv3D -> BN -> ReLU -> max-pooling
    """
    
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        conv_x = self.block(x)
        maxp_x = nn.functional.max_pool3d(conv_x, kernel_size=2)
        return conv_x, maxp_x
    

class UpBlock(nn.Module):
    """
    TConv3D -> BN -> concat skip-conn -> Conv3D -> BN -> ReLU -> Conv3D -> BN -> ReLU  
    """

    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        
        self.tconv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(num_features=in_channels),
        )
        
        self.dconv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels*2, out_channels=out_channels, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x, skip_connection):
        x = self.tconv(x)
        x = torch.concat([x, skip_connection], axis=1)
        x = self.dconv(x)
        return x
    


class MidBlock(nn.Module):
    """
    Add covariates effect to image.
    """
    
    def __init__(self, 
            input_shape, 
            latent_dim,
            age_dim, 
            diagnosis_dim,
            feature_maps=32, 
            down_levels=3
        ):
        super(MidBlock, self).__init__()
        
        k = 2 ** down_levels            # reduction factor
        f = 2 ** (down_levels - 1)      # feature maps multiplier

        # flattened dimension of the input
        projector1_input = np.prod([ s // k for s in input_shape[-3:] ]) * feature_maps

        self.projector1 = nn.Sequential(
            nn.Conv3d(in_channels=feature_maps * f, out_channels=feature_maps, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(), 
            nn.Flatten(start_dim=1), 
            nn.Linear(in_features=projector1_input, out_features=latent_dim), 
            nn.Sigmoid(),
        )

        # input will be the latent dimension plus the covariates dimensions
        projector2_input  = latent_dim + age_dim + diagnosis_dim
        h, w, d = [ s // k for s in input_shape[-3:] ]

        self.projector2 = nn.Sequential(
            nn.Linear(in_features=projector2_input, out_features=projector1_input),
            Rearrange("b (c h w d) -> b c h w d", h=h, w=w, d=d)
        )

        # back to `feature_maps * 2^down_levels` features
        self.assembler = nn.Conv3d(
                in_channels=5*feature_maps, 
                out_channels=4*feature_maps, 
                kernel_size=3, 
                stride=1, 
                padding=1
        )
        
        
    def forward(self, x, a, h):
        m = self.projector1(x)
        m = torch.concat([m, a, h], axis=1)
        m = self.projector2(m)
        x = torch.concat([x, m], axis=1)
        x = self.assembler(x)
        return x
    

class Generator(nn.Module):
    """
    Generator from (Xia et Al, 2019)
    """

    def __init__(self, 
            input_shape, 
            latent_dim,
            age_dim, 
            diagnosis_dim,
            feature_maps=32
        ):
        super(Generator, self).__init__()

        self.down_blocks = nn.ModuleList([
            DownBlock(in_channels=1, out_channels=feature_maps), 
            DownBlock(in_channels=feature_maps, out_channels=feature_maps*2),
            DownBlock(in_channels=feature_maps*2, out_channels=feature_maps*4)
        ])

        self.mid_block = MidBlock(
            input_shape=input_shape,
            latent_dim=latent_dim,
            age_dim=age_dim, 
            diagnosis_dim=diagnosis_dim,
            feature_maps=feature_maps,
        )
        
        self.up_blocks = nn.ModuleList([
            UpBlock(in_channels=4*feature_maps, out_channels=2*feature_maps), 
            UpBlock(in_channels=2*feature_maps, out_channels=feature_maps),
            UpBlock(in_channels=feature_maps, out_channels=1)
        ])
        
        # He initialization
        he_init(self)


    def forward(self, x, a, h):
        down_outputs =  []
        for down_block in self.down_blocks:
            conv_x, x = down_block(x)
            down_outputs.append(conv_x)
        
        x = self.mid_block(x, a, h)
        
        down_outputs.reverse()
        for up_block, skip_connection in zip(self.up_blocks, down_outputs):
            x = up_block(x, skip_connection)
        return x
    


class Critic(nn.Module):
    """
    Critic from (Xia et Al, 2019)
    """
    
    def __init__(self, 
            input_shape, 
            latent_dim,
            age_dim, 
            diagnosis_dim,
            feature_maps=32
        ):
        super(Critic, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=feature_maps, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2), 
            nn.Conv3d(in_channels=feature_maps, out_channels=feature_maps * 2, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2), 
            nn.Conv3d(in_channels=feature_maps * 2, out_channels=feature_maps * 4, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2), 
        )

        # flattened dimension of the input
        projector1_input = np.prod([ s // 8 for s in input_shape[-3:] ]) * feature_maps
        
        self.projector_1 = nn.Sequential(
            nn.Conv3d(in_channels=feature_maps * 4, out_channels=feature_maps, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(), 
            nn.Flatten(start_dim=1), 
            nn.Linear(in_features=projector1_input, out_features=latent_dim), 
            nn.Sigmoid(),
        )
        
        projector2_input  = latent_dim + age_dim + diagnosis_dim
        h, w, d = [ s // 8 for s in input_shape[-3:] ]
        self.projector_2 = nn.Sequential(
            nn.Linear(in_features=projector2_input, out_features=projector1_input),
            Rearrange("b (c h w d) -> b c h w d", h=h, w=w, d=d)
        )
        
        self.downstream = nn.Sequential(
            nn.Conv3d(in_channels=feature_maps*5, out_channels=feature_maps*8, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(), 
            nn.Conv3d(in_channels=feature_maps*8, out_channels=feature_maps*8, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(), 
            nn.Conv3d(in_channels=feature_maps*8, out_channels=feature_maps*8, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(), 
            nn.Conv3d(in_channels=feature_maps*8, out_channels=1, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(), 
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        # He initialization
        he_init(self)

        
    def forward(self, x, a, h):
        x = self.encoder(x)
        m = self.projector_1(x)
        m = torch.concat([ m, a, h ], axis=1)
        m = self.projector_2(m)
        x = torch.concat([x, m], axis=1)
        x = self.downstream(x).view(-1, 1)
        return x