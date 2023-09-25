# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 14:15:06 2022

@author: hafsa
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:18:01 2022

@author: hafsa
"""
from typing import Optional, Tuple
import numpy as np
import torch

import matplotlib.pyplot as plt



class Permute(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,  # extra comma

    def forward(self, x):
        return torch.permute(x, *self.shape)


class ConvBlock(torch.nn.Module):
    """1D Convolutional block.

    Args:
        io_channels (int): The number of input/output channels, <B, Sc>
        hidden_channels (int): The number of channels in the internal layers, <H>.
        kernel_size (int): The convolution kernel size of the middle layer, <P>.
        padding (int): Padding value of the convolution in the middle layer.
        dilation (int, optional): Dilation value of the convolution in the middle layer.
        no_redisual (bool, optional): Disable residual block/output.

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.
    """

    def __init__(
        self,
        io_channels: int,
        hidden_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int = 1,
        no_residual: bool = False,
    ):
        super().__init__()

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=io_channels, out_channels=hidden_channels, kernel_size=1),
            torch.nn.PReLU(),
            torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08),
            torch.nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=hidden_channels,
            ),
            torch.nn.PReLU(),
            torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08),
        )

        self.res_out = (
            None
            if no_residual
            else torch.nn.Conv1d(in_channels=hidden_channels, out_channels=io_channels, kernel_size=1)
        )
        self.skip_out = torch.nn.Conv1d(in_channels=hidden_channels, out_channels=io_channels, kernel_size=1)

    def forward(self, input: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        feature = self.conv_layers(input)
        if self.res_out is None:
            residual = None
        else:
            residual = self.res_out(feature)
        skip_out = self.skip_out(feature)
        return residual, skip_out




class ConvBlock_Reduit(torch.nn.Module):
    """1D Convolutional block.

    Args:
        io_channels (int): The number of input/output channels, <B, Sc>
        hidden_channels (int): The number of channels in the internal layers, <H>.
        kernel_size (int): The convolution kernel size of the middle layer, <P>.
        padding (int): Padding value of the convolution in the middle layer.
        dilation (int, optional): Dilation value of the convolution in the middle layer.
        no_redisual (bool, optional): Disable residual block/output.

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.
    """

    def __init__(
        self,
        ico_channels: int,
        hiddenc_channels: int,
        t_channels: int,
        n_features: int,
        t_features: int,
        Type: int,
        kernel_size: int,
        padding: int,
        dilation: int = 1,
        no_residual: bool = False,
        no_residual_layers: bool = False 
    ):
        super().__init__()
        self.Type = Type
        self.hiddenc_channels = 128
        self.no_residual_layers =no_residual_layers
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ico_channels, out_channels=self.hiddenc_channels, kernel_size=1),
            torch.nn.PReLU(),
            Permute((0,2,1,3)),
            torch.nn.GroupNorm(num_groups=1, num_channels=n_features, eps=1e-08),
            Permute((0,2,1,3)),
            torch.nn.Conv2d(
                in_channels=self.hiddenc_channels,
                out_channels=self.hiddenc_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=self.hiddenc_channels,
            ),
            torch.nn.PReLU(),
            Permute((0,2,1,3)),
            torch.nn.GroupNorm(num_groups=1, num_channels=n_features, eps=1e-08),
            Permute((0,2,1,3))
        )
        
        self.res_out = (
            None
            if no_residual
            else torch.nn.Conv2d(in_channels=self.hiddenc_channels, out_channels=ico_channels, kernel_size=1)
        )
        self.skip_out = torch.nn.Conv2d(in_channels=self.hiddenc_channels, out_channels=ico_channels, kernel_size=1)
        if Type == 0 or Type == 1:
            self.input_conv_Features_res = torch.nn.Conv2d(in_channels = n_features, out_channels = int(n_features/2), kernel_size=1)
            self.input_conv_Channels_res = torch.nn.Conv2d(in_channels = ico_channels, out_channels =  int(ico_channels/2), kernel_size=1)
        if Type == 1 or Type == 2:
            self.input_conv_Features_skip = torch.nn.Conv2d(in_channels = n_features, out_channels = t_features, kernel_size=1)
            self.input_conv_Channels_skip = torch.nn.Conv2d(in_channels = ico_channels, out_channels =  t_channels, kernel_size=1)

    def forward(self, input: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        feature = self.conv_layers(input)
        if self.res_out is None:
            residual = None
        else:
            residual = self.res_out(feature)
            """
            if (self.Type == 0 or self.Type == 1) and self.no_residual_layers:
               residual = torch.permute(residual, (0,2,1,3))
               residual = self.input_conv_Features_res(residual) 
               residual = torch.permute(residual, (0,2,1,3))
               residual = self.input_conv_Channels_res(residual) 
            """  
        skip_out = self.skip_out(feature)
        
        if (self.Type == 1 or self.Type == 2):
            skip_out = self.input_conv_Channels_skip (skip_out)
            skip_out = torch.permute(skip_out, (0,2,1,3))
            skip_out = self.input_conv_Features_skip (skip_out) 
            skip_out = torch.permute(skip_out, (0,2,1,3))
        
        return residual, skip_out






class MaskGenerator(torch.nn.Module):
    """TCN (Temporal Convolution Network) Separation Module

    Generates masks for separation.

    Args:
        input_dim (int): Input feature dimension, <N>.
        num_sources (int): The number of sources to separate.
        kernel_size (int): The convolution kernel size of conv blocks, <P>.
        num_featrs (int): Input/output feature dimenstion of conv blocks, <B, Sc>.
        num_hidden (int): Intermediate feature dimention of conv blocks, <H>
        num_layers (int): The number of conv blocks in one stack, <X>.
        num_stacks (int): The number of conv block stacks, <R>.
        msk_activate (str): The activation function of the mask output.

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.
    """

    def __init__(
        self,
        input_dim: int,
        num_sources: int,
        kernel_size: int,
        num_feats: int,
        num_hidden: int,
        num_layers: int,
        num_stacks: int,
        msk_activate: str,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_sources = num_sources

        self.input_norm = torch.nn.GroupNorm(num_groups=1, num_channels=input_dim, eps=1e-8)
        self.input_conv = torch.nn.Conv1d(in_channels=input_dim, out_channels=num_feats, kernel_size=1)

        self.receptive_field = 0
        self.conv_layers = torch.nn.ModuleList([])
        for s in range(num_stacks):
            for l in range(num_layers):
                multi = 2**l
                self.conv_layers.append(
                    ConvBlock(
                        io_channels=num_feats,
                        hidden_channels=num_hidden,
                        kernel_size=kernel_size,
                        dilation=multi,
                        padding=multi,
                        # The last ConvBlock does not need residual
                        no_residual=(l == (num_layers - 1) and s == (num_stacks - 1)),
                    )
                )
                self.receptive_field += kernel_size if s == 0 and l == 0 else (kernel_size - 1) * multi
        self.output_prelu = torch.nn.PReLU()
        self.output_conv = torch.nn.Conv1d(
            in_channels=num_feats,
            out_channels=input_dim * num_sources,
            kernel_size=1,
        )
        if msk_activate == "sigmoid":
            self.mask_activate = torch.nn.Sigmoid()
        elif msk_activate == "relu":
            self.mask_activate = torch.nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation {msk_activate}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Generate separation mask.

        Args:
            input (torch.Tensor): 3D Tensor with shape [batch, features, frames]

        Returns:
            Tensor: shape [batch, num_sources, features, frames]
        """
        batch_size = input.shape[0]
        feats = self.input_norm(input)
        feats = self.input_conv(feats)
        output = 0.0
        for layer in self.conv_layers:
            residual, skip = layer(feats)
            
            if residual is not None:  # the last conv layer does not produce residual
                feats = feats + residual
            output = output + skip
        output = self.output_prelu(output)
        output = self.output_conv(output)
        output = self.mask_activate(output)
        #print(output.shape)
        #print( output.view(batch_size, self.num_sources, self.input_dim, -1).shape)
        return output.view(batch_size, self.num_sources, self.input_dim, -1)


class MaskGenerator_SS(torch.nn.Module):
    """
    
    TCN (Temporal Convolution Network) Separation Module
    Generates masks for separation.
    
    Args:
        input_dim (int): Input feature dimension, <N>.
        num_sources (int): The number of sources to separate.
        kernel_size (int): The convolution kernel size of conv blocks, <P>.
        num_featrs (int): Input/output feature dimenstion of conv blocks, <B, Sc>.
        num_hidden (int): Intermediate feature dimention of conv blocks, <H>
        num_layers (int): The number of conv blocks in one stack, <X>.
        num_stacks (int): The number of conv block stacks, <R>.
        msk_activate (str): The activation function of the mask output.

    Note:
    
        This implementation corresponds to the "non-causal" setting in the paper.
    
    """
    
    
    
    def __init__(
        self,
        input_dim: int,
        num_sources: int,
        kernel_size: int,
        num_feats: int,
        num_layers: int,
        num_stacks: int,
        msk_activate: str,
        NumberOfChannels: int,
        MtoN_channels: int,
        ico_channels: int,
        hiddenc_channels: int,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_sources = num_sources

        self.input_norm = torch.nn.GroupNorm(num_groups=1, num_channels = input_dim, eps=1e-8)
        self.input_conv_Features = torch.nn.Conv2d(in_channels = input_dim, out_channels = 128, kernel_size=1)
        self.input_conv_Channels = torch.nn.Conv2d(in_channels = NumberOfChannels, out_channels =  MtoN_channels, kernel_size=1)

        self.receptive_field = 0
        
        self.conv_layers = torch.nn.ModuleList([])
        
        t_channels = MtoN_channels
        t_features = 128
        icoo_channels = [ico_channels, int(ico_channels/2),int(ico_channels/4) ]
        n_features = [t_features, int(t_features/2), int(t_features/4)]
        for s in range(num_stacks):
            for l in range(num_layers):
                multi = 2**l
                """
                self.conv_layers.append(
                    ConvBlock(
                        ico_channels=ico_channels,
                        hiddenc_channels= ico_channels*2 ,
                        kernel_size=kernel_size,
                        dilation=multi,
                        padding=multi,
                        # The last ConvBlock does not need residual
                        no_residual=(l == (num_layers - 1) and s == (num_stacks - 1)),
                    )
                )
                """
                
                self.conv_layers.append(
                    ConvBlock_Reduit(
                        ico_channels=icoo_channels[s],
                        hiddenc_channels= 128,
                        t_channels = t_channels,
                        n_features = n_features[s],
                        t_features = t_features,
                        Type = s,
                        kernel_size=kernel_size,
                        dilation=multi,
                        padding=multi,
                        # The last ConvBlock does not need residual
                        no_residual=(l == (num_layers - 1) and s == (num_stacks - 1)),
                        no_residual_layers = l == (num_layers - 1)
                    )
                )
        
                self.receptive_field += kernel_size if s == 0 and l == 0 else (kernel_size - 1) * multi
        self.output_prelu = torch.nn.PReLU()
        self.output_conv_Features = torch.nn.Conv2d(
            in_channels=128,
            out_channels=input_dim,
            kernel_size=1,
        )
        
        self.output_conv_Channels = torch.nn.Conv2d(
            in_channels  = MtoN_channels,
            out_channels = self.num_sources,
            kernel_size=1,
        ) 
        
        
        if msk_activate == "sigmoid":
            self.mask_activate =  torch.nn.Sigmoid()
        elif msk_activate == "relu":
            self.mask_activate = torch.nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation {msk_activate}")
        
        
        self.input_conv_Features_res_0 = torch.nn.Conv2d(in_channels = n_features[0], out_channels = n_features[1], kernel_size=1)
        self.input_conv_Channels_res_0 = torch.nn.Conv2d(in_channels = icoo_channels[0], out_channels =  int(icoo_channels[1]), kernel_size=1)
        
        self.input_conv_Features_res_1 = torch.nn.Conv2d(in_channels = n_features[1], out_channels = int(n_features[2]), kernel_size=1)
        self.input_conv_Channels_res_1 = torch.nn.Conv2d(in_channels = icoo_channels[1], out_channels =  icoo_channels[2], kernel_size=1)
        
        

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Generate separation mask.

        Args:
            input (torch.Tensor): 3D Tensor with shape [batch, features, frames]

        Returns:
            Tensor: shape [batch, num_sources, features, frames]
        
        """
        batch_size = input.shape[0]
        feats = torch.permute(input, (0,2,1,3))
        feats = self.input_norm(feats)
        feats = self.input_conv_Features(feats)
        feats = torch.permute(feats, (0,2,1,3))
        feats = self.input_conv_Channels(feats)
        output = 0.0
        cpt = 0
        for layer in self.conv_layers:
            
            residual, skip = layer(feats)
            
            if residual is not None:
                feats = feats + residual 
      # the last conv layer does not produce residual
                if cpt == 4:
                        feats = torch.permute(feats, (0,2,1,3))
                        feats = self.input_conv_Features_res_0(feats)
                        feats = torch.permute(feats, (0,2,1,3))
                        feats = self.input_conv_Channels_res_0(feats)
                elif cpt == 4+5:
                        feats = torch.permute(feats, (0,2,1,3))
                        feats = self.input_conv_Features_res_1(feats)
                        feats = torch.permute(feats, (0,2,1,3))
                        feats = self.input_conv_Channels_res_1(feats)
                    
            output = output + skip
            cpt = cpt+1
        output_ = self.output_prelu(output)
        
        
        output = self.output_conv_Channels(output_)
        output = torch.permute(output, (0,2,1,3))
        output = self.output_conv_Features(output)
        output = torch.permute(output, (0,2,1,3))
        output = self.mask_activate(output)
        #print(output.shape)
        return output#.view(batch_size, self.num_sources, self.input_dim, -1)
class ConvTasNet(torch.nn.Module):
    """Conv-TasNet: a fully-convolutional time-domain audio separation network
    *Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation*
    [:footcite:`Luo_2019`].

    Args:
        num_sources (int, optional): The number of sources to split.
        enc_kernel_size (int, optional): The convolution kernel size of the encoder/decoder, <L>.
        enc_num_feats (int, optional): The feature dimensions passed to mask generator, <N>.
        msk_kernel_size (int, optional): The convolution kernel size of the mask generator, <P>.
        msk_num_feats (int, optional): The input/output feature dimension of conv block in the mask generator, <B, Sc>.
        msk_num_hidden_feats (int, optional): The internal feature dimension of conv block of the mask generator, <H>.
        msk_num_layers (int, optional): The number of layers in one conv block of the mask generator, <X>.
        msk_num_stacks (int, optional): The numbr of conv blocks of the mask generator, <R>.
        msk_activate (str, optional): The activation function of the mask output (Default: ``sigmoid``).

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.
    """
    
    
    def __init__(
        self,
        num_sources: int = 2,
        # encoder/decoder parameters
        enc_kernel_size: int = 16,
        enc_num_feats: int = 512,
        # mask generator parameters
        msk_kernel_size: int = 3,
        msk_num_hidden_feats: int = 512,
        msk_num_layers: int = 8,
        msk_num_stacks: int = 3,
        NumberOfChannels: int = 4,
        MtoN_channels: int =  12,
        ico_channels : int = 12, # The same as MtoN
        hiddenc_channels: int= 64,
        #N_cahnnels : int=4,
        
        msk_activate: str = "sigmoid",
    ):
        super().__init__()

        self.num_sources = num_sources
        self.enc_num_feats = enc_num_feats
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_kernel_size // 2
        self.encoder = torch.nn.Conv1d(
            in_channels=1,
            out_channels=enc_num_feats,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )
        self.mask_generator_SS = MaskGenerator_SS(
            input_dim=enc_num_feats,
            num_sources=num_sources,
            kernel_size=msk_kernel_size,
            num_feats=enc_num_feats,
            num_layers=msk_num_layers,
            num_stacks=msk_num_stacks,
            msk_activate=msk_activate,
            NumberOfChannels= NumberOfChannels,
            MtoN_channels =  MtoN_channels,
            ico_channels= ico_channels,
            hiddenc_channels= hiddenc_channels,
        )
        """
        self.mask_generator_SE = MaskGenerator_SS(
            input_dim=enc_num_feats,
            num_sources=NumberOfChannels,
            kernel_size=msk_kernel_size,
            num_feats=enc_num_feats,
            num_layers=5,
            num_stacks=2,
            msk_activate=msk_activate,
            NumberOfChannels= NumberOfChannels,
            MtoN_channels =  MtoN_channels,
            ico_channels= ico_channels,
            hiddenc_channels= 64,
        )
        """







        """
        self.LastConv_4 = torch.nn.Conv2d(
            in_channels=hiddenc_channels,
            out_channels=4,
            kernel_size=1,
        )
        self.LastActivation_4 = torch.nn.Sigmoid()
        
        self.LastConv_18 = torch.nn.Conv2d(
            in_channels=hiddenc_channels,
            out_channels=18,
            kernel_size=1,
        )
        self.LastActivation_18 = torch.nn.Softmax(dim = 1)
        """
                
        
        

        
        
        self.decoder = torch.nn.ConvTranspose1d(
            in_channels=enc_num_feats,
            out_channels=1,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )


    def _align_num_frames_with_strides(self, input: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pad input Tensor so that the end of the input tensor corresponds with

        1. (if kernel size is odd) the center of the last convolution kernel
        or 2. (if kernel size is even) the end of the first half of the last convolution kernel

        Assumption:
            The resulting Tensor will be padded with the size of stride (== kernel_width // 2)
            on the both ends in Conv1D

        |<--- k_1 --->|
        |      |            |<-- k_n-1 -->|
        |      |                  |  |<--- k_n --->|
        |      |                  |         |      |
        |      |                  |         |      |
        |      v                  v         v      |
        |<---->|<--- input signal --->|<--->|<---->|
         stride                         PAD  stride

        Args:
            input (torch.Tensor): 3D Tensor with shape (batch_size, channels==1, frames)

        Returns:
            Tensor: Padded Tensor
            int: Number of paddings performed
        """
        batch_size, num_channels, num_frames = input.shape
        is_odd = self.enc_kernel_size % 2
        num_strides = (num_frames - is_odd) // self.enc_stride
        num_remainings = num_frames - (is_odd + num_strides * self.enc_stride)
        if num_remainings == 0:
            return input, 0

        num_paddings = self.enc_stride - num_remainings
        pad = torch.zeros(
            batch_size,
            num_channels,
            num_paddings,
            dtype=input.dtype,
            device=input.device,
        )
        return torch.cat([input, pad], 2), num_paddings

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Perform source separation. Generate audio source waveforms.

        Args:
            input (torch.Tensor): 3D Tensor with shape [batch, channel==1, frames]

        Returns:
            Tensor: 3D Tensor with shape [batch, channel==num_sources, frames]
        """
        
        #♠if input.ndim != 4 or input.shape[1] != 1:
          #  raise ValueError(f"Expected 3D tensor (batch, channel==1, frames). Found: {input.shape}")

        # B: batch size
        # L: input frame length
        # L': padded input frame length
        # F: feature dimension
        # M: feature frame length
        # S: number of sources
        
        
        
        padded, num_pads = self._align_num_frames_with_strides(input)  # B, 1, L'
        """
        #print('padded', padded.shape)
        #print('num_pads', num_pads)
        plt.plot(padded[0,0,:].cpu().detach().numpy())
        plt.show()
        plt.plot(padded[0,1,:].cpu().detach().numpy())
        plt.show()
        """
        
        batch_size, num_padded_frames, channels = padded.shape[0], padded.shape[2], padded.shape[1]
        padded = padded.reshape(batch_size * padded.shape[1], padded.shape[2]).unsqueeze(1)  
        feats = self.encoder(padded)

        feats = feats.reshape(batch_size, channels,self.enc_num_feats,-1  )
        print(feats.shape)
    
        """
        ListOfFetas = []
        for il in range(padded.shape[1]):
            feats = self.encoder(torch.unsqueeze(padded[:,il,:],1))  # B, F, M
            ListOfFetas.append(feats) 
        feats = torch.stack(ListOfFetas)
        feats = torch.permute(feats, (1,0,2,3))
        #masked1 = masked1.repeat(1, 2, 1, 1)
        """

        #maskSE = self.mask_generator_SE(feats)
        #MaskedInput = maskSE*feats
        maskSS  = self.mask_generator_SS(feats)
        print(maskSS.shape)
        print(feats.shape)
        Speakers_f_first = maskSS*(feats[:,0,:,:]).unsqueeze(1)#.repeat(1,  self.num_sources, 1, 1) 
        #print(Speakers_f_first.shape)
        
        
        """
        maskNN = 1 - maskSS
        
        #Speakers = maskSS * (feats[:,0,:,:]).unsqueeze(1).repeat(1,  self.num_sources, 1, 1)
   
        Speakers = maskSS.unsqueeze(2) * feats.unsqueeze(1)
        Noises = maskNN.unsqueeze(2) * feats.unsqueeze(1)

        Phi_s = torch.mean(torch.einsum('ijklmn,ijlemn->ijkemn', torch.unsqueeze(Speakers,3) , torch.unsqueeze(Speakers,2)),-1)
        Phi_N = torch.mean(torch.einsum('ijklmn,ijlemn->ijkemn', torch.unsqueeze(Noises,3) , torch.unsqueeze(Noises,2)),-1)
   
        A = torch.permute(Phi_s + Phi_N, (0,1,4,2,3))

        PinvA = torch.linalg.inv(A)
        PinvA = torch.permute(PinvA, (0,1,3,4,2))
        W = torch.einsum('ijklm,ijlem->ijkem', PinvA , Phi_s) 
        W_rank1 = W[:,:,:,:,:]
        W_rank1 = torch.permute(W_rank1 , (0,1,3,2,4))
        Speakers_f = torch.einsum('ijlnm, inkmf->ijlkmf', W_rank1 , torch.unsqueeze(feats,2) )
        Speakers_f_first = Speakers_f[:,:,0,0,:,:]  
        """
        
        
        Separated = Speakers_f_first.reshape(batch_size * self.num_sources, self.enc_num_feats, -1)  
        output2 = self.decoder(Separated).reshape(batch_size, self.num_sources, num_padded_frames)  # B, S, L'
 
    
 
        if num_pads > 0:
            #output1 = output1[..., :-num_pads]  # B, S, L
            output2 = output2[..., :-num_pads]  # B, S, L
  
        return  output2

    
    
import torchaudio   

if __name__ == '__main__':
    with torch.no_grad():
        model = ConvTasNet(num_sources=1 ,enc_kernel_size = 256, enc_num_feats = 256, msk_kernel_size = 3, msk_num_hidden_feats = 128, msk_num_layers = 5, msk_num_stacks = 3,NumberOfChannels = 3, MtoN_channels=  16, ico_channels = 16,  hiddenc_channels= 64).cuda()
        
        model.cpu()
        #model = ConvTasNet(num_sources=24, enc_kernel_size = 256, enc_num_feats = 512, msk_kernel_size = 3, msk_num_hidden_feats = 512, msk_num_layers = 8, msk_num_stacks = 3,NumberOfChannels = 4, MtoN_channels=  24, ico_channels = 24,  hiddenc_channels= 24).cuda()
        import time
        Input = torch.randn((1,3,16000)).cpu()
        start = time.time()
        Output1 = model(Input) 
        print(time.time()-start)
        
        
        model = torchaudio.models.ConvTasNet(num_sources=1, enc_kernel_size = 256, enc_num_feats = 256, msk_kernel_size = 3, msk_num_feats= 128, msk_num_hidden_feats = 256, msk_num_layers = 8, msk_num_stacks = 3)
        
        model.cpu()
        #model = ConvTasNet(num_sources=24, enc_kernel_size = 256, enc_num_feats = 512, msk_kernel_size = 3, msk_num_hidden_feats = 512, msk_num_layers = 8, msk_num_stacks = 3,NumberOfChannels = 4, MtoN_channels=  24, ico_channels = 24,  hiddenc_channels= 24).cuda()
        import time
        Input = torch.randn((1,1,16000)).cpu()
        start = time.time()
        Output1 = model(Input) 
        print(time.time()-start)        
        
        