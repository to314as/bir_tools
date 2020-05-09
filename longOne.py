import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np
from complexFunctions import complex_relu, complex_max_pool2d,complex_dropout, complex_dropout2d
from complexLayers import ComplexConv2d,ComplexConvTranspose2d,ComplexConvTranspose3d,ComplexSequential

class ComplexFourier(nn.Module):
   
    def __init__(self, in_chans, out_chans, drop_prob, resolution):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.resolution= resolution
        self.layer1=ComplexConv2d(in_channels=1, out_channels=resolution, kernel_size=(1,resolution),padding=(0,0), stride=1, bias=False)
        self.layer2=ComplexConv2d(in_channels=1, out_channels=resolution, kernel_size=(1,resolution),padding=(0,0), stride=1, bias=False)
        
    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        if len(input.shape)>5:
            input.squeeze(1)
        input_r=input[...,0]
        input_i=input[...,1]
        #print("in",input_r.shape)
        output_r,output_i=self.layer1(input_r,input_i)
        output_r,output_i=output_r.squeeze(-1).unsqueeze(1),output_i.squeeze(-1).unsqueeze(1)
        #print("out",output_r.shape)
        output_r,output_i=self.layer2(output_r,output_i)
        output_r,output_i=output_r.squeeze(-1),output_i.squeeze(-1)
        #print("out2",output_r.shape)
        return (output_r**2+output_i**2)**(1/2)