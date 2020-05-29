import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn import functional as F
from complexLayers import ComplexSequential,ComplexConv2d,ComplexBatchNorm2d,ComplexConvTranspose2d,ComplexMaxPool2d,ComplexReLU,ComplexDropout2d
from  complexFunctions import complex_relu, complex_max_pool2d


class ComplexConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = int(in_chans)
        self.out_chans = int(out_chans)
        self.drop_prob = drop_prob

        self.layers = ComplexSequential(
            ComplexConv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False,setW=False),
            ComplexBatchNorm2d(out_chans),
            ComplexReLU(),
            ComplexDropout2d(p=drop_prob,inplace=True),
            ComplexConv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False,setW=False),
            ComplexBatchNorm2d(out_chans),
            ComplexReLU(),
            ComplexDropout2d(p=drop_prob,inplace=True)
        )

    def forward(self, input_r,input_i):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input_r,input_i)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class ComplexTransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose layers followed by
    instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = int(in_chans)
        self.out_chans = int(out_chans)

        self.layers = ComplexSequential(
            ComplexConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, bias=False,setW=False),
            ComplexBatchNorm2d(out_chans),
            ComplexReLU(),
        )

    def forward(self, input_r,input_i):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input_r,input_i)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans})'


class ComplexEndToEnd(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    Incremental improvements carried out by FastMRI(https://fastmri.org/) and the author.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, resolution):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = int(in_chans)
        self.out_chans = int(out_chans)
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.resolution= resolution
        self.f_layer1=ComplexConv2d(in_channels=1, out_channels=resolution, kernel_size=(1,resolution),padding=(0,0), stride=1, bias=False)
        self.f_layer2=ComplexConv2d(in_channels=1, out_channels=resolution, kernel_size=(1,resolution),padding=(0,0), stride=1, bias=False)

        self.down_sample_layers = nn.ModuleList([ComplexConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ComplexConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ComplexConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_transpose_conv += [ComplexTransposeConvBlock(ch * 2, ch)]
            self.up_conv += [ComplexConvBlock(ch * 2, ch, drop_prob)]
            ch //= 2

        self.up_transpose_conv += [ComplexTransposeConvBlock(ch * 2, ch)]
        self.up_conv += [
            ComplexSequential(
                ComplexConvBlock(ch * 2, ch, drop_prob),
                ComplexConv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )]

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output_r = input[...,0]
        output_i = input[...,1]

        # Apply down-sampling layers
        for i, layer in enumerate(self.down_sample_layers):
            output_r,output_i = layer(output_r,output_i)
            stack.append([output_r,output_i])
            #print(stack)
            output_r, output_i = complex_max_pool2d(output_r, output_i, kernel_size=2, stride=2, padding=0)

        output_r,output_i = self.conv(output_r,output_i)

        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output_r,output_i = transpose_conv(output_r,output_i)

            # Reflect pad on the right/botton if needed to handle odd input dimensions.
            padding = [0, 0, 0, 0]
            if output_r.shape[-1] != downsample_layer[0].shape[-1]:
                padding[1] = 1 # Padding right
            if output_r.shape[-2] != downsample_layer[0].shape[-2]:
                padding[3] = 1 # Padding bottom
            if sum(padding) != 0:
                output_r = F.pad(output_r, padding, "reflect")
                output_i = F.pad(output_i, padding, "reflect")

            output_r = torch.cat([output_r, downsample_layer[0]], dim=1)
            output_i = torch.cat([output_i, downsample_layer[1]], dim=1)
            output_r,output_i = conv(output_r,output_i)
            
        output_r,output_i=self.f_layer1(output_r,output_i)
        output_r,output_i=output_r.squeeze(-1).unsqueeze(1),output_i.squeeze(-1).unsqueeze(1)
        #print("out",output_r.shape)
        output_r,output_i=self.f_layer2(output_r,output_i)
        output_r,output_i=output_r.squeeze(-1).unsqueeze(1),output_i.squeeze(-1).unsqueeze(1)
        
        return output_r,output_i
    
    
    
class td_fourier_net(nn.Module):
    """
    Two dimensional convolutional fourier transform approximation.
    """

    def __init__(self, in_chans=1,out_chans=1,drop_prob=0.05,resolution=128):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = int(in_chans)
        self.out_chans = int(out_chans)
        self.drop_prob = drop_prob
        self.resolution= resolution
        self.c_layer1=ComplexConv2d(in_channels=1, out_channels=resolution, kernel_size=(1,resolution),padding=(0,0), stride=1, bias=False)
        self.c_layer2=ComplexConv2d(in_channels=1, out_channels=resolution, kernel_size=(1,resolution),padding=(0,0), stride=1, bias=False)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        output_r = input[...,0]
        output_i = input[...,1]
        output_r,output_i=self.c_layer1(output_r,output_i)
        output_r,output_i=output_r.squeeze(-1).unsqueeze(1),output_i.squeeze(-1).unsqueeze(1)
        #print("out",output_r.shape)
        output_r,output_i=self.c_layer2(output_r,output_i)
        output_r,output_i=output_r.squeeze(-1).unsqueeze(1),output_i.squeeze(-1).unsqueeze(1)
        return output_r,output_i

    
    
class td_fourier_net_real(nn.Module):
    """
    Two dimensional convolutional fourier transform approximation.
    """

    def __init__(self, in_chans=1,out_chans=1,drop_prob=0.05,resolution=128):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = int(in_chans)
        self.out_chans = int(out_chans)
        self.drop_prob = drop_prob
        self.resolution= resolution
        self.c_layer1=Conv2d(in_channels=1, out_channels=2*resolution, kernel_size=(1,2*resolution),padding=(0,0), stride=1, bias=False)
        self.c_layer2=Conv2d(in_channels=1, out_channels=2*resolution, kernel_size=(1,resolution),padding=(0,0), stride=1, bias=False)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        output = input.flatten(start_dim=-2)
        #print(output.shape)
        output=self.c_layer1(output)
        #print(output.shape)
        output=output.squeeze(-1).unsqueeze(1)
        #print("out",output_r.shape)
        output=self.c_layer2(output)
        output=output.squeeze(-1).unsqueeze(1)
        #print(output.shape)
        return output[...,:self.resolution,:self.resolution],output[...,self.resolution:,self.resolution:]