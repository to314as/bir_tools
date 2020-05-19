import torch
from torch import nn
from torch.nn import functional as F
from complexLayers import ComplexSequential,ComplexConv2d,ComplexBatchNorm2d,ComplexConvTranspose2d,ComplexMaxPool2d


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

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = ComplexSequential(
            ComplexConv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            ComplexBatchNorm2d(out_chans),
            ComplexReLU(),
            ComplexDropout2d(drop_prob),
            ComplexConv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            ComplexBatchNorm2d(out_chans),
            ComplexReLU(),
            ComplexDropout2d(drop_prob)
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

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = ComplexSequential(
            ComplexConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
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


class ComplexUnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

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
            nn.Sequential(
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
            stack.append(output_r,output_i)
            output_r,output_i = ComplexMaxPool2d(output_r,output_i, kernel_size=2, stride=2, padding=0)

        output_r,output_i = self.conv(output_r,output_i)

        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output_r,output_i = transpose_conv(output_r,output_i)

            # Reflect pad on the right/botton if needed to handle odd input dimensions.
            padding = [0, 0, 0, 0]
            if output_r.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1 # Padding right
            if output_r.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1 # Padding bottom
            if sum(padding) != 0:
                output_r = F.pad(output_r, padding, "reflect")
                output_i = F.pad(output_i, padding, "reflect")

            output_r = torch.cat([output_r, downsample_layer], dim=1)
            output_i = torch.cat([output_i, downsample_layer], dim=1)
            output_r,output_i = conv(output_r,output_i)

        return output_r,output_i