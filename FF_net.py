import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np
from complexFunctions import complex_relu, complex_max_pool2d
from complexFunctions import complex_dropout, complex_dropout2d

class CoilWeight_net(nn.Module):
    def __init__(self,nodes,channels):
        super(CoilWeight_net, self).__init__()
        #5x4xchannels
        #self.fc1 = nn.Linear(nodes, 20)
        #self.fc2 = nn.Linear(20, 1)       
        self.layers = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(4,5,channels), padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        
    def forward(self, x):
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))            
        x=self.layers(x)               
        return x

class FF_net(nn.Module):
    def __init__(self,nodes):
        super().__init__()
        #self.fc1 = nn.Linear(nodes, nodes//2)
        #self.fc2 = nn.Linear(nodes//2, nodes//2)
        #self.fc3 = nn.Linear(nodes//2, nodes//2)
        self.ff_block=nn.Sequential(
            nn.Linear(nodes, nodes//2),
            nn.LeakyReLU(),
            nn.Linear(nodes//2, nodes//2),
            nn.LeakyReLU()
        )
        
    def forward(self, x):
        #x = F.tanh(self.fc1(x))
        #x = F.tanh(self.fc2(x))
        #x = F.tanh(self.fc3(x))
        x=self.ff_block(x)
        return x
    
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features,bias=False)
        self.fc_i = nn.Linear(in_features, out_features,bias=False)
        #IF_R=torch.Tensor([[np.real(np.exp(2j*(j)*(i)*np.pi/in_features)) for j in range(in_features)] for i in range(in_features)])
        #IF_IMAG=torch.Tensor([[np.imag(np.exp(2j*(j)*(i)*np.pi/in_features)) for j in range(in_features)] for i in range(in_features)])
        IF_R=torch.Tensor([[np.cos(2*(j)*(i)*np.pi/in_features)/in_features for j in range(in_features)] for i in range(in_features)]).double()
        IF_IMAG=torch.Tensor([[np.sin(2*(j)*(i)*np.pi/in_features)/in_features for j in range(in_features)] for i in range(in_features)]).double()
        IFB=torch.Tensor([0 for i in range(in_features)]).double()
        with torch.no_grad():
            self.fc_r.weight = torch.nn.Parameter(IF_R)
            self.fc_i.weight = torch.nn.Parameter(IF_IMAG)
            #self.fc_r.bias = torch.nn.Parameter(IFB)
            #self.fc_i.bias = torch.nn.Parameter(IFB)

    def forward(self,input_r, input_i):
        return self.fc_r(input_r)-self.fc_i(input_i), \
               self.fc_r(input_i)+self.fc_i(input_r)
    
class ComplexSequential(nn.Sequential):
    def forward(self, input_r, input_t):
        for module in self._modules.values():
            input_r, input_t = module(input_r, input_t)
        return input_r, input_t

class Complex_net_ext(nn.Module):
    def __init__(self, n_input,args):
        super().__init__()
        #IF=torch.Tensor([[1 if j==i else 0 for j in range(n_input)] for i in range(n_input)])
        self.n_input=n_input
        self.layer = ComplexLinear(n_input, n_input)
        self.layer2 = ComplexLinear(n_input, n_input)
        self.args=args
        
    def forward(self, x):
        for i in range(1,self.n_input):
            x_r=x[...,i,:,:]
            xr = x_r[...,:,0]
            xi = x_r[...,:,1]
            out_r,out_i = self.layer(xr, xi)
            x[...,i,:,0]=out_r
            x[...,i,:,1]=out_i
        for i in range(1,self.n_input):
            y_r=x[...,:,i,:]
            yr = y_r[...,:,0]
            yi = y_r[...,:,1]
            out_r,out_i = self.layer2(yr, yi)
            x[...,:,i,0]=out_r+out_i
        return torch.abs(x[...,:,i,0])

class Complex_net(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        #IF=torch.Tensor([[1 if j==i else 0 for j in range(n_input)] for i in range(n_input)])
        self.n_input=n_input
        self.layer = ComplexLinear(n_input, n_input)
        
    def forward(self, x):
        xr = x_r[...,:self.n_input]
        xi = x_r[...,self.n_input:]
        out_r,out_i = self.layer(xr, xi)
        return (out_r**2+out_i**2)**(0.5)
    
class FTI_net(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        #IF=torch.Tensor([[1 if j==i else 0 for j in range(n_input)] for i in range(n_input)])
        self.n_input=n_input
        self.layer = ComplexLinear(n_input, n_input)
        self.layer2 = ComplexLinear(n_input, n_input)
        
    def forward(self, x):
        xr = x[...,:self.n_input]
        xi = x[...,self.n_input:]
        out_r,out_i = self.layer(xr, xi)
        out_r,out_i = self.layer2(out_r,out_i)
        return (out_r**2+out_i**2)**(0.5)
    
class FF_net_tanh(nn.Module):
    def __init__(self,nodes):
        super().__init__()
        #self.fc1 = nn.Linear(nodes, nodes//2)
        #self.fc2 = nn.Linear(nodes//2, nodes//2)
        #self.fc3 = nn.Linear(nodes//2, nodes//2)
        self.ff_block=nn.Sequential(
            nn.Linear(nodes, nodes//2),
            nn.ELU(),
            nn.Linear(nodes//2, nodes//2),
            nn.ReLU(),
        )
        
    def forward(self, x):
        #x = F.tanh(self.fc1(x))
        #x = F.tanh(self.fc2(x))
        #x = F.tanh(self.fc3(x))
        x=self.ff_block(x)
        return x
    
class FF_simple_net(nn.Module):
    def __init__(self,nodes):
        super().__init__()
        #self.fc1 = nn.Linear(nodes, nodes//2)
        #self.fc2 = nn.Linear(nodes//2, nodes//2)
        #self.fc3 = nn.Linear(nodes//2, nodes//2)
        self.ffs_block=nn.Sequential(
            nn.Linear(nodes, nodes//2),
            nn.LeakyReLU()
        )
        
    def forward(self, x):
        #x = F.tanh(self.fc1(x))
        #x = F.tanh(self.fc2(x))
        #x = F.tanh(self.fc3(x))
        x=self.ffs_block(x)
        return x
    
class ComplexDropout(nn.Module):
    def __init__(self,p=0.5, inplace=False):
        super(ComplexDropout,self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self,input_r,input_i):
        return complex_dropout(input_r,input_i,self.p,self.inplace)

class ComplexDropout2d(nn.Module):
    def __init__(self,p=0.5, inplace=False):
        super(ComplexDropout2d,self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self,input_r,input_i):
        return complex_dropout2d(input_r,input_i,self.p,self.inplace)

class ComplexMaxPool2d(nn.Module):
    def __init__(self,kernel_size, stride= None, padding = 0,
                 dilation = 1, return_indices = False, ceil_mode = False):
        super(ComplexMaxPool2d,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self,input_r,input_i):
        return complex_max_pool2d(input_r,input_i,kernel_size = self.kernel_size,stride = self.stride, padding = self.padding,dilation = self.dilation, ceil_mode = self.ceil_mode,return_indices = self.return_indices)

class ComplexReLU(nn.Module):
     def forward(self,input_r,input_i):
            return complex_relu(input_r,input_i)

class ComplexConvTranspose2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super(ComplexConvTranspose2d, self).__init__()
        self.conv_tran_r = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,output_padding, groups, bias, dilation, padding_mode)

    def forward(self,input_r,input_i):
        return self.conv_tran_r(input_r)-self.conv_tran_i(input_i), \
               self.conv_tran_r(input_i)+self.conv_tran_i(input_r)

class ComplexConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    def forward(self,input_r, input_i):
#        assert(input_r.size() == input_i.size())
        return self.conv_r(input_r)-self.conv_i(input_i), \
               self.conv_r(input_i)+self.conv_i(input_r)

class Combiner(nn.Module):
    def __init__(self,input_nodes):
        super(Combiner, self).__init__()
        self.networks=[]
        self.input_nodes = input_nodes        
        for i in range(len(input_nodes)):
            self.networks.append(CoilWeight_net(input_nodes[i]))
            self.networks = nn.ModuleList(self.networks)
        
        self.fc1 = nn.Linear(len(input_nodes), 40)
        self.fc2 = nn.Linear(40, 10)
        self.fc_out = nn.Linear(10, 1)
        
    def forward(self, input_):
        
        x_list=[]
        for i in range(len(self.input_nodes)):
            x_list.append(F.relu(self.networks[i](input_[:,i])))#input_[:,i] shape 500 * 200
            
        x = torch.cat((x_list), 1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc_out(x))
        return x