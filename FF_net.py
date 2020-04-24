import torch.nn as nn
import torch
from torch.nn import functional as F

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
    
class FF_net_tanh(nn.Module):
    def __init__(self,nodes):
        super().__init__()
        #self.fc1 = nn.Linear(nodes, nodes//2)
        #self.fc2 = nn.Linear(nodes//2, nodes//2)
        #self.fc3 = nn.Linear(nodes//2, nodes//2)
        self.ff_block=nn.Sequential(
            nn.Linear(nodes, nodes//2),
            nn.Tanh(),
            nn.Linear(nodes//2, nodes//2),
            nn.Tanh(),
            nn.Linear(nodes//2, nodes//2),
            nn.Tanh(),
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