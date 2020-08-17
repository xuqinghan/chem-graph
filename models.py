import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, CGConv, ChebConv
from torch_geometric.utils import to_dense_batch
#https://github.com/rusty1s/pytorch_geometric/issues/1502

class Net(torch.nn.Module):
    '''带有graph输入的'''
    def __init__(self, node_atom:int, n_h1:int, dim_out):
        super(Net, self).__init__()
        #node 特征1维 原子类型  edge特征 1 维 bond 键数
        #graph卷积不提升维数
        self.conv1 = CGConv(1, dim=1)
        #self.conv1 = ChebConv(1, 1, K=5)
        self.conv2 = GCNConv(1, 1, add_self_loops =False)
        #self.conv2 = GCNConv(dim_output1, dim_out)
        #之前输出的是 50 node * 1维
        #普通线性层
        self.layer3 = nn.Sequential(nn.Linear(1*node_atom, n_h1), nn.BatchNorm1d(n_h1))
        self.layer4 = nn.Sequential(nn.Linear(n_h1, dim_out))

        #self.conv2 = CGConv(dim_output1, dim_output2)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        #print('x ', x.shape)
        #print('edge_index ', edge_index.shape)
        #print('edge_attr ', edge_attr.shape)
        #print('conv1 weight ', self.conv1.weight.shape)
        x = self.conv1(x, edge_index, edge_attr)
        #print('conv1 out x ', x.shape)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        #print('conv1 out x ', x.shape)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        #print('conv2 out x ', x.shape)
        #转为普通1D
        x, mask = to_dense_batch(x, data.batch)
        x = x.transpose(1, 2)  # [batch_size, in_channels, num_nodes]
        #print('to_dense_batch out x ', x.shape)
        #展平
        #x = x.view(x.size(0), -1)
        x = x.reshape(x.size(0), x.size(1)*x.size(2))
        #print('layer2 in x ', x.shape)
        #x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(self.layer3(x))
        #print('layer2 out x ', x.shape)
        x = F.relu(self.layer4(x))

        return x