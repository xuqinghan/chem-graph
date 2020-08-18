import pickle
import torch
from torch import nn
from models import Net
import nmrshiftdb2
import dataset

from torch_geometric.data import Batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#----------------------
print('读取模型')
#PATH = 'model_mol2spectrum_20200817.pkl'
PATH = 'model_dim50.pkl'
model = torch.load(PATH)
#测试模式
model.eval()

#----------------------
print('读取原始数据')
#读取全部数据
with open('./nmrshiftdb2_orgin.pkl', 'rb') as file:
    id_abs = pickle.load(file)

num_atom_max = 50
print(f'规范化分子中原子数={num_atom_max} 补齐不足、去掉原子数大于的')
id_abs = nmrshiftdb2.fill_num_atom(id_abs, num_atom_max=num_atom_max)
print(f'共 {len(id_abs)}个结构式')


print('构造1个样本')
#指定1个id的
idms = list(id_abs.keys())
idm = idms[0]
#模型接受的data
data1 = dataset.create_Data1_from_abs(id_abs[idm], idm)

batch = Batch.from_data_list([data1]).to(device)

criterion = nn.HingeEmbeddingLoss(reduction='sum')
#forward
out = model(batch)
#loss =criterion(out, batch.y)
#loss_ts = loss.item()


print('实际:', batch.y)
print('预测:', out)
#print(loss_ts)