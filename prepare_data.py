from dataset import NMRShiftDB2
from torch_geometric.data import DataLoader
#print('run')
dataset_tr = NMRShiftDB2(root='./dataset/NMRShiftDB2', is_train=True)
dataset_ts = NMRShiftDB2(root='./dataset/NMRShiftDB2', is_train=False)

#print(dataset_ts)
loader_tr = DataLoader(dataset_ts, batch_size=128, shuffle=True)
loader_ts = DataLoader(dataset_ts, batch_size=128, shuffle=False)
#print(loader_ts)
# for batch in loader_ts:
#     print(batch)