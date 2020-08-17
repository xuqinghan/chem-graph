import os.path as osp
import nmrshiftdb2
from torch_geometric.data import Data
import numpy as np

import torch
#from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset

def create_Data1_from_abs(record1_abs:dict, idx):
    '''records_abs 形如 {'atoms': atoms, 'bonds': bonds, 'spectrums': dict_spectrum}'''
    #2D array
    print(idx)
    edges = nmrshiftdb2.read_edges_to_ptG(record1_abs)
    if edges is None:
        #甲烷
        edge_index = torch.tensor([[], []], dtype=torch.long)
        bonds_order = torch.tensor([], dtype=torch.float)
        return None
    else:
        edge_index = torch.tensor(edges[0:2, :], dtype=torch.long)
        #键数
        bonds_order = torch.tensor(edges[2, :], dtype=torch.float)

    bonds_order = torch.unsqueeze(bonds_order, 1)
    #print('edge_attr', bonds_order)
    atom_features = nmrshiftdb2.read_atom_features_to_ptG(record1_abs['atoms'])

    atom_features = torch.tensor(atom_features, dtype=torch.float)
    #print(atom_features)
    #print('atom_features', atom_features)
    #谱线
    spectrum2500 = nmrshiftdb2.get_spectrum_to_ptG(record1_abs)
    #转成tensor 0 1 稀疏
    spectrum2500 = torch.tensor(spectrum2500, dtype=torch.float)
    return Data(x=atom_features, edge_index=edge_index, edge_attr=bonds_order, y=spectrum2500)


#class NMRShiftDB2(Dataset):
class NMRShiftDB2(InMemoryDataset):
    def __init__(self, root, is_train= True , transform=None, pre_transform=None):
        super(NMRShiftDB2, self).__init__(root, transform, pre_transform)
        idx_fname = 0 if is_train else 1
        self.data, self.slices = torch.load(self.processed_paths[idx_fname])
        self.root = root

    @property
    def raw_file_names(self):
        return ['nmrshiftdb2withsignals.sd']

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     pass

    def process(self):
        print(f'读取原始文件 {self.raw_dir}/{self.raw_file_names[0]}...')
       
        with open(osp.join(self.raw_dir, self.raw_file_names[0]), 'r', encoding='utf-8') as file:
            id_abs = nmrshiftdb2.split_sd2_by_nmrshiftdb2(file)
        print('读取原始文件 完毕!')
        # Read data into huge `Data` list.
        #print(id_abs)
        num_atom_max = 50
        print(f'规范化分子中原子数={num_atom_max} 补齐不足、去掉原子数大于的')
        id_abs = nmrshiftdb2.fill_num_atom(id_abs, num_atom_max=num_atom_max)
        print(f'共 {len(id_abs)}个结构式')
        data_list = []
        for idx, records_abs in id_abs.items():
            data = create_Data1_from_abs(records_abs, idx)
            if data is not None:
                data_list.append(data)
        #print(data_list)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        #划分train test 9:1
        N = len(data_list)
        N_train = int(N * 0.9)
        data_list_train = data_list[0:N_train]
        data, slices = self.collate(data_list_train)
        torch.save((data, slices), self.processed_paths[0])
        data_list_test = data_list[N_train:N]
        data, slices = self.collate(data_list_test)
        torch.save((data, slices), self.processed_paths[1])



    # def process(self):
    #     print(f'读取原始文件 {self.root}/{self.raw_file_names[0]}...')
       
    #     with open(f'{self.root}/{self.raw_file_names[0]}', 'r', encoding='utf-8') as file:
    #         id_abs = nmrshiftdb2.split_sd2_by_nmrshiftdb2(file)
    #     print('读取原始文件 完毕!')
    #     # Read data into huge `Data` list.
    #     #print(id_abs)
    #     #data_list = [create_Data1_from_abs(records_abs) for idx, records_abs in id_abs.items()]
    #     #print(data_list)
    #     for records_abs in id_abs.values():
    #         # Read data from `raw_path`.
    #         data = create_Data1_from_abs(records_abs)

    #         if self.pre_filter is not None and not self.pre_filter(data):
    #             continue

    #         if self.pre_transform is not None:
    #             data = self.pre_transform(data)

    #         torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
    #         i += 1



    # def len(self):
    #     return len(self.processed_file_names)

    # def get(self, idx):
    #     data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
    #     return data