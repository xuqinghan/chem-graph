import os.path as osp
import nmrshiftdb2
from torch_geometric.data import Data
import numpy as np

import torch
#from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset

def create_Data1_from_abs(records_abs:dict, idx):
    '''records_abs 形如 {'atoms': atoms, 'bonds': bonds, 'spectrums': dict_spectrum}'''
    #2D array
    print(idx)
    edges = nmrshiftdb2.read_edges_to_ptG(records_abs)
    if edges is None:
        #甲烷
        edge_index = torch.tensor([[], []], dtype=torch.long)
        bonds_order = torch.tensor([], dtype=torch.long)
    else:
        edge_index = torch.tensor(edges[0:2, :], dtype=torch.long)
        #键数
        bonds_order = torch.tensor(edges[2, :], dtype=torch.long)
    #print(edge_index)
    atom_features = torch.tensor(nmrshiftdb2.read_atom_features_to_ptG(records_abs), dtype=torch.float)

    #谱线
    spectrum2500 = nmrshiftdb2.get_spectrum_to_ptG(records_abs)
    return Data(x=atom_features, edge_index=edge_index, edge_attr=bonds_order, y=spectrum2500)


#class NMRShiftDB2(Dataset):
class NMRShiftDB2(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(NMRShiftDB2, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root

    @property
    def raw_file_names(self):
        return ['nmrshiftdb2withsignals.sd']

    @property
    def processed_file_names(self):
        return ['data.pt']

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
        data_list = [create_Data1_from_abs(records_abs, idx) for idx, records_abs in id_abs.items()]
        #print(data_list)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

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