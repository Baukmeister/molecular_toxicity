import os
import os.path as osp
from typing import List
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset, data


class MolecularToxicityDataset(InMemoryDataset):
    def __init__(self, root, graphs: [], labels: [], test=False, transform=None, pre_transform=None,
                 pre_filter=None):

        self.graphs = graphs
        self.toxicity_labels = labels
        self.test = test

        super(MolecularToxicityDataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return "TOX"

    @property
    def processed_file_names(self):
        if self.test:
            return [f'data_test_{i}.pt' for i in range(len(self.graphs))]
        else:
            return [f'data_{i}.pt' for i in range(len(self.graphs))]

    def download(self):
        pass


    def process(self):

        for idx, graph in tqdm(enumerate(self.graphs)):

            graph.y = self.toxicity_labels[idx]
            if self.test:
                torch.save(graph,
                           os.path.join(self.processed_dir,
                                        f'data_test_{idx}.pt'))
            else:
                torch.save(graph,
                           os.path.join(self.processed_dir,
                                        f'data_{idx}.pt'))


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_{idx}.pt'))
        return data
