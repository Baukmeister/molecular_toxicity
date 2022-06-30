import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm


class MolecularToxicityDataset(InMemoryDataset):

    def __init__(self, root, molecule_graphs, test=False, transform=None, pre_transform=None):
        self.molecule_graphs = molecule_graphs
        super(MolecularToxicityDataset, self).__init__(root, transform, pre_transform)
        self.data = molecule_graphs


    @property
    def raw_file_names(self):
        return ['TOX']

    @property
    def processed_file_names(self):

        self.data = self.molecule_graphs
        return [f'data_{i}.pt' for i in range(len(self.data))]

    def download(self):
       pass

    def process(self):
        # Store the processed data
        self.data = self.molecule_graphs
        data, slices = self.collate(self.data)
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_{idx}.pt'))
        return data