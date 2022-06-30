import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm


class MolecularToxicityDataset(InMemoryDataset):
    def __init__(self, root, molecules, filename="TOX", test=False, transform=None, pre_transform=None):

        self.test = test
        self.molecules = molecules
        self.filename = filename

        super(MolecularToxicityDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        if self.test:
            return [f'data_test_{i}.pt' for i in range(len(self.molecules))]
        else:
            return [f'data_{i}.pt' for i in range(len(self.molecules))]

    def download(self):
        pass

    def process(self):
        for index, mol in tqdm(enumerate(self.molecules)):
            # Get node features
            node_feats = self._get_node_features(mol)
            # Get edge features
            edge_feats = self._get_edge_features(mol)
            # Get adjacency info
            edge_index = self._get_adjacency_info(mol)
            # Get labels info
            label = 1  # self._get_labels(mol["HIV_active"])

            # Create data object
            data = Data(x=node_feats,
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=label,
                        smiles=mol["smiles"]
                        )
            if self.test:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_test_{index}.pt'))
            else:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_{index}.pt'))

    def _get_node_features(self, mol):
        """
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """

        all_node_feats = np.asarray(
            [
                mol.node_stores[0]['element'],
                mol.node_stores[0]['charge'].tolist(),
                mol.node_stores[0]['aromatic'].tolist(),
                mol.node_stores[0]['hcount'].tolist(),
            ]
        )

        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mol):
        """
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        all_edge_feats = []

        for bond in mol.GetBonds():
            edge_feats = []
            # Feature 1: Bond type (as double)
            edge_feats.append(bond.GetBondTypeAsDouble())
            # Feature 2: Rings
            edge_feats.append(bond.IsInRing())
            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, mol):
        """
        We could also use rdmolops.GetAdjacencyMatrix(mol)
        but we want to be sure that the order of the indices
        matches the order of the edge features
        """
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

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
