from src.dataset import MolecularToxicityDataset
from src.molecule_parsing.molecule_data_handler import MoleculeDataHandler
from torch_geometric.loader import DataLoader, ImbalancedSampler
from src.train import *

from torch_geometric.datasets import Planetoid
planetoid = Planetoid(root='/tmp/Cora', name='Cora')


parser = MoleculeDataHandler(load_cache=True)
parser.load_molecules_from_neo4j()
graphs = parser.convert_molecules_to_pyG()

tox_dataset = MolecularToxicityDataset(root="../data/torch_data_set", molecule_graphs=graphs)
split_point = 2000
train_dataset = tox_dataset.data[:split_point]
test_dataset = tox_dataset.data[split_point:]

train_sampler = ImbalancedSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

best_test_acc = 0
for epoch in range(1, 400):

    train(train_loader)
    train_acc = test(train_loader, best_test_acc=None)
    test_acc = test(test_loader, best_test_acc)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

