from src.dataset import MolecularToxicityDataset
from src.molecule_parsing.molecule_data_handler import MoleculeDataHandler
from torch_geometric.loader import DataLoader
from src.train import *

from torch_geometric.datasets import Planetoid
planetoid = Planetoid(root='/tmp/Cora', name='Cora')


parser = MoleculeDataHandler(load_cache=True)
parser.load_molecules_from_neo4j()

graphs = parser.convert_molecules_to_pyG()
tox_dataset = MolecularToxicityDataset(root="../data/torch_data_set", molecule_graphs=graphs)

train_dataset = tox_dataset.data[:200]
test_dataset = tox_dataset.data[200:]

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


for epoch in range(1, 171):
    train(train_loader)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
