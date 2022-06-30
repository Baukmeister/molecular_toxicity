from src.molecule_parsing.molecule_data_handler import MoleculeDataHandler
from src.train import *

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')

parser = MoleculeDataHandler(load_cache=True)
parser.load_molecules_from_neo4j()

graphs = parser.convert_molecules_to_pyG()
tox_dataset = MolecularToxicityDataset(root="../data/torch_data_set", molecules=graphs)

print()