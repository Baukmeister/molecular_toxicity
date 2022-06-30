from src.molecule_parsing.molecule_data_handler import MoleculeDataHandler
from src.train import *
parser = MoleculeDataHandler(load_cache=True)
parser.load_molecules_from_neo4j()

graphs = parser.convert_molecules_to_pyG()

run_training(graphs, [1 for i in range(len(graphs))])

print()