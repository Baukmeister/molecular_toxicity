from src.molecule_parsing.molecule_data_handler import MoleculeDataHandler

parser = MoleculeDataHandler(load_cache=True)
parser.load_molecules_from_neo4j()

graphs = parser.convert_molecules_to_pyG()
print()