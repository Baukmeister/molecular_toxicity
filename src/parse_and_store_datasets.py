from src.molecule_parsing.molecule_data_handler import MoleculeDataHandler

parser = MoleculeDataHandler()

parser.load_tox21_smiles_list("../data/tox21.csv")
parser.store_smiles_list_in_neo4j()


