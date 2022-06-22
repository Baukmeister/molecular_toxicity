from src.molecule_parsing.simles_parser import SmilesParser

parser = SmilesParser()

parser.load_tox21_smiles_list("../data/tox21.csv")
parser.store_smiles_list_in_neo4j()
print()