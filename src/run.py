from src.molecule_parsing.simles_parser import SmilesParser

parser = SmilesParser()

parser.load_tox21_smiles_list("../data/tox21.csv")
print()