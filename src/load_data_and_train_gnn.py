from src.molecule_parsing.molecule_data_handler import MoleculeDataHandler

parser = MoleculeDataHandler(load_cache=False)

parser.load_molecules_from_neo4j()