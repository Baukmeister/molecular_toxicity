

# https://github.com/pckroon/pysmiles
# https://neo4j.com/developer/python/
import os
from typing import List
from pysmiles import read_smiles
import pandas as pd
import networkx as nx

class SmilesParser:

    def __init__(self):
        self.molecules: List[nx.Graph] = None

    def parse_smiles_molecule(self, smiles: str, explicit_hydrogen=True):

        return read_smiles(smiles, explicit_hydrogen)

    def parse_smiles_list(self, smiles_list: List[str]):

        return [self.parse_smiles_molecule(smiles) for smiles in smiles_list]

    def store_smiles_list_in_neo4j(self, smiles_list: List[str], neo4j_con):
        pass

    def load_tox21_smiles_list(self, file_path: str) -> None:

        if not os.path.isfile(file_path):
            raise ValueError(f"Path {file_path} is invalid!")

        tox_21_data = pd.read_csv(file_path)
        smiles_codes = list(tox_21_data['smiles'])
        self.molecules = self.parse_smiles_list(smiles_codes)