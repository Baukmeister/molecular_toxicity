# https://github.com/pckroon/pysmiles
# https://neo4j.com/developer/python/
import json
import os
import pickle
import uuid
from typing import List

from neo4j import GraphDatabase
from pysmiles import read_smiles
import pandas as pd
import networkx as nx
from tqdm import tqdm


class MoleculeDataHandler:

    def __init__(self, load_cache=True):
        self.molecules: List[nx.Graph] = []
        self.database_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "postgres"))
        self.molecule_cache = {}
        self.cache_location = "./molecule_cache.pkl"
        if load_cache:
            if os.path.isfile(self.cache_location):
                with open(self.cache_location, 'rb') as f:
                    print("Using cached molecules...")
                    self.molecule_cache = pickle.load(f)

    def parse_smiles_molecule(self, smiles: str, explicit_hydrogen=False):

        if smiles in self.molecule_cache.keys():
            molecule = self.molecule_cache[smiles]
        else:
            molecule = read_smiles(smiles, explicit_hydrogen)
            self.molecule_cache[smiles] = molecule

        return (
            smiles,
            molecule
        )

    def parse_smiles_list(self, smiles_list: List[str], cache_result=True):

        molecule_list = [self.parse_smiles_molecule(smiles) for smiles in smiles_list]

        for smiles, molecule in zip(smiles_list, molecule_list):

            if smiles not in self.molecule_cache.keys():
                self.molecule_cache[smiles] = molecule

        if cache_result:
            with open(self.cache_location, 'wb') as f:
                pickle.dump(self.molecule_cache, f)

        return molecule_list

    def store_smiles_list_in_neo4j(self):

        transaction_execution_commands = []
        print(f"Creating CREATE statements for {len(self.molecules)} molecules ...")
        for molecule_smiles, molecule in self.molecules:
            for idx, atom_data in molecule.nodes(data=True):
                # Add atoms
                atom_create_command = f"create (t:Atom " \
                                      f"{{" \
                                      f"atom_id: '{str(molecule_smiles) + '-' + str(idx)}'," \
                                      f"molecule_smiles: '{molecule_smiles}'," \
                                      f"element: '{atom_data['element']}'" \
                                      f"}}" \
                                      f")"
                transaction_execution_commands.append(atom_create_command)

            for start, end in molecule.edges:
                start_id = str(molecule_smiles) + '-' + str(start)
                end_id = str(molecule_smiles) + '-' + str(end)

                # the direction in this relationship is required because of neo4j restrictions, but is ignored later on
                edge_create_command = f"match (a:Atom), (b:Atom) where a.atom_id = '{start_id}' and b.atom_id = '{end_id}'" \
                                      f"create (a)-[r:Bond]->(b)"

                transaction_execution_commands.append(edge_create_command)

        self._execute_neo4j_transactions(transaction_execution_commands)

    def load_tox21_smiles_list(self, file_path: str) -> None:

        if not os.path.isfile(file_path):
            raise ValueError(f"Path {file_path} is invalid!")

        tox_21_data = pd.read_csv(file_path)
        smiles_codes = list(tox_21_data['smiles'])
        self.molecules = self.parse_smiles_list(smiles_codes)

    def load_molecules_from_neo4j(self) -> List[nx.Graph]:

        molecules = []

        with self.database_driver.session() as session:
            smiles_list = session.run("MATCH (a:Atom) RETURN DISTINCT a.molecule_smiles as smiles").to_df()["smiles"].tolist()

            for smiles_string in tqdm(smiles_list):

                results = session.run("Match (a:Atom)-[b]-(x:Atom) where a.molecule_smiles = $smiles return a,b", smiles=smiles_string)

                G = nx.MultiDiGraph()

                nodes = list(results.graph()._nodes.values())
                for node in nodes:
                    G.add_node(node.id, labels=node._labels, properties=node._properties)

                rels = list(results.graph()._relationships.values())
                for rel in rels:
                    G.add_edge(rel.start_node.id, rel.end_node.id, key=rel.id, type=rel.type, properties=rel._properties)

                molecules.append(G)

            self.molecules = molecules
            print(f"Completed loading molecules from neo4j database!")

    def _execute_neo4j_transactions(self, transaction_execution_commands: List[str]):
        print(f"Executing {len(transaction_execution_commands)} neo4j commands ...")
        with self.database_driver.session() as session:
            for idx, transaction in enumerate(transaction_execution_commands):


                if idx % 100 == 0:
                    print(
                        f"{idx}/{len(transaction_execution_commands)} ({round((idx / len(transaction_execution_commands)) * 100, 2) }%) of commands done ...")

                session.run(transaction)

        print(f"Completed writing to neo4j database!")
