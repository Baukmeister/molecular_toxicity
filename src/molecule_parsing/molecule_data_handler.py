# https://github.com/pckroon/pysmiles
# https://neo4j.com/developer/python/
import json
import os
import pickle
import random
import uuid
from typing import List

from neo4j import GraphDatabase
from pysmiles import read_smiles
import pandas as pd
import networkx as nx
from tqdm import tqdm
from torch_geometric.utils import convert
from torch_geometric.data import dataset


class MoleculeDataHandler:

    def __init__(self, load_cache=True):
        self.molecules: List[nx.Graph] = []
        self.toxicity_dict = {}
        self.database_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "postgres"))
        self.molecule_cache = {}
        self.neo4j_cache = {}
        self.neo4j_cache_location = "./neo4j_graph_cache.pkl"
        self.molecule_cache_location = "./molecule_cache.pkl"
        self.toxicity_dict_location = "./toxicity_dict.pkl"

        if os.path.isfile(self.toxicity_dict_location):
            with open(self.toxicity_dict_location, 'rb') as f:
                print("Loading toxicity dict...")
                self.toxicity_dict = pickle.load(f)
        else:
            print("Toxicity dict file is missing. Make sure to properly load the data first!")

        if load_cache:
            if os.path.isfile(self.molecule_cache_location):
                with open(self.molecule_cache_location, 'rb') as f:
                    print("Using cached molecules...")
                    self.molecule_cache = pickle.load(f)
                    self.molecules = self.parse_smiles_list([smiles for smiles in self.molecule_cache.keys()])

            if os.path.isfile(self.neo4j_cache_location):
                with open(self.neo4j_cache_location, 'rb') as f:
                    print("Using cached neo4j molecules...")
                    self.neo4j_cache = pickle.load(f)

    def parse_smiles_molecule(self, smiles: str, explicit_hydrogen=False):

        if smiles in self.molecule_cache.keys():
            molecule = self.molecule_cache[smiles]
        else:
            molecule = read_smiles(smiles, explicit_hydrogen)
            self.molecule_cache[smiles] = molecule

        for idx, data in molecule.nodes(data=True):
            for atomic_property in ["stereo", "isotope"]:
                if atomic_property in data.keys():
                    del molecule.nodes[idx][atomic_property]

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
            with open(self.molecule_cache_location, 'wb') as f:
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
                                      f"atomic_element: '{atom_data['element']}'," \
                                      f"charge: '{atom_data['charge']}'," \
                                      f"hcount: '{atom_data['hcount']}'," \
                                      f"aromatic: '{atom_data['aromatic']}'" \
                                      f"}}" \
                                      f")"
                transaction_execution_commands.append(atom_create_command)

            for start, end, edge_data in molecule.edges(data=True):
                start_id = str(molecule_smiles) + '-' + str(start)
                end_id = str(molecule_smiles) + '-' + str(end)

                # the direction in this relationship is required because of neo4j restrictions, but is ignored later on
                edge_create_command = f"match (a:Atom), (b:Atom) where a.atom_id = '{start_id}' and b.atom_id = '{end_id}'" \
                                      f"create (a)-[r:Bond {{ order: '{edge_data['order']}' }}]->(b)"

                transaction_execution_commands.append(edge_create_command)

        self._execute_neo4j_transactions(transaction_execution_commands)

    def load_tox21_smiles_list(self, file_path: str) -> None:

        if not os.path.isfile(file_path):
            raise ValueError(f"Path {file_path} is invalid!")

        tox_21_data = pd.read_csv(file_path)

        toxicity_cols = [col_name for col_name in tox_21_data.columns if col_name not in ["mol_id", "smiles"] ]
        toxicity_values = tox_21_data[toxicity_cols]
        is_toxic = (toxicity_values.sum(axis=1) >= 1).tolist()
        smiles_codes = list(tox_21_data['smiles'])
        self.toxicity_dict = dict(zip(smiles_codes, is_toxic))

        with open(self.toxicity_dict_location, 'wb') as f:
            pickle.dump(self.toxicity_dict, f)

        self.molecules = self.parse_smiles_list(smiles_codes)

    def load_molecules_from_neo4j(self, cache_results=True) -> List[nx.Graph]:

        molecules = []

        with self.database_driver.session() as session:
            smiles_list = session.run("MATCH (a:Atom) RETURN DISTINCT a.molecule_smiles as smiles").to_df()[
                "smiles"].tolist()

            for smiles_string in tqdm(smiles_list):

                if smiles_string in self.neo4j_cache.keys():
                    G = self.neo4j_cache[smiles_string]
                else:
                    results = session.run("Match (a:Atom)-[b]-(x:Atom) where a.molecule_smiles = $smiles return a,b",
                                          smiles=smiles_string)

                    G = nx.Graph()

                    nodes = list(results.graph()._nodes.values())
                    for node in nodes:
                        G.add_node(node.id, **self._convert_node_properties_to_basic_type(node._properties))

                    rels = list(results.graph()._relationships.values())
                    for rel in rels:
                        G.add_edge(rel.start_node.id, rel.end_node.id, key=rel.id, type=rel.type,
                                   **self._convert_edge_properties_to_basic_type(rel._properties))

                    self.neo4j_cache[smiles_string] = G

                molecules.append((smiles_string, G))

            self.molecules = molecules

            if cache_results:
                with open(self.neo4j_cache_location, 'wb') as f:
                    pickle.dump(self.neo4j_cache, f)

            print(f"Completed loading molecules from neo4j database!")

    def _convert_node_properties_to_basic_type(self, node_props):


        output_dict = {
            "atomic_element": int("".join([str(ord(c)) for c in node_props["atomic_element"]])),
            "charge": int(node_props["charge"]),
            "aromatic": bool(node_props["aromatic"]),
            "hcount": int(node_props["hcount"]),
        }

        return output_dict

    def _convert_edge_properties_to_basic_type(self, edge_props):

        output_dict = {
            "order": float(edge_props["order"])
        }

        return output_dict

    def convert_molecules_to_pyG(self, single_molecule=None):
        py_torch_graphs = []

        if single_molecule:

            for _, node in single_molecule.nodes(data=True):
                node["atomic_element"] = int("".join([str(ord(c)) for c in node["element"]]))

            pyG_graph = convert.from_networkx(single_molecule,
                                              group_node_attrs=["atomic_element", "charge", "aromatic", "hcount"],
                                              group_edge_attrs=["order"])
            return pyG_graph
        else:
            for smiles, molecule in tqdm(self.molecules):
                if isinstance(molecule, nx.Graph) and len(molecule.nodes) > 0:

                    pyG_graph = convert.from_networkx(molecule,
                                                      group_node_attrs=["atomic_element", "charge", "aromatic", "hcount"],
                                                      group_edge_attrs=["order"])

                    pyG_graph.y = 1 if self.toxicity_dict[smiles] else 0
                    py_torch_graphs.append(pyG_graph)


            return py_torch_graphs

    def _execute_neo4j_transactions(self, transaction_execution_commands: List[str]):
        print(f"Executing {len(transaction_execution_commands)} neo4j commands ...")
        with self.database_driver.session() as session:
            for idx, transaction in enumerate(transaction_execution_commands):

                if idx % 100 == 0:
                    print(
                        f"{idx}/{len(transaction_execution_commands)} ({round((idx / len(transaction_execution_commands)) * 100, 2)}%) of commands done ...")
                session.run(transaction)

        print(f"Completed writing to neo4j database!")
