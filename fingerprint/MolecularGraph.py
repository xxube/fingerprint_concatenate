import numpy as np
from rdkit.Chem import MolFromSmiles
from features import atom_features, bond_features

degree_levels = [0, 1, 2, 3, 4, 5]

class GraphNode:
    __slots__ = ['node_type', 'features', 'neighbor_nodes', 'rdkit_index']
    
    def __init__(self, node_type, features, rdkit_index):
        self.node_type = node_type
        self.features = features
        self.neighbor_nodes = []
        self.rdkit_index = rdkit_index

    def add_neighbors(self, neighbors):
        for neighbor in neighbors:
            self.neighbor_nodes.append(neighbor)
            neighbor.neighbor_nodes.append(self)

    def get_neighbors(self, node_type):
        return [n for n in self.neighbor_nodes if n.node_type == node_type]

class MolecularGraph:
    def __init__(self):
        self.node_dict = {}  # dict of lists of nodes, keyed by node type

    def create_node(self, node_type, features=None, rdkit_index=None):
        new_node = GraphNode(node_type, features, rdkit_index)
        self.node_dict.setdefault(node_type, []).append(new_node)
        return new_node

    def merge_subgraph(self, subgraph):
        old_nodes = self.node_dict
        new_nodes = subgraph.node_dict
        for node_type in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(node_type, []).extend(new_nodes.get(node_type, []))

    def organize_nodes_by_degree(self, node_type):
        nodes_by_degree = {i: [] for i in degree_levels}
        for node in self.node_dict[node_type]:
            nodes_by_degree[len(node.get_neighbors(node_type))].append(node)

        new_node_list = []
        for degree in degree_levels:
            current_nodes = nodes_by_degree[degree]
            self.node_dict[(node_type, degree)] = current_nodes
            new_node_list.extend(current_nodes)

        self.node_dict[node_type] = new_node_list

    def get_feature_array(self, node_type):
        assert node_type in self.node_dict
        return np.array([node.features for node in self.node_dict[node_type]])

    def get_rdkit_index_array(self):
        return np.array([node.rdkit_index for node in self.node_dict['atom']])

    def get_neighbor_list(self, self_node_type, neighbor_node_type):
        assert self_node_type in self.node_dict and neighbor_node_type in self.node_dict
        neighbor_indices = {n: i for i, n in enumerate(self.node_dict[neighbor_node_type])}
        return [[neighbor_indices[neighbor]
                 for neighbor in self_node.get_neighbors(neighbor_node_type)]
                for self_node in self.node_dict[self_node_type]]

def create_graph_from_smiles(smiles):
    graph = MolecularGraph()
    mol = MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)
    
    atom_nodes = {}
    for atom in mol.GetAtoms():
        new_atom_node = graph.create_node('atom', features=atom_features(atom), rdkit_index=atom.GetIdx())
        atom_nodes[atom.GetIdx()] = new_atom_node

    for bond in mol.GetBonds():
        atom1_node = atom_nodes[bond.GetBeginAtom().GetIdx()]
        atom2_node = atom_nodes[bond.GetEndAtom().GetIdx()]
        new_bond_node = graph.create_node('bond', features=bond_features(bond))
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        atom1_node.add_neighbors((atom2_node,))

    molecule_node = graph.create_node('molecule')
    molecule_node.add_neighbors(graph.node_dict['atom'])
    return graph

def create_graph_from_smiles_tuple(smiles_tuple):
    graph_list = [create_graph_from_smiles(smiles) for smiles in smiles_tuple]
    combined_graph = MolecularGraph()
    for subgraph in graph_list:
        combined_graph.merge_subgraph(subgraph)

    combined_graph.organize_nodes_by_degree('atom')
    return combined_graph




