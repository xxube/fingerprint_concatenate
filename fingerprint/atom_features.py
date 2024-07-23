import autograd.numpy as np
from rdkit import Chem
from util import one_of_k_encoding, one_of_k_encoding_unk

def calculate_num_atom_features():
    molecule = Chem.MolFromSmiles('CC')
    atom_list = molecule.GetAtoms()
    atom = atom_list[0]
    return len(get_atom_features(atom))

def calculate_num_bond_features():
    simple_molecule = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_molecule)
    return len(get_bond_features(simple_molecule.GetBonds()[0]))

def get_atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al',
            'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
            'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
        ]) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
        one_of_k_encoding_unk(atom.GetFormalCharge(), [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]) +
        [atom.GetIsAromatic()]
    )

def get_bond_features(bond):
    bt = bond.GetBondType()
    return np.array([
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ])
