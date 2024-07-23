import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import autograd.numpy as np

def convert_stringlist_to_intarray(string_array):
    return np.array([list(bitstring) for bitstring in string_array], dtype=int)

def generate_fingerprint(smile, fp_length, fp_radius):
    molecule = Chem.MolFromSmiles(smile)
    return (AllChem.GetMorganFingerprintAsBitVect(molecule, fp_radius, nBits=fp_length)).ToBitString()

def convert_smiles_to_fingerprints(smiles_list, fp_length, fp_radius):
    fingerprints = [generate_fingerprint(smile, fp_length, fp_radius) for smile in smiles_list]
    return convert_stringlist_to_intarray(np.array(fingerprints))
