from rdkit import Chem
from rdkit.Chem import AllChem
from collections import deque

def get_molecule_smiles(mol_obj):
    return Chem.MolToSmiles(mol_obj)

def get_double_bond_atoms(mol_obj):
    pattern = Chem.MolFromSmarts('C=C')
    double_bond_list = mol_obj.GetSubstructMatches(pattern)
    return double_bond_list

def get_adjacent_halogen_atoms(mol_obj):
    pattern = Chem.MolFromSmarts('CC([Cl,Br,I])C')
    adjacent_hal_atoms = mol_obj.GetSubstructMatches(pattern)
    return (adjacent_hal_atoms[0][0], adjacent_hal_atoms[0][3])

def calculate_bulkiness(mol_obj, target_atom_id, pair_atom_id):
    radius = 3 
    target_atom = mol_obj.GetAtomWithIdx(target_atom_id)
    target_neighbors = [neighbor.GetIdx() for neighbor in target_atom.GetNeighbors()]
    neighbor_queue = deque(target_neighbors)
    neighbor_list = target_neighbors.copy()
    bulkiness_sum = 0

    while neighbor_queue:
        atom_id = neighbor_queue.popleft()
        if atom_id in (pair_atom_id, target_atom_id):
            neighbor_list.remove(atom_id)
            continue
        current_atom = mol_obj.GetAtomWithIdx(atom_id)
        new_neighbors = [neighbor.GetIdx() for neighbor in current_atom.GetNeighbors() if neighbor.GetIdx() not in neighbor_list]
        neighbor_list.extend(new_neighbors)
        neighbor_queue.extend(new_neighbors)
    
    for atom_id in neighbor_list:
        current_atom = mol_obj.GetAtomWithIdx(atom_id)
        bulkiness_sum += current_atom.GetMass()

    return neighbor_list, bulkiness_sum

def find_bulkier_atom(mol_obj, atom1_id, atom2_id):
    atom1_neighbors, atom1_bulkiness = calculate_bulkiness(mol_obj, atom1_id, atom2_id)
    atom2_neighbors, atom2_bulkiness = calculate_bulkiness(mol_obj, atom2_id, atom1_id)

    if len(atom1_neighbors) > len(atom2_neighbors):
        return atom1_id
    elif len(atom1_neighbors) == len(atom2_neighbors):
        if atom1_bulkiness >= atom2_bulkiness:
            return atom1_id
    return atom2_id

def mark_bulkier_atom(mol_obj, atom1_id, atom2_id):
    bulkier_atom_id = find_bulkier_atom(mol_obj, atom1_id, atom2_id)
    bulkier_atom = mol_obj.GetAtomWithIdx(bulkier_atom_id)
    bulkier_atom.SetAtomicNum(14)
    return mol_obj

def find_markovnikov_atom(mol_obj, atom1_id, atom2_id):
    atom1 = mol_obj.GetAtomWithIdx(atom1_id)
    atom2 = mol_obj.GetAtomWithIdx(atom2_id)
    if atom1.GetTotalNumHs() > atom2.GetTotalNumHs():
        return atom2_id
    return atom1_id

def mark_markovnikov_atom(mol_obj, atom1_id, atom2_id):
    markovnikov_atom_id = find_markovnikov_atom(mol_obj, atom1_id, atom2_id)
    markovnikov_atom = mol_obj.GetAtomWithIdx(markovnikov_atom_id)
    markovnikov_atom.SetAtomicNum(14)
    return mol_obj

def test_bulkiness():
    test_molecule = Chem.MolFromSmiles('C(C)C(C)=CC') 
    double_bonds = get_double_bond_atoms(test_molecule)
    atom1_id, atom2_id = double_bonds[0]

    print(f'Bulkier atom: {find_bulkier_atom(test_molecule, atom1_id, atom2_id)}')
    return 'done'

def test_mark_label():
    test_molecule = Chem.MolFromSmiles('C(C)CC=C(C)C') 
    pattern = Chem.MolFromSmarts('C=C')
    atom1_id, atom2_id = test_molecule.GetSubstructMatches(pattern)[0]
    marked_molecule = mark_markovnikov_atom(test_molecule, atom1_id, atom2_id)
    print(Chem.MolToSmiles(marked_molecule))

def test_mark_bulkiness():
    test_molecule = Chem.MolFromSmiles('C(C)CC=CC')
    pattern = Chem.MolFromSmarts('C=C')
    atom1_id, atom2_id = test_molecule.GetSubstructMatches(pattern)[0]
    marked_molecule = mark_bulkier_atom(test_molecule, atom1_id, atom2_id)
    print(Chem.MolToSmiles(marked_molecule))

def test_epoxide_reaction():
    epox_alkene = AllChem.ReactionFromSmarts('[C:1]=[C:2].[C:3][C:4](=[O:5])[O:6][O:7]>>[C:1]1[O:7][C:2]1.[C:3][C:4](=[O:6])[O:5]')
    epox_opening = AllChem.ReactionFromSmarts('[C:1]1[O:7][C:2]1>>[C:1]([O:7])[C:2]O')
    test_molecule = Chem.MolFromSmiles('C(C)C(C)=CC') 
    pattern = Chem.MolFromSmarts('CC(=O)OO')
    print(Chem.MolFromSmiles('CCC(=O)OO').HasSubstructMatch(pattern))
    print(test_molecule.HasSubstructMatch(Chem.MolFromSmarts('C=C')))
    products = epox_alkene.RunReactants((test_molecule, Chem.MolFromSmiles('CCC(=O)OO')))

    print(f'Number of products: {len(products[0])}')
    print(Chem.MolToSmiles(products[0][0]))
    print(Chem.MolToSmiles(products[0][1]))

    opened_products = epox_opening.RunReactants((products[0][0],))
    print(f'Number of opened products: {len(opened_products)}')
    print(Chem.MolToSmiles(opened_products[0][0]))

    return 'done'

def test_osmium_ozone_h2():
    osmium_ox = AllChem.ReactionFromSmarts('[C:1]=[C:2]>>[C:1](O)[C:2](O)')
    ozon_alkene = AllChem.ReactionFromSmarts('[C:1]=[C:2]>>[C:1](=O).[C:2](=O)')
    h2_alkene_add = AllChem.ReactionFromSmarts('[C:1]=[C:2]>>[C:1][C:2]')
    reactions = {'Os': osmium_ox, 'Oz': ozon_alkene, 'H2': h2_alkene_add}
    test_molecule = Chem.MolFromSmiles('C(C)C(C)=CC') 

    for rxn_name, reaction in reactions.items():
        print(rxn_name)
        products = reaction.RunReactants((test_molecule, ))
        for product in products[0]:
            print(Chem.MolToSmiles(product))

    return 'done'

def test_properities_of_alcohol():
    hoh_alkene_add = AllChem.ReactionFromSmarts('[Si:1]=[C:2].[OH2:3]>>[C:1]([O:3])[C:2]')
    test_molecule = Chem.MolFromSmiles('C(C)C(C)=CC')
    pattern = Chem.MolFromSmarts('C=C')
    atom1_id, atom2_id = test_molecule.GetSubstructMatches(pattern)[0]
    marked_molecule = mark_bulkier_atom(test_m
