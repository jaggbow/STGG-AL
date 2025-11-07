import argparse
import pickle
import sys
from pathlib import Path

from rdkit import Chem
from tqdm import tqdm

FRAGMENTS_DIR = Path("../resource/fragments.pkl")
MAX_RING_SIZE = 6


def get_largest_fused_ring_system(smiles):
    # Extract the largest fused ring system from a given molecule.
    mol = Chem.MolFromSmiles(smiles)
    ri = mol.GetRingInfo()
    if not ri.NumRings():
        return None  # No rings in the molecule

    # Get ring bonds and atoms
    ring_bonds = ri.BondRings()
    ring_atoms = ri.AtomRings()

    # Create a graph where nodes are rings and edges represent shared atoms or bonds
    ring_graph = {}
    for i in range(len(ring_atoms)):
        ring_graph[i] = set()
        for j in range(len(ring_atoms)):
            if i != j:
                # Check if rings share bonds or atoms
                if set(ring_bonds[i]).intersection(ring_bonds[j]) or set(
                    ring_atoms[i]
                ).intersection(ring_atoms[j]):
                    ring_graph[i].add(j)
    # Find the largest connected component of the ring graph

    visited = set()
    largest_component = set()

    def dfs(node, component):
        visited.add(node)
        component.add(node)
        for neighbor in ring_graph[node]:
            if neighbor not in visited:
                dfs(neighbor, component)

    for node in ring_graph:
        if node not in visited:
            current_component = set()
            dfs(node, current_component)
            if len(current_component) > len(largest_component):
                largest_component = current_component

    # Collect atoms and bonds from the largest fused ring component

    largest_ring_atoms = set()
    largest_ring_bonds = set()
    for ring_idx in largest_component:
        largest_ring_atoms.update(ring_atoms[ring_idx])
        largest_ring_bonds.update(ring_bonds[ring_idx])

    # Create a new RWMol containing only the largest fused ring system
    rw_mol = Chem.RWMol()
    atom_mapping = {}  # Map old atom indices to new ones
    for atom_idx in largest_ring_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        new_idx = rw_mol.AddAtom(atom)
        atom_mapping[atom_idx] = new_idx

    for bond_idx in largest_ring_bonds:
        bond = mol.GetBondWithIdx(bond_idx)
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        if begin_idx in atom_mapping and end_idx in atom_mapping:
            rw_mol.AddBond(
                atom_mapping[begin_idx], atom_mapping[end_idx], bond.GetBondType()
            )
    return Chem.MolToSmiles(rw_mol.GetMol())


FRAGMENTS = pickle.load(open(FRAGMENTS_DIR, "rb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_path", type=str)
    args = parser.parse_args()

    smiles_path = Path(args.smiles_path)

    existing_smiles = []
    for file in smiles_path.parent.glob("*.pkl"):
        if file != smiles_path:
            old_data = pickle.load(open(file, "rb"))
            if isinstance(old_data, dict) and "smiles" in old_data:
                existing_smiles += old_data["smiles"]
    data = pickle.load(open(smiles_path, "rb"))
    smiles_list = data["smiles"]

    novel_smiles = [smi for smi in smiles_list if smi not in existing_smiles]
    fragment_presence = []
    max_ring_size = []
    nice_smiles = []

    for smiles in tqdm(novel_smiles):
        mol = Chem.MolFromSmiles(smiles)
        core_structure = get_largest_fused_ring_system(smiles)
        fragments_in_mol = core_structure in FRAGMENTS
        fragment_presence.append(int(fragments_in_mol))

        ri = mol.GetRingInfo()
        max_ring_size_ = 0
        if ri.NumRings():
            for ring in ri.AtomRings():
                ring_size = len(ring)
                if ring_size > max_ring_size_:
                    max_ring_size_ = ring_size

        max_ring_size.append(max_ring_size_)

        if (
            fragments_in_mol
            and max_ring_size_ <= MAX_RING_SIZE
        ):
            nice_smiles.append(smiles)

    if len(novel_smiles) > 0:
        print(
            f"There are {len(nice_smiles)} molecules that passed the size and fragment presence check and they represent {(100 * len(nice_smiles) / len(novel_smiles)):.2f} % of the dataset."
        )
    else:
        print("There are no novel smiles.")
        sys.exit()
    data["smiles"] = nice_smiles
    data["statistics"]["num_efficient"] = len(novel_smiles)
    data["statistics"]["num_pass_fragments"] = len(nice_smiles)

    with open(smiles_path, "wb") as f:
        pickle.dump(data, f)
