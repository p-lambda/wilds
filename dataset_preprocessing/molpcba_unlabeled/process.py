import numpy as np
from wilds import get_dataset
from rdkit.Chem import AllChem
from rdkit import Chem
from tqdm import tqdm
import pandas as pd
import os
import torch

def compute_pcba_fingerprint():
    '''
        Compute the fingerprint features for molpcba molecules.
    '''
    os.makedirs('processed_fp', exist_ok = True)

    pcba_dataset = get_dataset(dataset = 'ogb-molpcba')
    smiles_list = pd.read_csv('data/ogbg_molpcba/mapping/mol.csv.gz')['smiles'].tolist()
    x_list = []
    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        x = np.array(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)), dtype=np.int8)
        x_list.append(x)

    x = np.stack(x_list)

    np.save('processed_fp/molpcba.npy', x)


def jaccard_similarity(vec, mat):
    AND = vec * mat
    OR = (vec + mat) > 0
    denom = np.sum(OR, axis = 1)
    nom = np.sum(AND, axis = 1)

    denom[denom==0] = 1
    return nom / denom


def assign_to_group():
    '''
        Assign unlabeled pubchem molecules to scaffold groups of molpcba.
    '''
    smiles_list = pd.read_csv('molpcba_unlabeled/mapping/unlabeled_smiles.csv', header = None)[0].tolist()
    
    x_pcba = np.load('processed_fp/molpcba.npy')
    print(x_pcba.shape)
    print((x_pcba > 1).sum())
    scaffold_group = np.load('data/ogbg_molpcba/raw/scaffold_group.npy')

    # ground-truth assignment
    group_assignment = np.load('molpcba_unlabeled/processed/group_assignment.npy')

    for i, smiles in tqdm(enumerate(smiles_list), total = len(smiles_list)):
        mol = Chem.MolFromSmiles(smiles)
        x = np.array(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)), dtype=np.int8)
        sim = jaccard_similarity(x, x_pcba)
        
        max_idx = np.argmax(sim)
        a = scaffold_group[max_idx]
        b = group_assignment[i]

        print(a, b)
        assert a == b # make sure they coincide each other
        

def test_jaccard():
    vec = np.random.randn(1024) > 0
    mat = np.random.randn(1000, 1024)
    mat[0] = vec

    sim = jaccard_similarity(vec, mat)
    print(sim)


if __name__ == '__main__':
    compute_pcba_fingerprint()
    assign_to_group()

