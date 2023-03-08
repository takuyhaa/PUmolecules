import numpy as np
import pandas as pd
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Draw,  ReducedGraphs, MACCSkeys
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.EState import Fingerprinter
from mordred import Calculator, descriptors

descriptor_names = [descriptor_name[0] for descriptor_name in Descriptors._descList]
descriptor_calculation = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

## Two files -> X,y
def makedata_cls(file1, file2, variable1, variable2, target, fptype):
    df1 = pd.read_csv(file1, index_col=0)
    df1[target] = 1
    df2 = pd.read_csv(file2, index_col=0)
    df2[target] = 0

    if fptype == "Mordred":
        X1 = Mol2Mordred(df1[variable1], ignore_3D=True)
        X2 = Mol2Mordred(df2[variable2], ignore_3D=True)
        X = pd.concat([X1, X2])
        X = X.dropna(axis=1)
        X = X.select_dtypes("number")
    else:
        X1 = [Mol2Vec(smiles, fptype=fptype) for smiles in df1[variable1]]
        X2 = [Mol2Vec(smiles, fptype=fptype) for smiles in df2[variable2]]
        X = X1 + X2
        X = pd.DataFrame(X)
        X = X.dropna(axis=1)
        X = X.select_dtypes("number")
    X = np.array(X)
    y1 = df1[target].tolist()
    y2 = df2[target].tolist()
    y = y1 + y2
    y = pd.DataFrame(y)
    y = np.array(y).flatten()
    return X, y

## One file -> X,y
def makedata_reg(file1, variable1, target, datasize, fptype):
    df = pd.read_csv(file1)  #, index_col=0)
#     df = df.sample(datasize)

    if fptype == "Mordred":
        X = Mol2Mordred(df[variable1], ignore_3D=True)
        X = X.dropna(axis=1)
        X = X.select_dtypes("number")
    else:
        X = [Mol2Vec(smiles, fptype=fptype) for smiles in df[variable1]]
        X = pd.DataFrame(X)
        X = X.dropna(axis=1)
        X = X.select_dtypes("number")
    X = np.array(X)
    y = df[target]
    y = np.array(y).flatten()
    return X, y


## One file -> X,y (dataframe for FI)
def makedata_reg_fi(file1, variable1, target, fptype, datasize, shuffle=False, random_state=0):
    df = pd.read_csv(file1)#, index_col=0)
#     df = df.sample(datasize)
    if fptype == "Mordred":
        X = Mol2Mordred(df[variable1], ignore_3D=True)
    else:
        X = [Mol2Vec(smiles, fptype=fptype) for smiles in df[variable1]]
        X = pd.DataFrame(X)
    X = X.dropna(axis=1)
    X = X.select_dtypes("number")
    X = X.reset_index(drop=True)
    
    if shuffle==True:
        y = df[target].sample(frac=1, random_state=random_state)
    else:
        y = np.array(df[target])
        
    return X, y


## One file -> smiles, X
def makedata_pred(file1, variable1, fptype):
    df = pd.read_csv(file1)
    
    if fptype == "Mordred":
        X = Mol2Mordred(df[variable1], ignore_3D=True)
        X = X.dropna(axis=1)
        X = X.select_dtypes("number")
    else:
        X = [Mol2Vec(smiles, fptype=fptype) for smiles in df[variable1]]
        X = pd.DataFrame(X)
        X = X.dropna(axis=1)
        X = X.select_dtypes("number")
    X = np.array(X)
    return df[variable1], X


## SMILES -> MOL object -> fingerprint
def Mol2Vec(smiles, fptype="ECFP", radius=2, bits = 1024):
    vector = np.zeros((1,))
    mol = Chem.MolFromSmiles(smiles)
    bitI_morgan = {}
    if fptype == "ECFP":
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius, bits), vector)
    elif fptype == "Avalon":
        DataStructs.ConvertToNumpyArray(GetAvalonFP(mol), vector)
    elif fptype == "MACCSKeys":
        DataStructs.ConvertToNumpyArray(AllChem.GetMACCSKeysFingerprint(mol), vector)
    elif fptype == "ErG":
        vector = ReducedGraphs.GetErGFingerprint(mol)
    elif fptype == "Estate":
        vector = Fingerprinter.FingerprintMol(mol)[0]
    elif fptype == "RDKit":
        vector = list(descriptor_calculation.CalcDescriptors(mol))
        for index, value in enumerate(vector):
            if index == 40: # IPC descriptor 
                vector[index] = Descriptors.Ipc(mol, avg=True)
        vector = np.array(vector)
    else:
        raise TypeError()
    return vector

## list -> DataFrame
def Mol2Mordred(smiles_list, ignore_3D=True):
    if ignore_3D == True:
        calc = Calculator(descriptors, ignore_3D=True)
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        mordred = calc.pandas(mols)
    elif ignore_3D == False:
        calc = Calculator(descriptors, ignore_3D=True)
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        mordred = calc.pandas(mols)
    return mordred