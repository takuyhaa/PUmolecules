# PUmolecules
![Abstract_1](https://user-images.githubusercontent.com/86113952/195810249-da6dc649-ff67-406c-a786-c297e140d8df.png)

## Dataset
Dataset_public_[date].xlsx contains the information of solid-solid phase transition we manually collected as of the date.  
Each variable means following.  
`ID`: Identifier  
`SMILES`: Simplified molecule input line entry system  
`CCDC`: CCDC number (Left and right columns store numbers assigned to phase before and after the transition, respectively.)  
`Phase`: Phase name assigned in the literture  
`T_endo (K)`: Transition temperature at endotherm  
`dH_endo (kJ/mol)`: Transition enthalpy at endotherm  
`T_exo (K)`: Transition temperature at exotherm  
`dH_exo (kJ/mol)`: Transition enthalpy at exotherm  
`T_melt (K)`: Melting point  
`dH_melt (kJ/mol)`: Enthalpy of fusion  
`Acquisition`: Whether we complemented data (0 is False, 1 is True)  
`Acquisition memo`: How to complement the data when `Acquisition` is 1

## Files
Each executable file has the following roles.  
`main.py`: Work mainly  
`train.py`: Train the machine learning models  
`process.py`: Calculate molecular descriptors  
`settings.py`: Input the setting  

The types and meanings of the arguments of `settings.py` correspond to the following, respectively.　　
- mode (str): "Scratch", "Transfer", "PU_learning"  
- file1 (str): Dataset file  
- file2 (str): Dataset file  
- variable1 (str): Explanatory varialbe in file1  
- variable2 (str): Explanatory varialbe in file2  
- target (str): target variable  
- fptype (str): "Mordred", "ECFP", "Avalon", "ErG", "RDKitDesc", "MACCSKeys", "Estate"   
- datasize (int): Datasize of regression data and unlabel data  
- model_path (str): Scratch model for TL  
- lr (float): learning rate  
- epochs (int): iteration of training  
- number (int): For iteration  

## Folder
Each folder stores following.  
`PU_models`: Binary classification (BC) and Positive-Unlabeled (PU) learning models  
`PU_results`: Classification results by PU models  
`PU_preds`: Screening results by PU models  
`Regression_results`: Regression results by   
`TL_models`: Fune-tuned transfer learning (TL) models  
`TL_results`: Regression results by TL models  
`Scratch_models`: Pre-train model on TL  
`Scratch_results`: Regression results by pre-train models

## Examples
To run:  
`python main.py`

By this code, `main.py` reads the argument of `setting.py` and start the calculation of molecular descriptors and training models.
