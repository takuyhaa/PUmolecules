# PUmolecules
![Abstract_2-01](https://user-images.githubusercontent.com/86113952/223691202-abf489aa-d220-45a0-a2a2-61ed3fca5979.png)

## Dataset
PositiveDataset_public_YYMMDD.xlsx contains the information of solid-solid phase transition we manually collected as of the date.  
Each variable means following.  
`ID`: Identifier  
`SMILES`: Simplified molecule input line entry system  
`CCDC`: CCDC number (Left and right columns store numbers assigned to phase before and after the transition, respectively.)  
`Phase`: Phase name assigned in the literture  
`T_endo (K)`: Temperature of endothermic transition  
`dH_endo (kJ/mol)`: Enthalpy of endothermic transition  
`T_exo (K)`: Temperature of exothermic transition  
`dH_exo (kJ/mol)`: Enthalpy of exothermic transition  
`T_melt (K)`: Melting point  
`dH_melt (kJ/mol)`: Enthalpy of melt  
`Acquisition`: Whether we complemented data (0 is False, 1 is True)  
`Acquisition memo`: How to complement the data when `Acquisition` is 1

## Files
Each executable file has the following roles.  
`main.py`: Work mainly  
`train.py`: Train the machine learning models  
`process.py`: Calculate molecular descriptors  
`settings.py`: Input the setting  

## Setting arguments
The types and meanings of the arguments of `settings.py` correspond to the following, respectively.　　
- mode (str): "PU_learning", "PU_predict", "Regression", "Scratch", "Transfer"   
- file1 (str): Dataset file for positive data in PU learning, and regression  
- file2 (str): Dataset file for unlabeled data in PU learning 
- variable1 (str): Explanatory varialbe in file1  
- variable2 (str): Explanatory varialbe in file2  
- target (str): target variable  
- fptype (str): "Mordred", "ECFP", "Avalon", "ErG", "RDKit", "MACCSKeys", "Estate"   
- model_path (str): Trained models for "PU_learning" and "Transfer" modes  
- lr (float): learning rate  
- epochs (int): iteration of training  
- number (int): For iteration  

## Default Output Folder 
`./PU_models/`: Folder where trained models are stored in "PU_learning" mode  
`./PU_results/`: Folder where classification results are stored in "PU_learning" mode  
`./PU_preds/`: Folder where prediction results are storesd in "PU_predict" mode  
`./Regression_results/`: Folder where regression results are stored in "Regression" mode    
`./TL_models/`: Folder where fune-tuned models are stored in "Transfer" mode 
`./TL_results/`: Folder where regression results in "Transfer" mode  
`./Scratch_models/`: Folder where scratch models are stored in "Scratch" mode  
`./Scratch_results/`: Folder where regression results are stored in "Scratch" mode

## Examples
To run:  
`python main.py`

By this code, `main.py` reads the argument of `setting.py` and start the calculation of molecular descriptors and training models.  
Please note that the preparation of unlabeled dataset for PU learning requires Cambridge Structural Database (CSD) license (https://www.ccdc.cam.ac.uk/solutions/csd-licence/). 
