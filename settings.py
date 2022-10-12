mode = 'Scratch'  # Scratch  # Regression  # Transfer  # PU_predict
file1 = 'data/csv/melting_point_by_enamine_training_.csv'  # data/csv/melting_point_by_enamine_training_.csv  # data/csv/PT_T_endo_max.csv
file2 = 'data/csv/nonPTclean---.csv'
variable1 = 'SMILES'
variable2 = 'SMILES---'
target = 'Melting Point {measured, converted}'  # Melting Point {measured, converted}  # T_endo (K)  # H_endo (kJ/mol)
fptype = 'RDKitDesc'   # 'Mordred','ECFP','Avalon','ErG','RDKitDesc','MACCSKeys','Estate'  Transfer, list
datasize = 22404   # 181385  # 22404  # 88,72,46,37
model_path = './Scratch_models/_mp_epoch200_lr0.001/Scratch_ECFP.h5---' #./Scratch_models/_mp_epoch200_lr0.001/Scratch_Estate.h5#./PU_models/*.pkl
lr = 0.001   ## only Transfer
epochs = 200
number = 5


##### 以下、変数の説明 #####
# mode (str): "Scratch", "Transfer", "PU_learning"
# file1 (str): Dataset file
# file2 (str): Dataset file
# variable1 (str): Explanatory varialbe in file1
# variable2 (str): Explanatory varialbe in file2
# target (str): target variable
# fptype (str): ECFP, Avalon, ErG, MACCSKeys, Estate, RDKitDesc, Mordred
# datasize (int): Datasize of regression data and unlabel data
# model_path (str): Scratch model for TL
# lr (float): learning rate
# epochs (int): iteration of training
# number (int): For iteration