from train import *
from process import *
import csv
import settings
import time

def main(mode, file1, file2=None, variable1=None, variable2=None, target=None, fptype=None, datasize=None, model_path=None, lr=None, epochs=None, number=None):
    ################################
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
    ################################
    time_start = time.time()
    
    ##### Choose a mode #####
    ## Scratch mode
    if mode == "Scratch":
        print(f"-----Scratch mode ({fptype})-----")
        X, y = makedata_reg(file1, variable1, target, datasize, fptype)
        result = train(X, y, fptype, epochs)
        folder = './Scratch_results/'
        os.makedirs(folder, exist_ok=True)
        result = pd.DataFrame(result)
#         results.to_csv(f"{folder}{fptype}_datasize{datasize}.csv", index=False)
        result.to_csv(f"{folder}{fptype}_datasize{datasize}.csv", index=False)
        time_end = time.time()
        time_exe = time_end-time_start
        print(f"Scratch done! (Execution time: {int(time_exe)} sec)")
    
    ## Direct regression mode
    if mode == "Regression":
        print(f"-----Regression mode ({fptype})-----")
        X, y = makedata_reg_fi(file1, variable1, target, fptype, datasize)
        folder = './Regression_results_test/'
        os.makedirs(folder, exist_ok=True)
        
        ml_list=['rf', 'nn'] ##
        for ml in ml_list:
            mae, fi, corrs = simple_model(X, y, ml, number)
            
            pd.DataFrame(mae).T.to_csv(f'Regression_results_test/{ml}/{fptype}_mae_all.csv')
            pd.DataFrame(corrs).T.to_csv(f'Regression_results_test/{ml}/{fptype}_corrs_all.csv')
            
            if ml == 'rf':
                fi.to_csv(f'Regression_results_test/{ml}/{fptype}_fi_all.csv')
                fi_mean = pd.DataFrame(fi.mean(axis=1))
                fi_std = pd.DataFrame(fi.std(axis=1))
                fi_all = pd.concat([fi_mean, fi_std], axis=1)
                fi_all = fi_all.set_axis(['mean', 'std'], axis=1)
                fi_s = fi_all.sort_values('mean', ascending=False)
                fi_s.to_csv(f'Regression_results_test/{ml}/{fptype}_fi_summary.csv')
        
        time_end = time.time()
        time_exe = time_end-time_start
        print(f"Regression done! (Execution time: {int(time_exe)} sec)")
    
    ## Transfer mode
    elif mode == "Transfer":
        print(f"-----Transfer mode ({fptype})-----")
        ##
        for fp in fptype:
            model_path = f'./Scratch_models/_mp_epoch200_lr0.001/Scratch_{fp}.h5'
            X, y = makedata_reg(file1, variable1, target, datasize, fptype=fp)
            # repeat number
            for num in range(number):
                name, error_train, error_test, error_test_all, corr_all = TL(X, y, model_path, number=num, lr=lr, epochs=epochs)
                results = pd.DataFrame([name, error_train, error_test]).T
                results = pd.concat([results, error_test_all, corr_all], axis=1)
                results.columns = ["Model", "MAE_train", "MAE_test", "MAE_test1", "MAE_test2", "MAE_test3", "MAE_test4", "MAE_test5", 
                                   "Corr1", "Corr2", "Corr3", "Corr4", "Corr5"]
                folder = './TL_results_test/'
                os.makedirs(folder, exist_ok=True)
                results.to_csv(f"{folder}{fp}_lr{lr}_epochs{epochs}_{num}.csv", index=False)
        ##
        time_end = time.time()
        time_exe = time_end-time_start
        print(f'Transfer done! (Execution time: {int(time_exe)} sec)')
    
    ## PU_learning mode
    elif mode == "PU_learning":
        print(f"-----PU_learning mode ({fptype})-----")
        X, y = makedata_cls(file1, file2, variable1, variable2, target, fptype, datasize)
        models = ["RF-PU", "RF", "NN-PU", "NN", "SVC-PU", "SVC", "GBDT-PU", "GBDT"]
        results=[]
        for model in models:
            result = PU(model, X, y, fptype)
            results.append(result)
        results = pd.DataFrame(results)
        results.columns = ["Model", "acc_ave", "acc_std", "pre_ave", "pre_std", "rec_ave", "rec_std", "f1_ave", "f1_std"]
        folder = './PU_results/'
        os.makedirs(folder, exist_ok=True)
        results.to_csv(f"{folder}{fptype}_{datasize}.csv", index=False)
        time_end = time.time()
        time_exe = time_end-time_start
        print(f'PU_learning done! (Execution time: {int(time_exe)} sec)')
        
    ## PU_predict mode
    elif mode == 'PU_predict':
        print(f"-----PU_predict mode ({fptype})-----")
        smiles, X = makedata_pred(file1, variable1, fptype)
        models = ["SVC-PU", "RF-PU", "NN-PU", "GBDT-PU"]#, "RF", "NN", "GBDT", "SVC",
        for model in models:
            results_all, ave, std = PU_pred(X, fptype, model, model_path)
            results = pd.DataFrame([smiles, ave, std]).T
            results.columns = ['SMILES', 'y_ave', 'y_std']
            results_all.columns = ['y0', 'y1', 'y2', 'y3', 'y4']
            r = pd.concat([results, results_all], axis=1)
            folder = './PU_preds_test/'
            os.makedirs(folder, exist_ok=True)
            file = os.path.basename(file1)
            file = file.rstrip(".csv")
            r.to_csv(f"{folder}{file}_{fptype}_{model}.csv", index=False)
            print(f'{model} done')
        time_end = time.time()
        time_exe = time_end-time_start
        print(f'PU_predict done! (Execution time: {int(time_exe)} sec)')
    else:
        print("Mode Error!")
    

if __name__ == "__main__":
    params = {
    'mode': settings.mode,
    'file1': settings.file1,
    'file2': settings.file2,
    'variable1': settings.variable1,
    'variable2': settings.variable2,
    'target': settings.target,
    'fptype': settings.fptype,
    'datasize': settings.datasize,
    'model_path': settings.model_path,
    'lr': settings.lr,
    'epochs': settings.epochs,
    'number': settings.number
}
    main(**params)