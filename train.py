import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Input
from tensorflow.keras.optimizers import Adam
from process import *
from keras import losses
from sklearn.model_selection import train_test_split, KFold
from pulearn import ElkanotoPuClassifier, WeightedElkanotoPuClassifier
import os
import glob
import optuna
mae = losses.mean_absolute_error
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
import pickle

def DNN(in_dim, n_layers=3, n_dim=3, lr=1e-3, activation_h='relu', activation_o='relu', loss=mae, save_name=None):
    
    model = Sequential()
    model.add(Input(in_dim,))

    for i in range(n_layers):
        if i==0 or i%2==1:
            model.add(Dense(n_dim, activation=activation_h))
        elif i%2 == 0:
            n_dim = n_dim//2
            model.add(Dense(n_dim, activation=activation_h))
    model.add(Dense(1, activation=activation_o))
    model.compile(optimizer=Adam(lr), loss=loss)
    return model


def train(X, y, fptype, epochs=10, batch_size=64):
    # Hyperparameter optimization
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
    best_params = Hyperparameter(X_train, y_train, batch_size, epochs)
    best_params['in_dim']=X.shape[1]
    
    # Training using optimized params
    model = DNN(**best_params)
    model.fit(X_train, y_train, epochs=epochs, verbose=0, batch_size=batch_size)
    
    # Model save
    folder = './Scratch_models/'
    os.makedirs(folder, exist_ok=True)
    model_name = f'Scratch_{fptype}.h5'
    model.save(folder + model_name)
    y_pred = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred)
    
    return mae_test, best_params


def simple_model(X, y, ml, folder, save_name, number=5):
    df_fi = pd.DataFrame(index=X.columns)
    kf = KFold(n_splits=5, shuffle=True)
    results = []
    corrs = []
    
    for i in range(number):
        count = 0
        for train_idx, test_idx in kf.split(X, y):
            train_l = train_idx.tolist()
            test_l = test_idx.tolist()
            if ml == 'rf':
                model = RandomForestRegressor()
            elif ml == 'nn':
                model = MLPRegressor()
            model.fit(X.iloc[train_l], y[train_l])
            y_pred = model.predict(X.iloc[test_l])
            results.append(mean_absolute_error(y[test_l], y_pred))
            
            df_all = pd.concat([pd.DataFrame(y[test_l]), pd.DataFrame(y_pred)], axis=1)
            corr = df_all.corr().iloc[0,1]
            corrs.append(corr)

            if ml == 'rf':
                fi = pd.DataFrame(model.feature_importances_, index=X.columns)
                df_fi = pd.concat([df_fi, fi], axis=1)
            pickle.dump(model, open(f'{folder}{save_name}-{i}-cv{count}.pkl', 'wb'))
            count += 1
    
    return results, df_fi, corrs


class Objective:
    def __init__(self, X, y, batch_size, epochs):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.epochs = epochs

    def __call__(self, trial):
        # Search space
        model_params = {
            'in_dim': self.X.shape[1],
            'n_layers': trial.suggest_int('n_layers', 1, 10),
            'n_dim': trial.suggest_categorical('n_dim', [50, 100, 200, 300, 400, 500, 750, 1000])
            }
        # Train-Val split
        model = DNN(**model_params)
        kf = KFold(n_splits=5, shuffle=True)
        valscores = []
        for train_idx, val_idx in kf.split(self.X, self.y):
            model.fit(self.X[train_idx], self.y[train_idx], verbose=0, epochs=self.epochs, batch_size=self.batch_size)
            y_pred = model.predict(self.X[val_idx])
            valscores.append(mean_absolute_error(self.y[val_idx], y_pred))
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return np.mean(valscores)


def Hyperparameter(X,y,batch_size,epochs):
    objective = Objective(X,y,batch_size,epochs)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=80)#, timeout=60)  ## n_trials=80
    print('params:', study.best_params)
    return study.best_params


def TL(X, y, model_path, number, lr=0.001, loss=mae, epochs=100):
    
    # Load model
    model = keras.models.load_model(model_path)
    name = os.path.splitext(os.path.basename(model_path))[0]
    n_layers = len(model.layers)
    print("Model loaded: ", name)

    # Prepare for prediction metrics
    df_train = pd.DataFrame(index=range(n_layers), columns=range(5))
    df_test = pd.DataFrame(index=range(n_layers), columns=range(5))
    df_corr = pd.DataFrame(index=range(n_layers), columns=range(5))
    kf = KFold(n_splits=5, shuffle=True)
    cv=0

    # Prepare for Fine-Tune
    df_FT = pd.DataFrame(index=range(n_layers), columns=range(n_layers))
    df_FT.fillna('False', inplace=True)

    # Run CV
    for train_idx, test_idx in kf.split(X, y):    
        for i in range(n_layers):
            # Make a list including True
            df_FT.iloc[i, -1-i:] = 'True'
            list_FT = df_FT.iloc[i, :].tolist()

            # Switch model layers trainable
            model = keras.models.load_model(model_path)
            for j in range(n_layers):
                model.layers[j].trainable = list_FT[j]

            # Train & Test
            model.compile(optimizer=Adam(lr), loss=loss)
            model.fit(X[train_idx], y[train_idx], verbose=0, epochs=epochs)
            y_train_pred = model.predict(X[train_idx])
            y_test_pred = model.predict(X[test_idx])

            # Store results
            df_train.iloc[i,cv] = mean_absolute_error(y[train_idx], y_train_pred)
            df_test.iloc[i,cv] = mean_absolute_error(y[test_idx], y_test_pred)
            y_all = pd.concat([pd.DataFrame(y[test_idx]), pd.DataFrame(y_test_pred)], axis=1)
            df_corr.iloc[i,cv] = y_all.corr().iloc[0,1]
            # Model save
            folder = './TL_models_test/'
            os.makedirs(folder, exist_ok=True)
            y_all.to_csv('./y_all_test/'+name+'_ft'+str(i+1)+'_cv'+str(cv)+'_'+str(number)+'.csv')
            model_name = name+'_ft'+str(i+1)+'_cv'+str(cv)+'_'+str(number)+'.h5'
            model.save(folder+model_name)
        cv+=1

    # Calculate CV ave & std
    list_mae_train = df_train.mean(axis='columns').tolist()
    list_mae_test = df_test.mean(axis='columns').tolist()
    list_name = [f'ft{i+1}' for i in range(n_layers)]
            
    return list_name, list_mae_train, list_mae_test, df_test, df_corr


def metrics(y1, y2):
    acc = accuracy_score(y1, y2)
    pre = precision_score(y1, y2)
    rec = recall_score(y1, y2)
    f1 = f1_score(y1, y2)
    return acc, pre, rec, f1


def PU(model,X,y,fptype):
    kf = KFold(n_splits=10, shuffle=True, random_state=2)
    results = []
    count = 1
    model_name = model
    for train_idx, test_idx in kf.split(X, y):
        if model == 'RF':
            model = RandomForestClassifier()
        elif model == 'RF-PU':
            # model = ElkanotoPuClassifier(estimator=RandomForestClassifier())
            model = WeightedElkanotoPuClassifier(estimator=RandomForestClassifier(), labeled=1, unlabeled=0)
        elif model == 'SVC':
            model = SVC(probability=True)#(C=1, kernel='rbf', gamma=0.2, probability=True)
        elif model == 'SVC-PU':
            # model = ElkanotoPuClassifier(estimator=SVC(probability=True))
            model = WeightedElkanotoPuClassifier(estimator=SVC(probability=True), labeled=1, unlabeled=0)
        elif model == 'NN':
            model = MLPClassifier(hidden_layer_sizes=(50,),max_iter=1000)
        elif model == 'NN-PU':
            # model = ElkanotoPuClassifier(estimator=MLPClassifier(hidden_layer_sizes=(50,)))
            model = WeightedElkanotoPuClassifier(estimator=MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000), labeled=1, unlabeled=0)
        elif model == 'GBDT':
            model = GradientBoostingClassifier()
        elif model == 'GBDT-PU':
            # model = ElkanotoPuClassifier(estimator=GradientBoostingClassifier())
            model = WeightedElkanotoPuClassifier(estimator=GradientBoostingClassifier(), labeled=1, unlabeled=0)

        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        y_pred[y_pred==-1] = 0
        results.append(metrics(y[test_idx], y_pred))

        # Model save
        folder = './PU_models/'
        os.makedirs(folder, exist_ok=True)
        save_name = f'{fptype}_{model_name}_{count}.pkl'
        pickle.dump(model, open(folder+save_name, 'wb'))
        count += 1
    
    results = np.array(results)
    acc_ave = np.mean(results[:,0])
    acc_std = np.std(results[:,0])
    pre_ave = np.mean(results[:,1])
    pre_std = np.std(results[:,1])
    rec_ave = np.mean(results[:,2])
    rec_std = np.std(results[:,2])
    f1_ave = np.mean(results[:,3])
    f1_std = np.std(results[:,3])
    
    return [model_name, acc_ave, acc_std, pre_ave, pre_std, rec_ave, rec_std, f1_ave, f1_std]


def PU_pred(X, fptype, model, model_path):
    files = glob.glob(model_path)
    results = []
    for file in files:
        if fptype in file and model in file:
            loaded_model = pickle.load(open(file, 'rb'))
            y_pred = loaded_model.predict(X)
            y_pred[y_pred==-1] = 0
            results.append(y_pred)
    results = pd.DataFrame(results).T
    ave = results.mean(axis=1)
    std = results.std(axis=1)
    return results, ave, std
