import lightgbm as lgbm
import catboost
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from copy import deepcopy
import numpy as np
import optuna
import random
import gc

def optuna_optimizer(trial, params_optuna, classifier, x_train, y_train, x_val, y_val):

    p_grid = params_optuna(trial)
    if isinstance(classifier, catboost.core.CatBoostClassifier):
        classifier = CatBoostClassifier(loss_function='CrossEntropy', eval_metric='AUC', verbose=False)         
    classifier.set_params(**p_grid)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict_proba(x_val)[:, 1]
    score = roc_auc_score(y_val, predictions)

    return score

def nestedcv(x, y, classifier, cv_outer, cv_inner, p_grid, groups, n_iter=10):
    clfs = []
    oof_preds = []
    oof_targets = []
    for pointer, (train_index, test_index) in enumerate(cv_outer.split(x, y, groups=groups)):                          
        print('\nNested CV: {} of {} outer fold'.format(pointer + 1, cv_outer.get_n_splits()))
        x_out_train, x_test = x.loc[train_index], x.loc[test_index]
        y_out_train, y_test = y.loc[train_index], y.loc[test_index]

        optuna.logging.set_verbosity(optuna.logging.INFO)
        best_score = 0
        best_params = None
        inner_groups = x_out_train['WEEK_NUM']
        for train_index, val_index in cv_inner.split(x_out_train, y_out_train, groups=inner_groups):
            x_train, x_val = x_out_train.iloc[train_index], x_out_train.iloc[val_index]
            y_train, y_val = y_out_train.iloc[train_index], y_out_train.iloc[val_index]
            
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction='maximize', sampler=sampler)
            study.optimize(lambda trial: optuna_optimizer(trial, p_grid, classifier, x_train, y_train, x_val, y_val), n_trials=n_iter)
            if study.best_value > best_score:
                print('Found new best score with score {:.6f}'.format(study.best_value))
                best_score = study.best_value
                best_params = study.best_params
        model = deepcopy(classifier)
        model.set_params(**best_params)
        model.fit(x_out_train, y_out_train)

        pred_test = model.predict_proba(x_test)[:, 1]              
        auc_test = roc_auc_score(y_test, pred_test)

        print("""
        Test AUC                           : {:.3f}
        """.format(
            auc_test,
            )
        )
        oof_preds.append(pred_test)
        oof_targets.append(y_test)
        clfs.append(model)
        
    return clfs, oof_preds, oof_targets

def find_best_ensemble(current_ensemble, best_models, oof_files, oof_csv, truth):
    best_weight = ''
    best_score = 0
    best_model = ''
    print('Searching for best model to add... ')
    for model_index in range(len(oof_files)):
        print(model_index, ', ', end='')
        if model_index in best_models:
            continue
        for w in np.arange(0, 1.01, 0.01):
            ensemble_pred = w * current_ensemble + (1-w) * oof_csv[model_index].oof
            ensemble_auc = roc_auc_score( truth , ensemble_pred )
            if ensemble_auc > best_score:
                best_weight = w
                best_score = ensemble_auc
                best_model = model_index
    return best_weight, best_model, best_score

def random_seed(seed_value): 
    np.random.seed(seed_value) 
    random.seed(seed_value) 

def predict_proba_in_batches(model, data, batch_size=100000):
    num_samples = len(data)
    num_batches = int(np.ceil(num_samples / batch_size))
    probabilities = np.zeros((num_samples,))

    for batch_idx in range(num_batches):
        print(f"Processing batch: {batch_idx+1}/{num_batches}")
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        X_batch = data.iloc[start_idx:end_idx]
        batch_probs = model.predict_proba(X_batch)[:, 1]
        probabilities[start_idx:end_idx] = batch_probs
        gc.collect()

    return probabilities