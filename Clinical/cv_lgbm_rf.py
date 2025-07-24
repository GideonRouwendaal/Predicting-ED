import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score
import optuna
import seaborn as sns
from pathlib import Path

from preprocessing_utils import clean_data, filter_data

import json

import random
from tqdm import tqdm
import torch

from sklearn.metrics import balanced_accuracy_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lgb_balanced_accuracy(y_true, y_pred):
    y_pred_labels = (y_pred >= 0.5).astype(int)
    y_true = y_true.astype(int)
    bal_acc = balanced_accuracy_score(y_true, y_pred_labels)
    return 'balanced_accuracy', bal_acc, True

# For logging the train and validation scores
class RecordCallback:
    def __init__(self):
        self.train_scores = []
        self.val_scores = []

    def __call__(self, env):
        self.train_scores.append(env.evaluation_result_list[0][2])
        self.val_scores.append(env.evaluation_result_list[1][2])

def objective(trial, inner_folds):
    inner_scores = []

    trial_params = {
        # 'boosting_type': trial.suggest_categorical('boosting_type', ['rf', 'gbdt']),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 30),
        'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 150),
        'cat_smooth': trial.suggest_int('cat_smooth', 8, 20),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
    }

    params = {**const_params, **trial_params}


    for inner_idx in range(3):
        record = RecordCallback()
        # Create the model
        clf = lgb.LGBMClassifier(**params)

        train_patients, val_patients = inner_folds[f"inner_{inner_idx}"]["train"], inner_folds[f"inner_{inner_idx}"]["val"]
        train_ds, val_ds = data[data["AnonymizedName"].isin(train_patients)], data[data["AnonymizedName"].isin(val_patients)]

        # For CAT_COLS, replace NaN with the mode of the column
        for col in CAT_COLS:
            train_ds = train_ds.fillna({col: train_ds[col].mode()[0]})
            val_ds = val_ds.fillna({col: val_ds[col].mode()[0]})

        # For the rest of USED_COLS, replace NaN with the median of the column
        for col in USED_COLS:
            if col not in CAT_COLS:
                train_ds = train_ds.fillna({col: train_ds[col].median()})
                val_ds = val_ds.fillna({col: val_ds[col].median()})

        X_train, X_val = train_ds[USED_COLS], val_ds[USED_COLS]
        y_train, y_val = train_ds[f'IIEF15_01_12m'].apply(lambda x: 0 if x < 4 else 1), val_ds[f'IIEF15_01_12m'].apply(lambda x: 0 if x < 4 else 1)

        X_train_np = X_train.values.astype(np.float32)
        y_train_np = y_train.values.astype(np.int64)
        X_val_np = X_val.values.astype(np.float32)
        y_val_np = y_val.values.astype(np.int64)


        # Fit the model
        clf.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_metric=lgb_balanced_accuracy,
                callbacks=[record],
                )

        # Report intermediate objective value
        for i in range(len(record.train_scores)):
            trial.report(record.val_scores[i], i)    
        preds = clf.predict(X_val, num_iteration=clf._best_iteration)
        balanced_acc = balanced_accuracy_score(y_val_np, preds)
        inner_scores.append(balanced_acc)

    return np.mean(inner_scores) 


def outer_cv():
    set_seed(42)
    outer_scores = []
    all_outer_trues, all_outer_preds = [], []

    for fold_idx in tqdm(range(5)):
        print(f"Outer Fold {fold_idx}")
        train_patients, val_patients = folds[f"fold_{fold_idx}"]["outer_train"], folds[f"fold_{fold_idx}"]["outer_val"]

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, folds[f"fold_{fold_idx}"]["inner_folds"]), n_trials=50)

        best_tuned_params = study.best_params

        best_params = {**const_params, **best_tuned_params}

        print("Best hyperparameters:\n")
        for key, value in best_tuned_params.items():
            print(f'{key}: {value}')

        full_out_path = OUT_PATH + f'fold_{fold_idx}/'
        Path(full_out_path).mkdir(parents=True, exist_ok=True)

        # Save the best hyperparameters to a file
        with open(full_out_path + 'best_hyperparameters.txt', 'w') as f:
            for key, value in best_params.items():
                f.write(f'{key}: {value}\n')


        record = RecordCallback()

        # Train the model with the best hyperparameters
        best_model = lgb.LGBMClassifier(**best_params)

        train_ds, val_ds = data[data["AnonymizedName"].isin(train_patients)], data[data["AnonymizedName"].isin(val_patients)]
        # For CAT_COLS, replace NaN with the mode of the column
        for col in CAT_COLS:
            train_ds = train_ds.fillna({col: train_ds[col].mode()[0]})
            val_ds = val_ds.fillna({col: val_ds[col].mode()[0]})

        # For the rest of USED_COLS, replace NaN with the median of the column
        for col in USED_COLS:
            if col not in CAT_COLS:
                train_ds = train_ds.fillna({col: train_ds[col].median()})
                val_ds = val_ds.fillna({col: val_ds[col].median()})
            
        X_train, X_val = train_ds[USED_COLS], val_ds[USED_COLS]
        y_train, y_val = train_ds[f'IIEF15_01_12m'].apply(lambda x: 0 if x < 4 else 1), val_ds[f'IIEF15_01_12m'].apply(lambda x: 0 if x < 4 else 1)

        X_train_np = X_train.values.astype(np.float32)
        y_train_np = y_train.values.astype(np.int64)
        X_val_np = X_val.values.astype(np.float32)
        y_val_np = y_val.values.astype(np.int64)


        best_model.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    eval_metric=lgb_balanced_accuracy,
                    callbacks=[record]
                    )
        

        best_model.booster_.save_model(full_out_path + 'best_model')

        y_pred = best_model.predict(X_val_np)
        y_pred_prob = best_model.predict_proba(X_val_np)[:, 1]

        # Save the predictions
        np.save(full_out_path + 'y_pred.npy', y_pred)
        np.save(full_out_path + 'y_pred_proba.npy', y_pred_prob)
        y_val.to_csv(full_out_path + 'y_test.csv', index=False)

        class_report = classification_report(y_val_np, y_pred)

        # AUC
        roc_auc = roc_auc_score(y_val_np, y_pred_prob)
        print(f'AUC: {roc_auc}')

        # Balanced accuracy
        bal_acc = balanced_accuracy_score(y_val_np, y_pred)
        print(f'Balanced accuracy: {bal_acc}')

        # Save classification report and AUC to a file
        with open(full_out_path + 'classification_report.txt', 'w') as f:
            f.write(class_report)
            f.write(f'\nAUC: {roc_auc}\n')
            f.write(f'Balanced accuracy: {bal_acc}\n')
        
        outer_scores.append(bal_acc)

    print(f"\nMean Balanced ACC: {np.mean(outer_scores):.4f}")



DATA_PATH = <path to data>
DATA_PATH_LABELS = <path to labels>
OUT_ROOT = <path to output folder>
folds_path = <path to K-folds>

# Upsampling
SAMPLE_METHOD = 'class_weights' # 'basic' | 'class_weights'

# Evaluation metric
EVAL_METRIC = 'binary_logloss' # 'binary_logloss' | 'auc'

# TUNING
N_TRIALS = 50

# Training
TARGET_TIME = '12m' # '6m' | '12m' | '24m' | '36m'
EARLY_STOPPING_ROUNDS = 50

# Plotting
sns.set_theme(style="darkgrid")

# Other
SEED = 42

OUT_PATH = OUT_ROOT + 'lgbm/' + 'RF_' + SAMPLE_METHOD + '_' + TARGET_TIME + '_' + EVAL_METRIC + '/'
Path(OUT_PATH).mkdir(parents=True, exist_ok=True)

# Columns to use for training
USED_COLS = [
    'age_at_diagnosis',     
    'lengte',               
    'gewicht',              
    'roken',                
    'roken_hoeveel',          
    'alkohol',              
    'alkohol_hoeveel',      
    'medicatie',            
    'andere_ziekten',        
    'IIEF15_01_preop',
]

# Categorical columns (for LightGBM models)
CAT_COLS = [
    'roken',
    'alkohol',
    'medicatie',
    'andere_ziekten',
]

# Possible target columns
TARGET_COLS = [
    'IIEF15_01_6m',
    'IIEF15_01_12m',
    'IIEF15_01_24m',
    'IIEF15_01_36m'
]

data = clean_data(DATA_PATH, DATA_PATH_LABELS)
data = filter_data(data, TARGET_TIME)


with open(folds_path, 'r') as f:
    folds = json.load(f)

if SAMPLE_METHOD == "basic":
    const_params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'metric': EVAL_METRIC, # 'binary_logloss', 'auc'
        'n_estimators': 50,
        'early_stopping_round': EARLY_STOPPING_ROUNDS,
        'seed': 42,
        'verbose': -1,
        'is_unbalance': False,
    }
else:
    const_params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'metric': EVAL_METRIC, # 'binary_logloss', 'auc'
        'n_estimators': 50,
        'early_stopping_round': EARLY_STOPPING_ROUNDS,
        'seed': 42,
        'verbose': -1,
        'is_unbalance': True,
    }

outer_cv()