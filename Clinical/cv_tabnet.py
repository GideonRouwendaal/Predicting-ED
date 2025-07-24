import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import log_loss, classification_report, roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn as nn

from preprocessing_utils import clean_data, filter_data
from pytorch_tabnet.metrics import Metric
import json

import random
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class BalancedMetric(Metric):
    def __init__(self):
        self._name = "BalancedMetric"
        self._maximize = True

    def __call__(self, y_true, y_pred):
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true = y_true.astype(int)
        bal_acc = balanced_accuracy_score(y_true, y_pred_labels)
        return bal_acc


def objective(trial, inner_folds):
    inner_scores = []

    # Define hyperparameter search space
    n_da = trial.suggest_int('n_d', 56, 64)
    # n_a = trial.suggest_int('n_a', 8, 64)
    n_steps = trial.suggest_int('n_steps', 1, 3, step=1)
    gamma = trial.suggest_float("gamma", 1., 1.4, step=0.2)
    # lambda_sparse = trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True)
    optimizer_params = dict(lr=2e-2,weight_decay=1e-5)
    # n_independent = trial.suggest_int('n_independent', 1, 5)
    n_shared = trial.suggest_int('n_shared', 1, 3)
    # momentum = trial.suggest_float('momentum', 0.01, 0.4)

    for inner_idx in range(3):
        # Create the model
        model = TabNetClassifier(
            n_d=n_da, n_a=n_da, n_steps=n_steps, gamma=gamma,
            optimizer_params=optimizer_params,
            n_shared=n_shared,
        )

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

        class_counts = np.bincount(y_train_np)
        print(f"Class counts: {class_counts}")
        total_samples = len(y_train_np)
        class_weights = total_samples / (len(class_counts) * class_counts)
        # Calculate class weights
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        criterion = nn.CrossEntropyLoss()
        if SAMPLE_METHOD == "class_weights":
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            # Fit the model
        model.fit(
            X_train_np, y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            eval_name=['valid'],
            eval_metric=[BalancedMetric],
            max_epochs=400,  # Increased epochs for better tuning
            patience=50,
            loss_fn=criterion,
        )
        preds = model.predict_proba(X_val_np)[:, 1].round().astype(int)
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

        print("Best hyperparameters:\n")
        for key, value in best_tuned_params.items():
            print(f'{key}: {value}')

        best_model = TabNetClassifier(
            n_d=best_tuned_params['n_d'],
            n_a=best_tuned_params['n_d'],
            n_steps=best_tuned_params['n_steps'],
            gamma=best_tuned_params['gamma'],
            # lambda_sparse=best_tuned_params['lambda_sparse'],
            optimizer_params = dict(lr=2e-2,weight_decay=1e-5),
            # n_independent=best_tuned_params['n_independent'],
            n_shared=best_tuned_params['n_shared'],
            # momentum=best_tuned_params['momentum']
        )

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

        class_counts = np.bincount(y_train_np)
        print(f"Class counts: {class_counts}")
        total_samples = len(y_train_np)
        criterion = nn.CrossEntropyLoss()
        if SAMPLE_METHOD == "class_weights":
            class_weights = total_samples / (len(class_counts) * class_counts)
            # Calculate class weights
            class_weights = torch.tensor(class_weights, dtype=torch.float)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_model.fit(
            X_train_np, y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            eval_name=['valid'],
            eval_metric=[BalancedMetric],
            max_epochs=400,  # Increased epochs for better tuning
            patience=50,
            loss_fn=criterion,
        )
        

        full_out_path = OUT_PATH + f'fold_{fold_idx}/'
        Path(full_out_path).mkdir(parents=True, exist_ok=True)

        best_model.save_model(full_out_path + 'best_model')

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
SAMPLE_METHOD = 'basic' # 'basic' | 'smote' | 'class_weights'

# Evaluation metric
EVAL_METRIC = 'binary_logloss' # 'binary_logloss' | 'auc'

# TUNING
N_TRIALS = 200

# Training
TARGET_TIME = '12m' # '6m' | '12m' | '24m' | '36m'
TRAIN_FRAC = 0.7
EARLY_STOPPING_ROUNDS = 50

# Plotting
sns.set_theme(style="darkgrid")

# Other
SEED = 42

OUT_PATH = OUT_ROOT + 'tabnet/' + 'TabNet_' + SAMPLE_METHOD + '_' + TARGET_TIME + '_' + EVAL_METRIC + '/'
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


print(data.head())

with open(folds_path, 'r') as f:
    folds = json.load(f)

outer_cv()