import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score
import optuna
import seaborn as sns
from pathlib import Path

from preprocessing_utils import clean_data, filter_data
import json

import random
from tqdm import tqdm

import torch
import random

from tensorflow.keras import backend as K


class BalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='balanced_accuracy', **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', shape=(), initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', shape=(), initializer='zeros')
        self.false_positives = self.add_weight(name='fp', shape=(), initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', shape=(), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.cast(K.greater_equal(y_pred, 0.5), K.floatx())  # binary threshold

        tp = K.sum(K.cast(K.equal(y_true, 1) & K.equal(y_pred, 1), K.floatx()))
        tn = K.sum(K.cast(K.equal(y_true, 0) & K.equal(y_pred, 0), K.floatx()))
        fp = K.sum(K.cast(K.equal(y_true, 0) & K.equal(y_pred, 1), K.floatx()))
        fn = K.sum(K.cast(K.equal(y_true, 1) & K.equal(y_pred, 0), K.floatx()))

        self.true_positives.assign_add(tp)
        self.true_negatives.assign_add(tn)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        sensitivity = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        specificity = self.true_negatives / (self.true_negatives + self.false_positives + K.epsilon())
        return (sensitivity + specificity) / 2

    def reset_states(self):
        self.true_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def objective(trial, inner_folds):
    inner_scores = []

    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    n_layers = trial.suggest_int('n_layers', 1, 4)
    n_units = [trial.suggest_int(f'n_units_l{i}', 16, 128) for i in range(n_layers)]


    for inner_idx in range(3):

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

        model = Sequential()
        model.add(Input(shape=(X_train_np.shape[1],))) 

        for i in range(1, n_layers):
            model.add(Dense(n_units[i], activation='relu'))
        
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[BalancedAccuracy(name='balanced_ACC')])

        class_counts = np.bincount(y_train_np)
        print(f"Class counts: {class_counts}")
        total_samples = len(y_train_np)
        class_weights = total_samples / (len(class_counts) * class_counts)
        print(f"Class weights: {class_weights}")

        if SAMPLE_METHOD == 'class_weights':
            model.fit(X_train_np, y_train_np, validation_data=(X_val_np, y_val_np), epochs=400, class_weight=dict(enumerate(class_weights)), verbose=0)
        else:
            model.fit(X_train_np, y_train_np, validation_data=(X_val_np, y_val_np), epochs=400, verbose=0)

        y_pred_prob = model.predict(X_val_np)
        preds = (y_pred_prob > 0.5).astype(int)
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

        train_ds, val_ds = data[data["AnonymizedName"].isin(train_patients)], data[data["AnonymizedName"].isin(val_patients)]

        # For CAT_COLS, replace NaN with the mode of the column
        for col in CAT_COLS:
            train_ds = train_ds.fillna({col: train_ds[col].mode()[0]})
            val_ds = val_ds.fillna({col: train_ds[col].mode()[0]})

        # For the rest of USED_COLS, replace NaN with the median of the column
        for col in USED_COLS:
            if col not in CAT_COLS:
                train_ds = train_ds.fillna({col: train_ds[col].median()})
                val_ds = val_ds.fillna({col: train_ds[col].median()})

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

        print("Best hyperparameters:\n")
        for key, value in best_tuned_params.items():
            print(f'{key}: {value}')

        best_model = Sequential()
        best_model.add(Input(shape=(X_train.shape[1],)))

        for i in range(1, best_tuned_params['n_layers']):
            best_model.add(Dense(best_tuned_params[f'n_units_l{i}'], activation='relu'))

        best_model.add(Dense(1, activation='sigmoid'))

        best_optimizer = Adam(learning_rate=best_tuned_params['learning_rate'])
        best_model.compile(loss='binary_crossentropy', optimizer=best_optimizer, metrics=[BalancedAccuracy(name='balanced_ACC')])

        if SAMPLE_METHOD == 'class_weights':
            history = best_model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_val_np, y_val_np), class_weight=dict(enumerate(class_weights)), verbose=0)
        else:
            history = best_model.fit(X_train_np, y_train_np, epochs=500, batch_size=32, validation_data=(X_val_np, y_val_np), verbose=0)
      

        full_out_path = OUT_PATH + f'fold_{fold_idx}/'
        Path(full_out_path).mkdir(parents=True, exist_ok=True)

        best_model.save(full_out_path + 'model.h5')

        y_pred_prob = best_model.predict(X_val_np)
        y_pred = (y_pred_prob > 0.5).astype(int)

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
N_TRIALS = 200

# Training
TARGET_TIME = '12m' # '6m' | '12m' | '24m' | '36m'

# Plotting
sns.set_theme(style="darkgrid")

# Other
SEED = 42

OUT_PATH = OUT_ROOT + 'mlp/' + 'MLP_' + SAMPLE_METHOD + '_' + TARGET_TIME + '_' + EVAL_METRIC + '/'
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