import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score
import optuna
import seaborn as sns
from pathlib import Path
from preprocessing_utils import clean_data, filter_data
import json
import random
from tqdm import tqdm
import torch



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# For logging the train and validation scores
class RecordCallback:
    def __init__(self):
        self.train_scores = []
        self.val_scores = []

    def __call__(self, env):
        self.train_scores.append(env.evaluation_result_list[0][2])
        self.val_scores.append(env.evaluation_result_list[1][2])


def balanced_accuracy_metric(y_true, y_pred):
    # Ensure predictions are in binary format (0 or 1)
    y_pred = (y_pred > 0.5).astype(int)
    y_true = y_true.astype(int)
    # Calculate the balanced accuracy score
    score = balanced_accuracy_score(y_true, y_pred)
    return score


def objective(trial, inner_folds):
    inner_scores = []

    trial_params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-3, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('lambda', 1e-3, 1.0, log=True),
        'reg_alpha': trial.suggest_float('alpha', 1e-3, 1.0, log=True),
    }

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

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        scale_pos_w = 1

        if SAMPLE_METHOD == 'class_weights':
            scale_pos_w = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

        const_params = {
            'objective': 'binary:logistic',
            'eval_metric' : balanced_accuracy_metric,
            'n_estimators': 50,
            'booster': 'gbtree',
            'seed': SEED,
            'verbosity': 0,
            'scale_pos_weight': scale_pos_w
        }

        params = {**const_params, **trial_params}

        clf = XGBClassifier(**params, use_label_encoder=False)

        # Fit the model
        clf.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                )

        
        preds = clf.predict(X_val)
        balanced_acc = balanced_accuracy_score(y_val, preds)
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

        full_out_path = OUT_PATH + f'fold_{fold_idx}/'
        Path(full_out_path).mkdir(parents=True, exist_ok=True)

        # Save the best hyperparameters to a file
        with open(full_out_path + 'best_hyperparameters.txt', 'w') as f:
            for key, value in best_tuned_params.items():
                f.write(f'{key}: {value}\n')


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

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        scale_pos_w = 1

        if SAMPLE_METHOD == 'class_weights':
            scale_pos_w = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

        const_params = {
            'objective': 'binary:logistic',
            'eval_metric' : balanced_accuracy_metric,
            'n_estimators': 50,
            'booster': 'gbtree',
            'seed': SEED,
            'verbosity': 0,
            'scale_pos_weight': scale_pos_w
        }

        params = {**const_params, **best_tuned_params}

        best_model = XGBClassifier(**params, use_label_encoder=False)

        # Fit the model
        best_model.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                )

        best_model.save_model(full_out_path + 'best_model')

        y_pred = best_model.predict(X_val)
        y_pred_prob = best_model.predict_proba(X_val)[:, 1]

        # Save the predictions
        np.save(full_out_path + 'y_pred.npy', y_pred)
        np.save(full_out_path + 'y_pred_proba.npy', y_pred_prob)
        y_val.to_csv(full_out_path + 'y_test.csv', index=False)

        class_report = classification_report(y_val, y_pred)

        # AUC
        roc_auc = roc_auc_score(y_val, y_pred_prob)
        print(f'AUC: {roc_auc}')

        # Balanced accuracy
        bal_acc = balanced_accuracy_score(y_val, y_pred)
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

OUT_PATH = OUT_ROOT + 'xgboost/' + 'GBDT_' + SAMPLE_METHOD + '_' + TARGET_TIME + '_' + EVAL_METRIC + '/'
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

outer_cv()