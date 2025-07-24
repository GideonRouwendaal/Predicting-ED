import argparse
from Dataset.dataset_2D import prepare_fold_dataset_multimodal
from torch.utils.data import DataLoader
from types import SimpleNamespace

import torch.nn as nn
import torch

from sklearn.metrics import f1_score, accuracy_score
from collections import Counter

from tqdm import tqdm
import numpy as np
import os
import random

from sklearn.metrics import balanced_accuracy_score

from preprocessing_utils import clean_data, filter_data

import re

import lightgbm as lgb
import shap
import pandas as pd
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['resnet', 'vit', 'hybrid_rvit'], required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--folds_path', type=str, required=True)
    parser.add_argument('--n_outer_folds', type=int, default=5)
    parser.add_argument('--n_inner_folds', type=int, default=3)
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--pretrained', type=str, default="pretrained")
    parser.add_argument('--data', type=str)
    return parser.parse_args()



def compute_weighted_accuracy(y_true, y_pred):
    counts = Counter(y_true)
    acc_ed = accuracy_score([y for y in y_true if y == 0], [y_pred[i] for i, y in enumerate(y_true) if y == 0])
    acc_no_ed = accuracy_score([y for y in y_true if y == 1], [y_pred[i] for i, y in enumerate(y_true) if y == 1])
    n_ed, n_no_ed = counts[0], counts[1]
    total = n_ed + n_no_ed
    weighted_acc = (n_ed * acc_ed + n_no_ed * acc_no_ed) / total
    return weighted_acc, acc_ed, acc_no_ed


def build_config(args, trial=None, fixed_params=None):
    cf = {
        "model_type": args.model_type,
        "data_path": args.data_path,
        "folds_path": args.folds_path,
        "augment": False,
        "pretrained": args.pretrained,
        "num_classes": 2,
        "n_epochs": args.n_epochs,
        "patience": args.patience,
    }
    if trial:
        cf.update({
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_int("batch_size", 8, 64, step=4),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        })


    elif fixed_params:
        cf.update(fixed_params)
        if args.model_type == "hybrid_rvit":
            try:
                hidden_dim, num_heads = fixed_params["hidden_heads_pair"]
            except KeyError:
                if "hidden_dim" not in fixed_params or "num_heads" not in fixed_params:
                    # If not provided, set default values
                    hidden_dim = 64
                    num_heads = 4
                else:
                    hidden_dim, num_heads = fixed_params["hidden_dim"], fixed_params["num_heads"]
            cf["hidden_dim"] = hidden_dim
            cf["num_heads"] = num_heads
            cf["img_size"] = 512
            cf["num_patches"] = int((cf["img_size"] // cf["patch_size"]) ** 2)
    return cf


class SimpleBinaryClassifier(nn.Module):
    def __init__(self):
        super(SimpleBinaryClassifier, self).__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        x = self.linear(x)
        # x = torch.sigmoid(x)
        
        return x

# For logging the train and validation scores
class RecordCallback:
    def __init__(self):
        self.train_scores = []
        self.val_scores = []

    def __call__(self, env):
        self.train_scores.append(env.evaluation_result_list[0][2])
        self.val_scores.append(env.evaluation_result_list[1][2])

def lgb_balanced_accuracy(y_true, y_pred):
    y_pred_labels = (y_pred >= 0.5).astype(int)
    y_true = y_true.astype(int)
    bal_acc = balanced_accuracy_score(y_true, y_pred_labels)
    return 'balanced_accuracy', bal_acc, True

def train_and_evaluate(model, train_loader, val_loader, cf, class_weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cf["learning_rate"], weight_decay=cf["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    best_balanced_acc = 0
    best_metrics = {}
    patience, n_patience = cf["patience"], 0

    all_preds, all_trues = [], []

    for epoch in range(cf["n_epochs"]):
        model.train()
        for sample in train_loader:
            imgs, labels, _ = sample 
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for sample in val_loader:
                imgs, labels, _ = sample
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds += outputs.argmax(dim=1).cpu().tolist()
                trues += labels.tolist()


        weighted_acc, acc_ed, acc_no_ed = compute_weighted_accuracy(trues, preds)
        f1 = f1_score(trues, preds, average="weighted")
        balanced_acc = balanced_accuracy_score(trues, preds)

        print(f"Epoch {epoch+1:03d} | Balanced ACC: {balanced_acc:.4f} | Weighted ACC: {weighted_acc:.4f} | F1: {f1:.4f} | ACC ED: {acc_ed:.4f} | ACC no ED: {acc_no_ed:.4f}") 

        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_metrics = {
                "balanced_acc": balanced_acc,
                "weighted_acc": weighted_acc,
                "acc_ed": acc_ed,
                "acc_no_ed": acc_no_ed,
                "f1": f1,
                "epoch": epoch,
            }
            all_preds = preds
            all_trues = trues
            n_patience = 0
        else:
            n_patience += 1
            if n_patience >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return best_metrics, all_trues, all_preds

def objective(trial, fold_idx, args, model_deep_learning, model_clinical):
    inner_scores = []
    cf = build_config(args, trial)
    for inner_idx in range(args.n_inner_folds):
        print(cf)

        train_ds, val_ds = prepare_fold_dataset_multimodal(
            args.folds_path, fold_idx, 'inner', inner_idx, args=SimpleNamespace(**cf)
        )
        print(f"Inner Fold {inner_idx} | Train size: {len(train_ds)}, Val size: {len(val_ds)}")
        adjust_imgs_and_labels(train_ds)
        adjust_imgs_and_labels(val_ds)
        print(f"Adjusted Train size: {len(train_ds)}, Val size: {len(val_ds)}")
        weights = compute_class_weights_from_dataset(train_ds)
        print(f"Class weights: {weights}")
        train_loader = DataLoader(train_ds, batch_size=cf["batch_size"], shuffle=True, num_workers=4, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=cf["batch_size"], num_workers=4, persistent_workers=True)

        # model = build_model(SimpleNamespace(**cf))
        best_metrics, trues, preds = train_and_evaluate(model_deep_learning, model_clinical, train_loader, val_loader, cf, class_weights=weights)
        inner_scores.append(best_metrics["balanced_acc"])

    return np.mean(inner_scores) 


def compute_class_weights_from_dataset(dataset):
    labels = [dataset.__label_extract__(path) for path in dataset.image_paths]
    counts = Counter(labels)
    total = sum(counts.values())
    num_classes = len(counts)
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)


def load_best_params(file_path):
    best_params = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Flag to indicate we are reading the Best Hyperparameters block
    reading_params = False
    for line in lines:
        line = line.strip()
        if line.startswith("Best Hyperparameters:"):
            reading_params = True
            continue
        if line.startswith("Evaluation Metrics:"):
            break
        if reading_params and line:
            key, value = line.split(":")
            key = key.strip()
            value = value.strip()
            # Try to parse the value
            try:
                # Handle tuples
                if value.startswith("(") and value.endswith(")"):
                    hidden_dim, num_heads = eval(value)
                    best_params["hidden_dim"] = hidden_dim
                    best_params["num_heads"] = num_heads
                else:
                    value = float(value) if key not in ["batch_size", "num_layers", "mlp_dim", "patch_size"] else int(value)
                    best_params[key] = value
            except:
                # If parsing fails, keep as string
                pass
    return best_params


def adjust_imgs_and_labels(ds):
    keep_indices = []

    pattern = re.compile(r'(Anonymized Patient \d+)(?:_slice_\d+)?\.nrrd$')

    for i, path in enumerate(ds.image_paths):
        filename = os.path.basename(path)
        match = pattern.search(filename)
        if match:
            patient_id = match.group(1)
            if patient_id in data["AnonymizedName"].values:
                keep_indices.append(i)
        else:
            print(f"Warning: filename '{filename}' did not match expected pattern. Skipping.")
    
    ds.image_paths = [ds.image_paths[i] for i in keep_indices]



def outer_cv(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_PATH = <PATH TO OUTPUT DIR>
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    pat = re.compile(r'([Aa]?nonymized Patient \d+)(?:_slice_\d+)?\.nrrd$')
    all_shap_values = []
    all_X = []
    feature_names = None
    feature_importances = []
    for fold_idx in tqdm(range(args.n_outer_folds)):
        print(f"Outer Fold {fold_idx}")

        pretrained = "pretrained" if ((args.pretrained == "pretrained") or (args.pretrained == True)) else "not_pretrained"
        best_params_file = <PATH TO BEST metrics_and_params FILE>
        best_params = load_best_params(best_params_file)
        
        cf = build_config(args, fixed_params=best_params)

        train_ds, val_ds = prepare_fold_dataset_multimodal(args.folds_path, fold_idx, 'outer', args=SimpleNamespace(**cf))
        adjust_imgs_and_labels(train_ds)
        adjust_imgs_and_labels(val_ds)


        train_ids = []
        for p in train_ds.image_paths:
            filename = os.path.basename(p)
            match = pat.search(filename)
            if match:
                train_ids.append(match.group(1))
            else:
                print(f"Warning: '{filename}' did not match expected pattern. Skipping.")

        val_ids = []
        for path in val_ds.image_paths:
            filename = os.path.basename(path)
            match = pat.search(filename)
            if match:
                val_ids.append(match.group(1))
            else:
                print(f"Warning: '{filename}' did not match expected pattern. Skipping.")
        train_df = data[data["AnonymizedName"].isin(train_ids)]
        val_df = data[data["AnonymizedName"].isin(val_ids)]
        # For CAT_COLS, replace NaN with the mode of the column
        for col in CAT_COLS:
            train_df = train_df.fillna({col: train_df[col].mode()[0]})
            val_df = val_df.fillna({col: val_df[col].mode()[0]})

        # For the rest of USED_COLS, replace NaN with the median of the column
        for col in USED_COLS:
            if col not in CAT_COLS:
                train_df = train_df.fillna({col: train_df[col].median()})
                val_df = val_df.fillna({col: val_df[col].median()})

      
        model_clinical_path = BEST_CLINICAL_MODEL_PATH + f"/fold_{fold_idx}/best_model"
        model_clinical = lgb.Booster(model_file=model_clinical_path)

        val_df_np = val_df[USED_COLS].values

        all_X.append(val_df_np)

        explainer = shap.TreeExplainer(model_clinical)
        shap_vals = explainer.shap_values(val_df_np)
        all_shap_values.append(shap_vals)

        if feature_names is None:
            feature_names = model_clinical.feature_name()
        feature_importances.append(model_clinical.feature_importance(importance_type='gain'))

    X_all = np.vstack(all_X)
    shap_all = np.concatenate(all_shap_values, axis=0)
    feature_importances_mean = np.mean(np.stack(feature_importances), axis=0)

    # Plot top-N feature importances
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importances_mean
    }).sort_values(by="importance", ascending=False).head(TOP_N)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.xlabel("Mean Gain Importance")
    plt.title("Top 10 Feature Importances (Aggregated over Folds)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    # save the plot in output directory
    plt.savefig(os.path.join(OUTPUT_PATH, "lgbm_feature_importance_top10.png"))
    plt.close()

    # SHAP summary plot (top N)
    shap.summary_plot(shap_all, X_all, feature_names=feature_names, max_display=TOP_N, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "lgbm_shap_summary_top10.png"))
    plt.close()


EARLY_STOPPING_ROUNDS = 50

EVAL_METRIC = 'binary_logloss' # 'binary_logloss' | 'auc'

DATA_PATH = <path to clinical data>
DATA_PATH_LABELS = <path to clinical labels>
TARGET_TIME = '12m'

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


const_params = {
    'objective': 'binary',
    'boosting_type': 'rf',
    'metric': 'binary_logloss', 
    'n_estimators': 50,
    'early_stopping_round': EARLY_STOPPING_ROUNDS,
    'seed': 42,
    'verbose': -1,
    'is_unbalance': True,
}


data = clean_data(DATA_PATH, DATA_PATH_LABELS)
data = filter_data(data, TARGET_TIME)

BEST_CLINICAL_MODEL_PATH = <PATH TO BEST CLINICAL MODEL>

TOP_N = 10  # Number of top features to display in the plots

if __name__ == "__main__":
    args = get_args()
    if args.pretrained == "pretrained":
        args.pretrained = True
    elif args.pretrained == "not_pretrained":
        args.pretrained = False
    print(args)
    outer_cv(args)