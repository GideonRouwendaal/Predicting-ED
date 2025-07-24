import argparse
from Dataset.dataset_2D_multi_task import prepare_fold_dataset, calculate_mean_std
from multi_task_models import build_model
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
import pandas as pd

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
        "augment": True,
        "pretrained": args.pretrained,
        "num_classes": 2,
        "n_epochs": args.n_epochs,
        "patience": args.patience,
    }

    if trial:
        cf.update({
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_int("batch_size", 4, 16, step=4),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "age_loss_weight" : trial.suggest_float("age_loss_weight", 1e-4, 1e-1, log=True)

        })

        if args.model_type == "hybrid_rvit":
            valid_combos = [
                (64, 4),
                (64, 8),
                (128, 4),
                (128, 8),
            ]
            hidden_dim, num_heads = trial.suggest_categorical("hidden_heads_pair", valid_combos)

            cf.update({
                "num_layers": trial.suggest_int("num_layers", 3, 5),
                "mlp_dim": trial.suggest_categorical("mlp_dim", [64, 128, 256]),
                "hidden_dim": hidden_dim,
                "num_heads": num_heads,
                "patch_size": trial.suggest_categorical("patch_size", [16, 32, 64]),
                "img_size": 512,
            })
            
            cf["num_patches"] = int((cf["img_size"] // cf["patch_size"]) ** 2)

            print(cf)
    elif fixed_params:
        cf.update(fixed_params)
        if args.model_type == "hybrid_rvit":
            try:
                hidden_dim, num_heads = fixed_params["hidden_heads_pair"]
            except KeyError:
                # Default values if not provided (for normalizing the data)
                hidden_dim = 64
                num_heads = 4
            cf["hidden_dim"] = hidden_dim
            cf["num_heads"] = num_heads
            cf["img_size"] = 512
            cf["num_patches"] = int((cf["img_size"] // cf["patch_size"]) ** 2)

    return cf



def train_and_evaluate(model, train_loader, val_loader, cf, class_weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cf["learning_rate"], weight_decay=cf["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    loss_reg_fn = nn.MSELoss()

    best_balanced_acc = 0
    best_metrics = {}
    patience, n_patience = cf["patience"], 0

    all_preds, all_trues = [], []

    alpha = cf["age_loss_weight"]

    for epoch in range(cf["n_epochs"]):
        model.train()
        for sample in train_loader:
            imgs, labels, ages = sample
            imgs, labels, ages = imgs.to(device), labels.to(device), ages.to(device)
            logits, age_preds = model(imgs)
            loss_cls = loss_fn(logits, labels)
            loss_reg = loss_reg_fn(age_preds, ages)

            loss = loss_cls + alpha * loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    preds, trues, age_errors, outputs_list = [], [], [], []
    with torch.no_grad():
        for sample in val_loader:
            imgs, labels, ages = sample
            imgs, ages = imgs.to(device), ages.to(device)
            logits, age_preds = model(imgs)
            preds += logits.argmax(dim=1).cpu().tolist()
            outputs_list += logits.cpu().tolist()
            trues += labels.tolist()
            age_errors += (age_preds - ages).abs().cpu().tolist()


    weighted_acc, acc_ed, acc_no_ed = compute_weighted_accuracy(trues, preds)
    f1 = f1_score(trues, preds, average="weighted")
    balanced_acc = balanced_accuracy_score(trues, preds)
    val_mae = float(np.mean(age_errors))

    print(f"Epoch {epoch+1:03d} | Balanced ACC: {balanced_acc:.4f} | Val MAE: {val_mae:.4f} | Weighted ACC: {weighted_acc:.4f} | F1: {f1:.4f} | ACC ED: {acc_ed:.4f} | ACC no ED: {acc_no_ed:.4f}") 

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

    return best_metrics, all_trues, all_preds, outputs_list


def load_best_params(file_path):
    best_params = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
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


def compute_class_weights_from_dataset(dataset):
    labels = [dataset.__label_extract__(path) for path in dataset.image_paths]
    counts = Counter(labels)
    total = sum(counts.values())
    num_classes = len(counts)
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)


def outer_cv(args):
    set_seed(42)
    outer_scores = []
    all_outer_trues, all_outer_preds = [], []

    for fold_idx in tqdm(range(args.n_outer_folds)):
        print(f"Outer Fold {fold_idx}")

        # TEMP: build a config to get a dataset and compute weights
        temp_cf = build_config(args, fixed_params={
            "batch_size": 8,
            "learning_rate": 1e-4,
            "dropout_rate": 0.1,
            "weight_decay": 1e-4,
            "img_size": 512,
            "patch_size": 32,
        })

        pretrained = "pretrained" if args.pretrained else "not_pretrained"
        best_params_file = f"Final/Models/Multi_Task/{args.model_type}/{pretrained}/fold_{fold_idx}/metrics_and_params.txt"
        best_params = load_best_params(best_params_file)

        cf = build_config(args, fixed_params=best_params)
        cf["n_epochs"] = epochs_dict[args.model_type][pretrained][fold_idx]
        args.n_epochs = epochs_dict[args.model_type][pretrained][fold_idx]
        cf["data_path"] = "/processing/g.rouwendaal/final/multiple_2D/IIEF15_01_12m/Train"
        train_ds, val_ds = prepare_fold_dataset(args.folds_path, fold_idx, 'outer', args=SimpleNamespace(**cf), age_dict=age_dict)
        if args.model_type == "hybrid_rvit" or args.model_type == "vit":
            mean, std = calculate_mean_std(train_ds)
            cf["normalize_override"] = (mean, std)
        model = build_model(cf)
        cf["data_path"] = "/processing/g.rouwendaal/final/single_2D/IIEF15_01_12m/Train"
        _, val_ds = prepare_fold_dataset(args.folds_path, fold_idx, 'outer', args=SimpleNamespace(**cf), age_dict=age_dict)


        train_loader = DataLoader(train_ds, batch_size=cf["batch_size"], shuffle=True, num_workers=4, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=cf["batch_size"], num_workers=4, persistent_workers=True)

        weights = compute_class_weights_from_dataset(train_ds)

        best_metrics, trues, preds, outputs_list = train_and_evaluate(model, train_loader, val_loader, cf, class_weights=weights)

        all_outer_trues.extend(trues)
        all_outer_preds.extend(preds)
        outer_scores.append(best_metrics["balanced_acc"])

        print(f'Fold {fold_idx} done | Balanced ACC: {best_metrics["balanced_acc"]:.4f}, Weighted ACC: {best_metrics["weighted_acc"]:.4f}, F1: {best_metrics["f1"]:.4f} | ACC ED: {best_metrics["acc_ed"]:.4f}, ACC no ED: {best_metrics["acc_no_ed"]:.4f}')

        pretrained = "pretrained" if ((cf["pretrained"] == "pretrained") or (cf["pretrained"] == True)) else "not_pretrained"


        save_dir = f"Final/Models/Multi_Task/multiple_2D_slices/{cf['model_type']}/{pretrained}/fold_{fold_idx}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "best_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model for Fold {fold_idx} at {save_path}")

        preds_path = os.path.join(save_dir, "predictions.npz")
        np.savez(preds_path, preds=preds, trues=trues, outputs=outputs_list)

        log_path = os.path.join(save_dir, "metrics_and_params.txt")
        with open(log_path, "w") as f:
            f.write(f"Best Hyperparameters:\n")
            for k, v in best_params.items():
                f.write(f"{k}: {v}\n")
            f.write("\nEvaluation Metrics:\n")
            f.write(f'Epoch: {best_metrics["epoch"]}\n')
            f.write(f'Balanced Accuracy: {best_metrics["balanced_acc"]:.4f}\n')
            f.write(f'Weighted Accuracy: {best_metrics["weighted_acc"]:.4f}\n')
            f.write(f'Weighted F1: {best_metrics["f1"]:.4f}\n')
            f.write(f'Accuracy ED: {best_metrics["acc_ed"]:.4f}\n')
            f.write(f'Accuracy no_ED: {best_metrics["acc_no_ed"]:.4f}\n')

    print(f"\nMean Balanced ACC: {np.mean(outer_scores):.4f}")


DATA_PATH = r"/data/groups/beets-tan/archive/edp_mri_seg/clinical.xlsx"
df = pd.read_excel(DATA_PATH)
df = df[["AnonymizedName", "age_at_diagnosis"]]

age_dict = dict(zip(df["AnonymizedName"], df["age_at_diagnosis"]))

if __name__ == "__main__":
    args = get_args()
    if args.pretrained == "pretrained":
        args.pretrained = True
    elif args.pretrained == "not_pretrained":
        args.pretrained = False
    print(args)
    outer_cv(args)