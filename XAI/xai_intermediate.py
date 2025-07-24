import argparse
# from Dataset.dataset_2D import prepare_fold_dataset, prepare_fold_dataset_multimodal
from Dataset.dataset_2D_1_3 import prepare_fold_dataset, calculate_mean_std
from models import build_model
from torch.utils.data import DataLoader
from types import SimpleNamespace
import pandas as pd
import torch.nn as nn
import torch

from sklearn.metrics import f1_score, accuracy_score
from collections import Counter

from tqdm import tqdm
import numpy as np
import os
import random

from preprocessing_utils import clean_data, filter_data
from sklearn.metrics import balanced_accuracy_score

import re
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import shap
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


def build_config_imaging(args, trial=None, fixed_params=None):
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


def build_config_clinical(args, trial=None, fixed_params=None):
    cf = {
        
    }

    if trial:
        cf.update({
            "learning_rate_clinical": trial.suggest_float("learning_rate_clinical", 1e-6, 1e-3, log=True),
            # "dropout_rate_clinical": trial.suggest_float("dropout_rate_clinical", 0.0, 0.5),
            "weight_decay_clinical": trial.suggest_float("weight_decay_clinical", 1e-6, 1e-2, log=True),
            "clinical_hidden_dim": trial.suggest_int("clinical_hidden_dim", 32, 128, step=32),
            "clinical_hidden_dim2": trial.suggest_int("clinical_hidden_dim2", 32, 128, step=32),
        })
        if cf["clinical_hidden_dim"] > cf["clinical_hidden_dim2"]:
            cf["clinical_hidden_dim"], cf["clinical_hidden_dim2"] = cf["clinical_hidden_dim2"], cf["clinical_hidden_dim"]
    elif fixed_params:
        cf.update(fixed_params)
    return cf


def build_config_fusion(args, trial=None, fixed_params=None):
    cf = {
        
    }

    if trial:
        cf.update({
            "learning_rate_classifier": trial.suggest_float("learning_rate_classifier", 1e-6, 1e-3, log=True),
            # "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
            "weight_decay_classifier": trial.suggest_float("weight_decay_classifier", 1e-6, 1e-2, log=True)
        })
    elif fixed_params:
        cf.update(fixed_params)

    return cf


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


def train_and_evaluate(model, train_loader, val_loader, args, cf_imaging, cf_clinical, cf_fusion, preprocessor, train_df, val_df, class_weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pattern = re.compile(r'(Anonymized Patient \d+)(?:_slice_\d+)?\.nrrd$')
    params = {
        "imaging_params" : model.imaging.parameters(),
        "clinical_params" : model.clinical.parameters(),
        "classifier_params" : model.classifier.parameters()
    }

    optimizer = torch.optim.AdamW([
    {'params': params['imaging_params'], 'lr': cf_imaging["learning_rate"], 'weight_decay': cf_imaging["weight_decay"]},
    {'params': params['clinical_params'], 'lr': cf_clinical["learning_rate_clinical"], 'weight_decay': cf_clinical["weight_decay_clinical"]},  
    {'params': params['classifier_params'], 'lr': cf_fusion["learning_rate_classifier"], 'weight_decay': cf_fusion["weight_decay_classifier"]},  
    ], weight_decay=0)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    best_balanced_acc = 0
    best_metrics = {}
    patience, n_patience = args.patience, 0

    all_preds, all_trues = [], []
    pat = re.compile(r'([Aa]?nonymized Patient \d+)(?:_slice_\d+)?\.nrrd$')
    for epoch in range(args.n_epochs):
        model.train()
        for sample in train_loader:
            imgs, labels, paths = sample
            patient_ids, new_keep_indices, clinial_features_list = [], [], []
            for i, path in enumerate(paths):
                filename = os.path.basename(path)
                match = pattern.search(filename)
                if match:
                    pid = match.group(1)
                    row = train_df[train_df["AnonymizedName"] == pid]
                    if not row.empty:
                        patient_ids.append(pid)
                        new_keep_indices.append(i)
                        clinial_features_list.append(row.iloc[0][USED_COLS])
                    else:
                        print(f"Warning: No clinical data found for patient ID '{pid}'. Skipping.")
            clinial_features = pd.DataFrame(clinial_features_list)
            clinial_features = preprocessor.transform(clinial_features).astype('float32')
            clinial_features = torch.tensor(clinial_features, dtype=torch.float32).to(device)
            imgs = imgs[new_keep_indices]
            labels = labels[new_keep_indices]
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs, clinial_features)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    preds, trues, outputs_list = [], [], []
    with torch.no_grad():
        for sample in val_loader:
            imgs, labels, paths = sample
            patient_ids, new_keep_indices, clinial_features_list = [], [], []
            for i, path in enumerate(paths):
                filename = os.path.basename(path)
                match = pattern.search(filename)
                if match:
                    pid = match.group(1)
                    row = val_df[val_df["AnonymizedName"] == pid]
                    if not row.empty:
                        patient_ids.append(pid)
                        new_keep_indices.append(i)
                        clinial_features_list.append(row.iloc[0][USED_COLS])
                    else:
                        print(f"Warning: No clinical data found for patient ID '{pid}'. Skipping.")
            clinial_features = pd.DataFrame(clinial_features_list)
            clinial_features = preprocessor.transform(clinial_features).astype('float32')
            clinial_features = torch.tensor(clinial_features, dtype=torch.float32).to(device)
            imgs = imgs[new_keep_indices]
            labels = labels[new_keep_indices]
            imgs= imgs.to(device)
            outputs = model(imgs, clinial_features)
            preds += outputs.argmax(dim=1).cpu().tolist()
            trues += labels.tolist()
            outputs_list += outputs.cpu().tolist()


        weighted_acc, acc_ed, acc_no_ed = compute_weighted_accuracy(trues, preds)
        f1 = f1_score(trues, preds, average="weighted")
        balanced_acc = balanced_accuracy_score(trues, preds)

        print(f"Epoch {epoch+1:03d} | Balanced ACC: {balanced_acc:.4f} | Weighted ACC: {weighted_acc:.4f} | F1: {f1:.4f} | ACC ED: {acc_ed:.4f} | ACC no ED: {acc_no_ed:.4f}") 


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
        

    return best_metrics, all_trues, all_preds, outputs_list


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


class ClinicalEncoder(nn.Module):
    def __init__(self, in_dim=12, emb_dim=64, hidden1=64, hidden2=128, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )

    def forward(self, x):
        return self.net(x.float())


class IntermediateFusionNet(nn.Module):
    def __init__(self, imaging_model, clinical_in=12, clinical_hidden1=64, clinical_hidden2=128, emb_dim=64, fusion_hidden=128, num_classes=2, freeze_imaging=False):
        super().__init__()
        self.imaging = imaging_model
        if freeze_imaging:
            for p in self.imaging.parameters():
                p.requires_grad_(False)

        self.clinical = ClinicalEncoder(clinical_in, emb_dim, clinical_hidden1, clinical_hidden2)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, num_classes)
        )

    def forward(self, imgs, clin_feats):
        img_emb  = self.imaging.forward_features(imgs)
        clin_emb = self.clinical(clin_feats)
        fused    = torch.cat([img_emb, clin_emb], dim=1)
        return self.classifier(fused)


def objective(trial, fold_idx, args, cf):
    inner_scores = []
    cf_imaging, cf_clinical, cf_fusion = build_config_imaging(args, trial), build_config_clinical(args, trial), build_config_fusion(args, trial)
    pat = re.compile(r'([Aa]?nonymized Patient \d+)(?:_slice_\d+)?\.nrrd$')
    for inner_idx in range(args.n_inner_folds):

        cf["data_path"] = <path to multiple slices>
        train_ds, val_ds = prepare_fold_dataset(args.folds_path, fold_idx, 'outer', args=SimpleNamespace(**cf))
        if args.model_type == "hybrid_rvit" or args.model_type == "vit":
            mean, std = calculate_mean_std(train_ds)
            cf["normalize_override"] = (mean, std)
        cf["data_path"] = <path to single slices>
        _, val_ds = prepare_fold_dataset(args.folds_path, fold_idx, 'outer', args=SimpleNamespace(**cf))
        adjust_imgs_and_labels(train_ds)
        adjust_imgs_and_labels(val_ds)

        weights = compute_class_weights_from_dataset(train_ds)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), NUM_COLS),                      
                ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS) 
            ],           
            sparse_threshold=0          
        )
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

        preprocessor.fit(train_df[USED_COLS])

        n_features = preprocessor.transform(train_df.iloc[:1]).shape[1]

        train_loader = DataLoader(train_ds, batch_size=cf_imaging["batch_size"], shuffle=True, num_workers=4, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=cf_imaging["batch_size"], num_workers=4, persistent_workers=True)

        imaging_model = build_model(SimpleNamespace(**cf_imaging))
        # extract the imaging model's feature extractor final shape
        example_img = next(iter(train_loader))[0]
        embedding_dim = None
        with torch.no_grad():
            embedding_dim = imaging_model.forward_features(example_img).shape[1]
            
        fusion_model = IntermediateFusionNet(imaging_model, clinical_in=n_features, clinical_hidden1=cf_clinical["clinical_hidden_dim"], 
                                            clinical_hidden2=cf_clinical["clinical_hidden_dim"], emb_dim=embedding_dim, fusion_hidden=2*embedding_dim, num_classes=2, freeze_imaging=False)

        best_metrics, trues, preds = train_and_evaluate(fusion_model, train_loader, val_loader, args, cf_imaging, cf_clinical, cf_fusion, preprocessor, train_df, val_df, class_weights=weights)
        inner_scores.append(best_metrics["balanced_acc"])

    return np.mean(inner_scores) 


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
    pattern = re.compile(r'(Anonymized Patient \d+)(?:_slice_\d+)?\.nrrd$')
    for fold_idx in tqdm(range(0, args.n_outer_folds)):
        print(f"Outer Fold {fold_idx}")

        temp_cf = build_config(args, fixed_params={
            "batch_size": 8,
            "learning_rate": 1e-4,
            "dropout_rate": 0.1,
            "weight_decay": 1e-4,
            "img_size": 512,
            "patch_size": 32,
        })

        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pretrained = "pretrained" if args.pretrained else "not_pretrained"
        best_params_file = f"Final/Models/Intermediate_Fusion/{args.model_type}/{pretrained}/fold_{fold_idx}/metrics_and_params.txt"
        best_params = load_best_params(best_params_file)

        cf = build_config(args, fixed_params=best_params)
        cf["n_epochs"] = 13
        cf["data_path"] = "/processing/g.rouwendaal/final/multiple_2D/IIEF15_01_12m/Train"
        train_ds, val_ds = prepare_fold_dataset(
            args.folds_path, fold_idx, 'outer', args=SimpleNamespace(**cf)
        )
        cf["data_path"] = "/processing/g.rouwendaal/final/single_2D/IIEF15_01_12m/Train"
        _, val_ds = prepare_fold_dataset(args.folds_path, fold_idx, 'outer', args=SimpleNamespace(**cf))
        adjust_imgs_and_labels(train_ds)
        adjust_imgs_and_labels(val_ds)
        
        weights = compute_class_weights_from_dataset(train_ds)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), NUM_COLS),                      
                ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS) 
            ],           
            sparse_threshold=0          
        )
        train_ids = []
        for p in train_ds.image_paths:
            filename = os.path.basename(p)
            match = pattern.search(filename)
            if match:
                train_ids.append(match.group(1))
            else:
                print(f"Warning: '{filename}' did not match expected pattern. Skipping.")

        val_ids = []
        for path in val_ds.image_paths:
            filename = os.path.basename(path)
            match = pattern.search(filename)
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

        preprocessor.fit(train_df[USED_COLS])

        n_features = preprocessor.transform(train_df.iloc[:1]).shape[1]

        train_loader = DataLoader(train_ds, batch_size=cf["batch_size"], shuffle=True, num_workers=4, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=cf["batch_size"], num_workers=4, persistent_workers=True)

        imaging_model = build_model(SimpleNamespace(**cf))
        # extract the imaging model's feature extractor final shape
        example_img = next(iter(train_loader))[0]
        embedding_dim = None
        with torch.no_grad():
            embedding_dim = imaging_model.forward_features(example_img).shape[1]
            
        fusion_model = IntermediateFusionNet(imaging_model, clinical_in=n_features, clinical_hidden1=cf["clinical_hidden_dim"], 
                                            clinical_hidden2=cf["clinical_hidden_dim"], emb_dim=embedding_dim, fusion_hidden=2*embedding_dim, num_classes=2, freeze_imaging=False)


        fusion_model.load_state_dict(torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))

        fusion_model.eval()
        fusion_model.to(device)

        def classifier_forward(x_numpy):
            x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
            x_tensor = x_tensor.to(device)
            with torch.no_grad():
                return fusion_model.classifier(x_tensor).cpu().numpy()


        fused_vectors = []
        labels = []

        with torch.no_grad():
            for imgs, lbls, paths in val_loader:
                imgs = imgs.to(device)
                clin_feats_batch = []
                for path in paths:
                    filename = os.path.basename(path)
                    match = pattern.search(filename)
                    pid = match.group(1)
                    row = val_df[val_df["AnonymizedName"] == pid]
                    clin_feats_batch.append(row.iloc[0][USED_COLS])

                clin_feats_batch = pd.DataFrame(clin_feats_batch)
                clin_feats_batch = preprocessor.transform(clin_feats_batch).astype("float32")
                clin_feats_batch = torch.tensor(clin_feats_batch).to(device)

                img_emb = fusion_model.imaging.forward_features(imgs)
                clin_emb = fusion_model.clinical(clin_feats_batch)
                fused = torch.cat([img_emb, clin_emb], dim=1)
                fused_vectors.append(fused.cpu())
                labels.extend(lbls.tolist())

        fused_input = torch.cat(fused_vectors, dim=0).numpy()

        explainer = shap.Explainer(classifier_forward, fused_input, max_evals=2050)
        shap_values = explainer(fused_input)

        D1 = img_emb.shape[1]
        D2 = clin_emb.shape[1]

        modality_contributions = {
            "imaging": np.abs(shap_values.values[:, :D1]).mean(),
            "clinical": np.abs(shap_values.values[:, D1:]).mean()
        }

        print("Modality Contributions (mean |SHAP|):", modality_contributions)

        OUTPUT_PATH = f"/home/g.rouwendaal/imgs/XAI/intermediate/ResNet_1_3/{fold_idx}"
        os.makedirs(OUTPUT_PATH, exist_ok=True)

        plt.bar(modality_contributions.keys(), modality_contributions.values())
        plt.title("Mean SHAP Contribution per Modality (Fold 4)")
        plt.ylabel("Mean |SHAP value|")
        plt.xlabel("Modality")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, f"modality_contributions_fold_{fold_idx}.png"))
        plt.close()


DATA_PATH = <path to clinical data>
DATA_PATH_LABELS = <path to labels>
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

NUM_COLS = [c for c in USED_COLS if c not in CAT_COLS]

data = clean_data(DATA_PATH, DATA_PATH_LABELS)
data = filter_data(data, TARGET_TIME)

if __name__ == "__main__":
    args = get_args()
    if args.pretrained == "pretrained":
        args.pretrained = True
    elif args.pretrained == "not_pretrained":
        args.pretrained = False
    print(args)
    outer_cv(args)