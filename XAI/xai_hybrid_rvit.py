import argparse
from Dataset.dataset_2D_1_3 import prepare_fold_dataset, calculate_mean_std
from models import build_model
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
            cf["hidden_dim"] = int(hidden_dim)
            cf["num_heads"] = int(num_heads)
            cf["img_size"] = 512
            cf["num_patches"] = int((cf["img_size"] // cf["patch_size"]) ** 2)
    return cf



def train_and_evaluate(model, train_loader, val_loader, cf, class_weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cf["learning_rate"], weight_decay=cf["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    best_weighted_acc = 0
    best_metrics = {}
    patience, n_patience = cf["patience"], 0

    all_preds, all_trues = [], []

    for epoch in range(cf["n_epochs"]):
        model.train()
        for imgs, labels in train_loader:
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
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds += outputs.argmax(dim=1).cpu().tolist()
                trues += labels.tolist()


        weighted_acc, acc_ed, acc_no_ed = compute_weighted_accuracy(trues, preds)
        f1 = f1_score(trues, preds, average="weighted")
        balanced_acc = balanced_accuracy_score(trues, preds)

        print(f"Epoch {epoch+1:03d} | Balanced ACC: {balanced_acc:.4f} | Weighted ACC: {weighted_acc:.4f} | F1: {f1:.4f} | ACC ED: {acc_ed:.4f} | ACC no ED: {acc_no_ed:.4f}") 

        if weighted_acc > best_weighted_acc:
            best_weighted_acc = weighted_acc
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

def objective(trial, fold_idx, args, weights):
    inner_scores = []

    for inner_idx in range(args.n_inner_folds):
        cf = build_config(args, trial)

        train_ds, val_ds = prepare_fold_dataset(
            args.folds_path, fold_idx, 'inner', inner_idx, args=SimpleNamespace(**cf)
        )

        train_loader = DataLoader(train_ds, batch_size=cf["batch_size"], shuffle=False, num_workers=4, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=cf["batch_size"], num_workers=4, persistent_workers=True)

        model = build_model(SimpleNamespace(**cf))
        best_metrics, trues, preds = train_and_evaluate(model, train_loader, val_loader, cf, class_weights=weights)
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



def generate_2d_heatmap(model, x, activation):
    feature_maps = activation['last_layer']  # [1, 512, 16, 16]
    attention_maps = model.attention_maps    # [1, num_layers, heads, 257, 257]
    
    # Take attention from CLS to patches
    cls_attention = model.attention_maps.mean(dim=1)
    
    # Normalize
    cls_attention = cls_attention / cls_attention.max(dim=1, keepdim=True)[0]
    
    # Reshape to match feature map spatial dimensions
    cls_attention = cls_attention.view(1, 1, 16, 16)  # [1, 1, 16, 16]
    
    # Apply attention
    weighted_feature_maps = feature_maps * cls_attention  # [1, 512, 16, 16]
    
    # Average across channels
    heatmap = weighted_feature_maps.mean(dim=1, keepdim=True)  # [1, 1, 16, 16]
    heatmap = torch.relu(heatmap)
    heatmap = heatmap / heatmap.max()
    # Optional: Upsample to match input image size
    heatmap_upsampled = torch.nn.functional.interpolate(heatmap, size=(512, 512), mode='bilinear', align_corners=False)
    
    # Return as numpy array
    return heatmap_upsampled.squeeze().detach().cpu().numpy()


def get_activation(name, activation_dict):
    def hook(model, input, output):
        activation_dict[name] = output.detach()
    return hook



def plot_grid_of_heatmaps(image_heatmap_list, save_path=None, alpha=0.5, cmap='jet', title=None):
    n_images = len(image_heatmap_list)
    n_rows = n_images
    n_cols = 3  # Original, Heatmap, Overlay

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    if n_rows == 1:
        axes = [axes]  # Single row case

    for row_idx in range(n_rows):
        heatmap, image = image_heatmap_list[row_idx]
        image = image.squeeze()
        heatmap = heatmap.squeeze()

        # Column 1: Original Image
        ax_img = axes[row_idx][0] if n_rows > 1 else axes[0]
        ax_img.imshow(image, cmap='gray')
        ax_img.axis('off')
        ax_img.set_title(f"Original {row_idx+1}")

        # Column 2: Heatmap Only
        ax_hm = axes[row_idx][1] if n_rows > 1 else axes[1]
        ax_hm.imshow(heatmap, cmap=cmap)
        ax_hm.axis('off')
        ax_hm.set_title(f"Heatmap {row_idx+1}")

        # Column 3: Overlay
        ax_overlay = axes[row_idx][2] if n_rows > 1 else axes[2]
        ax_overlay.imshow(image, cmap='gray')
        ax_overlay.imshow(heatmap, cmap=cmap, alpha=alpha)
        ax_overlay.axis('off')
        ax_overlay.set_title(f"Overlay {row_idx+1}")

    if title:
        fig.suptitle(title, fontsize=20)

    fig.subplots_adjust(wspace=0.15, hspace=0.25)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_heatmap_grid_by_class_transposed(columns, save_path=None, alpha=0.5, cmap='jet', col_titles=None, general_title=None):
    n_cols = len(columns)
    n_rows = 2  # Fixed for 4x4 layout

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows))

    # Ensure axes is always a 2D list
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for col_idx, column in enumerate(columns):
        for row_idx in range(min(len(column), n_rows)):
            heatmap, image = column[row_idx]
            image = image.squeeze()
            heatmap = heatmap.squeeze()

            ax = axes[row_idx][col_idx]
            ax.imshow(image, cmap='gray')
            ax.imshow(heatmap, cmap=cmap, alpha=alpha)
            ax.axis('off')

            if col_titles and row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=14, fontweight='bold')

    plt.subplots_adjust(wspace=0.05, hspace=0.1, top=0.88)  # leaves room for the suptitle

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def outer_cv(args):
    set_seed(42)
    for fold_idx in tqdm(range(args.n_outer_folds)):
        print(f"Outer Fold {fold_idx}")

        pretrained = "pretrained" if ((args.pretrained == "pretrained") or (args.pretrained == True)) else "not_pretrained"
        best_params_file = <PATH TO BEST PARAMS FILE>
        best_params = load_best_params(best_params_file)
        best_model_path = f"Final/Models/multiple_2D_slices_1_3/{args.model_type}/{pretrained}/fold_{fold_idx}/best_model.pth"
        cf = build_config(args, fixed_params=best_params)
        print(cf)

        model_deep_learning = build_model(SimpleNamespace(**cf))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(best_model_path, map_location=device)

        # Ensure positional embeddings are compatible
        num_patches = state_dict['pos_embed.weight'].shape[0]
        hidden_dim = state_dict['pos_embed.weight'].shape[1]
        model_deep_learning.pos_embed = nn.Embedding(num_patches, hidden_dim)
        model_deep_learning.load_state_dict(state_dict)

        cf["data_path"] = <path to multiple slices>
        train_ds, val_ds = prepare_fold_dataset(args.folds_path, fold_idx, 'outer', args=SimpleNamespace(**cf))
        if args.model_type == "hybrid_rvit" or args.model_type == "vit":
            mean, std = calculate_mean_std(train_ds)
            cf["normalize_override"] = (mean, std)
        cf["data_path"] = <path to single slices>
        _, val_ds = prepare_fold_dataset(args.folds_path, fold_idx, 'outer', args=SimpleNamespace(**cf))

        val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, shuffle=True, persistent_workers=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_deep_learning.to(device)
        model_deep_learning.eval()

        # Prepare output folder
        OUTPUT_PATH = f"imgs/XAI/hybrid_rvit/multiple_2D_slices_1_3/outer_fold_{fold_idx}/"
        os.makedirs(OUTPUT_PATH, exist_ok=True)

        # Initialize class-wise groups
        imgs_TN, imgs_TP, imgs_FN, imgs_FP = [], [], [], []
        activation = {}
        model_deep_learning.backbone[-1].register_forward_hook(get_activation('last_layer', activation))

        with torch.no_grad():
            for img, label in tqdm(val_loader):
                if all(len(lst) >= 5 for lst in [imgs_TN, imgs_TP, imgs_FN, imgs_FP]):
                    break

                img = img.to(device)
                output = model_deep_learning(img)
                pred = output.argmax(dim=1).item()
                label = label.item()

                heatmap = generate_2d_heatmap(model_deep_learning, img, activation)
                item = [heatmap, img.cpu().numpy()]

                if label == 0 and pred == 0 and len(imgs_TN) < 5:
                    imgs_TN.append(item)  # Correct ED = TN
                elif label == 1 and pred == 1 and len(imgs_TP) < 5:
                    imgs_TP.append(item)  # Correct no ED = TP
                elif label == 0 and pred == 1 and len(imgs_FN) < 5:
                    imgs_FN.append(item)  # Incorrect ED = FN
                elif label == 1 and pred == 0 and len(imgs_FP) < 5:
                    imgs_FP.append(item)  # Incorrect no ED = FP

        print(f"Collected | TN: {len(imgs_TN)}, TP: {len(imgs_TP)}, FN: {len(imgs_FN)}, FP: {len(imgs_FP)}")

        # Plot individual class grids
        plot_grid_of_heatmaps(imgs_TN,
                              save_path=os.path.join(OUTPUT_PATH, 'correct_ED.png'),
                              alpha=0.5, cmap='jet', title='Correct ED (True Negatives)')

        plot_grid_of_heatmaps(imgs_TP,
                              save_path=os.path.join(OUTPUT_PATH, 'correct_no_ED.png'),
                              alpha=0.5, cmap='jet', title='Correct no ED (True Positives)')

        plot_grid_of_heatmaps(imgs_FN,
                              save_path=os.path.join(OUTPUT_PATH, 'incorrect_ED.png'),
                              alpha=0.5, cmap='jet', title='Incorrect ED (False Negatives)')

        plot_grid_of_heatmaps(imgs_FP,
                              save_path=os.path.join(OUTPUT_PATH, 'incorrect_no_ED.png'),
                              alpha=0.5, cmap='jet', title='Incorrect no ED (False Positives)')

        plot_heatmap_grid_by_class_transposed([imgs_TN, imgs_TP, imgs_FN, imgs_FP],
            save_path=os.path.join(OUTPUT_PATH, 'final_summary_grid.svg'), alpha=0.5,
            col_titles=["Correct ED (TN)", "Correct no ED (TP)", "Incorrect ED (FN)", "Incorrect no ED (FP)"],
            general_title="Hybrid-RViT XAI Results")


if __name__ == "__main__":
    args = get_args()
    if args.pretrained == "pretrained":
        args.pretrained = True
    elif args.pretrained == "not_pretrained":
        args.pretrained = False
    print(args)
    outer_cv(args)