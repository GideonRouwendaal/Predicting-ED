import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import RandomizedSearchCV

from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tqdm import tqdm
import numpy as np


import warnings
from sklearn.model_selection import PredefinedSplit
import json

warnings.filterwarnings("ignore", category=FutureWarning)



def plot_dodecagon(img_output_path, rearranged_dict, largest_length, patient):
    labels = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'L6', 'L5', 'L4', 'L3', 'L2', 'L1']
    angles, values = list(rearranged_dict.keys()), list(rearranged_dict.values())

    values += values[:1]
    angles += angles[:1]
    values = [(value / largest_length) * 100 for value in values]

    shifted_angles = [(angle + np.pi / 12) % (2 * np.pi) for angle in angles]

    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(polar=True))

    ax.fill(shifted_angles, values, color='purple', alpha=0.25)
    ax.plot(shifted_angles, values, color='purple', linewidth=2)

    ax.set_xticks(shifted_angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    max_value = 100
    ax.set_ylim(0, max_value)
    ax.set_yticks(np.linspace(0, max_value, 5))
    ax.set_yticklabels([f'{int(i)}%' for i in np.linspace(0, max_value, 5)], fontsize=8, rotation=45)
    
    for grid_level in np.linspace(0, max_value, 5):
        ax.plot(angles, [grid_level] * len(angles), color='gray', linewidth=0.75)

    ax.spines['polar'].set_visible(False)  
    ax.grid(False)

    for angle in angles[:-1]: 
        ax.plot([angle, angle], [0, max_value], color='gray', linewidth=0.75, linestyle='-')

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_title(f"{patient}", fontweight='bold', pad=20)

    plt.savefig(img_output_path, bbox_inches='tight')


def flatten_fascia_thickness_dict(nested_dict):
    flattened_dict = {}

    for patient_id, slice_dict in nested_dict.items():
        flattened_patient = {}
        for slice_idx, regions in slice_dict.items():
            for region, value in regions.items():
                key = f"slice_{slice_idx}_{region}"
                flattened_patient[key] = value
        flattened_dict[patient_id] = flattened_patient

    return flattened_dict


def compute_stats(fascia_data, axis='slice'):
    stat_funcs = {'mean': np.mean, 'sum': np.sum, 'median': np.median}
    stats = {k: {} for k in stat_funcs}
    region_labels = ['R1','R2','R3','R4','R5','R6','L6','L5','L4','L3','L2','L1']

    if axis == 'slice':
        for patient in fascia_data.values():
            for slice_idx_str, region_dict in patient.items():
                slice_idx = int(slice_idx_str)
                values = list(region_dict.values())
                for stat, func in stat_funcs.items():
                    stats[stat].setdefault(slice_idx, []).append(func(values))

    elif axis == 'region':
        for patient in fascia_data.values():
            region_values = {region: [] for region in region_labels}
            for region_dict in patient.values():
                for region, value in region_dict.items():
                    region_values[region].append(value)
            for region in region_labels:
                for stat, func in stat_funcs.items():
                    stats[stat].setdefault(region, []).append(func(region_values[region]))

    return stats

def plot_grouped_comparison_boxplots_from_dicts(ed_dict, no_ed_dict, output_path="boxplot_comparison.png"):
    stats = ["mean", "sum", "median"]
    titles = ["Mean Fascia Thickness", "Sum of Fascia Thickness", "Median Fascia Thickness"]
    ed_color = "red"
    no_ed_color = "blue"

    labels = list(ed_dict["mean"].keys())
    labels.sort() 

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)

    for i, stat in enumerate(stats):
        ed_data = [ed_dict[stat][label] for label in labels]
        no_ed_data = [no_ed_dict[stat][label] for label in labels]

        positions_ed = x - width/2
        positions_no_ed = x + width/2

        bp_ed = axes[i].boxplot(ed_data, positions=positions_ed, widths=width,
                                patch_artist=True, showmeans=True)
        bp_no_ed = axes[i].boxplot(no_ed_data, positions=positions_no_ed, widths=width,
                                   patch_artist=True, showmeans=True)

        for patch in bp_ed['boxes']:
            patch.set_facecolor(ed_color)
            patch.set_alpha(0.6)
        for patch in bp_no_ed['boxes']:
            patch.set_facecolor(no_ed_color)
            patch.set_alpha(0.6)

        axes[i].set_title(titles[i])
        axes[i].set_ylabel("Fascia Thickness")
        axes[i].grid(axis='y', linestyle='--', alpha=0.6)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45)
    xlab = "Region" if "region" in output_path else "Slice"
    axes[-1].set_xlabel(xlab)

    fig.legend(["ED", "No ED"], loc="upper right", fontsize="large")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved grouped comparison boxplot to: {output_path}")


def get_region_stat_dict(region_data, stat='mean'):
    angle_mapping = dict(zip(
        ['R1','R2','R3','R4','R5','R6','L6','L5','L4','L3','L2','L1'],
        np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ))
    if stat == 'mean':
        reducer = np.mean
    elif stat == 'median':
        reducer = np.median
    else:
        raise ValueError("Use 'mean' or 'median'.")

    region_stats = {angle_mapping[region]: reducer([float(v) for v in values]) for region, values in region_data.items()}
    return region_stats


def compute_dodecagon_slice_stats(fascia_data):
    num_slices = max(int(s) for patient in fascia_data.values() for s in patient.keys()) + 1
    region_labels = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'L6', 'L5', 'L4', 'L3', 'L2', 'L1']

    slice_region_values = {slice_idx: {region: [] for region in region_labels} for slice_idx in range(num_slices)}

    for patient_data in fascia_data.values():
        for slice_idx_str, region_dict in patient_data.items():
            slice_idx = int(slice_idx_str)
            for region, value in region_dict.items():
                slice_region_values[slice_idx][region].append(value)

    return slice_region_values


def plot_dodecagon(img_output_path, rearranged_dict, largest_length, patient):
    labels = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'L6', 'L5', 'L4', 'L3', 'L2', 'L1']
    angles, values = list(rearranged_dict.keys()), list(rearranged_dict.values())

    values += values[:1]
    angles += angles[:1]
    values = [(value / largest_length) * 100 for value in values]

    shifted_angles = [(angle + np.pi / 12) % (2 * np.pi) for angle in angles]

    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(polar=True))

    ax.fill(shifted_angles, values, color='purple', alpha=0.25)
    ax.plot(shifted_angles, values, color='purple', linewidth=2)

    ax.set_xticks(shifted_angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    max_value = 100
    ax.set_ylim(0, max_value)
    ax.set_yticks(np.linspace(0, max_value, 5))
    ax.set_yticklabels([f'{int(i)}%' for i in np.linspace(0, max_value, 5)], fontsize=8, rotation=45)
    
    for grid_level in np.linspace(0, max_value, 5):
        ax.plot(angles, [grid_level] * len(angles), color='gray', linewidth=0.75)

    ax.spines['polar'].set_visible(False)  
    ax.grid(False)

    for angle in angles[:-1]: 
        ax.plot([angle, angle], [0, max_value], color='gray', linewidth=0.75, linestyle='-')

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_title(f"{patient}", fontweight='bold', pad=20)

    plt.savefig(img_output_path, bbox_inches='tight')


def plot_dodecagons_per_slice(slice_region_values, output_dir, largest_length, stat, l="ED", n=0):
    os.makedirs(output_dir, exist_ok=True)
    for slice_idx, region_dict in slice_region_values.items():
        rearranged = get_region_stat_dict(region_dict, stat=stat)
        plot_dodecagon(
            img_output_path=os.path.join(output_dir, f"slice_{slice_idx}.png"),
            rearranged_dict=rearranged,
            largest_length=largest_length,
            patient=f"Slice {slice_idx} - {l} - {stat} - n=({n})"
        )


def plot_overall_dodecagons(slice_region_values, output_dir, largest_length, l="ED", n=0):
    all_regions = {region: [] for region in ['R1','R2','R3','R4','R5','R6','L6','L5','L4','L3','L2','L1']}
    os.makedirs(output_dir, exist_ok=True)
    for region_dict in slice_region_values.values():
        for region, values in region_dict.items():
            all_regions[region].extend(values)

    mean_dict = get_region_stat_dict(all_regions, stat='mean')
    median_dict = get_region_stat_dict(all_regions, stat='median')

    plot_dodecagon(os.path.join(output_dir, "mean.png"), mean_dict, largest_length, patient=f"Overall slices - {l} - mean - n=({n})")
    plot_dodecagon(os.path.join(output_dir, "median.png"), median_dict, largest_length, patient=f"Overall slices - {l} - median - n=({n})")

def plot_ed_no_ed_features(df, fascia_thickness_dict, label, approach):

    full_img_output_path = os.path.join(img_output_path, approach, label)
    os.makedirs(full_img_output_path, exist_ok=True)

    ed_patients = set(df[df[label] < 4]["AnonymizedName"])
    no_ed_patients = set(df[df[label] >= 4]["AnonymizedName"])
    fascia_thickness_ed = {k: v for k, v in fascia_thickness_dict.items() if k in ed_patients}
    fascia_thickness_no_ed = {k: v for k, v in fascia_thickness_dict.items() if k in no_ed_patients}

    slice_stats_ed, slice_stats_no_ed = compute_stats(fascia_thickness_ed, axis='slice'), compute_stats(fascia_thickness_no_ed, axis='slice')
    region_stats_ed, region_stats_no_ed = compute_stats(fascia_thickness_ed, axis='region'), compute_stats(fascia_thickness_no_ed, axis='region')

    plot_grouped_comparison_boxplots_from_dicts(
    ed_dict=slice_stats_ed,
    no_ed_dict=slice_stats_no_ed,
    output_path=os.path.join(full_img_output_path, "Boxplot_slices.png")
    )

    plot_grouped_comparison_boxplots_from_dicts(
    ed_dict=region_stats_ed,
    no_ed_dict=region_stats_no_ed,
    output_path=os.path.join(full_img_output_path, "Boxplot_regions.png")
    )

    
    slice_region_values_ed, slice_region_values_no_ed = compute_dodecagon_slice_stats(fascia_thickness_ed), compute_dodecagon_slice_stats(fascia_thickness_no_ed)

    largest_length_ed = np.mean([
        max([v for region_dict in patient_slices.values() for v in region_dict.values()])
        for patient_slices in fascia_thickness_ed.values()
    ])

    largest_length_no_ed = np.mean([
        max([v for region_dict in patient_slices.values() for v in region_dict.values()])
        for patient_slices in fascia_thickness_no_ed.values()
    ])

    largest_length = max(largest_length_ed, largest_length_no_ed)

    plot_dodecagons_per_slice(slice_region_values_ed, output_dir=os.path.join(full_img_output_path, f"Mean_ED_patients"), largest_length=largest_length, stat='mean', l="ED", n=len(ed_patients))
    plot_dodecagons_per_slice(slice_region_values_ed, output_dir=os.path.join(full_img_output_path, f"Median_ED_patients"), largest_length=largest_length, stat='median', l="ED", n=len(ed_patients))
    plot_overall_dodecagons(slice_region_values_ed, output_dir=os.path.join(full_img_output_path, f"Overall_ED_patients"), largest_length=largest_length, l="ED", n=len(ed_patients))

    plot_dodecagons_per_slice(slice_region_values_no_ed, output_dir=os.path.join(full_img_output_path, f"Mean_no_ED_patients"), largest_length=largest_length, stat='mean', l="no_ED", n=len(no_ed_patients))
    plot_dodecagons_per_slice(slice_region_values_no_ed, output_dir=os.path.join(full_img_output_path, f"Median_no_ED_patients"), largest_length=largest_length, stat='median', l="no_ED", n=len(no_ed_patients))
    plot_overall_dodecagons(slice_region_values_no_ed, output_dir=os.path.join(full_img_output_path, f"Overall_no_ED_patients"), largest_length=largest_length, l="no_ED", n=len(no_ed_patients))


def prepare_df(fascia_thickness_dict, df, to_predict_label):
    features = list(fascia_thickness_dict[list(fascia_thickness_dict.keys())[0]].keys())
    feature_df = pd.DataFrame.from_dict(fascia_thickness_dict, orient="index").reset_index()
    feature_df.rename(columns={"index": "AnonymizedName"}, inplace=True)

    print(f"In total, there are {len(feature_df)} with an image and segmentation")

    print(f"In total, there are {len(df)} with an image, segmentation, and preop score >= 4")

    df = df.dropna(subset=[to_predict_label])
    df["label"] = (df[to_predict_label] >= 4).astype(int)

    print(f"In total, there are {len(df)} with an image, segmentation, preop score >= 4, and postop score after {to_predict_label} (can be any)")

    df_merged = df.merge(feature_df, on="AnonymizedName", how="inner")
    return features, df_merged


def find_best_model(subset_df, features, subset_df_test, features_test, folds):
    best_models = {}

    print(subset_df.head())

    # Iterate over outer folds
    for fold_name, fold_data in tqdm(folds.items(), desc="Outer Fold Progress"):
        outer_train_ids = fold_data["outer_train"]
        outer_val_ids = fold_data["outer_val"]
        inner_folds = fold_data["inner_folds"]

        # Extract outer train and validation data
        outer_train_df = subset_df[subset_df["AnonymizedName"].isin(outer_train_ids)]
        outer_val_df = subset_df[subset_df["AnonymizedName"].isin(outer_val_ids)]

        X_outer_train = outer_train_df[features]
        y_outer_train = outer_train_df["label"]
        X_outer_val = outer_val_df[features]
        y_outer_val = outer_val_df["label"]

        # Define models with pipelines
        possible_models = {
            "LogisticRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
            ]),
            "RandomForest": Pipeline([
                ("clf", RandomForestClassifier(class_weight="balanced"))
            ]),
            "SVM": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, class_weight="balanced"))
            ]),
            "KNN": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier())
            ]),
            "Naive Bayes": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GaussianNB())
            ])
        }

        # Randomized search parameter grids
        random_param_grid = {
            "LogisticRegression": {
                "clf__C": np.logspace(-3, 2, 10),
                "clf__solver": ["lbfgs", "liblinear"],
                "clf__penalty": ["l2"]
            },
            "RandomForest": {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [None, 10, 20, 30],
                "clf__min_samples_split": [2, 5, 10],
                "clf__max_features": ["sqrt", "log2"]
            },
            "SVM": {
                "clf__C": np.logspace(-2, 2, 8),
                "clf__kernel": ["linear", "rbf", "poly"],
                "clf__gamma": ["scale", "auto"]
            },
            "KNN": {
                "clf__n_neighbors": list(range(3, 15)),
                "clf__weights": ["uniform", "distance"],
                "clf__metric": ["euclidean", "manhattan"]
            },
            "Naive Bayes": {
                "clf__var_smoothing": np.logspace(-9, -6, 5)
            }
        }

        # Iterate over models
        for model_name, pipeline in possible_models.items():
            print(f"\nOuter Fold: {fold_name}, Model: {model_name}")

            # Collect training data for hyperparameter tuning
            X_inner_train_list, y_inner_train_list, fold_indices = [], [], []

            for i, (inner_name, inner_data) in enumerate(inner_folds.items()):
                inner_train_ids = inner_data["train"]
                inner_val_ids = inner_data["val"]
                fold_df = subset_df[subset_df["AnonymizedName"].isin(inner_train_ids + inner_val_ids)]
                X_fold = fold_df[features]
                y_fold = fold_df["label"]
                val_mask = fold_df["AnonymizedName"].isin(inner_val_ids)
                fold_index = np.full(len(fold_df), -1)
                fold_index[val_mask.values] = i
                X_inner_train_list.append(X_fold)
                y_inner_train_list.append(y_fold)
                fold_indices.extend(fold_index)
            X_inner_train = pd.concat(X_inner_train_list)
            y_inner_train = pd.concat(y_inner_train_list)
            ps = PredefinedSplit(test_fold=fold_indices)
            # Hyperparameter tuning using inner folds
            random_search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=random_param_grid[model_name],
                n_iter=50,
                scoring="balanced_accuracy",
                n_jobs=-1,
                random_state=42,
                cv=ps,
            )
            random_search.fit(X_inner_train, y_inner_train)
            # Best model after hyperparameter tuning
            best_params = random_search.best_params_

            best_model = clone(pipeline)
            best_model.set_params(**best_params)
            best_model.fit(X_outer_train, y_outer_train)

            # Evaluate on outer validation set
            y_pred_outer = best_model.predict(X_outer_val)
            y_pred_probs_outer = best_model.predict_proba(X_outer_val)[:, 1]

            acc_outer = accuracy_score(y_outer_val, y_pred_outer)
            bal_acc_outer = balanced_accuracy_score(y_outer_val, y_pred_outer)

            print(f"Outer Validation Accuracy for {model_name}: {acc_outer:.4f}")
            print(f"Outer Validation Balanced Accuracy for {model_name}: {bal_acc_outer:.4f}")

            # Save predictions for outer fold
            save_dir = os.path.join(f"Final/Models/FT_single/{model_name}", fold_name)
            os.makedirs(save_dir, exist_ok=True)
            np.savez(os.path.join(save_dir, "outer_predictions.npz"), trues=y_outer_val, preds=y_pred_outer, outputs=y_pred_probs_outer)

            # Save best model for outer fold
            best_models[(fold_name, model_name)] = best_model

    return best_models



def get_feature_importances(best_models, features):
    model_feature_importances = defaultdict(lambda: defaultdict(list))

    for (fold_name, model_name), model in best_models.items():
        clf = model.named_steps["clf"]

        if hasattr(clf, "coef_"):  # Logistic Regression or linear SVM
            importances = np.abs(clf.coef_[0])
        elif hasattr(clf, "feature_importances_"):  # Random Forest
            importances = clf.feature_importances_
        else:
            print(f"Skipping feature importance for {model_name} (not supported).")
            continue

        for feature, importance in zip(features, importances):
            model_feature_importances[model_name][feature].append(importance)

    # Average over folds
    averaged_importances = {}
    for model_name, feature_dict in model_feature_importances.items():
        averaged_importances[model_name] = {
            feature: np.mean(values)
            for feature, values in feature_dict.items()
        }

    return averaged_importances


from collections import defaultdict

def aggregate_importances(importances_dict, mode="slice"):
    aggregated = defaultdict(float)

    for feature_name, importance in importances_dict.items():
        parts = feature_name.split("_")
        if len(parts) >= 3:
            _, slice_num, region = parts
            key = slice_num if mode == "slice" else region
            aggregated[key] += importance

    return dict(aggregated)


def plot_importances(aggregated_importances, title, output_path, features):
    items = sorted(aggregated_importances.items(), key=lambda x: x[0])
    labels, values = zip(*items)
    plt.figure(figsize=(12, 5))
    sns.barplot(x=['R1','R2','R3','R4','R5','R6','L6','L5','L4','L3','L2','L1'], y=values)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(title)
    print(aggregated_importances)
    print(f"Saved plot: {output_path}")


def get_region_stat_dict_all(slice_region_values, stat='mean'):
    labels = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'L6', 'L5', 'L4', 'L3', 'L2', 'L1']
    regions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    region_values = {region: [] for region in regions}

    for patient_data in slice_region_values.values():
        for region, value in patient_data.items():
            if int(region) in region_values:
                region_values[int(region)].append(value)
    
    if stat == 'mean':
        summarized = {region: np.mean(values) if values else 0 for region, values in region_values.items()}
        summarized = {labels[i]: summarized[i] for i in range(len(labels))}
    elif stat == 'median':
        summarized = {region: np.median(values) if values else 0 for region, values in region_values.items()}
        summarized = {labels[i]: summarized[i] for i in range(len(labels))}
    return summarized


def plot_dodecagon(img_output_path, rearranged_dict, largest_length, patient):
    labels = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'L6', 'L5', 'L4', 'L3', 'L2', 'L1']
    angles, values = list(rearranged_dict.keys()), list(rearranged_dict.values())

    values += values[:1]
    angles += angles[:1]
    values = [(value / largest_length) * 100 for value in values]

    shifted_angles = [(angle + np.pi / 12) % (2 * np.pi) for angle in angles]

    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(polar=True))

    ax.fill(shifted_angles, values, color='purple', alpha=0.25)
    ax.plot(shifted_angles, values, color='purple', linewidth=2)

    ax.set_xticks(shifted_angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    max_value = 100
    ax.set_ylim(0, max_value)
    ax.set_yticks(np.linspace(0, max_value, 5))
    ax.set_yticklabels([f'{int(i)}%' for i in np.linspace(0, max_value, 5)], fontsize=8, rotation=45)
    
    for grid_level in np.linspace(0, max_value, 5):
        ax.plot(angles, [grid_level] * len(angles), color='gray', linewidth=0.75)

    ax.spines['polar'].set_visible(False)  
    ax.grid(False)

    for angle in angles[:-1]: 
        ax.plot([angle, angle], [0, max_value], color='gray', linewidth=0.75, linestyle='-')

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_title(f"Mean Fascia Thickness for Patient {patient}", fontweight='bold', pad=20)

    plt.savefig(img_output_path, bbox_inches='tight')



print("######################## RADIAL APPROACH nnUNet - SINGLE SLICE - predict ########################")

to_plot = True


# to_predict_label = ["IIEF15_01_6m", "IIEF15_01_12m", "IIEF15_01_24m", "IIEF15_01_36m"]
# approaches = ["mid_prostate", "circumference", "sum"]

num_lines = 12

to_predict_label = ["IIEF15_01_12m"]
approaches = ["mid_prostate"]

data_path = "/data/groups/beets-tan/archive/edp_mri_seg/"
img_output_path = "/home/g.rouwendaal/imgs/nnUNet_radial/final/single_slice/"
df_path = data_path + "/IIEF_labels.xlsx"

possible_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

param_distributions = {
    "LogisticRegression": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"]
    },
    "RandomForest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2"]
    },
    "SVM": {
        "clf__C": np.logspace(-2, 2, 8),
        "clf__kernel": ["linear"]
    },
    "KNN": {
        "n_neighbors": list(range(3, 20, 2)),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    },
    "Naive Bayes": {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
    }
}

test_patients_ED = <path to MRI images of separate group of patients with ED>
test_patients_no_ED = <path to MRI images of separate group of patients without ED>
test_patients = os.listdir(test_patients_ED) + os.listdir(test_patients_no_ED)
test_patients = [patient.split(".")[0] for patient in test_patients]

train_patients_ED = <path to MRI images of patients with ED>
train_patients_no_ED = <path to MRI images of patients without ED>
train_patients = os.listdir(train_patients_ED) + os.listdir(train_patients_no_ED)
selected_patients = [patient.split(".")[0] for patient in train_patients]

folds_path = <path to generated fold structure>
with open(folds_path, 'r') as f:
    folds = json.load(f)


for approach in approaches:
    ft_path = <path to ft dictionary of single slice>

    with open(ft_path, "r") as f:
        fascia_thickness_dict = json.load(f)

    selected_patients = [patient for patient in selected_patients if patient not in test_patients]

    df = pd.read_excel(df_path)
    # filter data
    df_test = df.copy()
    df = df[df["AnonymizedName"].isin(selected_patients)]
    df_test = df_test[df_test["AnonymizedName"].isin(test_patients)]
    df = df[df["IIEF15_01_preop"] >= 4]

    flattened_fascia_thickness_dict = fascia_thickness_dict

    if type(to_predict_label) == list:
        for label in to_predict_label:
            if to_plot:
                df = df.dropna(subset=[label])
                ed_patients = set(df[df[label] < 4]["AnonymizedName"])
                no_ed_patients = set(df[df[label] >= 4]["AnonymizedName"])
                fascia_thickness_ed = {k: v for k, v in flattened_fascia_thickness_dict.items() if k in ed_patients}
                fascia_thickness_no_ed = {k: v for k, v in flattened_fascia_thickness_dict.items() if k in no_ed_patients}
                overall_ed, overall_no_ed = get_region_stat_dict_all(fascia_thickness_ed, stat="median"), get_region_stat_dict_all(fascia_thickness_no_ed, stat="median")
                largest_length = max(max(overall_ed.values()), max(overall_no_ed.values())) + 1.0
                dodecagon_output_path = os.path.join(img_output_path, approach, label)
                os.makedirs(dodecagon_output_path, exist_ok=True)
                print(f"largest length: {largest_length}")
                overall_ed = dict(zip(np.linspace(0, 2 * np.pi, num=num_lines, endpoint=False), overall_ed.values()))
                overall_no_ed = dict(zip(np.linspace(0, 2 * np.pi, num=num_lines, endpoint=False), overall_no_ed.values()))
                plot_dodecagon(os.path.join(dodecagon_output_path, "overall_ed.png"), overall_ed, largest_length=largest_length, patient=f"Overall ED - {label} - n=({len(ed_patients)})")
                plot_dodecagon(os.path.join(dodecagon_output_path, "overall_no_ed.png"), overall_no_ed, largest_length=largest_length, patient=f"Overall no ED - {label} - n=({len(no_ed_patients)})")
                print(f"Saved dodecagon plots for {label} - {approach} in {dodecagon_output_path}")

                # plot_ed_no_ed_features(df, fascia_thickness_dict, label, approach)
            features, subset_df = prepare_df(flattened_fascia_thickness_dict, df.copy(), label)
            features_test, subset_df_test = prepare_df(flattened_fascia_thickness_dict, df_test.copy(), label)

            print(f"In total, there are {len(subset_df_test)} test cases")
            print(f"############################### {approach} - {label} ###############################")
            best_models = find_best_model(subset_df, features, subset_df_test, features_test, folds)
            print(best_models)

            feature_importances = get_feature_importances(best_models, features)

            importance_dir = os.path.join(img_output_path, approach, label, "feature_importance")
            os.makedirs(importance_dir, exist_ok=True)

            for model_name, importances in feature_importances.items():
                plot_importances(importances,
                                title=f"{model_name} - Feature Importance per Region",
                                output_path=os.path.join(importance_dir, f"region_{model_name}.png"),
                                features=features)