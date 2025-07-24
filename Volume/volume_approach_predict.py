import json
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit
from sklearn.base import clone

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
            save_dir = os.path.join(f"Final/Models/Volume/{model_name}", fold_name)
            os.makedirs(save_dir, exist_ok=True)
            np.savez(os.path.join(save_dir, "outer_predictions.npz"), trues=y_outer_val, preds=y_pred_outer, outputs=y_pred_probs_outer)

            # Save best model for outer fold
            best_models[(fold_name, model_name)] = best_model

    return best_models


def prepare_volume_df(volume_dict, df, to_predict_label):
    volume_df = pd.DataFrame.from_dict(volume_dict, orient="index").reset_index()
    volume_df.rename(columns={"index": "AnonymizedName"}, inplace=True)

    print(f"In total, there are {len(volume_df)} patients with volume features")

    df = df.dropna(subset=[to_predict_label])
    df["label"] = (df[to_predict_label] >= 4).astype(int)

    df_merged = df.merge(volume_df, on="AnonymizedName", how="inner")
    print(f"In total, there are {len(df_merged)} patients with volume + label")

    features = ["prostate_volume_ml", "fascia_volume_ml"]
    return features, df_merged


to_predict_label = ["IIEF15_01_12m"]
approaches = ["mid_prostate"]

img_output_path = <output path to save imgs>
df_path = <path to labels>

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
    volume_path = <path to volume dictionary of all slices>

    with open(volume_path, "r") as f:
        volume_dict = json.load(f)

    selected_patients = [patient for patient in selected_patients if patient not in test_patients]

    df = pd.read_excel(df_path)
    # filter data
    df_test = df.copy()
    df = df[df["AnonymizedName"].isin(selected_patients)]
    df_test = df_test[df_test["AnonymizedName"].isin(test_patients)]
    df = df[df["IIEF15_01_preop"] >= 4]

    volume_dict

    if type(to_predict_label) == list:
        for label in to_predict_label:
            features, subset_df = prepare_volume_df(volume_dict, df.copy(), label)
            features_test, subset_df_test = prepare_volume_df(volume_dict, df_test.copy(), label)

            print(f"In total, there are {len(subset_df_test)} test cases")
            print(f"############################### {approach} - {label} ###############################")
            best_models = find_best_model(subset_df, features, subset_df_test, features_test, folds)
            print(best_models)