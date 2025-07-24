import os

from sklearn.model_selection import StratifiedKFold
import json

def collect_patient_data(folder_path):
    patient_label_dict = {}
    for label_dir in ['ED', 'no_ED']:
        full_dir = os.path.join(folder_path, label_dir)
        for file in os.listdir(full_dir):
            if file.endswith('.nrrd'):
                patient_id = file.replace('.nrrd', '')
                label = 0 if label_dir == 'ED' else 1
                patient_label_dict[patient_id] = label
    return patient_label_dict


def create_folds(patient_label_dict, outer_k=5, inner_k=3, save_path='folds.json'):
    patient_ids = list(patient_label_dict.keys())
    labels = [patient_label_dict[pid] for pid in patient_ids]

    outer = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=42)
    fold_structure = {}

    for outer_idx, (train_idx, val_idx) in enumerate(outer.split(patient_ids, labels)):
        outer_train_ids = [patient_ids[i] for i in train_idx]
        outer_val_ids = [patient_ids[i] for i in val_idx]
        fold_structure[f'fold_{outer_idx}'] = {
            'outer_train': outer_train_ids,
            'outer_val': outer_val_ids,
            'inner_folds': {}
        }

        # inner folds on outer_train
        inner_labels = [patient_label_dict[pid] for pid in outer_train_ids]
        inner = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=outer_idx)
        for inner_idx, (in_train_idx, in_val_idx) in enumerate(inner.split(outer_train_ids, inner_labels)):
            in_train_ids = [outer_train_ids[i] for i in in_train_idx]
            in_val_ids = [outer_train_ids[i] for i in in_val_idx]
            fold_structure[f'fold_{outer_idx}']['inner_folds'][f'inner_{inner_idx}'] = {
                'train': in_train_ids,
                'val': in_val_ids
            }
    print(fold_structure)

    with open(save_path, 'w') as f:
        json.dump(fold_structure, f, indent=2)
    
    with open('folds.json', 'w') as f:
        json.dump(fold_structure, f, indent=2)
    

folder_path = <Path to imgs>
patient_label_dict = collect_patient_data(folder_path + '/Train')

print("Total number of training patients:")
print(len(patient_label_dict))

create_folds(patient_label_dict, outer_k=5, inner_k=3, save_path=folder_path + '/folds.json')