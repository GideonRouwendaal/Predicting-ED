import os
import glob
import json
import random
from collections import Counter
from types import SimpleNamespace

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

import SimpleITK as sitk

import random
random.seed(42)

def get_label_distribution_from_subset(subset):
    labels = []
    for idx in subset.indices:
        path = subset.dataset.image_paths[idx]
        label = subset.dataset.__label_extract__(path)
        labels.append(label)
    count = Counter(labels)
    return {'ED': count[0], 'no_ED': count[1]}


def get_patient_id_from_filename(path):
    filename = os.path.basename(path).replace('.nrrd', '')
    patient_id = filename.split('_slice')[0]
    return patient_id


def percentile_torch(tensor, q):
    k = 1 + round(0.01 * float(q) * (tensor.numel() - 1))
    result = tensor.view(-1).kthvalue(k).values.item()
    return result


class CustomTransformPipeline:
    def __init__(self, desired_translate_ratio=0.1, desired_scale_bounds=(0.9, 1.1), affine_prob=0.6):
        self.desired_translate_ratio = desired_translate_ratio
        self.desired_scale_bounds = desired_scale_bounds

        self.static_transforms = transforms.Compose([
            transforms.RandomApply([
                transforms.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR)
            ], p=0.6),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.3),
            transforms.CenterCrop(size=(520, 520)),
        ])
        self.affine_prob = affine_prob
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((224, 224))

    def __call__(self, img, original_size, model_type="not_vit"):
        orig_w, orig_h = original_size

        img = transforms.Pad(padding=100, padding_mode='reflect')(img)
        if random.random() < self.affine_prob:
            padded_w, padded_h = img.size

            desired_translate_px_x = self.desired_translate_ratio * orig_w
            desired_translate_px_y = self.desired_translate_ratio * orig_h

            translate_x = desired_translate_px_x / padded_w
            translate_y = desired_translate_px_y / padded_h

            affine = transforms.RandomAffine(
                degrees=0,
                translate=(translate_x, translate_y),
                scale=(self.desired_scale_bounds),
                interpolation=InterpolationMode.BILINEAR,
                fill=0
            )

            img = affine(img)
        img = self.static_transforms(img)
        if model_type == 'vit':
            img = self.resize(img)
        img = self.to_tensor(img)
        return img


class FolderDatasetResNet(Dataset):
    def __init__(self, folder, allowed_patient_ids=None, undersample=False, augment=True, age_dict=None):
        self.folder = folder
        self.image_paths = glob.glob(f'{self.folder}/*/*.nrrd')
        self.labels = {'ED': 0, 'no_ED': 1}
        self.augment = augment
        self.age_dict = age_dict
        
        if allowed_patient_ids is not None:
            self.image_paths = [
                p for p in self.image_paths
                if get_patient_id_from_filename(p) in allowed_patient_ids
            ]
        
        ed_images = [path for path in self.image_paths if 'ED' in path and 'no_ED' not in path]
        no_ed_images = [path for path in self.image_paths if 'no_ED' in path]
        if undersample:
            min_count = min(len(ed_images), len(no_ed_images))
            ed_images = random.sample(ed_images, min_count)
            no_ed_images = random.sample(no_ed_images, min_count)
        image_paths = ed_images + no_ed_images
        image_paths = [
            path for path in image_paths 
            if "slice_8" not in path 
            and "slice_9" not in path 
            and "slice_10" not in path 
            and "slice_11" not in path
        ]
        # Filter only those with an age in the age_dict
        if self.age_dict is not None:
            image_paths = [path for path in image_paths if get_patient_id_from_filename(path) in self.age_dict]
        self.image_paths = image_paths
        
        random.shuffle(self.image_paths)


        self.transform = CustomTransformPipeline()
        self.val_transform = transforms.Compose([
            transforms.CenterCrop(size=(512, 512)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __label_dist__(self):
        ed, no_ed = 0, 0
        for path in self.image_paths:
            if self.__label_extract__(path) == 0:
                ed += 1
            elif self.__label_extract__(path) == 1:
                no_ed += 1
        return {'ED': ed, 'no_ED': no_ed}
    
    def __label_extract__(self, path):
        return 1 if 'no_ED' in path else 0

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        # img, _ = nrrd.read(path)
        # img = img.astype(np.float32)
        img = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
        img = Image.fromarray(img.squeeze())


        if self.augment: 
            img = self.transform(img, img.size)           
        else:
            img = self.val_transform(img) 

        lower = percentile_torch(img, 0.5)
        upper = percentile_torch(img, 99.5)
        img = torch.clamp(img, min=lower, max=upper)

        img = (img - torch.mean(img)) / (torch.std(img) + 1e-8)

        label = self.__label_extract__(path)
        age = self.age_dict.get(get_patient_id_from_filename(path), None)
        return img, label, torch.tensor(age, dtype=torch.float)


class FolderDatasetViT(Dataset):
    def __init__(self, folder, allowed_patient_ids=None, undersample=False, augment=True, normalize_images=True, model_type='vit', age_dict=None):
        self.model_type = model_type
        self.folder = folder
        self.image_paths = glob.glob(f'{self.folder}/*/*.nrrd')
        self.age_dict = age_dict
        
        self.labels = {
            'ED': 0,
            'no_ED': 1
        }
        self.augment = augment
        
        if allowed_patient_ids is not None:
            self.image_paths = [
                p for p in self.image_paths
                if get_patient_id_from_filename(p) in allowed_patient_ids
            ]
        
        ed_images = [path for path in self.image_paths if 'ED' in path and 'no_ED' not in path]
        no_ed_images = [path for path in self.image_paths if 'no_ED' in path]
        
        if undersample:
            min_count = min(len(ed_images), len(no_ed_images))
            ed_images = random.sample(ed_images, min_count)
            no_ed_images = random.sample(no_ed_images, min_count)
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

        image_paths = ed_images + no_ed_images
        # Filter only those with an age in the age_dict

        image_paths = [
            path for path in image_paths 
            if "slice_8" not in path 
            and "slice_9" not in path 
            and "slice_10" not in path 
            and "slice_11" not in path
        ]

        if self.age_dict is not None:
            image_paths = [path for path in image_paths if get_patient_id_from_filename(path) in self.age_dict]

        self.image_paths = image_paths
        self.model_type = model_type
        random.shuffle(self.image_paths)
        
        self.transform = CustomTransformPipeline()

        if model_type == 'vit':
            self.val_transform = transforms.Compose([
                transforms.CenterCrop(size=(512, 512)),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        elif model_type == 'hybrid_rvit':
            self.val_transform = transforms.Compose([
                transforms.CenterCrop(size=(512, 512)),
                transforms.ToTensor(),
            ])

        self.normalize_images = normalize_images

    def __len__(self):
        return len(self.image_paths)
    
    def __label_dist__(self):
        ed, no_ed = 0, 0
        for path in self.image_paths:
            if self.__label_extract__(path) == 0:
                ed += 1
            elif self.__label_extract__(path) == 1:
                no_ed += 1
        return {'ED': ed, 'no_ED': no_ed}
    
    def __label_extract__(self, path):
        return 1 if 'no_ED' in path else 0
   

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        # img, _ = nrrd.read(path)
        # img = img.astype(np.float32)
        img = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
        img = Image.fromarray(img.squeeze())

        if self.augment: 
            if self.model_type == 'vit':
                img = self.transform(img, img.size, model_type=self.model_type)
            else:
                img = self.transform(img, img.size)             
        else:
            img = self.val_transform(img) 

        if self.model_type == 'vit':
            img = img.repeat(3, 1, 1)

        lower = percentile_torch(img, 0.5)
        upper = percentile_torch(img, 99.5)
        img = torch.clamp(img, min=lower, max=upper)

        if self.model_type == 'vit':
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        else:
            img = (img - torch.mean(img)) / (torch.std(img) + 1e-8)

        if self.normalize_images:
            img = self.normalize(img)

        label = self.__label_extract__(path)
        age = self.age_dict.get(get_patient_id_from_filename(path), None)
        return img, label, torch.tensor(age, dtype=torch.float)


def calculate_mean_std(dataset):
    mean, std, n = 0., 0., 0
    for sample in dataset:
        imgs, _, _ = sample
        imgs = imgs.squeeze(0)
        mean += torch.mean(imgs)
        std += torch.std(imgs)
        n += 1
    mean /= n
    std /= n
    return mean, std


def generate_dataset(args, allowed_patient_ids=None, augment=False, age_dict=None):
    mean_override, std_override = getattr(args, "normalize_override", (None, None))

    if args.model_type == 'resnet':
        dataset = FolderDatasetResNet(args.data_path, augment=augment, allowed_patient_ids=allowed_patient_ids, age_dict=age_dict)

    elif args.model_type in ['vit', 'hybrid_rvit']:
        dataset = FolderDatasetViT(args.data_path, augment=augment, model_type=args.model_type, normalize_images=False, allowed_patient_ids=allowed_patient_ids, age_dict=age_dict)

        if mean_override is None:
            mean, std = calculate_mean_std(dataset)
            print(f"Calculated mean: {mean}, std: {std}")
        else:
            mean, std = mean_override, std_override

        dataset.normalize_images = True
        if args.model_type == 'vit':
            dataset.normalize = transforms.Normalize(mean=[mean]*3, std=[std]*3)
        else:
            dataset.normalize = transforms.Normalize(mean=mean, std=std)

    return dataset


def prepare_fold_dataset(folds_path, fold_idx, split_level='outer', inner_idx=None, args=None, age_dict=None):
    with open(folds_path, 'r') as f:
        folds = json.load(f)

    if split_level == 'outer':
        train_ids = folds[f'fold_{fold_idx}']['outer_train']
        val_ids   = folds[f'fold_{fold_idx}']['outer_val']
    elif split_level == 'inner':
        assert inner_idx is not None, "inner_idx must be specified for inner split"
        train_ids = folds[f'fold_{fold_idx}']['inner_folds'][f'inner_{inner_idx}']['train']
        val_ids   = folds[f'fold_{fold_idx}']['inner_folds'][f'inner_{inner_idx}']['val']
    else:
        raise ValueError("split_level must be 'outer' or 'inner'")

    # Only compute mean/std for ViT or hybrid_rvit
    if args.model_type in ['vit', 'hybrid_rvit']:
        temp_train_ds = generate_dataset(args, allowed_patient_ids=train_ids, augment=False, age_dict=age_dict)
        mean, std = calculate_mean_std(temp_train_ds)
        print(f"[{split_level.capitalize()} Fold {fold_idx}"
              f"{'' if inner_idx is None else f' | Inner {inner_idx}'}] "
              f"Mean: {mean:.4f}, Std: {std:.4f}")
        args.normalize_override = (mean, std)
    else:
        # Clear any previous override
        args.normalize_override = (None, None)

    train_ds = generate_dataset(args, allowed_patient_ids=train_ids, augment=True, age_dict=age_dict)
    val_ds   = generate_dataset(args, allowed_patient_ids=val_ids, augment=False, age_dict=age_dict)

    return train_ds, val_ds