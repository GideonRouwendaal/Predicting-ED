import nrrd
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
import pandas as pd
import json
import json

from tqdm import tqdm


def load_img_and_segmentation(full_img_path, full_segmentation_path, patient):
    try:
        (img, img_header), (segmentation, seg_header) = nrrd.read(full_img_path), nrrd.read(full_segmentation_path)
        spacing = seg_header.get("space directions")
        spacing = [abs(spacing[i][i]) for i in range(3)]
        return img, segmentation
    except:
        print(f"There does not exist an image or segmentation for patient {patient}")
        return False, False


def resample_img(img, target_spacing=(139.9 / 512, 139.9 / 512, 45 / 19), size=(512, 512), interpolator=sitk.sitkLinear):
    original_spacing = np.array(img.GetSpacing())
    original_size = np.array(img.GetSize())

    # Adjust the order to (x, y, z) for consistency
    target_spacing = (target_spacing[1], target_spacing[0], target_spacing[2])

    # Calculate new size
    new_size = (original_size * (original_spacing / np.array(target_spacing))).astype(int)
    new_size[0] = max(new_size[0], size[0])
    new_size[1] = max(new_size[1], size[1])

    # Resample
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(interpolator)
    resample_filter.SetOutputSpacing(target_spacing)
    resample_filter.SetSize(new_size.tolist())
    resample_filter.SetOutputDirection(img.GetDirection())
    resample_filter.SetOutputOrigin(img.GetOrigin())

    resampled_img = resample_filter.Execute(img)

    return resampled_img


def resample_to_img_size(segmentation, target_img):
    # Get original size and spacing
    original_size = segmentation.GetSize()
    original_spacing = segmentation.GetSpacing()

    target_spacing = target_img.GetSpacing()

    # Adjust spacing to prevent excessive scaling
    adjusted_spacing = [max(s, t) for s, t in zip(original_spacing, target_spacing)]
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
    resample_filter.SetOutputSpacing(adjusted_spacing)

    # Calculate new size based on adjusted spacing
    new_size = [int(round(osz * osp / asp)) for osz, osp, asp in zip(original_size, original_spacing, adjusted_spacing)]
    resample_filter.SetSize(new_size)
    resample_filter.SetOutputDirection(target_img.GetDirection())
    resample_filter.SetOutputOrigin(target_img.GetOrigin())

    # Apply resampling
    resampled_segmentation = resample_filter.Execute(segmentation)

    return resampled_segmentation


def process_patient(patient, data_path_full):
    img, segmentation = load_img_and_segmentation(full_img_path, full_segmentation_path, patient)

    if type(segmentation) == bool:
        print(f"Could not load image or segmentation for patient {patient}")
        return False, False, False

    target_spacing, size = (140 / 512, 140 / 512, 45 / 19), (512, 512)
    
    # Load image and segmentation
    img, segmentation = sitk.ReadImage(data_path_full + "Image.nrrd"), sitk.ReadImage(full_segmentation_path)

    resampled_seg = resample_to_img_size(segmentation, img)

    resampled_img = resample_img(img, target_spacing=target_spacing, size=size, interpolator=sitk.sitkLinear)
    resampled_seg = resample_img(resampled_seg, target_spacing=target_spacing, size=size, interpolator=sitk.sitkNearestNeighbor)

    num_slices_with_prostate = np.sum(np.any(sitk.GetArrayFromImage(resampled_seg) == 1, axis=(1, 2)))

    if num_slices_with_prostate < smallest_n_prostate_slices or img.GetSize()[0] < min_depth:
        print(f"Patient {patient} has too few slices with prostate or is too thin...")
        return False, False, False
    
    # get indices of slices with prostate
    prostate_slices = np.where(np.any(sitk.GetArrayFromImage(resampled_seg) == 1, axis=(1, 2)))[0]
    prostate_slices = np.arange(prostate_slices[0], prostate_slices[-1] + 1)

    mid_prostate_slice_idx = prostate_slices[len(prostate_slices) // 2]
    
    prostate_start_idx = mid_prostate_slice_idx - (smallest_n_prostate_slices // 2)
    prostate_end_idx = mid_prostate_slice_idx + (smallest_n_prostate_slices // 2)
    prostate_start_idx = max(prostate_start_idx, 0)
    prostate_end_idx = min(prostate_end_idx, img.GetSize()[0] - 1)

    if prostate_end_idx - prostate_start_idx < smallest_n_prostate_slices:
        prostate_end_idx = prostate_start_idx + smallest_n_prostate_slices
        if prostate_end_idx > img.GetSize()[0] - 1:
            prostate_start_idx = prostate_end_idx - smallest_n_prostate_slices
    
    full_3D_start_idx = mid_prostate_slice_idx - (min_depth // 2)
    full_3D_end_idx = mid_prostate_slice_idx + (min_depth // 2)
    full_3D_start_idx = max(full_3D_start_idx, 0)
    full_3D_end_idx = min(full_3D_end_idx, img.GetSize()[0] - 1)
    if full_3D_end_idx - full_3D_start_idx < min_depth:
        full_3D_end_idx = full_3D_start_idx + min_depth
        if full_3D_end_idx > img.GetSize()[0] - 1:
            full_3D_start_idx = full_3D_end_idx - min_depth
    
    img_array = sitk.GetArrayFromImage(resampled_img)
    mid_prostate_img = sitk.GetImageFromArray(img_array[mid_prostate_slice_idx, :, :])
    prostate_imgs = sitk.GetImageFromArray(img_array[prostate_start_idx:prostate_end_idx, :, :])
    full_3D_img = sitk.GetImageFromArray(img_array[full_3D_start_idx:full_3D_end_idx, :, :])

    return mid_prostate_img, prostate_imgs, full_3D_img


img_path = <path to imgs>
segmentation_path = <path to segmentations>
input_data_path = <path to clinical data & labels>

smallest_n_prostate_slices = 12
min_depth = 24

mapping = <path to mapping>
with open(mapping, "r") as file:
    mapping = json.load(file)

reversed_mapping = {v: k for k, v in mapping.items()}

output_data_path_mid_prostate = <output path of mid prostate slice>
output_data_path_mid_prostate_multiple_slices = <output path of mid prostate slices>
output_data_path_mid_prostate_3D = <output path of 3D mid prostate slices>

to_predict_labels = ["IIEF15_01_12m"]

for label in to_predict_labels:
    labeled_data_path = label + "/"
    train_path, test_path = labeled_data_path + "Train/", labeled_data_path + "Test/"

    ED_NO_ED = ["ED/", "no_ED/"]
    for ed_no in ED_NO_ED:
        full_train_path, full_test_path = train_path + ed_no, test_path + ed_no
        os.makedirs(output_data_path_mid_prostate + full_train_path, exist_ok=True)
        os.makedirs(output_data_path_mid_prostate + full_test_path, exist_ok=True)
        os.makedirs(output_data_path_mid_prostate_multiple_slices + full_train_path, exist_ok=True)
        os.makedirs(output_data_path_mid_prostate_multiple_slices + full_test_path, exist_ok=True)
        os.makedirs(output_data_path_mid_prostate_3D + full_train_path, exist_ok=True)
        os.makedirs(output_data_path_mid_prostate_3D + full_test_path, exist_ok=True)


patients = [
    patient for patient in os.listdir(input_data_path)
    if "Patient" in patient and "Anonymized" in patient
    and len([f for f in os.listdir(os.path.join(input_data_path, patient))]) < 2
]

test_patients = [
    patient for patient in os.listdir(input_data_path)
    if "Patient" in patient and "Anonymized" in patient
    and len([f for f in os.listdir(os.path.join(input_data_path, patient))]) >= 2
]

# load data
df = pd.read_excel(input_data_path + "/IIEF_labels.xlsx")

print(f"Total number of patients in the excel = {len(df)}")

# filter data
df = df[df["IIEF15_01_preop"] >= 4]

df = df[df["IIEF15_01_12m"].notna()]

print(f"Total number of patients in the excel with preop ok = {len(df)}")

df_train = df[df["AnonymizedName"].isin(patients)]
patients = df_train["AnonymizedName"].tolist()

df_test = df[df["AnonymizedName"].isin(test_patients)]
test_patients = df_test["AnonymizedName"].tolist()

print(f"Total number of patients with a nnUNet segmentation and label = {len(patients)}")
print(f"Total number of test patients = {len(test_patients)}")

n_train, n_test = 0, 0
for file, patient in tqdm(mapping.items()):
    if patient not in patients and patient not in test_patients:
        continue
    data_path_full = input_data_path + patient + "/"
    splitted_patient = file.split(".")
    full_img_path, full_segmentation_path = img_path + f"{splitted_patient[0]}_0000.{splitted_patient[1]}", segmentation_path + file
    mid_prostate_img, prostate_imgs, full_3D_img = process_patient(patient, data_path_full)
    if type(mid_prostate_img) == bool:
        print(f"Could not load image or segmentation for patient {patient}")
        continue
    train_test = "Train" if patient in patients else "Test"
    if train_test == "Train":
        n_train += 1
    else:
        n_test += 1
    label_value = df.loc[df["AnonymizedName"] == patient, "IIEF15_01_12m"].values[0]
    if pd.isna(label_value):
        print(f"Could not label for patient {patient}")
        continue
    label_value = "ED" if int(label_value) <= 3 else "no_ED"
    sitk.WriteImage(mid_prostate_img, output_data_path_mid_prostate + f"IIEF15_01_12m/{train_test}/{label_value}/{patient}.nrrd")
    sitk.WriteImage(full_3D_img, output_data_path_mid_prostate_3D + f"IIEF15_01_12m/{train_test}/{label_value}/{patient}.nrrd")
    img_array = sitk.GetArrayFromImage(prostate_imgs)

    for i in range(img_array.shape[0]):
        img_slice = img_array[i, :, :]
        img_slice = sitk.GetImageFromArray(img_slice)
        sitk.WriteImage(img_slice, output_data_path_mid_prostate_multiple_slices + f"IIEF15_01_12m/{train_test}/{label_value}/{patient}_slice_{i}.nrrd")

print(f"Number of patients in the train set = {n_train}")
print(f"Number of patients in the test set = {n_test}")