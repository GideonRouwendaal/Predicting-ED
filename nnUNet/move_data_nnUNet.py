import numpy as np
from skimage.draw import polygon
from scipy.ndimage import binary_fill_holes
from skimage.measure import find_contours
import os
import pandas as pd
import SimpleITK as sitk
import pandas as pd
import json
from tqdm import tqdm

def retrieve_fascia_prostate_layer_value(segmentation):
    fascia_layer, prostate_layer, fascia_value, prostate_value = None, None, None, None
    for key in segmentation.GetMetaDataKeys():
        if key.endswith("_Name") and segmentation.GetMetaData(key).lower() == "fascia":
            segment_prefix = key.split("_Name")[0]
            fascia_layer = int(segmentation.GetMetaData(f"{segment_prefix}_Layer"))
            fascia_value = int(segmentation.GetMetaData(f"{segment_prefix}_LabelValue"))
        elif key.endswith("_Name") and segmentation.GetMetaData(key).lower() == "prostate":
            segment_prefix = key.split("_Name")[0]
            prostate_layer = int(segmentation.GetMetaData(f"{segment_prefix}_Layer"))
            prostate_value = int(segmentation.GetMetaData(f"{segment_prefix}_LabelValue"))
        if fascia_layer is not None and prostate_layer is not None:
            break
    return fascia_layer, prostate_layer, fascia_value, prostate_value


def preprocess_img_segmentation(img, segmentation, target_size=(640, 640)):
    original_size = img.GetSize()
    original_spacing = img.GetSpacing()

    new_spacing = (
        original_spacing[0] * original_size[0] / target_size[0],  
        original_spacing[1] * original_size[1] / target_size[1],
        original_spacing[2]  # Keep the original depth spacing
    )

    new_spacing = (
        original_spacing[0],  
        original_spacing[1],
        original_spacing[2]
    )

    resampled_img = sitk.Resample(img, target_size + (original_size[2],),
                                  sitk.Transform(),
                                  sitk.sitkLinear,
                                  img.GetOrigin(),
                                  new_spacing,
                                  img.GetDirection(),
                                  0,  # Default intensity value for out-of-bounds pixels
                                  img.GetPixelID())

    if type(segmentation) == bool:
        return sitk.GetArrayFromImage(resampled_img)

    resampled_segmentation = sitk.Resample(segmentation, target_size + (original_size[2],),
                                           sitk.Transform(),
                                           sitk.sitkNearestNeighbor,
                                           segmentation.GetOrigin(),
                                           new_spacing,
                                           segmentation.GetDirection(),
                                           0,  # Default value for out-of-bounds pixels
                                           segmentation.GetPixelID())

    img, segmentation = sitk.GetArrayFromImage(resampled_img), sitk.GetArrayFromImage(resampled_segmentation)
    return img, segmentation


def preprocess_segmentation(segmentation, img):
    segmentation = sitk.Cast(segmentation, sitk.sitkUInt8)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)  # This handles size, spacing, origin, direction
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(segmentation)


def load_img_and_segmentation(data_path, patient):
    try:
        img, segmentation = sitk.ReadImage(data_path + "Image.nrrd"), sitk.ReadImage(data_path + "Segmentation.seg.nrrd")

        fascia_layer, prostate_layer, fascia_value, prostate_value = retrieve_fascia_prostate_layer_value(segmentation)
        segmentation = process_prostate_fascia_segmentation(segmentation, prostate_layer, fascia_layer, prostate_value=prostate_value, fascia_value=fascia_value)

        segmentation = preprocess_segmentation(segmentation, img)
        return img, segmentation, prostate_layer, prostate_value, fascia_layer, fascia_value
    except:
        print(f"There does not exist a segmentation for patient {patient}")
        try:
            img = sitk.ReadImage(data_path + "Image.nrrd")
            return img, False, False, False, False, False
        except:
                print(f"There does not exist an image and segmentation for patient {patient}")
                return False, False, False, False, False, False


def correct_labels(segmentation, label_mapping):
    result = np.zeros(segmentation.shape)
    for key, value in label_mapping.items():
        result[segmentation == key] = value
    return result


def touches_prostate_or_fascia(contour, segment, width, height):
    contour_pixels = np.round(contour).astype(int)
    for y, x in contour_pixels:
        if 0 <= y < height and 0 <= x < width:
            # Check 8-neighbor pixels
            neighbors = segment[max(0, y-1):min(height, y+2), max(0, x-1):min(width, x+2)]
            if np.any((neighbors == 1)):
                return True
    return False


def process_segmentation(fascia_segment):
    prostate_mask, fascia_mask = (fascia_segment == 1), (fascia_segment == 2)
    valid_mask = np.zeros_like(fascia_segment, dtype=np.uint8)
    # if there are holes in the prostate, fix it
    prostate_filled = np.array([binary_fill_holes(prostate_mask[slice_idx]) for slice_idx in range(fascia_segment.shape[0])])
    valid_mask[prostate_filled] = 1

    for slice_idx in range(fascia_segment.shape[0]):
        contours = find_contours(fascia_mask[slice_idx], level=0.5)
        height, width = fascia_segment.shape[1:]
        for contour in contours:
            if touches_prostate_or_fascia(contour, fascia_segment[slice_idx], height, width):
                # Convert contour to a filled region
                contour_int = np.round(contour).astype(int)
                rr, cc = polygon(contour_int[:, 0], contour_int[:, 1], fascia_segment[slice_idx].shape)
                valid_mask[slice_idx, rr, cc] = 2

            # In rare cases, correct contours are not found, assume that fascia mask is correct
            if 1 not in np.unique(valid_mask[slice_idx]):
                temp_mask = np.zeros_like(valid_mask[slice_idx])
                temp_mask[prostate_filled[slice_idx] == 1] = 1
                temp_mask[fascia_mask[slice_idx] == 1] = 2
                valid_mask[slice_idx] = temp_mask

    return valid_mask



def process_prostate_fascia_segmentation(segmentation_img, prostate_layer, fascia_layer, prostate_value=1, fascia_value=2):
    original_seg_spacing = segmentation_img.GetSpacing()
    original_seg_origin = segmentation_img.GetOrigin()
    original_seg_direction = segmentation_img.GetDirection()
    

    segmentation = sitk.GetArrayFromImage(segmentation_img)
    
    # Handle separate layers for prostate and fascia
    if prostate_layer != fascia_layer:
        if len(segmentation.shape) != 4:
            return False
        prostate_segmentation = segmentation[:, :, :, prostate_layer]
        fascia_segmentation = segmentation[:, :, :, fascia_layer]

        # Initialize a combined segmentation map
        prostate_fascia_segmentation = np.zeros_like(prostate_segmentation)
        prostate_fascia_segmentation[prostate_segmentation == prostate_value] = 1
        prostate_fascia_segmentation[fascia_segmentation == fascia_value] = 2

    # Single layer case
    else:
        if len(segmentation.shape) == 4:
            prostate_fascia_segmentation = segmentation[:, :, :, prostate_layer]
        elif len(segmentation.shape) == 3 and int(prostate_layer) == 0:
            prostate_fascia_segmentation = segmentation[:, :, :]
        else:
            raise ValueError("There is no prostate and/or fascia segmentation for this case.")

    # Label mapping adjustment
    label_mapping = {prostate_value: 1, fascia_value: 2}
    prostate_fascia_segmentation = correct_labels(prostate_fascia_segmentation, label_mapping)
    # Apply further processing (e.g., morphological operations, etc.)
    prostate_fascia_segmentation = process_segmentation(prostate_fascia_segmentation)
    prostate_fascia_segmentation  = sitk.GetImageFromArray(prostate_fascia_segmentation)

    # Set the original spacing, size, and direction
    prostate_fascia_segmentation.SetSpacing(original_seg_spacing)
    prostate_fascia_segmentation.SetOrigin(original_seg_origin)
    prostate_fascia_segmentation.SetDirection(original_seg_direction)

    return prostate_fascia_segmentation


data_path = <path of MRIS>

nnUNet_raw_data_path = <nnUNet raw data path>
train_imgs_path = nnUNet_raw_data_path + "imagesTr/"
train_labels_path = nnUNet_raw_data_path + "labelsTr/"
test_imgs_path = nnUNet_raw_data_path + "imagesTs/"
test_labels_path = nnUNet_raw_data_path + "labelsTs/"

for path in [train_imgs_path, train_labels_path, test_imgs_path, test_labels_path]:
    if not os.path.exists(path):
        os.makedirs(path)

patients = [patient for patient in os.listdir(data_path) if "Patient" in patient]

df_path = <clinical data path>
# load data
df = pd.read_excel(df_path)
# filter data
df = df[df["IIEF15_01_preop"] >= 4]
df_with_labels = df[df[["IIEF15_01_6m", "IIEF15_01_12m", "IIEF15_01_24m", "IIEF15_01_36m"]].notna().any(axis=1)]

test_patients = list(set(patients) & set(df["AnonymizedName"].unique()))

n_train, n_test = 0, 0

train_cases, test_cases = {}, {}

for i in tqdm(range(len(patients))): 
    patient = patients[i]
    data_path_full = data_path + patient + "/"
    img, segmentation, prostate_layer, prostate_value, fascia_layer, fascia_value = load_img_and_segmentation(data_path_full, patient)

    if type(img) is bool and type(segmentation) is bool:
        # There is both no image and no segmentation
        continue
    elif type(segmentation) is bool:
        # There is only no segmentation
        img_file_name = f"prostate_{n_test+1:03}_0000.nrrd"
        seg_file_name = f"prostate_{n_test+1:03}.nrrd"
        # save image
        sitk.WriteImage(img, test_imgs_path + img_file_name)
        test_cases[seg_file_name] = patient
        n_test += 1
        continue

    if type(segmentation) is bool:
        print(f"There is no consistent/correct prostate and/or fascia segmentation for patient {patient}")
        # There is no prostate and/or fascia segmentation
        img_file_name = f"prostate_{n_test+1:03}_0000.nrrd"
        seg_file_name = f"prostate_{n_test+1:03}.nrrd"
        # save image
        sitk.WriteImage(img, test_imgs_path + img_file_name)
        test_cases[seg_file_name] = patient
        n_test += 1
        continue

    if patient not in test_patients:
        img_file_name = f"prostate_{n_train+1:03}_0000.nrrd"
        seg_file_name = f"prostate_{n_train+1:03}.nrrd"
        # save image
        sitk.WriteImage(img, train_imgs_path + img_file_name)
        # save prostate-fascia segmentation
        sitk.WriteImage(segmentation, train_labels_path + seg_file_name)
        train_cases[seg_file_name] = patient
        n_train += 1
    else:
        img_file_name = f"prostate_{n_test+1:03}_0000.nrrd"
        seg_file_name = f"prostate_{n_test+1:03}.nrrd"
        sitk.WriteImage(img, test_imgs_path + img_file_name)
        # save prostate-fascia segmentation
        sitk.WriteImage(segmentation, test_labels_path + seg_file_name)
        test_cases[seg_file_name] = patient
        n_test += 1

print(train_cases)
print(test_cases)

dataset_json = { 
 "name": "PROSTATE AND FASCIA", 
 "description": "Prostate and fascia segmentation",
"reference": <Reference of Dataset>,
"channel_names": { 
   "0": "MRI"
 }, 
 "labels": { 
   "background" : 0,
   "prostate" : 1,
   "fascia" : 2
 }, 
 "numTraining": len(train_cases), 
 "numTest": len(test_cases),
 "file_ending": ".nrrd"
 }

with open(nnUNet_raw_data_path + "dataset.json", "w") as file:
    json.dump(dataset_json, file, indent=4)

with open(nnUNet_raw_data_path + "train_cases.json", "w") as file:
    json.dump(train_cases, file, indent=4)

with open(nnUNet_raw_data_path + "test_cases.json", "w") as file:
    json.dump(test_cases, file, indent=4)