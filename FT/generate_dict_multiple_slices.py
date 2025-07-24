import nrrd
import numpy as np
from shapely.geometry import LineString, Point, box
from skimage.draw import line_aa
import os
import pandas as pd
import SimpleITK as sitk
from collections import defaultdict
import pandas as pd
import json
from tqdm import tqdm



def find_center(mask):
    rows, cols = np.where(mask == 1)

    if len(rows) > 0:
        middle_row = np.mean(rows)
        middle_col = np.mean(cols)
        inner_center = Point(middle_col, middle_row)
        return inner_center
    else:
        print("The mask has no `1` values.")
        return False



def create_main_and_sub_angles(num_lines, num_sub_lines):
    main_angles = np.linspace(0, 2 * np.pi, num=num_lines, endpoint=False)
    sub_angles = np.linspace(0, 2 * np.pi, num=num_sub_lines, endpoint=False)

    sub_angle_dict = {}
    for sub_angle in sub_angles:
        added = False
        for i in range(len(main_angles) - 1):
            if sub_angle >= main_angles[i] and sub_angle < main_angles[i + 1]:
                sub_angle_dict[main_angles[i]] = [sub_angle] if main_angles[i] not in sub_angle_dict else sub_angle_dict[main_angles[i]] + [sub_angle]
                added = True
                break
        if not added:
            sub_angle_dict[main_angles[num_lines - 1]] = [sub_angle] if main_angles[num_lines - 1] not in sub_angle_dict else sub_angle_dict[main_angles[num_lines - 1]] + [sub_angle]
    
    return sub_angle_dict


def filter_line_coordinates(max_line_coords, fascia_mask, height=520, width=520):
    int_max_line_coords = max_line_coords.astype(int)

    rr, cc, val = line_aa(
        int_max_line_coords[1, 0], int_max_line_coords[0, 0],
        int_max_line_coords[1, 1], int_max_line_coords[0, 1]
    )

    rr = np.clip(rr, 0, height - 1)
    cc = np.clip(cc, 0, width - 1)

    valid_indices = fascia_mask[rr, cc] == 1
    filtered_rr = np.asarray(rr[valid_indices])
    filtered_cc = np.asarray(cc[valid_indices])

    return filtered_rr, filtered_cc


def create_lines(angles, center, boundary, fascia_mask, height, width):
    lines = []
    for angle in angles:
        dx, dy = np.cos(angle), np.sin(angle)
        extended_line = LineString([
            (center.x, center.y),
            (center.x + 1000 * dx, center.y + 1000 * dy)
        ])
        clipped_line = extended_line.intersection(boundary)
        if not clipped_line.is_empty:
            x, y = clipped_line.xy
            rr, cc = filter_line_coordinates(np.array([x, y]), fascia_mask, height, width)
            if len(rr) > 0:
                lines.append([rr, cc])
    return lines


def calculate_median_lengths(lines_sub_angles, spacing):
    median_length = {}
    largest_length = -1
    for angle in lines_sub_angles:
        lengths = []
        for l in lines_sub_angles[angle]:
            rr, cc = l
            x1, x2, y1, y2 = min(cc), max(cc), min(rr), max(rr)
            length = ((spacing[0] * (x2 - x1)) ** 2 + (spacing[1] * (y2 - y1)) ** 2) ** 0.5
            largest_length = length if length > largest_length else largest_length
            median_length[angle] = length
            lengths.append(length)
        if len(lengths) > 0:
            median_length[angle] = np.median(lengths)
        else:
            median_length[angle] = 0
    
    return median_length, largest_length


def rearrange_dictionary(original_dict):
    index_to_region_mapping = {
        9: 0,
        10: 1,
        11: 2,
        0: 3,
        1: 4,
        2: 5,
        3: 6,
        4: 7,
        5: 8,
        6: 9,
        7: 10,
        8: 11,
    }

    items = list(original_dict.items())
    
    rearranged_dict = {}
    for old_idx, new_idx in index_to_region_mapping.items():
        rearranged_dict[items[new_idx][0]] = items[old_idx][1]
    
    return rearranged_dict


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

    # Get target size and spacing
    target_size = target_img.GetSize()
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


def load_img_and_segmentation(full_img_path, full_segmentation_path, patient):
    try:
        (img, img_header), (segmentation, seg_header) = nrrd.read(full_img_path), nrrd.read(full_segmentation_path)
        spacing = seg_header.get("space directions")
        spacing = [abs(spacing[i][i]) for i in range(3)]
        return img, segmentation, spacing
    except:
        print(f"There does not exist an image or segmentation for patient {patient}")
        return False, False, False


def process_patient(patient, full_img_path, full_segmentation_path, data_path_full):
    labels = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'L6', 'L5', 'L4', 'L3', 'L2', 'L1']

    img, segmentation, spacing = load_img_and_segmentation(full_img_path, full_segmentation_path, patient)
    
    if type(segmentation) == bool:
        print(f"For {patient}, no segmentation can be found" )
        return False

    target_spacing, size = (140 / 512, 140 / 512, 45 / 19), (512, 512)

    img, segmentation = sitk.ReadImage(data_path_full + "Image.nrrd"), sitk.ReadImage(full_segmentation_path)

    resampled_seg = resample_to_img_size(segmentation, img)

    resampled_img = resample_img(img, target_spacing=target_spacing, size=size, interpolator=sitk.sitkLinear)
    resampled_seg = resample_img(resampled_seg, target_spacing=target_spacing, size=size, interpolator=sitk.sitkNearestNeighbor)

    num_slices_with_prostate = np.sum(np.any(sitk.GetArrayFromImage(resampled_seg) == 1, axis=(1, 2)))

    if num_slices_with_prostate < smallest_n_prostate_slices:
        print(f"Patient {patient} has no prostate slices...")
        return False
    
    # get indices of slices with prostate
    prostate_slices = np.where(np.any(sitk.GetArrayFromImage(resampled_seg) == 1, axis=(1, 2)))[0]
    prostate_slices = np.arange(prostate_slices[0], prostate_slices[-1] + 1)

    selected_indices = np.linspace(0, len(prostate_slices) - 1, smallest_n_prostate_slices, dtype=int)
    prostate_indices = prostate_slices[selected_indices]

    prostate_fascia_segmentation = sitk.GetArrayFromImage(resampled_seg)
    result_dict = defaultdict(dict)
    slice_idx = 0
    spacing = resampled_seg.GetSpacing()
    for i in prostate_indices:
        fascia_segment_slice = prostate_fascia_segmentation[i]
        height, width = fascia_segment_slice.shape[0], fascia_segment_slice.shape[1]

        prostate_mask = fascia_segment_slice == prostate_label
        fascia_mask = (fascia_segment_slice == fascia_label).astype(int)

        prostate_center = find_center(prostate_mask)

        if prostate_center != False:
            sub_angle_dict = create_main_and_sub_angles(num_lines, num_sub_lines)

            image_boundary = box(0, 0, width, height)

            lines_sub_angles = {angle: create_lines(sub_angles, prostate_center, image_boundary, fascia_mask, height, width) for angle, sub_angles in sub_angle_dict.items()}

            median_lengths, largest_length = calculate_median_lengths(lines_sub_angles, spacing)
            # Rearrange dictionary for plotting
            rearranged_dict = rearrange_dictionary(median_lengths)
            
            result_dict[slice_idx] = {labels[i]: value for i, (_, value) in enumerate(rearranged_dict.items())}
        else:
            result_dict[slice_idx] = {labels[i]: 0 for i in range(num_lines)}
        slice_idx += 1
    
    return result_dict



img_path = <nnUNet img path>
segmentation_path = <nnUNet segmentation path>
dict_output_path = <FT dict output path>
input_data_path = <original imgs path>


if not os.path.exists(dict_output_path):
    os.makedirs(dict_output_path)

prostate_label = 1
fascia_label = 2

num_lines = 12 
num_sub_lines = 360

smallest_n_prostate_slices = 12

mapping = <mapping to nnUNet test cases>
with open(mapping, "r") as file:
    mapping = json.load(file)

fascia_thickness_dict = defaultdict(dict)
no_middle_slice = []

for file in tqdm(mapping):
    patient = mapping[file]
    splitted_patient = file.split(".")
    full_img_path, full_segmentation_path = img_path + f"{splitted_patient[0]}_0000.{splitted_patient[1]}", segmentation_path + file
    data_path_full = input_data_path + patient + "/"
    fascia_thickness = process_patient(patient, full_img_path, full_segmentation_path, data_path_full)
    if not fascia_thickness:
        no_middle_slice.append(patient)
    else:
        fascia_thickness_dict[patient] = fascia_thickness

with open(os.path.join(dict_output_path, "fascia_thickness_all_slices_mid_prostate.json"), "w") as f:
    json.dump(dict(fascia_thickness_dict), f, indent=4)