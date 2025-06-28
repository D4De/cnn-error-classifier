import os
import numpy as np
import logging as log

from tqdm import tqdm
from collections import defaultdict

from args import Args
from classes import generate_classes_models
from visualizer import visualize
from coordinates import map_to_coordinates, numpy_coords_to_python_coord

from dataclasses import dataclass
from analyzed_tensor import AnalyzedTensorConv, AnalyzedTensorFC
from domain_classifier import ValueClass, domain_classification, value_classification
from spatial_classifier.spatial_classifier import spatial_classification

# ANALYSIS FUNCTIONS ---------------------------------------------------------------------------------------------------

def analyze_hw_unit_directory(unit_dir: str, group_name: str, unit_name: str, args: Args, golden: np.ndarray):
    log.info(f'Starting analysis of {group_name}-{unit_name}')

    errors_path = os.path.join(unit_dir, args.errors_archive_filename)
    if not os.path.exists(errors_path):
        log.error(f"Skipping {unit_dir} hardware unit directory. Errors archive not found.")
        return []
    
    # load errors archive
    try:
        errors = np.load(errors_path)
    except:
        log.error(f'Errors archive {errors_path} could not be loaded.')
        raise ValueError()

    # compute the total number of tensors in the archive
    total_tensors = precalculate_unit_workload(errors)

    golden_range_min = float(np.min(golden))
    golden_range_max = float(np.max(golden))

    classified_tensors = []

    # disable console logging
    set_console_logging_level(log.WARN)

    with tqdm(total=total_tensors) as progress_bar:
        # determine layer type based on the golden size
        if len(golden.shape) == 4:
            # assume convolutional
            classified_tensors = analyze_errors_conv(
                golden=golden,
                golden_range_min=golden_range_min,
                golden_range_max=golden_range_max,
                errors=errors,
                args=args,
                group_name=group_name,
                unit_name=unit_name,
                progress_bar=progress_bar,
                errors_path=errors_path
            )
            
            # generate classes models and reports
            if args.classes:
                unit_dir = os.path.join(args.classes_output_dir, f'{group_name}_{unit_name}')
                if not os.path.isdir(unit_dir):
                    os.makedirs(unit_dir, exist_ok=True)

                generate_classes_models(classified_tensors, args, unit_dir)
                generate_unit_report_conv(unit_dir, classified_tensors)
                report_uncategorized_tensors(unit_dir, classified_tensors)

        elif len(golden.shape) == 2:
            # assume fully connected
            classified_tensors = analyze_errors_fc(
                golden=golden,
                errors=errors,
                args=args,
                group_name=group_name,
                unit_name=unit_name,
                progress_bar=progress_bar,
                errors_path=errors_path
            )

            unit_dir = os.path.join(args.classes_output_dir, unit_name)
            if not os.path.isdir(unit_dir):
                os.makedirs(unit_dir, exist_ok=True)

            generate_unit_report_fc(unit_dir, classified_tensors)

        else:
            log.error(f'Skipping {unit_dir} hardware unit directory. Unsupported golden shape {golden.shape}.')
    
    # enable console logging
    set_console_logging_level(log.INFO)
    
    return classified_tensors

# INDIVIDUAL TENSORS ANALYSIS ---------------------------------------------------------------------------------------------

@dataclass
class ProcessArgs:
    group_name: str
    unit_name: str
    errors_file_idx: str
    errors_file: np.ndarray

def analyze_errors_file_conv(
    other_args: ProcessArgs,
    args: Args,
    golden: np.ndarray
):
    log.info(f'-->(Process) Analyzing {other_args.group_name}/{other_args.unit_name}, file number {other_args.errors_file_idx}. Tensors to analyze: {other_args.errors_file.shape[0]}')

    my_results: list[AnalyzedTensorConv] = []

    # from the whole golden batch, take the tensor associated to this file
    my_golden = golden[int(other_args.errors_file_idx)]
    golden_shape = map_to_coordinates(my_golden.shape)
    golden_range_min = my_golden.min()
    golden_range_max = my_golden.max()

    # the errors file is a 5D tensor with this shape (injection_number, batch_number, C, H, W)
    # we want to iterate through the injection numbers (and always get the first batch number, since it's 1)
    for injection_num, error in enumerate(other_args.errors_file):
        error: np.ndarray = error[0]
        error_shape = map_to_coordinates(error.shape)

        # perform error analysis
        if golden_shape != error_shape:
            log.warning(
                f"Skipping {other_args.group_name}/{other_args.unit_name} number {other_args.errors_file_idx} injection {injection_num}. Invalid shape (Faulty has shape: {error_shape}, Golden has shape: {golden_shape})"
            )
        else:
            value_class_count = defaultdict(int)

            # generate a list of all coordinates where a difference is observed (Sparse matrix)
            if args.almost_same:
                sparse_diff_native_coords = list(zip(*np.nonzero(error - my_golden)))
            else:
                sparse_diff_native_coords = list(zip(*np.where(np.abs(error - my_golden) >= args.epsilon)))

            tensor_diff = np.zeros(error.shape, dtype=np.int8)    

            # value classification
            for coord in sparse_diff_native_coords:
                val_class = value_classification(my_golden[coord[0], coord[1], coord[2]], error[coord[0], coord[1], coord[2]], golden_range_min, golden_range_max, args.epsilon, args.almost_same)
                tensor_diff[coord[0], coord[1], coord[2]] = val_class.value
                value_class_count[val_class] += 1

            value_class_count[ValueClass.SAME] = my_golden.size - sum(value_class_count.values())

            # proceed only if there are real differences
            if len(sparse_diff_native_coords) != 0:
                sparse_diff = [
                    map_to_coordinates(numpy_coords_to_python_coord(coords))
                    for coords in sparse_diff_native_coords
                ]

                # spatial classifcation
                spatial_class, pattern_params, faulty_channels = spatial_classification(sparse_diff, golden_shape)
                domain_class = domain_classification(value_class_count)

                # create visualization
                if args.visualize:
                    folder_path = spatial_class.class_folder(args.visualize_path)
                    file_count = len([x for x in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, x))])
                    if args.visualize_limit == 0 or file_count < args.visualize_limit:
                        visualize(
                            tensor_diff,
                            faulty_channels,
                            spatial_class.output_path(
                                args.visualize_path, f'{other_args.group_name}_{other_args.unit_name}_{other_args.errors_file_idx}_{injection_num}'
                            ),
                            save=True,
                            show=False,
                            suptitile=f'{other_args.group_name}/{other_args.unit_name} {other_args.errors_file_idx} {injection_num} {my_golden.shape[0]}x{my_golden.shape[1]}x{my_golden.shape[2]}',
                            invalidate=True,
                        )

                # produce analyzed tensor
                my_results.append(
                    AnalyzedTensorConv(
                        group=other_args.group_name,
                        hw_unit=other_args.unit_name,
                        error_number=int(other_args.errors_file_idx),
                        injection_number=injection_num,
                        shape=error.shape,
                        spatial_class=spatial_class,
                        spatial_class_params=pattern_params,
                        value_classes_counts=value_class_count,
                        domain_class=domain_class,
                        corrupted_values_count=len(sparse_diff_native_coords),
                        corrupted_channels_count=len(faulty_channels)
                    )
                )
    
    log.info(f'<--(Process) Analysis done for {other_args.group_name}/{other_args.unit_name}, file number {other_args.errors_file_idx}.')

    return my_results

def analyze_errors_file_fc(
    other_args: ProcessArgs,
    args: Args,
    golden: np.ndarray
):
    log.info(f'-->(Process) Analyzing {other_args.group_name}/{other_args.unit_name}, file number {other_args.errors_file_idx}.')

    my_results: list[AnalyzedTensorConv] = []

    # from the whole golden batch, take the tensor associated to this file
    my_golden = golden[other_args.errors_file_idx]

    # the errors file is a 3D tensor with this shape (injection_number, batch_number, N)
    # we want to iterate through the injection numbers (and always get the first batch number, since it's 1)
    for injection_num, error in enumerate(other_args.errors_file):
        error: np.ndarray = error[0]

        # perform error analysis
        if my_golden.shape != error.shape:
            log.warning(
                f"Skipping {other_args.group_name}/{other_args.unit_name} number {other_args.errors_file_idx} injection {injection_num}. Invalid shape (Faulty has shape: {error_shape}, Golden has shape: {golden_shape})"
            )
        else:
            num_equal_elements = np.sum(error == my_golden)
            num_different_elements = error.size - num_equal_elements

            L1_dist = np.linalg.norm(error-my_golden, 1)
            L2_dist = np.linalg.norm(error-my_golden)

            # Per tensor report generator
            my_results.append(
                AnalyzedTensorFC(
                    group=other_args.group_name,
                    hw_unit=other_args.unit_name,
                    error_number=int(other_args.errors_file_idx),
                    injection_number=injection_num,
                    shape=error.shape,
                    corrupted_values_count=num_different_elements,
                    L1_distance=L1_dist,
                    L2_distance=L2_dist
                )
            )
    
    log.info(f'<--(Process) Analysis done for {other_args.group_name}/{other_args.unit_name}, file number {other_args.errors_file_idx}.')

    return my_results
