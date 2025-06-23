import os
import sys
import json
import numpy as np
import logging as log
import traceback

from statistics import mean, stdev
from collections import OrderedDict

from db import create_db, delete_db, put_experiment_data
from args import Args, create_parser
from classes import generate_classes_models
from aggregators import cardinalities_counts_by_sp_class, spatial_classes_counts, tensor_count_by_shape
from analyzed_tensor import AnalyzedTensorConv, AnalyzedTensorFC
from hw_unit_analyzer import analyze_hw_unit_directory

from spatial_classifier.spatial_classifier import (
    clear_spatial_classification_folders,
    create_visual_spatial_classification_folders,
)


def setup_logging(log_path: str):
    root = log.getLogger()
    root.setLevel(log.INFO)

    console_handler = log.StreamHandler(sys.stdout)
    console_handler.setLevel(log.DEBUG)
    formatter = log.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    root.addHandler(console_handler)

    file_handler = log.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(log.INFO)
    file_handler.setFormatter(formatter)

    root.addHandler(file_handler)


def main():
    # Parse CLI arguments
    parser = create_parser()
    args = Args.from_argparse(parser.parse_args())

    # Prepare output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    log_path = os.path.join(args.output_dir, 'classification_log.out')

    # Prepare logging
    setup_logging(log_path)


    # Load the golden file
    golden_path = os.path.join(args.root_path, args.golden_path)
    if not os.path.exists(golden_path):
        log.error(f'Golden file at {golden_path} does not exist.')
        raise FileNotFoundError()

    try:
        golden = np.load(golden_path)
    except:
        log.error(f'Golden file {golden_path} cannot be loaded.')
        raise ValueError()

    # The hardware unit directories are grouped wrt the hardware path for the injection (control or data)
    ctrl_dir = os.path.join(args.root_path, 'ctrl')
    data_dir = os.path.join(args.root_path, 'data')

    ctrl_dirs_names = []
    data_dirs_names = []

    # Find the hardware unit directories
    if os.path.isdir(ctrl_dir):
        ctrl_dirs_names += [dir for dir in os.listdir(ctrl_dir) if os.path.isdir(os.path.join(ctrl_dir, dir))]
        log.info(f'Found ctrl directory: {len(ctrl_dirs_names)} unit directories within.')
    else:
        log.warning(f'Control path directory {ctrl_dir} is missing.')

    if os.path.isdir(data_dir):
        data_dirs_names += [dir for dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, dir))]
        log.info(f'Found data directory: {len(data_dirs_names)} unit directories within.')
    else:
        log.warning(f'Data path directory {data_dir} is missing.')

    if len(ctrl_dirs_names) == 0 and len(data_dirs_names) == 0:
        log.error('No hardware unit directories found.')
        raise FileNotFoundError()


    # Create additional directories
    if args.classes and not os.path.isdir(args.classes_output_dir):
        os.makedirs(args.classes_output_dir, exist_ok=True)


    # Create/clear the output folder for spatial class visualizations
    if args.visualize:
        clear_spatial_classification_folders(args.visualize_path)
        # Create output folder structure (if it does not exist already)
        create_visual_spatial_classification_folders(args.visualize_path)
    
    # Prepare the database if requested
    if args.database:
        db_path = os.path.join(args.output_dir, 'experiments.sqlite')
        delete_db(db_path)


    # START ANALYSIS
    analyzed_tensors = []

    for ctrl_dir_name in ctrl_dirs_names:
        unit_path = os.path.join(ctrl_dir, ctrl_dir_name)
        analyzed_tensors += analyze_hw_unit_directory(unit_path, 'ctrl', ctrl_dir_name, args, golden)

    for data_dir_name in data_dirs_names:
        unit_path = os.path.join(data_dir, data_dir_name)
        analyzed_tensors += analyze_hw_unit_directory(unit_path, 'data', data_dir_name, args, golden)

    # prepare global report
    global_report = OrderedDict()
    
    result_count = len(analyzed_tensors)

    # determine what to produce based on the type of analyzed operator
    result_type = type(analyzed_tensors[0])
    # CONVOLUTIONAL LAYER
    if result_type == AnalyzedTensorConv:
        # Generate the json files of errors models needed in the CLASSES framework (if option --classes is specified in arguments)
        if args.classes is not None and result_count > 0:
            generate_classes_models(analyzed_tensors, args)
        
        if args.database:
            db_path = os.path.join(args.output_dir, 'experiments.sqlite')
            try:
                create_db(db_path)
                put_experiment_data(db_path, analyzed_tensors)
                log.info(f"Saved experiments in {db_path}")
            except Exception as e:
                log.error(f"Exception {e} happened while saving to the db")
                traceback.print_exc(e)


        global_report["classified_tensors"] = result_count
        global_report["tensors_by_shape"] = tensor_count_by_shape(analyzed_tensors)
        global_report["spatial_classes"] = spatial_classes_counts(analyzed_tensors)
        global_report["class_cardinalites"] = cardinalities_counts_by_sp_class(analyzed_tensors)

    # FULLY CONNECTED LAYER
    elif result_type == AnalyzedTensorFC:
        global_report["classified_tensors"] = result_count
        global_report["tensor_shape"] = analyzed_tensors[0].shape

        # determine min, max, avg and std. dev. of number of corrupted values, L1 distance and L2 distance
        corrupted_values_counts = []
        L1_distances = []
        L2_distances = []

        for tensor in analyzed_tensors:
            corrupted_values_counts.append(tensor.corrupted_values_count)
            L1_distances.append(tensor.L1_distance)
            L2_distances.append(tensor.L2_distance)

        global_report["num_corrupted_values"] = {
            "min": min(corrupted_values_counts),
            "max": max(corrupted_values_counts),
            "mean": mean(corrupted_values_counts),
            "stdev": stdev(corrupted_values_counts)
        }
        global_report["L1_distance"] = {
            "min": min(L1_distances),
            "max": max(L1_distances),
            "mean": mean(L1_distances),
            "stdev": stdev(L1_distances)
        }
        global_report["L2_distance"] = {
            "min": min(L2_distances),
            "max": max(L2_distances),
            "mean": mean(L2_distances),
            "stdev": stdev(L2_distances)
        }

    else:
        raise TypeError(f"Results are of unknown type {result_type}")


    with open(os.path.join(args.output_dir, "global_report.json"), "w") as rf:
        json.dump(global_report, rf, indent=2)


if __name__ == "__main__":
    main()