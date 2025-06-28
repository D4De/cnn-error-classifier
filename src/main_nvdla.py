import os
import json
import numpy as np
import logging as log
import traceback
import multiprocessing

import dill
dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
multiprocessing.reduction.ForkingPickler = dill.Pickler

from tqdm import tqdm
from statistics import mean, stdev
from collections import OrderedDict

from db import create_db, delete_db, put_experiment_data
from args import Args, create_parser
from utils import precalculate_unit_workload
from classes import generate_classes_models
from functools import partial
from aggregators import cardinalities_counts_by_sp_class, spatial_classes_counts, tensor_count_by_shape
from report_utils import generate_unit_report_conv, generate_unit_report_fc, report_uncategorized_tensors
from logging_utils import setup_logging, set_console_logging_level
from analyzed_tensor import AnalyzedTensorConv, AnalyzedTensorFC
from hw_unit_analyzer import ProcessArgs, analyze_errors_file_conv, analyze_errors_file_fc

from spatial_classifier.spatial_classifier import (
    clear_spatial_classification_folders,
    create_visual_spatial_classification_folders,
)


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
        golden: np.ndarray = np.load(golden_path)
        log.info(f'Loaded golden file. Shape is {golden.shape}.')
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
        log.info('Creating classes directory.')
        os.makedirs(args.classes_output_dir, exist_ok=True)


    # Create/clear the output folder for spatial class visualizations
    if args.visualize:
        log.info('Creating visualization directories.')
        clear_spatial_classification_folders(args.visualize_path)
        # Create output folder structure (if it does not exist already)
        create_visual_spatial_classification_folders(args.visualize_path)
    
    # Prepare the database if requested
    if args.database:
        log.info('Preparing output database.')
        db_path = os.path.join(args.output_dir, 'experiments.sqlite')
        delete_db(db_path)


    # START ANALYSIS
    global_results = []

    def _generate_reports_conv(group_name: str, unit_name: str, tensors: list):
        if args.classes:
            unit_dir = os.path.join(args.classes_output_dir, f'{group_name}_{unit_name}')
            if not os.path.isdir(unit_dir):
                os.makedirs(unit_dir, exist_ok=True)

            generate_classes_models(tensors, args, unit_dir)
            generate_unit_report_conv(unit_dir, tensors)
            report_uncategorized_tensors(unit_dir, tensors)

    def _generate_reports_fc(group_name: str, unit_name: str, tensors: list):
        unit_dir = os.path.join(args.classes_output_dir, unit_name)
        if not os.path.isdir(unit_dir):
            os.makedirs(unit_dir, exist_ok=True)

        generate_unit_report_fc(unit_dir, tensors)

    # determine the type of operator according to the golden shape
    if len(golden.shape) == 4:
        # assume convolutional
        log.info('Golden is 4D, assuming convolutional layer.')
        analysis_function = analyze_errors_file_conv
        report_function = _generate_reports_conv
    elif len(golden.shape == 2):
        # assume fully connected
        log.info('Golden is 2D, assuming fully connected layer.')
        analysis_function = analyze_errors_file_fc
        report_function = _generate_reports_fc
    else:
        log.error(f'Unrecognized golden shape: {golden.shape}. Analysis is impossible.')
        raise ValueError()


    # proceed one unit directory at a time
    def _analyze_unit_directory(group_name: str, unit_name: str, dir_path: str):
        log.info(f'---Starting analysis of {group_name}/{unit_name}.')
        errors_archive_path = os.path.join(dir_path, args.errors_archive_filename)

        if not os.path.exists(errors_archive_path):
            log.warning(f'Errors archive {errors_archive_path} does not exist. Skipping.')
            return []

        errors_archive = np.load(errors_archive_path)
        num_errors = precalculate_unit_workload(errors_archive)

        log.info(f'Loaded errors archive: number of errors is {num_errors}.')

        # get the error files: archive_files is a list of tuples (file_id, numpy error file)
        archive_files = errors_archive.items()

        # prepare list of arguments to map to processes
        process_arguments: list[ProcessArgs] = []
        for file in archive_files:
            process_arguments.append(
                ProcessArgs(
                    group_name=group_name,
                    unit_name=unit_name,
                    errors_file_idx=file[0],
                    errors_file=file[1]
                )
            )

        # partially map the analysis function with the common arguments
        analysis_partial = partial(analysis_function, args=args, golden=golden)

        # start parallel analysis
        with multiprocessing.Pool(args.parallel) as pool:
            result = pool.map_async(analysis_partial, process_arguments, chunksize=1)
            final_result = result.get()

        # collect results
        unit_analyzed_tensors = []

        for process_result in final_result:
            if process_result is not None:
                unit_analyzed_tensors += process_result

        # create unit reports
        report_function(group_name=group_name, unit_name=unit_name, tensors=unit_analyzed_tensors)
        
        return unit_analyzed_tensors


    for ctrl_dir_name in ctrl_dirs_names:
        dir_path = os.path.join(ctrl_dir, ctrl_dir_name)
        global_results += _analyze_unit_directory('ctrl', ctrl_dir_name, dir_path)
    for data_dir_name in data_dirs_names:
        dir_path = os.path.join(data_dir, data_dir_name)
        global_results += _analyze_unit_directory('data', data_dir_name, dir_path)


    # ANALYSIS DONE
    log.info('---All units analyzed. Preparing global model and reports.')
    # prepare global report
    global_report = OrderedDict()
    
    result_count = len(global_results)

    # determine what to produce based on the type of analyzed operator
    result_type = type(global_results[0])
    # CONVOLUTIONAL LAYER
    if result_type == AnalyzedTensorConv:
        # Generate the json files of errors models needed in the CLASSES framework (if option --classes is specified in arguments)
        if args.classes is not None and result_count > 0:
            log.info('Generating classes model.')
            generate_classes_models(global_results, args)
        
        if args.database:
            log.info('Filling database.')
            db_path = os.path.join(args.output_dir, 'experiments.sqlite')
            try:
                create_db(db_path)
                put_experiment_data(db_path, global_results)
                log.info(f"Saved experiments in {db_path}")
            except Exception as e:
                log.error(f"Exception {e} happened while saving to the db")
                traceback.print_exc(e)


        global_report["classified_tensors"] = result_count
        global_report["tensors_by_shape"] = tensor_count_by_shape(global_results)
        global_report["spatial_classes"] = spatial_classes_counts(global_results)
        global_report["class_cardinalites"] = cardinalities_counts_by_sp_class(global_results)

    # FULLY CONNECTED LAYER
    elif result_type == AnalyzedTensorFC:
        global_report["classified_tensors"] = result_count
        global_report["tensor_shape"] = global_results[0].shape

        # determine min, max, avg and std. dev. of number of corrupted values, L1 distance and L2 distance
        corrupted_values_counts = []
        L1_distances = []
        L2_distances = []

        for tensor in global_results:
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

    log.info('Saving global report.')
    with open(os.path.join(args.output_dir, "global_report.json"), "w") as rf:
        json.dump(global_report, rf, indent=2)

    log.info('>>>ANALYSIS DONE<<<')

if __name__ == "__main__":
    main()