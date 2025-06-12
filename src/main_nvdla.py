import os
import json
import sys
import traceback
import logging as log

from functools import partial
from multiprocessing import Manager, Pool, Process, Queue
from queue import Empty
from typing import Dict, List, Tuple
from collections import OrderedDict
from tqdm import tqdm

from aggregators import cardinalities_counts_by_sp_class, spatial_classes_counts, tensor_count_by_shape
from analyzed_tensor import AnalyzedTensor
from args import Args, create_parser
from batch_analyzer_nvdla import analyze_batch
from classes import generate_classes_models
from db import create_db, delete_db, put_experiment_data
from utils import read_npz_sizes

from spatial_classifier.spatial_classifier import (
    clear_spatial_classification_folders,
    create_visual_spatial_classification_folders,
)


def setup_logging():
    """
    Configure logger for printing to the console.
    """
    root = log.getLogger()
    root.setLevel(log.INFO)

    handler = log.StreamHandler(sys.stdout)
    handler.setLevel(log.DEBUG)
    formatter = log.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)


def precalculate_workload(
    hw_unit_paths: List[str], errors_filename: str = 'errors.npz'
) -> Tuple[int, Dict[str, int]]:
    """
    Calculates the number of tensors to analyze in the subdirectories. Returns a tuple containing the total number of tensors in all
    subdirectories and a dict that contains the number of tensors for each one.
    """
    if not errors_filename.endswith('.npz'):
        errors_filename = errors_filename + '.npz'

    total_tensors = 0
    sizes = {}

    for unit_path in hw_unit_paths:
        num_tensors = 0

        path_to_errors = os.path.join(unit_path, errors_filename)

        for tensor_shape in read_npz_sizes(path_to_errors):
            # tensor_shape here is a 5-uple with this interpretation: (num_injections, num_batches, num_channels, height, width)
            # so, the amount of "single" output tensors for each file in the npz archive is injection_number * batch_number
            num_tensors += (tensor_shape[0] * tensor_shape[1])

        log.info(f'\t{unit_path}: found {num_tensors} tensors to analyze')

        sizes[unit_path] = num_tensors
        total_tensors += num_tensors

    return total_tensors, sizes


def progress_handler(queue: Queue, work: int):
    """
    A process that updates the progress bar.
    The working threads send a "processed" message to this thread, using the queue, whenever they complete processing a tensor.
    The responsibility of this process is to update the progress bar when a "processed" is received.
    """
    with tqdm(total=work) as pbar:
        work_count = 0
        while True:
            try:
                message, args = queue.get(True, 15)
                if message == "exit":
                    break
                elif message == "processed":
                    work_count += args
                    pbar.update(args)
                    sys.stdout.flush()
                    if work_count >= work:
                        print("Work complete!")
                        break
            except Empty:
                print("No updates received. Quitting")
                break


def main():
    # Parsing command line arguments
    parser = create_parser()
    argparse_args = parser.parse_args()
    # The args variable contains the configuration of this application given by the user via CLI arguments
    args = Args.from_argparse(argparse_args)

    setup_logging()


    # The error batches are grouped wrt the hardware path for the injection (control or data)
    ctrl_dir = os.path.join(args.root_path, 'ctrl')
    data_dir = os.path.join(args.root_path, 'data')

    hw_unit_dirs = []

    if os.path.isdir(ctrl_dir):
        log.info('Found ctrl directory')
        hw_unit_dirs += [os.path.join(ctrl_dir, dir) for dir in os.listdir(ctrl_dir) if os.path.isdir(os.path.join(ctrl_dir, dir))]
    else:
        log.warning(f'Control path directory {ctrl_dir} is missing.')

    if os.path.isdir(data_dir):
        log.info('Found data directory')
        hw_unit_dirs += [os.path.join(data_dir, dir) for dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, dir))]
    else:
        log.warning(f'Data path directory {data_dir} is missing.')

    log.info(f"Found {len(hw_unit_dirs)} hardware units directories to analyze.")

    # Create additional directories
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if args.classes and not os.path.isdir(args.classes_output_dir):
        os.makedirs(args.classes_output_dir, exist_ok=True)

    if args.partial_reports and not os.path.exists(args.reports_path):
        os.makedirs(args.reports_path, exist_ok=True)

    # Folder for visualizations must be erased only if new one are generated
    if args.visualize:
        clear_spatial_classification_folders(args.visualize_path)
        # Create output folder structure (if not exists already)
        create_visual_spatial_classification_folders(args.visualize_path)
    
    if args.database:
        db_path = os.path.join(args.output_dir, 'experiments.sqlite')
        delete_db(db_path)


    # workload == total number of tensors to analyze (for progress bar)
    total_tensors, unit_dirs_sizes = precalculate_workload(hw_unit_dirs)
    hw_unit_dirs = sorted(hw_unit_dirs, key=lambda x: unit_dirs_sizes[x], reverse=True)
    log.info(f"Found {total_tensors} total tensors to analyze")

    # Mute logger to avoid interferences with tqdm
    log.getLogger().setLevel(log.WARN)


    # Initialize global dictionaries
    global_report = OrderedDict()

    # Multiprocessing setup
    manager = Manager()
    progress_queue = manager.Queue()
    # Pre-Configure the arguments of analyze_batch that are the same for all batches
    # batch_partial accepts only one parameter
    batch_partial = partial(analyze_batch, args=args, queue=progress_queue)

    progress_process = Process(
        target=progress_handler, args=(progress_queue, total_tensors)
    )
    # Start the progress bar process
    progress_process.start()
    # Start the worker process
    with Pool(args.parallel) as pool:
        result = pool.map_async(batch_partial, hw_unit_dirs, chunksize=1)

        final_result = result.get()
    progress_process.join()

    analyzed_tensors : List[AnalyzedTensor]= []
    metadata_dicts = []


    for batch_result in final_result:
        if batch_result is not None:
            tensor_list, metadata = batch_result
            analyzed_tensors += tensor_list
            metadata_dicts.append(metadata)
    
    # Calculate cumulative metrics
    result_count = len(analyzed_tensors)
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
    #global_report["domain_classes_types_per_tensor"] = domain_classes_types_counts(analyzed_tensors)
    #global_report["domain_classes_types_per_sp_class"] = domain_class_type_per_spatial_class(analyzed_tensors)
    #global_report["domain_classes_counts"] = domain_classes_counts(analyzed_tensors)
    global_report["class_cardinalites"] = cardinalities_counts_by_sp_class(analyzed_tensors)

    with open(os.path.join(args.output_dir, "global_report.json"), "w") as rf:
        json.dump(global_report, rf, indent=2)


if __name__ == "__main__":
    main()
