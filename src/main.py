from functools import partial
from multiprocessing import Manager, Pool, Process, Queue
from queue import Empty
from typing import Dict, List, Tuple
import logging as log
from collections import OrderedDict
import os
from tqdm import tqdm
import json
import sys
from aggregators import cardinalities_counts, domain_class_type_per_spatial_class, domain_classes_counts, domain_classes_types_counts, experiment_counts, spatial_classes_counts, tensor_count_by_shape, tensor_count_by_sub_batch
from analyzed_tensor import AnalyzedTensor
from args import Args, create_parser
from batch_analyzer import analyze_batch
import numpy as np
from classes import generate_classes_models
from db import create_db, delete_db, put_experiment_data
import traceback

REPORT_FILE = "report.json"
TOP_PATTERNS_PCT = 5

from spatial_classifier.spatial_classifier import (
    clear_spatial_classification_folders,
    create_visual_spatial_classification_folders,
)


def setup_logging():
    """
    Configure logger for printing in the console
    """
    root = log.getLogger()
    root.setLevel(log.INFO)

    handler = log.StreamHandler(sys.stdout)
    handler.setLevel(log.DEBUG)
    formatter = log.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)


def precalculate_workload(
    batch_paths: List[str], faulty_path: str, golden_path: str
) -> Tuple[int, int, Dict[str, int]]:
    """
    Calculate the number of tensors to analyze in the subdirectories. Returns a tuple containing the number of tensors in all
    batches, the number of values inside all faulty tensors (obtaining summing the .size attribute of all tensors) and a dict
    that contains the number of tensors for each batch

    batch_paths
    ---
    Paths to the batchs to analyze

    faulty_path
    ---
    Relative path from all the batches' home folders to the folders that contains the sub batches of faulty values

    golden_path
    ---
    Relative path from all the batches' home folders to the golden .npy file
    """
    tensors = 0
    values = 0
    batch_sizes = {}
    # Iterate for each batch
    for batch_path in batch_paths:
        batch_values_count = 0
        batch_tensors_count = 0
        path_to_golden = os.path.join(batch_path, golden_path)
        golden_size = np.load(path_to_golden).size

        faulty_dir_path = os.path.join(batch_path, faulty_path)
        # Detect sub batches inside the batches
        sub_batch_dirs = [
            os.path.join(faulty_dir_path, dir)
            for dir in os.listdir(faulty_dir_path)
            if os.path.isdir(os.path.join(faulty_dir_path, dir))
        ]
        for sub_batch_path in sub_batch_dirs:
            # Count tensors inside the sub batch
            sub_batch_tensor_count = len(
                [
                    os.path.join(sub_batch_path, entry)
                    for entry in os.listdir(sub_batch_path)
                    if entry.split(".")[1] == "npy"
                ]
            )
            sub_batch_value_count = sub_batch_tensor_count * golden_size
            batch_tensors_count += sub_batch_tensor_count
            tensors += sub_batch_tensor_count
            values += sub_batch_value_count
            batch_values_count += sub_batch_value_count
        batch_sizes[batch_path] = batch_tensors_count
    return tensors, values, batch_sizes


def progress_handler(queue: Queue, work: int):
    """
    A process that updates the progress bar.
    The working threads send a "processed" message to the this thread, using the queue, whenever they complete processing a tensor.
    The responsibility of this process is to update the progress bar when a "processed" is received

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
    # Get all the tests batches paths (a batch is a folder inside the root path)
    test_batches_paths = [
        os.path.join(args.root_path, dir)
        for dir in os.listdir(args.root_path)
        if os.path.isdir(os.path.join(args.root_path, dir))
    ]

    # Slice the batches (for testing purposes)
    if args.limit is not None:
        test_batches_paths = test_batches_paths[: args.limit]

    test_batches_paths = [
        path
        for path in test_batches_paths
        if not os.path.basename(path).startswith("_")
    ]

    log.info(f"Found {len(test_batches_paths)} batches to analyze")

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if not os.path.exists(args.reports_path):
        os.mkdir(args.reports_path)

    # Folder for visualizations must be erased only if new one are generated
    if args.visualize:
        clear_spatial_classification_folders(args.visualize_path)
        # Create output folder structure (if not exists already)
        create_visual_spatial_classification_folders(args.visualize_path)
    
    if args.database:
        db_path = os.path.join(args.output_dir, 'experiments.sqlite')
        delete_db(db_path)

    # workload == number of tensors to analyze in all batches (for progress bar)
    n_tensors, n_values, batch_sizes = precalculate_workload(
        test_batches_paths, args.faulty_path, args.golden_path
    )
    test_batches_paths = sorted(
        test_batches_paths, key=lambda x: batch_sizes[x], reverse=True
    )
    log.info(f"Found {n_tensors} tensors to analyze")
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
        target=progress_handler, args=(progress_queue, n_tensors)
    )
    # Start the progress bar process
    progress_process.start()
    # Start the worker process
    with Pool(args.parallel) as pool:
        result = pool.map_async(batch_partial, test_batches_paths, chunksize=1)

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



    total_experiments, experiments_by_types = experiment_counts(metadata_dicts)
    global_report["total_experiments"] = total_experiments
    global_report["experiment_by_types"] = experiments_by_types
    global_report["tensors_by_sub_batch"] = tensor_count_by_sub_batch(analyzed_tensors)
    global_report["tensors_by_shape"] = tensor_count_by_shape(analyzed_tensors)
    global_report["classified_tensors"] = result_count
    global_report["spatial_classes"] = spatial_classes_counts(analyzed_tensors)
    global_report["domain_classes_types_per_tensor"] = domain_classes_types_counts(analyzed_tensors)
    global_report["domain_classes_types_per_sp_class"] = domain_class_type_per_spatial_class(analyzed_tensors)
    global_report["domain_classes_counts"] = domain_classes_counts(analyzed_tensors)
    global_report["cardinalities"] = cardinalities_counts(analyzed_tensors)


    with open(os.path.join(args.output_dir, "global_report.json"), "w") as f:
        f.writelines(json.dumps(global_report, indent=2))
    


if __name__ == "__main__":
    main()
