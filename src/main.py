
from functools import partial
from multiprocessing import Manager, Pool, Process, Queue
from operator import itemgetter
from queue import Empty
from typing import List
import logging as log
from collections import OrderedDict, defaultdict
import os
from tqdm import tqdm
import json
import sys
from args import Args, create_parser
from batch_analyzer import analyze_batch
from domain_classifier import DomainClass

REPORT_FILE = "report.json"
TOP_PATTERNS_PCT = 5

from spatial_classifier import (
    accumulate_max,
    clear_spatial_classification_folders,
    create_visual_spatial_classification_folders,
    to_classes_id,
)


def setup_logging():
    root = log.getLogger()
    root.setLevel(log.INFO)

    handler = log.StreamHandler(sys.stdout)
    handler.setLevel(log.DEBUG)
    formatter = log.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)


def precalculate_workload(batch_paths: List[str], faulty_path: str):
    tensors = 0
    for batch_path in batch_paths:
        faulty_dir_path = os.path.join(batch_path, faulty_path)
        sub_batch_dirs = [
            os.path.join(faulty_dir_path, dir)
            for dir in os.listdir(faulty_dir_path)
            if os.path.isdir(os.path.join(faulty_dir_path, dir))
        ]
        for sub_batch_path in sub_batch_dirs:
            tensors += len(
                [
                    os.path.join(sub_batch_path, entry)
                    for entry in os.listdir(sub_batch_path)
                    if entry.split(".")[1] == "npy"
                ]
            )
    return tensors

def progress_handler(queue: Queue, work: int):
    with tqdm(total=work) as pbar:
        work_count = 0
        while True:
            try:
                message = queue.get(True, 10)
                if message == "processed":
                    work_count += 1
                    pbar.update(1)
                    sys.stdout.flush()
                    if work_count >= work:
                        print("Work complete!")
                        break
                elif message == "exit":
                    break
            except Empty:
                print("No updates received. Quitting")
                break 




def main():

    parser = create_parser()
    argparse_args = parser.parse_args()
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
    
    test_batches_paths = [path for path in test_batches_paths if not os.path.basename(path).startswith('_')]

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
    # workload == number of tensors to analyze in all batches (for progress bar)
    workload = precalculate_workload(test_batches_paths, args.faulty_path)

    log.info(f"Found {workload} tensors to analyze")
    # Mute logger to avoid interferences with tqdm
    log.getLogger().setLevel(log.WARN)

    # Initialize global dictionaries
    global_report = OrderedDict()
    global_dom_classes = defaultdict(lambda: 0)
    global_sp_classes = defaultdict(lambda: 0)
    # global_error_patterns[cardinality][spatial class name][pattern (stringfied tuple)] -> frequency of the pattern
    global_error_patterns = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: 0))
    )
    global_class_params = defaultdict(
        lambda: defaultdict(list)
    )
    # global_error_patterns[cardinality][spatial class name] -> frequency of the spatial class with that cardinailty
    global_cardinalities = defaultdict(lambda: defaultdict(lambda: 0))
    # Global bind dictionaries to global dict (that will be ouptu in json)
    global_report["domain_classes"] = global_dom_classes
    global_report["spatial_classes"] = global_sp_classes
    
    manager = Manager()
    progress_queue = manager.Queue()
    batch_partial = partial(analyze_batch, args=args, queue=progress_queue)
    
    progress_process = Process(target=progress_handler, args=(progress_queue, workload))
    progress_process.start()
    with Pool(args.parallel) as pool:
        result = pool.map_async(batch_partial, test_batches_paths, chunksize=1)

        final_result = result.get()
    progress_process.join()   

    for batch, data in zip(test_batches_paths, final_result):
        if data is not None:
        # accumulate values to add in globa dicts
            for dom_class, count in data.batch_dom_classes.items():
                global_dom_classes[dom_class] += count
            for sp_class, count in data.batch_sp_classes.items():
                global_sp_classes[sp_class] += count
            for cardinality, sp_classes in data.batch_error_patterns.items():
                total = 0
                for sp_class, patterns in sp_classes.items():
                    total += len(patterns)
                    global_cardinalities[cardinality][sp_class] += len(patterns)
                    for pattern, freq in patterns.items():
                        global_error_patterns[cardinality][sp_class][
                            pattern
                        ] += freq
                global_cardinalities[cardinality]["sum"] = total
            for cardinality, sp_classes in data.batch_class_params.items():
                for sp_class, max_val in sp_classes.items():
                    global_class_params[cardinality][sp_class] = accumulate_max(global_class_params[cardinality][sp_class], max_val)
#            global_report[data.batch_name] = data.batch_report
        else:
            log.WARN(f"Batch {batch} returned None")
        
    global_cardinalities_count = OrderedDict(
        sorted(
            (
                (
                    cardinality,
                    sum(
                        sum(freq for freq in patterns.values())
                        for patterns in sp_classes.values()
                    ),
                )
                for cardinality, sp_classes in global_error_patterns.items()
            ),
            key=itemgetter(1),
            reverse=True,
        )
    )

    global_cardinalities_sorted = {
        cardinality: {
            sp_class: sum(freq for freq in patterns.values())
            for sp_class, patterns in sorted(sp_classes.items(), key=itemgetter(0))
        }
        for cardinality, sp_classes in global_error_patterns.items()
    }

    total_classified_tensors = sum(
        pattern_freq for pattern_freq in global_cardinalities_count.values()
    )

    global_error_patterns_sorted = {
        cardinality: {
            sp_class: OrderedDict(
                [
                    (pattern, freq / global_cardinalities_sorted[cardinality][sp_class])
                    for pattern, freq in sorted(
                        patterns.items(), key=itemgetter(1), reverse=True
                    )
                    if freq / global_cardinalities_sorted[cardinality][sp_class] >= 0.05
                ]
                + [
                    (
                        "RANDOM",
                        sum(
                            freq / global_cardinalities_sorted[cardinality][sp_class]
                            for freq in patterns.values()
                            if freq / global_cardinalities_sorted[cardinality][sp_class]
                            <= 0.05
                        ),
                    )
                ]
            )
            for sp_class, patterns in sp_classes.items()
        }
        for cardinality, sp_classes in sorted(
            global_error_patterns.items(), key=itemgetter(0)
        )
    }

    # Generate the json files of errors models needed in the CLASSES framework (if option --classes is specified in arguments)
    if args.classes is not None:
        # Cardinality file
        # Keys: All the cardinalities (number of errors in each tensors)
        # Values: Relative frequency of the cardinality [0,1]
        classes_cardinalities = {
            cardinality: [freq, freq / total_classified_tensors]
            for cardinality, freq in global_cardinalities_count.items()
        }

        # Spatial model file
        # Keys: All the cardinalities (number of errors in each tensors)
        # Values: A dict containing two keys
            # Keys:
                # "FF": A dict:
                    # Keys: The CLASSES id of a spatial class of error patterns (see SpatialClass.to_classes_id()) 
                    # Values: relative frequency of the pattern given the cardinality (all values must sum to 1)
                # "PF": A dict:
                    # Keys: The CLASSES id of a spatial class of error patterns (see SpatialClass.to_classes_id()) 
                    # Values: A dict
                        # Keys: The string representing the pattern (usually multiple nested tuple converted to str, structure varies depending on spatial class)
                        #       "RANDOM", "MAX" are other two keys that may appear
                        # Values: The relative frequency of the pattern. All the values associated to a tuple + the value of "RANDOM" MUST sum to 1
                        #       "RANDOM" value is the probability that the pattern is different from all the listed patterns
                        #       "MAX" values contains max parameters of the pattern, may be an int or a list 
        classes_spatial_models = {
            cardinality: {
                "FF": {
                    to_classes_id(sp_class): global_cardinalities_sorted[cardinality][
                        sp_class
                    ]
                    / global_cardinalities_count[cardinality]
                    for sp_class in sp_classes
                },
                "PF": {
                    to_classes_id(sp_class): {
                        pattern: freq
                        for pattern, freq in global_error_patterns_sorted[cardinality][
                            sp_class
                        ].items()
                    }| {"MAX": global_class_params[cardinality][sp_class] if len(global_class_params[cardinality][sp_class]) != 1 else global_class_params[cardinality][sp_class][0]}
                    for sp_class in sp_classes
                },
            }
            for cardinality, sp_classes in sorted(
                global_error_patterns.items(), key=itemgetter(0)
            )
        }

        classes_spatial_models[1] = {"RANDOM": 1}

        dom_class = {
            "plus_minus_one": global_dom_classes[DomainClass.OFF_BY_ONE.display_name()],
            "others": global_dom_classes[DomainClass.RANDOM.display_name()],
            "zeros": global_dom_classes[DomainClass.ZERO.display_name()],
            "NaN": global_dom_classes[DomainClass.NAN.display_name()],
        }

        total = sum(freq for freq in dom_class.values())

        dom_class = {dom_class: freq / total for dom_class, freq in dom_class.items()}

        dom_class["total"] = total
        # A string containing data about domains of the errors. Bitflip and Same must be excluded from the domains even if it figures in DomainClasses
        classes_domain_models = """There have been {total} faults
            [-1, 1]: {plus_minus_one}
            Others: {others}
            NaN: {NaN}
            Zeros: {zeros}
            Valid: 1.00000
        """.format(
            **dom_class
        )
        with open(os.path.join(args.output_dir, "value_analysis.txt"), "w") as f:
            f.write(classes_domain_models)
        with open(
            os.path.join(
                args.output_dir,
                f"{args.classes[0]}_{args.classes[1]}_spatial_model.json",
            ),
            "w",
        ) as f:
            f.write(json.dumps(classes_spatial_models, indent=2))
        with open(
            os.path.join(
                args.output_dir,
                f"{args.classes[1]}_{args.classes[0]}_anomalies_count.json",
            ),
            "w",
        ) as f:
            f.write(json.dumps(classes_cardinalities, indent=2))

    global_report["classified_tensors"] = total_classified_tensors
    global_report["classes_by_cardinalities"] = global_cardinalities_sorted
    global_report["cardinalities"] = global_cardinalities_count
    with open(os.path.join(args.output_dir, "global_report.json"), "w") as f:
        f.writelines(json.dumps(global_report, indent=2))


if __name__ == "__main__":
    main()
