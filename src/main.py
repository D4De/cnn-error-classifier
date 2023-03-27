import argparse
from typing import List
import numpy as np
import logging as log
from collections import OrderedDict
import os
from tqdm import tqdm
import json
import sys
from tensor_analyzer import analyze_tensor_directory

REPORT_FILE = "report.json"
TOP_PATTERNS_PCT = 5

from spatial_classifier import (
    clear_spatial_classification_folders,
    create_visual_spatial_classification_folders,
)

from coordinates import TensorLayout, map_to_coordinates


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


def main():
    parser = argparse.ArgumentParser(
        prog="Tensor Error Classifier",
        description="Compares Faulty Tensors with a golden one, and classifies them",
    )
    parser.add_argument(
        "root_path", help="A path to the root folder of the test results"
    )
    parser.add_argument(
        "golden_path",
        help="A relative path that reaches the golden file from the test batch home folder",
    )
    parser.add_argument(
        "faulty_path",
        help="A relative path that reaches the folders containing faulty files from the test batch home folder",
    )
    parser.add_argument(
        "output_dir", help="Path where to write the ouput of the analysis"
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        help="Limit the number of tensor to process",
        metavar="N",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Generate images that show visually the differences between tensors (overwriting the image generated before)",
    )

    parser.add_argument(
        "-as",
        "--almost-same",
        action="store_true",
        help="Include in the plot the values that are very close to golden value (< EPS)",
    )
    parser.add_argument(
        "-eps",
        "--epsilon",
        type=float,
        default=10e-3,
        help="Set epsilon value. Differences below epsilon are treated as almost same value and are not plotted (unless -as is enabled)",
    )

    tensor_format_group = parser.add_mutually_exclusive_group()
    tensor_format_group.add_argument(
        "-nchw",
        action="store_true",
        help="Loaded tensors are stored using NCHW dimensional order (default)",
    )
    tensor_format_group.add_argument(
        "-nhwc",
        action="store_true",
        help="Loaded tensors are stored using NHWC dimensional order",
    )

    args = parser.parse_args()

    setup_logging()

    # Read layout from args (default is NCHW)
    layout = TensorLayout.NHWC if args.nhwc else TensorLayout.NCHW

    test_batches_paths = [
        os.path.join(args.root_path, dir)
        for dir in os.listdir(args.root_path)
        if os.path.isdir(os.path.join(args.root_path, dir))
    ]

    if args.limit is not None:
        test_batches_paths = test_batches_paths[: args.limit]

    log.info(f"Found {len(test_batches_paths)} batches to analyze")

    visualize_path = os.path.join(args.output_dir, "visualize")
    reports_path = os.path.join(args.output_dir, "reports")

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if not os.path.exists(reports_path):
        os.mkdir(reports_path)

    if args.visualize:
        clear_spatial_classification_folders(visualize_path)
        # Create output folder structure (if not exists already)
        create_visual_spatial_classification_folders(visualize_path)

    workload = precalculate_workload(test_batches_paths, args.faulty_path)

    log.info(f"Found {workload} tensors to analyze")

    log.getLogger().setLevel(log.WARN)

    global_report = OrderedDict()
    with tqdm(total=workload) as prog_bar:
        for batch_path in sorted(test_batches_paths):
            golden_path = os.path.join(batch_path, args.golden_path)
            batch_name = os.path.basename(batch_path)

            if not os.path.exists(golden_path):
                print(
                    f"Skipping {batch_name} batch since it does not contain golden tensor"
                )

            # Load golden from file
            try:
                golden: np.ndarray = np.load(golden_path)
            except:
                log.error(f"Skipping {batch_name} batch. Could not read golden")
                continue

            if golden is not None and len(golden.shape) != 4:
                log.error(
                    f"Skipping {batch_name} batch. Dimension of golden not supported {golden.shape}"
                )
                continue

            # Remap coordinates using Coordinate namedtuple
            golden_shape = map_to_coordinates(golden.shape, layout)
            log.info(f"Golden tensor ({args.golden_path}) loaded. Shape {golden_shape}")

            faulty_path = os.path.join(batch_path, args.faulty_path)

            if not os.path.exists(faulty_path):
                log.warning(
                    f"Skipping {batch_name} batch. Could not open path to faulty path"
                )
                continue

            sub_batch_dir = [
                os.path.join(faulty_path, dir)
                for dir in os.listdir(faulty_path)
                if os.path.isdir(os.path.join(faulty_path, dir))
            ]

            global_report[batch_name] = OrderedDict()

            for faulty_path in sorted(sub_batch_dir):
                sub_batch_name = os.path.basename(faulty_path)
                report = analyze_tensor_directory(
                    faulty_path=faulty_path,
                    golden=golden,
                    report_output_path=os.path.join(
                        reports_path, f"report_{batch_name}_{sub_batch_name}.json"
                    ),
                    layout=layout,
                    epsilon=args.epsilon,
                    almost_same=args.almost_same,
                    image_output_dir=visualize_path,
                    visualize_errors=args.visualize,
                    save_report=True,
                    prog_bar=prog_bar,
                )
                global_report[batch_name][sub_batch_name] = report["global_data"]
                del global_report[batch_name][sub_batch_name]["raveled_patterns"]

    with open(os.path.join(args.output_dir, "global_report.json"), "w") as f:
        f.writelines(json.dumps(global_report, indent=2))


if __name__ == "__main__":
    main()
