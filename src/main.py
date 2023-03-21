import argparse
import numpy as np
import logging as log
from collections import OrderedDict, defaultdict
import traceback
import os
from tqdm import tqdm
import json
import sys
from domain_classifier import DomainClass, domain_classification_vect
from visualizer import visualize

REPORT_FILE = "report.json"

from spatial_classifier import (
    SpatialClass,
    clear_spatial_classification_folders,
    create_spatial_classification_folders,
    spatial_classification,
)

from coordinates import TensorLayout, map_to_coordinates


def setup_logging():
    root = log.getLogger()
    root.setLevel(log.INFO)

    handler = log.StreamHandler(sys.stdout)
    handler.setLevel(log.DEBUG)
    formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)


def main():
    parser = argparse.ArgumentParser(
        prog="Tensor Error Classifier",
        description="Compares Faulty Tensors with a golden one, and classifies them",
    )
    parser.add_argument("golden_path", help="A path to the golden .npy file")
    parser.add_argument(
        "faulty_path", help="A path to a folder containing faulty .npy files"
    )
    parser.add_argument(
        "output_dir", help="Path where to write the ouput of the analysis"
    )
    parser.add_argument(
        "-l", "--limit", type=int, help="Limit the number of tensor to process", metavar='N'
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

    # Load golden from file
    try:
        golden : np.ndarray = np.load(args.golden_path)
    except:
        log.error("Could not read golden")
        traceback.print_exc()
        exit(1)

    if golden is not None and len(golden.shape) != 4:
        log.error("Invalid shape of golden tensor")
        exit(1)

    # Remap coordinates using Coordinate namedtuple
    golden_shape = map_to_coordinates(golden.shape, layout)

    if not os.path.exists(args.faulty_path):
        log.error("Could not open path to faulty path")
        exit(1)
    log.info(f"Golden tensor ({args.golden_path}) loaded. Shape {golden_shape}")

    if os.path.exists(os.path.join(args.output_dir, REPORT_FILE)):
        os.remove(os.path.join(args.output_dir, REPORT_FILE))
    clear_spatial_classification_folders(args.output_dir)
    # Create output folder structure (if not exists already)
    create_spatial_classification_folders(args.output_dir)

    # Consider only files ending in .npy
    faulty_files_path = [
        os.path.join(args.faulty_path, entry)
        for entry in os.listdir(args.faulty_path)
        if entry.split(".")[1] == "npy"
    ]

    if args.limit is not None:
        faulty_files_path = faulty_files_path[:args.limit]
    log.info(f"Found {len(faulty_files_path)} faulty tensors to analize")

    # Initialize data structures for reports
    sp_class_count = defaultdict(lambda: 0)
    dom_class_count = np.int64(np.zeros(len(DomainClass)))
    temp_dom_class_count = np.int64(np.zeros(len(DomainClass)))
    tensor_report = OrderedDict()

    # Iterate over all tensor files
    for file_path in tqdm(sorted(faulty_files_path)):
        temp_dom_class_count = np.int64(np.zeros(len(DomainClass)))
        file_name = os.path.basename(file_path).split(".")[0]

        log.debug(f"Opening {file_path}")
        try:
            faulty : np.ndarray = np.load(file_path)
        except:
            log.error(f"Could not read {file_path}")
            sp_class_count["skipped"] += 1
            continue
        
        faulty_shape = map_to_coordinates(faulty.shape, layout)

        if faulty_shape != golden_shape:
            log.warn(
                f"Skipping {file_path}. Invalid shape (Faulty has shape: {faulty_shape}, Golden has shape: {golden_shape})"
            )
            sp_class_count["skipped"] += 1
            continue

        if faulty_shape.N != 1:
            log.warn(f"Skipping {file_path} not supported tensor (Batch dim > 1)")
            sp_class_count["skipped"] += 1
            continue

        tensor_diff = domain_classification_vect(golden, faulty, args.epsilon, args.almost_same)

        cat, counts = np.unique(tensor_diff, return_counts=True)

        for i in range(len(counts)):
            temp_dom_class_count[cat[i]] = counts[i]
        dom_class_count += temp_dom_class_count
            

        sparse_diff_native_coords = list(zip(*np.where(np.abs(golden - faulty) > args.epsilon)))
        raveld_coords, = np.where(np.ravel(tensor_diff) > 1)
        if len(sparse_diff_native_coords) == 0:
            log.info(f"{file_path} has no diffs with golden")
            sp_class_count["masked"] += 1
            continue
        sparse_diff = [
            map_to_coordinates(coords, layout) for coords in sparse_diff_native_coords
        ]
        spatial_class = spatial_classification(sparse_diff)
        sp_class_count[spatial_class.name] += 1

        # faulty_channels = {coord.C for coord in sparse_diff}
        if layout == TensorLayout.NCHW:
            channel_sums = np.sum(tensor_diff, axis=(0,2,3))
        elif layout == TensorLayout.NHWC:
            channel_sums = np.sum(tensor_diff, axis=(0,1,2))

        faulty_channels, = np.where(channel_sums != 0)
        visualize(
            tensor_diff,
            faulty_channels.tolist(),
            layout,
            spatial_class.output_path(args.output_dir, file_name),
            save=True,
            show=False,
        )


        # Per tensor report generator
        tensor_report[file_name] = {
            "class": spatial_class.display_name(),
            "domain_classes": {clz.display_name() : temp_dom_class_count[i].item() for i, clz in enumerate(DomainClass)},
            "corrupted_values": sum(temp_dom_class_count[2:]).item(),
            "corrupted_values_pct": sum(temp_dom_class_count[2:]).item() / golden.size * 100,
            "affected_channels": faulty_channels.tolist(),
            "faulty_channels_count": len(faulty_channels),
            "raveled_pos": raveld_coords.tolist()
        }
    classified_tensors = len(faulty_files_path) - sp_class_count["masked"] - sp_class_count["skipped"]

    if classified_tensors == 0:
        log.warn("No tensors were classified")
        exit(0)
    # Main Report Generation
    main_report = OrderedDict()
    global_data = OrderedDict()

    global_data["tensor_shape"] = golden_shape._asdict()
    global_data["tensor_size"] = golden.size
    global_data["tensors"] = len(faulty_files_path)
    global_data["masked"] = sp_class_count["masked"]
    global_data["skipped"] = sp_class_count["skipped"]
    global_data["classified_tensors"] = classified_tensors
    global_data["corrupted_values"] = sum(dom_class_count[2:]).item()
    global_data["average_corrupted_values_pct"] = float(sum(dom_class_count[2:]).item()) / golden.size / classified_tensors * 100
    global_data["spatial_classes"] = {
        "absolute": {sp_class.display_name() : sp_class_count[sp_class.name] for sp_class in SpatialClass},
        "pct": {sp_class.display_name() : sp_class_count[sp_class.name] / classified_tensors * 100 for sp_class in SpatialClass}
    }
    global_data["domain_classes"] = {
        "absolute": {dom_class.display_name() : dom_class_count[dom_class.value].item() for dom_class in DomainClass},
        "pct": {dom_class.display_name() : float(dom_class_count[dom_class.value].item()) / golden.size / classified_tensors * 100 for dom_class in DomainClass}
    }
    main_report["global_data"] = global_data
    main_report["tensors"] = tensor_report

    report_path = os.path.join(args.output_dir, REPORT_FILE)

    with open(report_path, 'w') as f:
        f.writelines(json.dumps(main_report, indent=2))


if __name__ == "__main__":
    main()
