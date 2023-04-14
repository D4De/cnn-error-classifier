from collections import OrderedDict, defaultdict
import itertools
import json
from operator import itemgetter
import os

from coordinates import TensorLayout, map_to_coordinates
from domain_classifier import DomainClass
from spatial_classifier import SpatialClass, accumulate_max, spatial_classification
from domain_classifier import domain_classification_vect
import logging as log
import numpy as np

from visualizer import visualize


def sort_dict(data: dict, sort_key=lambda x: x[1], reverse=True):
    return OrderedDict(
        sorted(
            [(key, value) for key, value in data.items()], key=sort_key, reverse=reverse
        )
    )


def numpy_coords_to_python_coord(coords: tuple):
    return tuple(coord.item() for coord in coords)


def analyze_tensor(
    file_path: str,
    golden: np.ndarray,
    layout: TensorLayout,
    epsilon: float,
    almost_same: bool,
    visualize_errors: bool,
    output_dir: str,
    metadata: dict = {},
):
    # Initialization
    temp_dom_class_count = np.int64(np.zeros(len(DomainClass)))

    # Opening file
    file_name = os.path.basename(file_path).split(".")[0]

    log.debug(f"Opening {file_path}")
    try:
        faulty: np.ndarray = np.load(file_path)
    except:
        log.error(f"Could not read {file_path}")
        return "skipped", {}

    faulty_shape = map_to_coordinates(faulty.shape, layout)
    golden_shape = map_to_coordinates(golden.shape, layout)

    # Check shape correctness
    if faulty_shape != golden_shape:
        log.warn(
            f"Skipping {file_path}. Invalid shape (Faulty has shape: {faulty_shape}, Golden has shape: {golden_shape})"
        )
        return "skipped", {}

    if faulty_shape.N != 1:
        log.warn(f"Skipping {file_path} not supported tensor (Batch dim > 1)")
        return "skipped", {}

    # Execute error domain classification on the whole tensor
    tensor_diff = domain_classification_vect(golden, faulty, epsilon, almost_same)

    # Count occourences of each domain class
    cat, counts = np.unique(tensor_diff, return_counts=True)

    for i in range(len(counts)):
        temp_dom_class_count[cat[i]] = counts[i]

    # Generate a list of all coordinates where a difference is observed (Sparse matrix)
    sparse_diff_native_coords = list(zip(*np.where(tensor_diff > 0)))
    (raveld_coords,) = np.where(np.ravel(tensor_diff) > 1)

    # No diff = masked
    if len(sparse_diff_native_coords) == 0:
        log.info(f"{file_path} has no diffs with golden")
        return "masked", {}
    sparse_diff = [
        map_to_coordinates(numpy_coords_to_python_coord(coords), layout)
        for coords in sparse_diff_native_coords
    ]

    # Pefmorm spatial classifcation
    spatial_class, pattern_params = spatial_classification(sparse_diff, golden_shape)

    # Get faults for each channel
    # faulty_channels = {coord.C for coord in sparse_diff}
    if layout == TensorLayout.NCHW:
        channel_sums = np.sum(tensor_diff, axis=(0, 2, 3))
    elif layout == TensorLayout.NHWC:
        channel_sums = np.sum(tensor_diff, axis=(0, 1, 2))

    (faulty_channels,) = np.where(channel_sums != 0)

    if visualize_errors:
        # Create error visualization
        suptitle_id = metadata["test_campaign"] if "test_campaign" in metadata else metadata["test_id"] if "test_id" in metadata else '?'
        filt_size = map_to_coordinates(metadata["filter_size"], layout)
        visualize(
            tensor_diff,
            faulty_channels.tolist(),
            layout,
            spatial_class.output_path(output_dir, file_name),
            save=True,
            show=False,
            suptitile=f'conv_{suptitle_id} {metadata["igid"]} {metadata["bfm"]} {golden_shape.C}x{golden_shape.H}x{golden_shape.W} filt: {filt_size.C}x{filt_size.H}x{filt_size.W}',
            invalidate=True,
        )

    raveled_offsets = (raveld_coords - raveld_coords[0]).tolist()
    # Per tensor report generator
    return spatial_class.display_name(), {
        "class": spatial_class.display_name(),
        "class_params": pattern_params,
        "domain_classes": {
            clz.display_name(): temp_dom_class_count[i].item()
            for i, clz in enumerate(DomainClass)
        },
        "corrupted_values": sum(temp_dom_class_count[2:]).item(),
        "corrupted_values_pct": sum(temp_dom_class_count[2:]).item()
        / golden.size
        * 100,
        "affected_channels": faulty_channels.tolist(),
        "faulty_channels_count": len(faulty_channels),
        "raveled_start": raveld_coords[0].item(),
        "raveled_offsets": sorted(raveled_offsets),
        "error_pattern": pattern_params["error_pattern"],
        "block_align": pattern_params["align"]
        if spatial_class == SpatialClass.CHANNEL_ALIGNED_SAME_BLOCK
        else None,
    }


def analyze_tensor_directory(
    faulty_path: str,
    golden: np.ndarray,
    report_output_path: str,
    layout: TensorLayout,
    epsilon: int,
    almost_same: bool,
    image_output_dir: str = "",
    visualize_errors: bool = False,
    save_report: bool = True,
    metadata: dict = {},
    prog_bar=None,
):
    golden_shape = map_to_coordinates(golden.shape, layout)

    # Get list of all file ending in .npy
    faulty_files_path = [
        os.path.join(faulty_path, entry)
        for entry in os.listdir(faulty_path)
        if entry.split(".")[1] == "npy"
    ]

    log.info(f"Found {len(faulty_files_path)} faulty tensors to analize")

    # Initialize data structures for reports
    sp_class_count = defaultdict(lambda: 0)
    dom_class_count = defaultdict(lambda: 0)
    tensor_report = OrderedDict()
    corrupt_cardinality = defaultdict(lambda: 0)
    raveled_patterns = defaultdict(lambda: 0)
    error_patterns = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    class_params = defaultdict(lambda: defaultdict(list))
    corrupted_values = 0

    # Iterate over all tensor files
    for file_path in sorted(faulty_files_path):
        sp_class, tensor_dict = analyze_tensor(
            file_path=file_path,
            golden=golden,
            layout=layout,
            epsilon=epsilon,
            almost_same=almost_same,
            visualize_errors=visualize_errors,
            output_dir=image_output_dir,
            metadata=metadata,
        )
        if prog_bar is not None:
            prog_bar.update(1)
        sp_class_count[sp_class] += 1

        if sp_class != "masked" and sp_class != "skipped":
            error_pattern = str(tensor_dict["error_pattern"])
            corr_val_count = tensor_dict["corrupted_values"]

            tensor_report[os.path.basename(file_path)] = tensor_dict
            raveled_patterns[str(tensor_dict["raveled_offsets"])] += 1
            corrupt_cardinality[tensor_dict["corrupted_values"]] += 1
            corrupted_values += tensor_dict["corrupted_values"]
            error_patterns[corr_val_count][sp_class][error_pattern] += 1
            class_params[corr_val_count][sp_class] = accumulate_max(
                class_params[corr_val_count][sp_class],
                tensor_dict["class_params"]["MAX"] if "MAX" in tensor_dict["class_params"] else [],
            )
            for dom_class, freq in tensor_dict["domain_classes"].items():
                dom_class_count[dom_class] += freq

    classified_tensors = (
        len(faulty_files_path) - sp_class_count["masked"] - sp_class_count["skipped"]
    )

    if classified_tensors == 0:
        log.warn("No tensors were classified")
        return {}
    # Main Report Generation
    main_report = OrderedDict()
    global_data = OrderedDict()

    global_data["tensor_shape"] = golden_shape._asdict()
    global_data["tensor_size"] = golden.size
    global_data["tensors"] = len(faulty_files_path)
    global_data["masked"] = sp_class_count["masked"]
    global_data["skipped"] = sp_class_count["skipped"]
    global_data["classified_tensors"] = classified_tensors
    global_data["corrupted_values"] = corrupted_values
    global_data["error_patterns"] = error_patterns
    global_data["class_params"] = class_params
    global_data["average_corrupted_values_pct"] = (
        float(corrupted_values) / golden.size / classified_tensors * 100
    )
    global_data["spatial_classes"] = sort_dict(
        {
            sp_class.display_name(): sp_class_count[sp_class.display_name()]
            for sp_class in SpatialClass
        }
    )
    global_data["spatial_classes_pct"] = sort_dict(
        {
            sp_class.display_name(): sp_class_count[sp_class.display_name()]
            for sp_class in SpatialClass
        }
    )

    global_data["domain_classes"] = sort_dict(dom_class_count)
    global_data["domain_classes_pct"] = sort_dict(
        {
            dom_class: float(freq) / golden.size / classified_tensors * 100
            for dom_class, freq in dom_class_count.items()
        }
    )
    global_data["corrupt_cardinality"] = sort_dict(corrupt_cardinality)

    global_data["corrupt_cardinality_pct"] = (
        {
            n_errors: freq / global_data["corrupted_values"] * 100
            for n_errors, freq in corrupt_cardinality.items()
        },
    )
    global_data["raveled_patterns"] = OrderedDict(
        sorted(
            [
                (pattern, freq)
                for pattern, freq in raveled_patterns.items()
                if freq > 5 / 100 * len(raveled_patterns)
            ],
            key=lambda x: x[1],
            reverse=True,
        )
    )
    main_report["metadata"] = metadata
    main_report["global_data"] = global_data
    main_report["tensors"] = tensor_report

    if save_report:
        report_path = report_output_path

        with open(report_path, "w") as f:
            f.writelines(json.dumps(main_report, indent=2))

    return main_report
